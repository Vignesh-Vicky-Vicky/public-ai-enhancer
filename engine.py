"""GPU photographic restoration and micro-detail engine.
Stage 1: Dedicated Real-ESRGAN super-resolution/restoration directly at target resolution (1x, 2x, 4x).
Stage 2 (Optional): High-frequency diffusion micro-detail generator (pores, fine hairs, iris detail, fabric weave).
Stage 3: Structure-preserving frequency blend anchoring identity and geometry while transferring rich high-frequency textures.
Accurate end-to-end 0-100% progress weighting across all pipeline phases.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

import math
import threading
import time
from pathlib import Path
import numpy as np
from PIL import Image

from sr_model import RRDBNet, tile_sr_inference

ROOT = Path(__file__).resolve().parent

DETAIL_PROMPT = (
    'masterpiece 8k uhd hyperrealistic 35mm photograph, tack sharp focus, authentic human skin texture, '
    'visible fine skin pores, natural cellular skin relief, delicate facial peach fuzz, subtle fine expression lines, '
    'realistic skin micro-irregularities and uneven microscopic texture, subtle tonal variation, '
    'individual hair strands, crisp eyelashes and eyebrow hairs, sharp iris limbal ring, intricate iris stroma, '
    'natural lip vermilion creases, authentic textile weave, individual cloth fibers, crisp stitching, '
    'material-specific realistic surfaces, cinematic directional lighting, RAW photo quality'
)
NEGATIVE_PROMPT = (
    'smooth, airbrushed, plastic doll skin, waxy complexion, porcelain skin, blurred, filtered, '
    'denoised, flat shading, cartoon, illustration, painting, drawing, lowres, soft focus, out of focus, '
    'over-smoothed, artificial skin, makeup blur, muddy textures, artifacts'
)


def encode_full_prompt(pipe, guidance):
    """Encode prompt chunks to prevent truncation beyond 77 tokens."""
    import torch
    positive = DETAIL_PROMPT + ('\nAdditional details: ' + guidance.strip() if guidance.strip() and guidance.strip() != DETAIL_PROMPT else '')
    negative = NEGATIVE_PROMPT
    tokenizer = pipe.tokenizer
    capacity = tokenizer.model_max_length - 2
    positive_ids = tokenizer(positive, add_special_tokens=False, truncation=False)['input_ids']
    negative_ids = tokenizer(negative, add_special_tokens=False, truncation=False)['input_ids']
    chunks = max(math.ceil(len(positive_ids)/capacity), math.ceil(len(negative_ids)/capacity), 1)
    def encode(ids):
        encoded = []
        for index in range(chunks):
            content = ids[index*capacity:(index+1)*capacity]
            tokens = [tokenizer.bos_token_id] + content + [tokenizer.eos_token_id]
            tokens += [tokenizer.pad_token_id] * (tokenizer.model_max_length-len(tokens))
            tensor = torch.tensor([tokens], device='cuda', dtype=torch.long)
            encoded.append(pipe.text_encoder(tensor)[0])
        return torch.cat(encoded, dim=1)
    return encode(positive_ids), encode(negative_ids), positive


PRESETS = {'Detail · 896': (896, 6), 'Fast · 512': (512, 8), 'Balanced · 640': (640, 10), 'Fine · 768': (768, 12)}
EFFICIENT_MODES = {'Fast Detail': (0, 0, 0.0), 'Maximum Detail': (768, 8, 4.0)}


def tile_starts(length, tile=512, overlap=96):
    if length <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return sorted(set(starts))


class Cancelled(RuntimeError):
    pass


def smooth_gpu(t, passes=1):
    """Gaussian smoothing kernel for frequency separation (supports 1 or 3 channels)."""
    import torch
    import torch.nn.functional as F
    channels = t.shape[1]
    kernel = torch.tensor([1, 4, 6, 4, 1], device=t.device, dtype=t.dtype) / 16.0
    k_h = kernel.view(1, 1, 1, 5).expand(channels, 1, 1, 5)
    k_v = kernel.view(1, 1, 5, 1).expand(channels, 1, 5, 1)
    out = t
    for _ in range(passes):
        out = F.conv2d(F.pad(out, (2, 2, 0, 0), mode='replicate'), k_h, groups=channels)
        out = F.conv2d(F.pad(out, (0, 0, 2, 2), mode='replicate'), k_v, groups=channels)
    return out


def clarity_gpu(original, strength):
    """Luminance micro-contrast clarity enhancing fine high frequencies."""
    if strength <= 0:
        return original
    import torch
    base = smooth_gpu(original, passes=1)
    fine = original - base
    medium = base - smooth_gpu(base, passes=2)
    delta = (fine * 1.0 + medium * 0.5).mean(1, keepdim=True)
    delta = torch.sign(delta) * (delta.abs() - 0.001).clamp(min=0)
    return (original + (delta * strength)).clamp(0, 1)


def detect_material_masks(base_tensor):
    """Detect material zones using colorimetry and local gradient statistics on GPU.
    Returns:
      skin_mask: [0, 1] soft mask for human skin tones
      edge_mask: [0, 1] soft mask for sharp high-contrast structural edges/hair/eyes
    """
    import torch
    import torch.nn.functional as F

    r, g, b = base_tensor[:, 0:1], base_tensor[:, 1:2], base_tensor[:, 2:3]
    # Skin tone detection in normalized RGB space:
    # Typical skin: R > G > B, (R - G) > 0.03, (R + G + B) in realistic skin range
    skin_cond1 = (r > g) & (g > b)
    skin_cond2 = (r - g > 0.02) & (r - b > 0.04)
    skin_cond3 = (r > 0.15) & (r < 0.95) & (b < 0.82)
    raw_skin = (skin_cond1 & skin_cond2 & skin_cond3).float()

    # Smooth the skin mask to create organic, gradual transitions
    skin_mask = smooth_gpu(raw_skin, passes=3)

    # Detect high-contrast edges (hair strands, eyebrows, eyelashes, iris boundary, textile edges)
    lum = (base_tensor * base_tensor.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)).sum(1, keepdim=True)
    kx = base_tensor.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8.0
    ky = base_tensor.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3) / 8.0
    gx = F.conv2d(F.pad(lum, (1, 1, 1, 1), mode='replicate'), kx)
    gy = F.conv2d(F.pad(lum, (1, 1, 1, 1), mode='replicate'), ky)
    grad = torch.sqrt(gx * gx + gy * gy)
    edge_mask = (grad * 12.0).clamp(0, 1)

    return skin_mask.clamp(0, 1), edge_mask.clamp(0, 1)


def structure_preserving_blend(base_reference, enhanced_texture, amount=1.0):
    """Material-aware structure-preserving frequency blend.
    Strictly preserves identity, macro face geometry, lighting, and global color,
    while transferring material-tailored photographic texture:
      - Skin: receives genuine fine pores, cellular relief, subtle tonal variation, and peach fuzz
      - Hair/Eyes: preserves crisp strand sharpness without harsh over-accentuation
      - Clothing/Surfaces: authentic fabric fibers, weave, and appropriate physical texture
    """
    import torch
    import torch.nn.functional as F

    def luminance(t):
        return (t * t.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)).sum(1, keepdim=True)

    # 1. Multi-scale frequency separation
    # base_low: macro geometry, lighting, and expression
    base_low = smooth_gpu(base_reference, passes=2)
    enh_low = smooth_gpu(enhanced_texture, passes=2)

    # High frequencies (pores, hairs, cloth weave, fine micro-relief)
    base_high = base_reference - base_low
    enh_high = enhanced_texture - enh_low

    # 2. Material-aware spatial guidance masks
    skin_mask, edge_mask = detect_material_masks(base_reference)

    # 3. Macro consistency check: safeguard against spatial drift or hallucinated structures
    diff_lum = luminance(enh_low - base_low).abs()
    confidence = torch.exp(-diff_lum * 6.0)

    # 4. Material-adaptive modulation:
    # Hair and high-contrast lines already have strong gradient energy; don't over-sharpen them.
    # Skin has lower natural gradient energy and needs full pore/micro-relief transfer + subtle micro-tonal variation.
    # Weave and textiles naturally occupy mid-to-high frequencies with moderate edge energy.
    material_gain = 1.0 + (skin_mask * 0.45) - (edge_mask * 0.20)
    material_gain = material_gain.clamp(0.6, 1.5)

    novel_high = (enh_high - base_high) * confidence
    blended_high = base_high + novel_high * (amount * material_gain)

    # 5. Extract natural micro-tonal variation on skin (subtle cellular relief & microscopic unevenness)
    # This prevents skin from looking artificially waxy or flat beneath the pores
    enh_mid = smooth_gpu(enhanced_texture, passes=1) - enh_low
    base_mid = smooth_gpu(base_reference, passes=1) - base_low
    novel_mid = (enh_mid - base_mid) * confidence * skin_mask * 0.35 * amount
    blended_high = blended_high + novel_mid

    # 6. Recombine: strictly anchor to base_low (preserves 100% identity, geometry, and color)
    result = base_low + blended_high

    # 7. Guard global color balance: match luminance delta while keeping base chrominance intact
    result_lum = luminance(result)
    base_lum = luminance(base_reference)
    lum_delta = result_lum - base_lum

    final_output = base_reference + (result - base_reference) * 0.85 + lum_delta * 0.15
    return final_output.clamp(0, 1)


# Maintain backward compatibility
def protected_detail(original, source, generated, amount):
    import torch.nn.functional as F
    if generated.shape[-2:] != original.shape[-2:]:
        generated = F.interpolate(generated, size=original.shape[-2:], mode='bilinear', align_corners=False)
    return structure_preserving_blend(original, generated, amount)


def working_size(size, edge):
    w, h = size
    scale = min(1.0, edge / max(w, h))
    return max(8, round(w * scale)), max(8, round(h * scale))


class DetailEngine:
    def __init__(self):
        self.pipe = None
        self.sr_model = None
        self.lock = threading.Lock()
        self.prompt_cache = None

    def load_sr(self, report_sub):
        """Load Real-ESRGAN super-resolution/restoration weights on GPU."""
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU unavailable. An NVIDIA GPU is required.')
        if self.sr_model is not None:
            return self.sr_model

        report_sub('Loading photographic super-resolution weights…', 0.5)
        sr_path = ROOT / 'models' / 'sr' / 'RealESRGAN_x4.pth'
        if not sr_path.exists():
            from huggingface_hub import hf_hub_download
            report_sub('Downloading Real-ESRGAN weights (~64MB)…', 0.2)
            sr_path.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(repo_id='ai-forever/Real-ESRGAN', filename='RealESRGAN_x4.pth', local_dir=str(sr_path.parent), token=False)
            (sr_path.parent / '.ready').write_text('Downloaded successfully\n')

        model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32)
        state = torch.load(str(sr_path), map_location='cpu')
        key = 'params_ema' if 'params_ema' in state else ('params' if 'params' in state else None)
        state = state[key] if key else state
        model.load_state_dict(state, strict=True)
        model.to('cuda').eval()
        self.sr_model = model
        report_sub('Photographic super-resolution model ready on GPU', 1.0)
        return model

    def load_sd(self, report_sub):
        """Load Stable Diffusion 1.5 pipeline on GPU (FP32 safe for GTX 1660 Ti)."""
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU unavailable.')
        torch.set_num_threads(2)
        if self.pipe is not None:
            return self.pipe
        model = ROOT / 'models' / 'sd15'
        if not (model / '.ready').exists():
            raise RuntimeError('Diffusion model is missing. Run download_model.py once with internet access.')
        report_sub('Loading diffusion micro-detail pipeline…', 0.5)
        from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
        class ConsistentPipeline(StableDiffusionImg2ImgPipeline):
            section_noise = None

            def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
                if self.section_noise is None:
                    return super().prepare_latents(image, timestep, batch_size, num_images_per_prompt, dtype, device, generator)
                if batch_size * num_images_per_prompt != 1:
                    raise ValueError('Consistent section generation supports one image at a time.')
                clean = self.vae.encode(image.to(device=device, dtype=dtype)).latent_dist.mode()
                clean = clean * self.vae.config.scaling_factor
                return self.scheduler.add_noise(clean, self.section_noise.to(dtype=dtype), timestep)

        pipe = ConsistentPipeline.from_pretrained(
            str(model), torch_dtype=torch.float32, local_files_only=True,
            use_safetensors=True, safety_checker=None, feature_extractor=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_attention_slicing('auto')
        pipe.set_progress_bar_config(disable=True)
        pipe.to('cuda')
        self.pipe = pipe
        report_sub('Diffusion pipeline ready on GPU', 1.0)
        return pipe

    def run(self, image, *, mode='Fast Detail', preset='Detail · 896', creativity=0.25, amount=1.2, prompt='', seed=42, cancel=None, report=None, clarity=0.35, output_scale=2, time_budget=170):
        started = time.perf_counter()
        import torch
        import torch.nn.functional as F

        if report is None:
            report = lambda text, progress: None
        if cancel is None:
            cancel = threading.Event()

        if not self.lock.acquire(blocking=False):
            raise RuntimeError('An enhancement is already running.')

        try:
            if time_budget is not None and time_budget <= 0:
                raise ValueError('Time budget must be positive or disabled.')
            if not 0.05 <= creativity <= 0.50 or not 0.1 <= amount <= 3.0:
                raise ValueError('Use creativity 0.05–0.50 and texture intensity 0.1–3.0.')
            if not 0 <= clarity <= 2.0:
                raise ValueError('Clarity must be between 0 and 2.0.')
            if output_scale not in (1, 2, 4):
                raise ValueError('Output scale must be 1, 2, or 4.')
            if not torch.cuda.is_available():
                raise RuntimeError('An NVIDIA CUDA GPU is required.')

            torch.set_num_threads(2)
            torch.cuda.reset_peak_memory_stats()
            if image.width * image.height > 16_000_000:
                raise ValueError('Please use an image of 16 megapixels or less.')
            if image.width * image.height * output_scale**2 > 64_000_000:
                raise ValueError('Output exceeds 64 megapixels. Choose a smaller scale or input image.')

            # Job Phase Progress Allocations (True 0.0 to 1.0 weighted progress)
            # Fast Detail:
            #  0.00 - 0.05: Prep & Image to GPU
            #  0.05 - 0.15: Model Load (if not cached)
            #  0.15 - 0.85: Photographic Super-Resolution Reconstruction
            #  0.85 - 0.95: Blending & Micro-Contrast Clarity
            #  0.95 - 1.00: GPU->CPU Transfer and Output PNG Prep
            # Maximum Detail:
            #  0.00 - 0.03: Prep & Image to GPU
            #  0.03 - 0.10: Model Loads (SR & SD)
            #  0.10 - 0.45: Photographic Super-Resolution Reconstruction
            #  0.45 - 0.85: Generative Micro-Detail Diffusion Refinement
            #  0.85 - 0.95: Blending & Micro-Contrast Clarity
            #  0.95 - 1.00: GPU->CPU Transfer and Output PNG Prep
            is_max = (mode == 'Maximum Detail' or mode == 'AI micro-detail at output resolution')

            def check():
                if cancel.is_set():
                    raise Cancelled('Cancelled. No partial image was saved.')
                if time_budget is not None and time.perf_counter() - started > time_budget:
                    raise Cancelled('Time budget reached. Reduce output scale or choose Fast Detail.')

            with torch.inference_mode():
                check()
                report('Preparing image on GPU…', 0.02)
                img_array = np.array(image.convert('RGB'), copy=True)
                orig_tensor = torch.from_numpy(img_array).to('cuda', dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

                target_w = image.width * output_scale
                target_h = image.height * output_scale

                # STAGE 1: Dedicated Super-Resolution & Restoration
                if mode == 'GPU texture boost (no new detail)':
                    report('Scaling image canvas…', 0.20)
                    if output_scale > 1:
                        sr_canvas = F.interpolate(orig_tensor, size=(target_h, target_w), mode='bicubic', align_corners=False).clamp(0, 1)
                    else:
                        sr_canvas = orig_tensor
                    report('Applying clarity on GPU…', 0.70)
                    result = clarity_gpu(sr_canvas, clarity)
                    pipe = None
                    actual_prompt = ''
                else:
                    check()
                    # Loading SR weights
                    sr_model = self.load_sr(lambda txt, p: report(txt, 0.03 + (0.07 if is_max else 0.12) * p))
                    check()

                    sr_start_p = 0.10 if is_max else 0.15
                    sr_end_p = 0.45 if is_max else 0.85

                    def sr_progress(tile_idx, total):
                        check()
                        frac = tile_idx / max(1, total)
                        curr_p = sr_start_p + (sr_end_p - sr_start_p) * frac
                        report(f'Photographic SR reconstruction ({output_scale}×) · section {min(tile_idx+1, total)}/{total}', curr_p)

                    # Real-ESRGAN tiled GPU reconstruction
                    sr_4x = tile_sr_inference(sr_model, orig_tensor, tile_size=384, overlap=32, target_scale=4, progress_cb=sr_progress)
                    check()

                    if output_scale == 4:
                        sr_canvas = sr_4x
                    elif output_scale == 2:
                        sr_canvas = F.interpolate(sr_4x, size=(target_h, target_w), mode='area').clamp(0, 1)
                    else: # 1x
                        sr_canvas = F.interpolate(sr_4x, size=(target_h, target_w), mode='area').clamp(0, 1)

                    # STAGE 2: Optional Generative Micro-Detail Diffusion Refinement
                    pipe = None
                    actual_prompt = ''
                    if is_max:
                        check()
                        pipe = self.load_sd(lambda txt, p: report(txt, 0.45 + 0.05 * p))
                        check()
                        report('Encoding micro-detail guidance on GPU…', 0.51)
                        if self.prompt_cache is None or self.prompt_cache[0] != prompt:
                            self.prompt_cache = (prompt, encode_full_prompt(pipe, prompt))
                        pos_emb, neg_emb, actual_prompt = self.prompt_cache[1]

                        # Working resolution for diffusion refinement (optimized for GTX 1660 Ti speed: 640px edge)
                        edge = 640
                        dw, dh = working_size((target_w, target_h), edge)
                        diff_input = F.interpolate(sr_canvas, size=(dh, dw), mode='bilinear', align_corners=False)

                        # Detect material masks at working resolution to filter out uninformative tiles (e.g. flat background)
                        work_skin_mask, work_edge_mask = detect_material_masks(diff_input)
                        detail_energy_map = (work_skin_mask + work_edge_mask).clamp(0, 1)

                        # Efficient 576px tiles with 48px overlap
                        tile_dim = 576
                        overlap_dim = 48
                        tiled = (dh > tile_dim or dw > tile_dim)
                        all_tiles = [(x, y) for y in tile_starts(dh, tile=tile_dim, overlap=overlap_dim) for x in tile_starts(dw, tile=tile_dim, overlap=overlap_dim)] if tiled else [(0, 0)]

                        # Filter tiles: prioritize tiles that contain skin, facial features, hair, eyes, or clothing textures
                        active_tiles = []
                        for x, y in all_tiles:
                            th = min(tile_dim, dh - y)
                            tw = min(tile_dim, dw - x)
                            tile_score = detail_energy_map[:, :, y:y+th, x:x+tw].mean().item()
                            # If image has multiple tiles, skip pure featureless background tiles
                            if len(all_tiles) > 1 and tile_score < 0.04:
                                continue
                            active_tiles.append((x, y))

                        if not active_tiles:
                            active_tiles = [(0, 0)]

                        diff_output = torch.zeros_like(diff_input)
                        diff_weights = torch.zeros_like(diff_input[:, :1])
                        shared_noise = torch.randn((1, 4, math.ceil(dh/8), math.ceil(dw/8)), device='cuda',
                                                   generator=torch.Generator(device='cuda').manual_seed(seed))

                        # 10 DPMSolver steps with small creativity yields 2-4 fast denoising steps (~3-4x faster per tile)
                        total_steps = 10
                        num_steps = math.ceil(total_steps * max(0.12, min(0.35, creativity)))
                        guidance_scale = 3.5

                        diff_start_p = 0.52
                        diff_end_p = 0.85

                        report(f'Generating micro-detail texture ({len(active_tiles)} section(s), {num_steps} steps)…', diff_start_p)
                        for tile_idx, (x, y) in enumerate(active_tiles):
                            check()
                            th = min(tile_dim, dh - y)
                            tw = min(tile_dim, dw - x)
                            patch = diff_input[:, :, y:y+th, x:x+tw]
                            ph, pw = patch.shape[-2:]
                            padded = F.pad(patch, (0, (-pw) % 8, 0, (-ph) % 8), mode='replicate')
                            pipe.section_noise = shared_noise[:, :, y//8:y//8+math.ceil(ph/8), x//8:x//8+math.ceil(pw/8)]

                            def step_cb(_p, step, _ts, kwargs):
                                check()
                                step_frac = (tile_idx + (step + 1) / num_steps) / len(active_tiles)
                                curr_p = diff_start_p + (diff_end_p - diff_start_p) * step_frac
                                report(f'Micro-detail section {tile_idx+1}/{len(active_tiles)} · step {step+1}/{num_steps}', curr_p)
                                return kwargs

                            gen_patch = pipe(
                                prompt_embeds=pos_emb, negative_prompt_embeds=neg_emb,
                                image=padded, strength=min(0.35, creativity), num_inference_steps=total_steps,
                                guidance_scale=guidance_scale, generator=torch.Generator(device='cuda').manual_seed(seed),
                                output_type='pt', callback_on_step_end=step_cb,
                            ).images[:, :, :ph, :pw]

                            # Color & illumination match to original patch
                            gen_patch = gen_patch + smooth_gpu(patch, passes=2) - smooth_gpu(gen_patch, passes=2)
                            wy = torch.sin(torch.linspace(0.05, math.pi - 0.05, ph, device='cuda')) ** 2
                            wx = torch.sin(torch.linspace(0.05, math.pi - 0.05, pw, device='cuda')) ** 2
                            wgt = (wy[:, None] * wx[None, :])[None, None]
                            diff_output[:, :, y:y+ph, x:x+pw] += gen_patch * wgt
                            diff_weights[:, :, y:y+ph, x:x+pw] += wgt

                        # Blend diff_output with diff_input where weights exist, keep original diff_input elsewhere
                        has_weight = (diff_weights > 1e-6)
                        diff_output = torch.where(has_weight, diff_output / diff_weights.clamp_min(1e-8), diff_input)
                        diff_upscaled = F.interpolate(diff_output, size=(target_h, target_w), mode='bilinear', align_corners=False)

                        # Extract high-frequency generative detail and transfer onto the super-resolved canvas
                        report('Transferring micro-detail textures…', 0.86)
                        diff_high = diff_upscaled - smooth_gpu(diff_upscaled, passes=1)
                        # Material-adaptive injection: skin receives enhanced micro-relief & pores
                        # while hair keeps its current strand quality without harsh over-accentuation
                        skin_mask_inj, edge_mask_inj = detect_material_masks(sr_canvas)
                        diff_gain = 1.0 + (skin_mask_inj * 0.40) - (edge_mask_inj * 0.15)
                        sr_canvas = (sr_canvas + diff_high * (amount * 0.9 * diff_gain.clamp(0.7, 1.4))).clamp(0, 1)

                    # STAGE 3: Final structure-preserving blend & micro-contrast
                    check()
                    report('Applying structure-preserving detail blend on GPU…', 0.88)
                    if output_scale > 1:
                        base_ref = F.interpolate(orig_tensor, size=(target_h, target_w), mode='bicubic', align_corners=False).clamp(0, 1)
                    else:
                        base_ref = orig_tensor

                    blended = structure_preserving_blend(base_ref, sr_canvas, amount=amount)

                    # Micro-contrast clarity
                    if clarity > 0:
                        report('Enhancing micro-contrast clarity…', 0.92)
                        result = clarity_gpu(blended, clarity)
                    else:
                        result = blended

                check()
                report('Transferring image from GPU and preparing output…', 0.96)
                result = result.clamp(0, 1)
                array = (result.squeeze(0).permute(1, 2, 0) * 255.0).round().to(torch.uint8).cpu().numpy()
                torch.cuda.synchronize()

                elapsed = time.perf_counter() - started
                info = dict(
                    pipeline_version=10,
                    architecture='RealESRGAN-RRDBNet + Optional-Diffusion-MicroDetail',
                    mode=mode,
                    preset=preset,
                    creativity=creativity,
                    amount=amount,
                    clarity=clarity,
                    output_scale=output_scale,
                    output_size=[target_w, target_h],
                    prompt=actual_prompt if pipe is not None else '',
                    seed=seed,
                    seconds=round(elapsed, 2),
                    gpu=torch.cuda.get_device_name(0),
                    peak_vram_mb=round(torch.cuda.max_memory_allocated() / 1024**2),
                    working_size=[target_w, target_h],
                )
                report(f'Finished in {elapsed:.1f}s', 1.0)
                return Image.fromarray(array), info

        except torch.cuda.OutOfMemoryError as exc:
            self.pipe = None
            self.sr_model = None
            self.prompt_cache = None
            raise RuntimeError('GPU memory is full. Fast Detail uses minimal memory; try Fast Detail or reduce output scale.') from exc
        finally:
            if self.pipe is not None:
                self.pipe.section_noise = None
            self.lock.release()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

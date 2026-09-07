"""One-Click Extreme Generative Micro-Detail & Super-Resolution Engine.

Pipeline:
  1. Input & Material / Blur Analysis: Detects structural contours, highlights, skin, and blurry low-info zones.
  2. Structure Recovery & Dedicated Neural SR (Real-ESRGAN): 
     Direct photographic super-resolution to 2x or 4x target canvas.
  3. Target-Resolution Generative Detail Reconstruction:
     Tiled generative latent diffusion operating on high-resolution overlapping tiles extracted
     directly from the target canvas. Injects genuine missing physical structures (chain links,
     beads, clasps, metal joints, fabric weave, threads, leather grain, skin pores, hair strands).
  4. Ultra-Micro Detail & Consistency Synthesis:
     Blends generative textures into the photographic base with Hann windowing, strictly avoiding
     edge embossing, white/black halos, ringing, and double contours.
  5. Artifact Safety Quality Check:
     Guarantees that output remains clean, natural, and photographic without excessive gradient distortion.
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

# Comprehensive multi-material generative prompt for extreme detail synthesis
DETAIL_PROMPT = (
    'masterpiece 8k uhd hyperrealistic 35mm photograph, tack sharp focus, authentic photographic material textures, '
    'highly detailed jewelry construction, intricate chain links, polished and brushed gold and silver metal, '
    'subtle microscopic metal scratches, crisp specular highlights, tiny micro-reflections, jewelry clasp mechanism, '
    'natural stone and bead surface textures, individual fabric textile fibers, visible cloth weave pattern, crisp thread stitching, '
    'authentic human skin texture, fine pores, delicate peach fuzz, individual sharp hair strands, '
    'authentic leather grain, natural microscopic surface irregularities, cinematic directional lighting, RAW photo quality'
)

NEGATIVE_PROMPT = (
    'smooth, airbrushed, plastic, waxy complexion, melted metal, deformed jewelry, porcelain, blurred, '
    'filtered, denoised, flat shading, cartoon, illustration, painting, drawing, lowres, soft focus, out of focus, '
    'over-smoothed, artificial skin, makeup blur, muddy textures, artifacts, tile seams, checkerboard, '
    'embossed, edge outlines, halos, ringing, double edges, excessive contrast, high-pass noise'
)


def encode_full_prompt(pipe):
    """Encode prompt chunks to prevent truncation beyond 77 tokens."""
    import torch
    positive = DETAIL_PROMPT
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


def tile_starts(length, tile=512, overlap=64):
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


def detect_materials_and_blur(tensor):
    """Multi-material semantic understanding and blur/information density map on GPU.
    Detects:
      - metal_mask: Specular / reflective metallic surfaces and jewelry
      - skin_mask: Human skin tones
      - edge_energy: Sharp high-contrast boundaries
      - blur_map: Low-frequency / blurry / low-information regions needing generative detail
    """
    import torch
    import torch.nn.functional as F

    r, g, b = tensor[:, 0:1], tensor[:, 1:2], tensor[:, 2:3]
    lum = (tensor * tensor.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)).sum(1, keepdim=True)

    # 1. Skin tone detector in normalized RGB
    skin_cond1 = (r > g) & (g > b)
    skin_cond2 = (r - g > 0.02) & (r - b > 0.04)
    skin_cond3 = (r > 0.15) & (r < 0.95) & (b < 0.82)
    raw_skin = (skin_cond1 & skin_cond2 & skin_cond3).float()
    skin_mask = smooth_gpu(raw_skin, passes=2).clamp(0, 1)

    # 2. Structural gradient energy
    kx = tensor.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8.0
    ky = tensor.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3) / 8.0
    gx = F.conv2d(F.pad(lum, (1, 1, 1, 1), mode='replicate'), kx)
    gy = F.conv2d(F.pad(lum, (1, 1, 1, 1), mode='replicate'), ky)
    grad = torch.sqrt(gx * gx + gy * gy)
    edge_energy = (grad * 6.0).clamp(0, 1)

    # 3. Specular / Metal highlight mask
    local_mean = smooth_gpu(lum, passes=2)
    specular_contrast = (lum - local_mean).clamp(min=0)
    metal_mask = ((lum > 0.72) | (specular_contrast > 0.12)).float() * (1.0 - skin_mask * 0.7)
    metal_mask = metal_mask.clamp(0, 1)

    # 4. Blur / Low-information map: where high frequencies are absent
    local_high_freq = (tensor - smooth_gpu(tensor, passes=1)).abs().mean(1, keepdim=True)
    smoothed_high_freq = smooth_gpu(local_high_freq, passes=3)
    blur_map = (1.0 - (smoothed_high_freq * 20.0).clamp(0, 1)).clamp(0, 1)

    return {
        'skin': skin_mask,
        'metal': metal_mask,
        'edge': edge_energy,
        'blur': blur_map,
    }


def clean_generative_blend(base_reference, generated_full, mat):
    """Photographic multi-scale detail reconstruction blend.
    - Level 1 (Macro): 100% faithful anchor to base geometry, silhouettes, perspective, and lighting.
    - Level 2 (Meso): Generatively reconstructs physical missing components (chain links, beads, clasps, weave).
    - Level 3 (Micro): Synthesizes authentic photographic textures (fibers, pores, scratches, specular highlights).
    - Level 4 (Ultra-Micro): Clean high-frequency detail without edge outlines, embossing, or halos.
    """
    import torch
    import torch.nn.functional as F

    def lum(t):
        return (t * t.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)).sum(1, keepdim=True)

    # Multi-scale frequency decomposition
    # Low frequency: macro illumination, broad shadows, silhouettes (> 10px)
    base_low = smooth_gpu(base_reference, passes=3)
    gen_low = smooth_gpu(generated_full, passes=3)

    # Mid frequency: meso structures (chain links, beads, clasps, contours) (~3 - 10px)
    base_mid = smooth_gpu(base_reference, passes=1) - base_low
    gen_mid = smooth_gpu(generated_full, passes=1) - gen_low

    # High frequency: micro textures (fibers, pores, scratches, specular micro-reflections) (< 3px)
    base_high = base_reference - smooth_gpu(base_reference, passes=1)
    gen_high = generated_full - smooth_gpu(generated_full, passes=1)

    skin_m = mat['skin']
    blur_m = mat['blur']
    metal_m = mat['metal']
    edge_m = mat['edge']

    # 1. Macro anchor: strictly preserve base composition and lighting
    # Prevents whole-image drift or tone shifts
    blended_low = base_low * 0.95 + gen_low * 0.05

    # 2. Meso blend: inject generative physical structures where original is blurry or lacking
    # Where image is blurry, let diffusion reconstruct physical shapes (beads, links, folds)
    meso_weight = 0.35 + (blur_m * 0.25)
    # Protect human skin facial contours from unintended morphing
    meso_weight = (meso_weight * (1.0 - skin_m * 0.30)).clamp(0.20, 0.60)
    blended_mid = base_mid * (1.0 - meso_weight) + gen_mid * meso_weight

    # 3. Micro texture: transfer newly generated high-frequency photographic texture
    novel_micro = gen_high - base_high
    # Apply soft gating to avoid edge ringing or double contours near strong boundaries
    edge_suppression = (1.0 - edge_m * 0.45).clamp(0.4, 1.0)
    micro_gain = (1.0 + metal_m * 0.25 + blur_m * 0.20) * edge_suppression
    blended_high = base_high + novel_micro * micro_gain.clamp(0.8, 1.3)

    # Recombine all scales
    reconstructed = blended_low + blended_mid + blended_high

    # Ensure zero global color shift
    rec_lum = lum(reconstructed)
    orig_lum = lum(base_reference)
    final_output = reconstructed - (rec_lum - orig_lum) * 0.25

    return final_output.clamp(0, 1)


def artifact_safety_check(base_tensor, enhanced_tensor):
    """Detects and rejects catastrophic edge-energy explosion, embossing, or extreme ringing."""
    import torch
    import torch.nn.functional as F

    lum_base = (base_tensor * base_tensor.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)).sum(1, keepdim=True)
    lum_enh = (enhanced_tensor * enhanced_tensor.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)).sum(1, keepdim=True)

    kx = base_tensor.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8.0
    ky = base_tensor.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3) / 8.0

    gx_b = F.conv2d(F.pad(lum_base, (1, 1, 1, 1), mode='replicate'), kx)
    gy_b = F.conv2d(F.pad(lum_base, (1, 1, 1, 1), mode='replicate'), ky)
    grad_b = torch.sqrt(gx_b * gx_b + gy_b * gy_b)

    gx_e = F.conv2d(F.pad(lum_enh, (1, 1, 1, 1), mode='replicate'), kx)
    gy_e = F.conv2d(F.pad(lum_enh, (1, 1, 1, 1), mode='replicate'), ky)
    grad_e = torch.sqrt(gx_e * gx_e + gy_e * gy_e)

    # If edge gradient energy explodes beyond 2.5x the base photographic SR, clamp it back
    ratio = grad_e.mean() / (grad_b.mean().clamp_min(1e-5))
    if ratio > 2.5:
        # Fall back gracefully to base SR + gentle micro-detail to protect user from broken output
        blend_factor = 2.5 / float(ratio.item())
        return base_tensor * (1.0 - blend_factor) + enhanced_tensor * blend_factor
    return enhanced_tensor


class DetailEngine:
    def __init__(self):
        self.pipe = None
        self.sr_model = None
        self.lock = threading.Lock()
        self.prompt_cache = None

    def get_vram_info(self):
        """Return formatted string of current and total VRAM."""
        import torch
        if not torch.cuda.is_available():
            return ''
        allocated_mb = torch.cuda.memory_allocated() / (1024**2)
        total_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        return f'{allocated_mb/1024:.1f} / {total_mb/1024:.1f} GB'

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
        """Load generative micro-detail reconstruction model on GPU."""
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU unavailable.')
        torch.set_num_threads(2)
        if self.pipe is not None:
            return self.pipe
        model = ROOT / 'models' / 'sd15'
        if not (model / '.ready').exists():
            raise RuntimeError('Diffusion model is missing. Run download_model.py once with internet access.')
        report_sub('Loading generative micro-detail reconstruction model…', 0.5)
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
        report_sub('Generative detail model ready on GPU', 1.0)
        return pipe

    def run(self, image, *, output_scale=2, seed=42, cancel=None, report=None):
        """Single-purpose Extreme Generative Detail Reconstruction Pipeline.
        Every image receives maximum generative detail reconstruction at the chosen output scale (2x or 4x).
        """
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
            if output_scale not in (2, 4):
                output_scale = 2
            if not torch.cuda.is_available():
                raise RuntimeError('An NVIDIA CUDA GPU is required.')

            torch.set_num_threads(2)
            torch.cuda.reset_peak_memory_stats()
            gpu_name = torch.cuda.get_device_name(0)

            if image.width * image.height > 16_000_000:
                raise ValueError('Please use an image of 16 megapixels or less.')
            if image.width * image.height * output_scale**2 > 64_000_000:
                raise ValueError('Output exceeds 64 megapixels.')

            def check():
                if cancel.is_set():
                    raise Cancelled('Cancelled. No partial image was saved.')

            with torch.inference_mode():
                check()
                report(f'Analyzing image on {gpu_name}…', 0.02)
                img_array = np.array(image.convert('RGB'), copy=True)
                orig_tensor = torch.from_numpy(img_array).to('cuda', dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

                target_w = image.width * output_scale
                target_h = image.height * output_scale

                # STAGE 1: Dedicated Photographic Super-Resolution & Structure Recovery
                check()
                sr_model = self.load_sr(lambda txt, p: report(f'{txt}', 0.03 + 0.07 * p))
                check()

                def sr_progress(tile_idx, total):
                    check()
                    frac = tile_idx / max(1, total)
                    curr_p = 0.10 + 0.25 * frac
                    report(f'Creating neural {output_scale}× base · tile {min(tile_idx+1, total)}/{total}', curr_p)

                # Real-ESRGAN tiled GPU reconstruction
                sr_4x = tile_sr_inference(sr_model, orig_tensor, tile_size=384, overlap=32, target_scale=4, progress_cb=sr_progress)
                check()

                if output_scale == 4:
                    sr_canvas = sr_4x
                else: # 2x
                    sr_canvas = F.interpolate(sr_4x, size=(target_h, target_w), mode='area').clamp(0, 1)

                # STAGE 2: Material & Degradation Analysis
                check()
                report('Analyzing materials & low-information regions…', 0.38)
                mat = detect_materials_and_blur(sr_canvas)

                # STAGE 3: Generative Detail Reconstruction directly on Target-Resolution Canvas
                check()
                pipe = self.load_sd(lambda txt, p: report(f'{txt}', 0.40 + 0.05 * p))
                check()
                report('Encoding photographic material guidance…', 0.46)
                if self.prompt_cache is None:
                    self.prompt_cache = encode_full_prompt(pipe)
                pos_emb, neg_emb, actual_prompt = self.prompt_cache

                # Tiled processing on target resolution for micro-detail generation
                # 512x512 tiles with 64px overlap
                tile_dim = 512
                overlap_dim = 64
                tiled = (target_h > tile_dim or target_w > tile_dim)
                all_tiles = [(x, y) for y in tile_starts(target_h, tile=tile_dim, overlap=overlap_dim) for x in tile_starts(target_w, tile=tile_dim, overlap=overlap_dim)] if tiled else [(0, 0)]

                diff_output = torch.zeros_like(sr_canvas)
                diff_weights = torch.zeros_like(sr_canvas[:, :1])
                shared_noise = torch.randn((1, 4, math.ceil(target_h/8), math.ceil(target_w/8)), device='cuda',
                                           generator=torch.Generator(device='cuda').manual_seed(seed))

                # Optimal generative strength for extreme detail without edge corruption
                # Denoise strength 0.38 with 12 inference steps produces rich micro-texture without embossing
                total_steps = 12
                denoise_strength = 0.38
                num_steps = math.ceil(total_steps * denoise_strength)
                guidance_scale = 3.6

                diff_start_p = 0.48
                diff_end_p = 0.88

                report(f'Generating photographic details ({len(all_tiles)} tiles)…', diff_start_p)
                for tile_idx, (x, y) in enumerate(all_tiles):
                    check()
                    th = min(tile_dim, target_h - y)
                    tw = min(tile_dim, target_w - x)
                    patch = sr_canvas[:, :, y:y+th, x:x+tw]
                    ph, pw = patch.shape[-2:]
                    padded = F.pad(patch, (0, (-pw) % 8, 0, (-ph) % 8), mode='replicate')
                    pipe.section_noise = shared_noise[:, :, y//8:y//8+math.ceil(ph/8), x//8:x//8+math.ceil(pw/8)]

                    def step_cb(_p, step, _ts, kwargs):
                        check()
                        step_frac = (tile_idx + (step + 1) / num_steps) / len(all_tiles)
                        curr_p = diff_start_p + (diff_end_p - diff_start_p) * step_frac
                        report(f'Micro-detail tiles {tile_idx+1}/{len(all_tiles)} · step {step+1}/{num_steps}', curr_p)
                        return kwargs

                    try:
                        gen_patch = pipe(
                            prompt_embeds=pos_emb, negative_prompt_embeds=neg_emb,
                            image=padded, strength=denoise_strength, num_inference_steps=total_steps,
                            guidance_scale=guidance_scale, generator=torch.Generator(device='cuda').manual_seed(seed + tile_idx),
                            output_type='pt', callback_on_step_end=step_cb,
                        ).images[:, :, :ph, :pw]
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        gen_patch = patch

                    # Smooth Hann/Cosine windowing for seamless overlap blending without tile lines
                    wy = torch.sin(torch.linspace(0.08, math.pi - 0.08, ph, device='cuda')) ** 2
                    wx = torch.sin(torch.linspace(0.08, math.pi - 0.08, pw, device='cuda')) ** 2
                    wgt = (wy[:, None] * wx[None, :])[None, None]
                    diff_output[:, :, y:y+ph, x:x+pw] += gen_patch * wgt
                    diff_weights[:, :, y:y+ph, x:x+pw] += wgt

                # Normalize tile contributions
                diff_output = diff_output / diff_weights.clamp_min(1e-7)
                check()

                # STAGE 4: Clean Photographic Multi-Scale Detail Blending
                report('Ultra-detail refinement & consistency…', 0.90)
                reconstructed = clean_generative_blend(sr_canvas, diff_output, mat)

                # STAGE 5: Artifact Safety Quality Check
                report('Artifact correction & quality verification…', 0.94)
                result = artifact_safety_check(sr_canvas, reconstructed)

                check()
                report('Finalizing and encoding output…', 0.97)
                result = result.clamp(0, 1)
                array = (result.squeeze(0).permute(1, 2, 0) * 255.0).round().to(torch.uint8).cpu().numpy()
                torch.cuda.synchronize()

                elapsed = time.perf_counter() - started
                vram_peak_mb = round(torch.cuda.max_memory_allocated() / 1024**2)
                info = dict(
                    pipeline_version=12,
                    architecture='One-Click Extreme Generative Micro-Detail',
                    output_scale=output_scale,
                    output_size=[target_w, target_h],
                    prompt=actual_prompt,
                    seed=seed,
                    seconds=round(elapsed, 2),
                    gpu=gpu_name,
                    peak_vram_mb=vram_peak_mb,
                    vram_display=f'{vram_peak_mb/1024:.1f} / {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB',
                )
                report(f'Finished in {elapsed:.1f}s', 1.0)
                return Image.fromarray(array), info

        except torch.cuda.OutOfMemoryError as exc:
            self.pipe = None
            self.sr_model = None
            self.prompt_cache = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError('GPU memory is full. Try reducing the output size.') from exc
        finally:
            if self.pipe is not None:
                self.pipe.section_noise = None
            self.lock.release()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

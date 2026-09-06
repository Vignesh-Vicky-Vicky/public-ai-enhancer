"""GPU-only enhancement with overlapping diffusion sections and texture blending."""
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

ROOT = Path(__file__).resolve().parent
DETAIL_PROMPT = '''Upscale the entire image with strong, natural micro-detail enhancement everywhere. Recover realistic skin pores, tiny facial hairs, fine lines, subtle skin texture, individual hair strands, eyelashes, eyebrows, iris detail, lip texture, and small facial features. Reveal dense fabric detail in all clothing, including weave, fibers, stitching, seams, folds, wrinkles, and surface texture. Enhance fine detail in the background and every visible object as well. Keep the original person, face, expression, proportions, pose, clothing, colors, lighting, and composition unchanged. The result should look like a genuinely higher-resolution version of the same photo, not a redesigned or beautified image. Avoid plastic skin, oversharpening, fake patterns, invented objects, altered facial features, or artificial-looking texture.'''
NEGATIVE_PROMPT = 'plastic skin, airbrushed, beauty filter, smooth skin, fake texture, repeating patterns, oversharpened, halos, altered face, altered expression, distorted anatomy, extra objects, changed clothing, changed colors, blurry, text, watermark'


def encode_full_prompt(pipe, guidance):
    """Encode every word in CLIP-sized chunks instead of silently truncating at 77 tokens."""
    import torch
    positive = DETAIL_PROMPT + '\nPrioritize identity and structural consistency over aggressive detail generation. Preserve the original face, expression, clothing design, proportions, lighting and composition.' + ('\nAdditional image guidance: ' + guidance.strip() if guidance.strip() and guidance.strip() != DETAIL_PROMPT else '')
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
EFFICIENT_MODES = {'Fast Detail': (640, 4, 1.0), 'Maximum Detail': (768, 8, 3.0)}


def tile_starts(length, tile=512, overlap=128):
    if length <= tile:
        return [0]
    # Align all sections to the VAE's 8-pixel grid so shared noise lines up.
    return sorted(set(list(range(0, length - tile + 1, tile - overlap)) + [math.ceil((length - tile) / 8) * 8]))


def clarity_gpu(original, strength):
    """Native-resolution luminance clarity; this enhances existing detail."""
    import torch
    base = smooth_gpu(original)
    fine = original - base
    medium = base - smooth_gpu(smooth_gpu(base))
    delta = (fine * .9 + medium * .6).mean(1, keepdim=True)
    delta = torch.sign(delta) * (delta.abs() - .002).clamp(min=0)
    return original + (delta * strength).clamp(-.07, .07)


class Cancelled(RuntimeError):
    pass


def smooth_gpu(t):
    import torch
    import torch.nn.functional as F
    kernel = torch.tensor([1, 4, 6, 4, 1], device=t.device, dtype=t.dtype) / 16
    t = F.conv2d(F.pad(t, (2, 2, 0, 0), mode='replicate'), kernel.view(1, 1, 1, 5).expand(3, 1, 1, 5), groups=3)
    return F.conv2d(F.pad(t, (0, 0, 2, 2), mode='replicate'), kernel.view(1, 1, 5, 1).expand(3, 1, 5, 1), groups=3)


def protected_detail(original, source, generated, amount):
    """Transfer novel fine-scale luminance texture rather than sharpening source edges."""
    import torch
    import torch.nn.functional as F
    def luminance(t):
        return (t * t.new_tensor([.2126, .7152, .0722]).view(1, 3, 1, 1)).sum(1, keepdim=True)
    def local_mean(t):
        return F.avg_pool2d(F.pad(t, (4, 4, 4, 4), mode='replicate'), 9, stride=1)
    def fine_band(t):
        return luminance(t - smooth_gpu(smooth_gpu(t)))
    source_band = fine_band(source)
    generated_band = fine_band(generated)
    # Remove the locally correlated source-edge component. A sharpened copy
    # of the source should not be mistaken for newly synthesized texture.
    covariance = local_mean(generated_band * source_band)
    source_energy = local_mean(source_band.square())
    projection = covariance / (source_energy + 1e-6)
    novel = generated_band - projection * source_band
    novel_energy = local_mean(novel.square()).sqrt()
    # Lift existing AI texture, but never manufacture a random grain layer
    # where the model supplied no texture. Cap amplification and amplitude.
    gain = (.012 / (novel_energy + 1e-6)).clamp(1., 3.)
    presence = (novel_energy / .002).clamp(0., 1.)
    base = smooth_gpu(smooth_gpu(source))
    mismatch = (smooth_gpu(smooth_gpu(generated)) - base).abs().mean(1, keepdim=True)
    confidence = torch.exp(-mismatch * 18)
    edges = F.max_pool2d(source_band.abs(), 5, stride=1, padding=2)
    confidence *= (1 - edges / .10).clamp(0., 1.)
    novel = novel * gain * presence * confidence
    novel = F.interpolate(novel, size=original.shape[-2:], mode='bilinear', align_corners=False)
    addition = (novel * amount).clamp(-.05, .05)
    # Remove broad tone changes introduced by spatial masks and clipping.
    addition_rgb = addition.expand(-1, 3, -1, -1)
    addition = addition_rgb - smooth_gpu(smooth_gpu(addition_rgb))
    return original + addition


def working_size(size, edge):
    w, h = size
    scale = min(1.0, edge / max(w, h))
    return max(8, round(w * scale)), max(8, round(h * scale))


class DetailEngine:
    def __init__(self):
        self.pipe = None
        self.lock = threading.Lock()
        self.prompt_cache = None

    def load(self, report):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU unavailable. Run setup.bat and check the NVIDIA driver. CPU fallback is disabled.')
        torch.set_num_threads(2)
        if self.pipe is not None:
            return self.pipe
        model = ROOT / 'models' / 'sd15'
        if not (model / '.ready').exists():
            raise RuntimeError('AI model is missing. Run setup.bat once with internet access.')
        report('Loading model into GPU memory…', 0)
        from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
        class ConsistentPipeline(StableDiffusionImg2ImgPipeline):
            section_noise = None

            def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
                if self.section_noise is None:
                    return super().prepare_latents(image, timestep, batch_size, num_images_per_prompt, dtype, device, generator)
                if batch_size * num_images_per_prompt != 1:
                    raise ValueError('Consistent section generation supports one image at a time.')
                # Deterministic encoding and coordinate-aligned noise for overlaps.
                clean = self.vae.encode(image.to(device=device, dtype=dtype)).latent_dist.mode()
                clean = clean * self.vae.config.scaling_factor
                return self.scheduler.add_noise(clean, self.section_noise.to(dtype=dtype), timestep)
        # FP32 avoids the half-precision black-image problem on some GTX 16 cards.
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
        return pipe

    def run(self, image, *, mode, preset, creativity, amount, prompt, seed, cancel, report, clarity=0.0, output_scale=1, time_budget=170):
        started = time.perf_counter()
        import torch
        import torch.nn.functional as F
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('An enhancement is already running.')
        try:
            native = mode == 'AI micro-detail at output resolution'
            efficient = mode in EFFICIENT_MODES
            if not efficient and mode not in ('AI micro-detail at output resolution', 'AI detail generation', 'GPU texture boost (no new detail)'):
                raise ValueError('Unknown enhancement mode.')
            if time_budget is not None and time_budget <= 0:
                raise ValueError('Time budget must be positive or disabled.')
            if not .10 <= creativity <= .30 or not .2 <= amount <= 2.0:
                raise ValueError('Use creativity 0.10–0.30 and texture intensity 0.20–2.00.')
            if not 0 <= clarity <= 2:
                raise ValueError('Clarity must be between 0 and 2.')
            if output_scale not in (1, 2, 4):
                raise ValueError('Output scale must be 1, 2, or 4.')
            if not torch.cuda.is_available():
                raise RuntimeError('An NVIDIA CUDA GPU is required. CPU fallback is disabled.')
            torch.set_num_threads(2)
            torch.cuda.reset_peak_memory_stats()
            if image.width * image.height > 16_000_000:
                raise ValueError('Please use an image of 16 megapixels or less.')
            if image.width * image.height * output_scale**2 > 64_000_000:
                raise ValueError('Output exceeds 64 megapixels. Choose a smaller scale or input image.')
            def check():
                if cancel.is_set():
                    raise Cancelled('Cancelled. No partial image was saved.')
                if time_budget is not None and time.perf_counter() - started > time_budget:
                    raise Cancelled('Time budget reached. Reduce output scale, choose the faster AI mode, or disable the time limit. No partial image was saved.')
            with torch.inference_mode():
                pipe = self.load(report) if mode != 'GPU texture boost (no new detail)' else None
                check()
                report('Preparing image on GPU…', 0.05)
                original = torch.from_numpy(np.array(image.convert('RGB'), copy=True)).to('cuda', dtype=torch.float32)
                original = original.permute(2, 0, 1).unsqueeze(0) / 255
                if native and output_scale > 1:
                    # This is the input canvas, not the final result: diffusion operates
                    # on every enlarged pixel afterwards, without a working-size reduction.
                    original = F.interpolate(original, scale_factor=output_scale, mode='bicubic', align_corners=False).clamp(0, 1)
                def smooth(t):
                    return smooth_gpu(t)
                if pipe is None:
                    report('Boosting existing texture on GPU…', 0.5)
                    result = clarity_gpu(original, clarity)
                    work = image.size
                else:
                    report('Encoding full enhancement instructions on GPU…', .06)
                    if self.prompt_cache is None or self.prompt_cache[0] != prompt:
                        self.prompt_cache = (prompt, encode_full_prompt(pipe, prompt))
                    positive_embeddings, negative_embeddings, actual_prompt = self.prompt_cache[1]
                    check()
                    if efficient:
                        edge, actual_steps, guidance_scale = EFFICIENT_MODES[mode]
                    else:
                        edge, actual_steps = PRESETS[preset]
                        guidance_scale = 3.0
                    work = (original.shape[-1], original.shape[-2]) if native else working_size(image.size, edge)
                    w, h = work
                    source = original if native else F.interpolate(original, size=(h, w), mode='bilinear', align_corners=False, antialias=True)
                    steps = math.ceil(actual_steps / creativity)
                    tiled = not efficient and (native or edge >= 896)
                    tiles = [(x, y) for y in tile_starts(h) for x in tile_starts(w)] if tiled else [(0, 0)]
                    report(f'AI work: {w} × {h} pixels · {len(tiles)} pass(es) · {actual_steps} steps', .08)
                    output = torch.zeros_like(source)
                    weights = torch.zeros_like(source[:, :1])
                    shared_noise = torch.randn((1, 4, math.ceil(h/8), math.ceil(w/8)), device='cuda',
                                               generator=torch.Generator(device='cuda').manual_seed(seed))
                    tile_index = 0
                    def callback(_pipe, step, timestep, kwargs):
                        check()
                        progress = (tile_index + (step + 1) / actual_steps) / len(tiles)
                        report(f'Generating detail · section {tile_index + 1}/{len(tiles)} · step {step + 1}/{actual_steps}', min(.9, .1 + .8 * progress))
                        return kwargs
                    for tile_index, (x, y) in enumerate(tiles):
                        check()
                        patch = source[:, :, y:y + (512 if tiled else h), x:x + (512 if tiled else w)]
                        ph, pw = patch.shape[-2:]
                        padded = F.pad(patch, (0, (-pw) % 8, 0, (-ph) % 8), mode='replicate')
                        pipe.section_noise = shared_noise[:, :, y//8:y//8+math.ceil(ph/8), x//8:x//8+math.ceil(pw/8)]
                        generated = pipe(
                            prompt_embeds=positive_embeddings,
                            negative_prompt_embeds=negative_embeddings,
                            image=padded, strength=creativity, num_inference_steps=steps,
                            guidance_scale=guidance_scale, generator=torch.Generator(device='cuda').manual_seed(seed),
                            output_type='pt', callback_on_step_end=callback,
                        ).images[:, :, :ph, :pw]
                        check()
                        # Anchor broad color and illumination before overlap blending.
                        generated = generated + smooth_gpu(smooth_gpu(patch)) - smooth_gpu(smooth_gpu(generated))
                        # Blend overlapping generated sections before extracting texture.
                        wy = torch.sin(torch.linspace(.05, math.pi - .05, ph, device='cuda')) ** 2
                        wx = torch.sin(torch.linspace(.05, math.pi - .05, pw, device='cuda')) ** 2
                        weight = (wy[:, None] * wx[None, :])[None, None]
                        output[:, :, y:y + ph, x:x + pw] += generated * weight
                        weights[:, :, y:y + ph, x:x + pw] += weight
                    output /= weights.clamp_min(1e-8)
                    check()
                    if not torch.isfinite(output).all().item():
                        raise RuntimeError('GPU produced invalid values. Try Fast after restarting the app.')
                    report('Blending generated texture on GPU…', .93)
                    result = protected_detail(original, source, output, amount) + (clarity_gpu(original, clarity) - original)
                check()
                if output_scale > 1 and not native:
                    report('Resizing enhanced image on GPU…', .97)
                    result = F.interpolate(result, scale_factor=output_scale, mode='bicubic', align_corners=False)
                check()
                array = (result.clamp(0, 1).squeeze(0).permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                info = dict(pipeline_version=5, mode=mode, preset=preset, creativity=creativity, amount=amount, clarity=clarity, prompt=actual_prompt if pipe is not None else '', additional_guidance=prompt, negative_prompt=NEGATIVE_PROMPT if pipe is not None else '', seed=seed, output_scale=output_scale, native_output_ai=native, time_budget=time_budget,
                            working_size=list(work), output_size=[image.width * output_scale, image.height * output_scale], seconds=round(elapsed, 2),
                            gpu=torch.cuda.get_device_name(0), peak_vram_mb=round(torch.cuda.max_memory_allocated() / 1024**2))
                info['pipeline_version'] = 8
                info['ai_passes'] = len(tiles) if pipe is not None else 0
                info['denoising_steps_per_pass'] = actual_steps if pipe is not None else 0
                info['guidance_scale'] = guidance_scale if pipe is not None else None
                info['output_scale_independent_ai'] = efficient
                info['texture_transfer'] = 'fine-band novel luminance residual; local source correlation removed'
                info['section_consistency'] = 'coordinate-aligned noise, deterministic VAE encoding, source color anchoring, feathered overlap' if pipe is not None else None
                report(f'Finished in {elapsed:.1f}s', 1)
                return Image.fromarray(array), info
        except torch.cuda.OutOfMemoryError as exc:
            self.pipe = None
            self.prompt_cache = None
            raise RuntimeError('GPU memory is full. Close GPU-heavy apps, reduce output scale, and use Fast Detail. No work was moved to the CPU.') from exc
        finally:
            if self.pipe is not None:
                self.pipe.section_noise = None
            self.lock.release()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

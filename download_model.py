"""Explicit, anonymous one-time model download for SD1.5 and Real-ESRGAN. The app itself stays offline."""
import os
from pathlib import Path

os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

ROOT = Path(__file__).resolve().parent
SD_MODEL_DIR = ROOT / 'models' / 'sd15'
SR_MODEL_DIR = ROOT / 'models' / 'sr'

if __name__ == '__main__':
    from huggingface_hub import snapshot_download, hf_hub_download
    
    # 1. Real-ESRGAN Super-Resolution weights
    SR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    sr_pth = SR_MODEL_DIR / 'RealESRGAN_x4.pth'
    if not sr_pth.exists() or not (SR_MODEL_DIR / '.ready').exists():
        print('Downloading Real-ESRGAN photographic restoration model weights (~64MB)...')
        hf_hub_download(
            repo_id='ai-forever/Real-ESRGAN',
            filename='RealESRGAN_x4.pth',
            local_dir=str(SR_MODEL_DIR),
            token=False,
        )
        (SR_MODEL_DIR / '.ready').write_text('Downloaded successfully\n')
        print('Real-ESRGAN model ready.')
    else:
        print('Real-ESRGAN model already downloaded and ready.')

    # 2. Stable Diffusion 1.5 weights (optional micro-detail generation stage)
    if not (SD_MODEL_DIR / '.ready').exists():
        print('Downloading Stable Diffusion 1.5 weights...')
        snapshot_download(
            'stable-diffusion-v1-5/stable-diffusion-v1-5',
            local_dir=str(SD_MODEL_DIR), token=False,
            allow_patterns=[
                'model_index.json', 'scheduler/*', 'tokenizer/*',
                'text_encoder/config.json', 'text_encoder/model.safetensors',
                'unet/config.json', 'unet/diffusion_pytorch_model.safetensors',
                'vae/config.json', 'vae/diffusion_pytorch_model.safetensors',
                'README.md', 'LICENSE*',
            ],
        )
        (SD_MODEL_DIR / '.ready').write_text('Downloaded successfully\n')
        print('Stable Diffusion model ready.')
    else:
        print('Stable Diffusion model already downloaded and ready.')

    print('All models ready. Detail Lab can run offline.')

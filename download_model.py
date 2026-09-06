"""Explicit, anonymous one-time model download. The app itself stays offline."""
import os
from pathlib import Path

os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

MODEL_DIR = Path(__file__).resolve().parent / 'models' / 'sd15'

if __name__ == '__main__':
    from huggingface_hub import snapshot_download
    print('Downloading public Stable Diffusion 1.5 weights. No login or token needed.')
    snapshot_download(
        'stable-diffusion-v1-5/stable-diffusion-v1-5',
        local_dir=str(MODEL_DIR), token=False,
        allow_patterns=[
            'model_index.json', 'scheduler/*', 'tokenizer/*',
            'text_encoder/config.json', 'text_encoder/model.safetensors',
            'unet/config.json', 'unet/diffusion_pytorch_model.safetensors',
            'vae/config.json', 'vae/diffusion_pytorch_model.safetensors',
            'README.md', 'LICENSE*',
        ],
    )
    (MODEL_DIR / '.ready').write_text('Downloaded successfully\n')
    print('Model ready. The app can now run offline.')

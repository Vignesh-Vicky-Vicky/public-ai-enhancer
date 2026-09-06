"""Run a reproducible GPU AI smoke test, or benchmark your own image."""
import argparse
import json
import threading
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from engine import DetailEngine

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image')
    parser.add_argument('--preset', default='Fast · 512')
    parser.add_argument('--size', type=int, default=512)
    args = parser.parse_args()
    if args.image:
        with Image.open(args.image) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
    else:
        image = Image.new('RGB', (args.size, args.size), '#9e8164')
        draw = ImageDraw.Draw(image)
        for y in range(0, args.size, 32):
            for x in range(-32 if y % 64 else 0, args.size, 64):
                draw.rectangle((x + 2, y + 2, x + 62, y + 30), fill='#b49373', outline='#675643', width=2)
    output = Path(__file__).parent / 'outputs'
    output.mkdir(exist_ok=True)
    image.save(output / 'benchmark-input.png')
    result, info = DetailEngine().run(image, mode='AI detail generation', preset=args.preset, creativity=.28, amount=1.,
        prompt='a brick wall, rough brick texture, tiny mineral grains, natural surface detail', seed=42,
        cancel=threading.Event(), report=lambda text, progress: print(text, flush=True))
    result.save(output / 'benchmark-output.png')
    (output / 'benchmark.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))

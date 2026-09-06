import threading
import unittest
import numpy as np
from PIL import Image
from engine import DetailEngine, Cancelled, working_size, protected_detail, smooth_gpu, tile_starts, clarity_gpu


class EngineTests(unittest.TestCase):
    def test_tile_coverage(self):
        for length in [32, 512, 513, 769, 1024]:
            covered = np.zeros(length)
            for start in tile_starts(length):
                covered[start:start+512] += 1
            self.assertTrue((covered > 0).all())

    def test_clarity_increases_texture_but_leaves_flat_color(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA required')
        flat = torch.full((1, 3, 64, 64), .5, device='cuda')
        self.assertTrue(torch.equal(clarity_gpu(flat, 1.), flat))
        textured = flat.clone()
        textured[:, :, ::3] += .03
        self.assertGreater(clarity_gpu(textured, 1.).std().item(), textured.std().item())

    def test_generated_smoothing_cannot_erase_input(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA required')
        source = torch.full((1, 3, 64, 64), .5, device='cuda')
        source[:, :, ::2, ::2] += .02
        flat = torch.full_like(source, .5)
        result = protected_detail(source, source, flat, 1.25)
        self.assertTrue(torch.equal(result, source))

    def test_shifted_edges_and_changes_are_bounded(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA required')
        source = torch.full((1, 3, 64, 64), .3, device='cuda')
        source[:, :, :, 32:] = .8
        shifted = torch.roll(source, 5, dims=-1)
        result = protected_detail(source, source, shifted, 1.25)
        self.assertLess((result - source).abs().max().item(), .02501)
        self.assertLess((result[:, :, :, 27:38] - source[:, :, :, 27:38]).abs().max().item(), .001)

    def test_working_size_preserves_aspect(self):
        self.assertEqual(working_size((1920, 1080), 512), (512, 288))
        self.assertEqual(working_size((200, 100), 512), (200, 100))

    def test_cuda_texture_and_cancellation(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA required')
        rng = np.random.default_rng(42)
        image = Image.fromarray(rng.integers(30, 220, (128, 192, 3), dtype=np.uint8))
        engine = DetailEngine()
        options = dict(mode='GPU texture boost (no new detail)', preset='Fast · 512', creativity=.28, amount=1., prompt='', seed=42, cancel=threading.Event(), report=lambda *args: None)
        result, info = engine.run(image, **options)
        self.assertEqual(result.size, image.size)
        self.assertGreater(np.abs(np.asarray(result).astype(float) - np.asarray(image)).mean(), 1)
        self.assertIn('gpu', info)
        options['cancel'].set()
        with self.assertRaises(Cancelled):
            engine.run(image, **options)
        self.assertFalse(engine.lock.locked())


if __name__ == '__main__':
    unittest.main()

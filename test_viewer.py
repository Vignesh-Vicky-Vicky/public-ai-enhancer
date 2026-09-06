import unittest
import tkinter as tk
from types import SimpleNamespace
from PIL import Image
from app import App


class ViewerTests(unittest.TestCase):
    def test_linked_zoom_pan_and_fit(self):
        root = tk.Tk()
        try:
            app = App(root)
            app.original = Image.new('RGB', (1200, 1600), 'gray')
            app.result = app.original.copy()
            root.update()
            app.set_zoom(1.)
            self.assertEqual(app.view_scale(), 1.)
            app.change_zoom(2.)
            self.assertEqual(app.view_scale(), 2.)
            app.start_pan(SimpleNamespace(x=100, y=100))
            app.pan(SimpleNamespace(x=180, y=160))
            self.assertLess(app.center[0], .5)
            self.assertEqual(len(app.photos), 2)
            app.set_zoom(None)
            self.assertEqual(app.center, [.5, .5])
            self.assertLess(app.view_scale(), 1.)
        finally:
            root.destroy()

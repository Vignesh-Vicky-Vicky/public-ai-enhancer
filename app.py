"""Local desktop interface for Detail Lab - Extreme Generative Micro-Detail Reconstruction."""
import json
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageOps, ImageTk
from engine import DetailEngine, Cancelled

BG, PANEL, FG, MUTED, ACCENT = '#090e18', '#151f30', '#f1f5ff', '#92a4bf', '#69dfce'
BORDER, WELL = '#2b3c54', '#0c1422'


class App:
    def __init__(self, root):
        self.root = root
        root.title('Detail Lab · Extreme Micro-Detail & Super-Resolution')
        root.geometry('1340x900')
        root.minsize(980, 720)
        root.configure(bg=BG)
        self.engine = DetailEngine()
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.original = self.result = None
        self.info = {}
        self.busy = False
        self.reset_pending = False
        self.started = 0
        self.photos = []
        self.zoom = None
        self.center = [0.5, 0.5]
        self.drag_origin = None
        self.zoom_text = tk.StringVar(value='Fit')

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', background=BG, foreground=FG, font=('Segoe UI', 10))
        style.configure('TButton', background=PANEL, padding=9)
        style.map('TButton', background=[('active', '#34445c')])
        style.configure('Primary.TButton', background=ACCENT, foreground='#082621', font=('Segoe UI', 11, 'bold'), padding=(20, 12), borderwidth=0)
        style.map('Primary.TButton', background=[('disabled', '#243c43'), ('active', '#9aefdf')], foreground=[('disabled', '#758a96')])
        style.configure('Stop.TButton', background='#3a2431', foreground='#ffbbca', padding=(14, 12))
        style.map('Stop.TButton', background=[('disabled', PANEL), ('active', '#573244')])
        style.configure('TRadiobutton', background=PANEL, foreground=FG, font=('Segoe UI', 10, 'bold'))
        style.map('TRadiobutton', background=[('active', PANEL)])
        style.configure('Horizontal.TProgressbar', troughcolor=PANEL, background=ACCENT)

        # Header
        header = tk.Frame(root, bg=BG)
        header.pack(fill='x', padx=28, pady=(20, 12))
        tk.Label(header, text='◈', bg=BG, fg=ACCENT, font=('Segoe UI', 30)).pack(side='left', padx=(0, 12))
        tk.Label(header, text='DETAIL LAB', bg=BG, fg=FG, font=('Segoe UI', 26, 'bold')).pack(side='left')
        tk.Label(header, text='EXTREME AI DETAIL ENHANCER', bg=BG, fg=MUTED, font=('Segoe UI', 10)).pack(side='left', padx=18)
        self.gpu_badge = tk.Label(header, text='●  LOCAL GPU (GTX 1660 Ti)', bg='#173331', fg=ACCENT, padx=16, pady=8, font=('Segoe UI', 10, 'bold'))
        self.gpu_badge.pack(side='right')

        tk.Label(root, text='One-click photographic super-resolution & generative microstructure synthesis. Hallucinates missing details.', bg=BG, fg=MUTED, font=('Segoe UI', 11)).pack(anchor='w', padx=28)

        body = tk.Frame(root, bg=BG)
        body.pack(fill='both', expand=True, padx=28, pady=14)

        # Simplified Sidebar
        sidebar = tk.Frame(body, bg=PANEL, highlightbackground=BORDER, highlightthickness=1, padx=14, pady=16, width=320)
        sidebar.pack(side='left', fill='y', padx=(0, 16))
        sidebar.pack_propagate(False)

        # Section 01: Source
        self.section(sidebar, '01', 'SOURCE')
        self.open_button = ttk.Button(sidebar, text='+  Open photograph', command=self.open_image)
        self.open_button.pack(fill='x', pady=(0, 10))
        self.source_dim_label = tk.Label(sidebar, text='No photograph loaded', bg=WELL, fg=MUTED, pady=8, font=('Segoe UI', 9))
        self.source_dim_label.pack(fill='x', pady=(0, 20))

        # Section 02: Output Size (The ONLY user setting)
        self.section(sidebar, '02', 'OUTPUT SIZE')
        self.output_scale = tk.IntVar(value=2)
        scale_frame = tk.Frame(sidebar, bg=PANEL)
        scale_frame.pack(fill='x', pady=(0, 10))

        self.r2 = ttk.Radiobutton(scale_frame, text='2×', variable=self.output_scale, value=2, command=self.update_expected_dims)
        self.r2.pack(side='left', padx=(10, 20))
        self.r4 = ttk.Radiobutton(scale_frame, text='4×', variable=self.output_scale, value=4, command=self.update_expected_dims)
        self.r4.pack(side='left', padx=10)

        self.target_dim_label = tk.Label(sidebar, text='Target: —', bg=WELL, fg=ACCENT, pady=8, font=('Segoe UI', 9, 'bold'))
        self.target_dim_label.pack(fill='x', pady=(0, 20))

        # Real Pipeline Activity Panel
        self.section(sidebar, '03', 'PIPELINE ACTIVITY')
        self.activity_frame = tk.Frame(sidebar, bg=WELL, padx=10, pady=10)
        self.activity_frame.pack(fill='both', expand=True, pady=(0, 10))

        self.stage_labels = {}
        stages = [
            ('analyze', 'Analyzing image'),
            ('sr_base', 'Neural super-resolution base'),
            ('gen_detail', 'Generating photographic details'),
            ('materials', 'Material microstructure'),
            ('refine', 'Ultra-detail refinement'),
            ('safety', 'Artifact safety check'),
        ]
        for key, name in stages:
            row = tk.Frame(self.activity_frame, bg=WELL)
            row.pack(fill='x', pady=3)
            name_lbl = tk.Label(row, text=name, bg=WELL, fg=MUTED, font=('Segoe UI', 9), anchor='w')
            name_lbl.pack(side='left')
            status_lbl = tk.Label(row, text='—', bg=WELL, fg=MUTED, font=('Segoe UI', 9), width=8, anchor='e')
            status_lbl.pack(side='right')
            self.stage_labels[key] = (name_lbl, status_lbl)

        # Hardware & telemetry display in sidebar
        self.hw_info_label = tk.Label(
            sidebar,
            text='GPU: GTX 1660 Ti 6GB\nMode: Extreme Micro-Detail\nTiling: 512px Hann Overlap',
            bg=PANEL, fg=MUTED, font=('Segoe UI', 8), justify='left'
        )
        self.hw_info_label.pack(anchor='w', pady=(8, 0))

        # Right Preview Area
        self.preview = tk.Frame(body, bg=BG)
        self.preview.pack(side='left', fill='both', expand=True)

        toolbar = tk.Frame(self.preview, bg=PANEL, padx=10, pady=8, highlightbackground=BORDER, highlightthickness=1)
        toolbar.pack(fill='x', pady=(0, 8))
        for label, command in [
            ('−', lambda: self.change_zoom(.8)), ('+', lambda: self.change_zoom(1.25)),
            ('100%', lambda: self.set_zoom(1.0)), ('200%', lambda: self.set_zoom(2.0)),
            ('332%', lambda: self.set_zoom(3.32)), ('400%', lambda: self.set_zoom(4.0)),
            ('Fit', lambda: self.set_zoom(None))
        ]:
            ttk.Button(toolbar, text=label, command=command, width=5).pack(side='left', padx=2)
        tk.Label(toolbar, textvariable=self.zoom_text, bg=PANEL, fg=ACCENT).pack(side='left', padx=8)

        image_row = tk.Frame(self.preview, bg=BG)
        image_row.pack(fill='both', expand=True)
        self.canvases = []
        for title in ['ORIGINAL', 'EXTREME DETAIL RECONSTRUCTION']:
            frame = tk.Frame(image_row, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
            frame.pack(side='left', fill='both', expand=True, padx=4)
            tk.Frame(frame, bg=ACCENT if 'RECONSTRUCTION' in title else '#506383', height=3).pack(fill='x')
            tk.Label(frame, text=title, bg=PANEL, fg=ACCENT if 'RECONSTRUCTION' in title else MUTED, font=('Segoe UI', 10, 'bold')).pack(pady=10)
            canvas = tk.Canvas(frame, bg=WELL, highlightthickness=0, width=250, cursor='fleur')
            canvas.pack(fill='both', expand=True)
            canvas.bind('<Configure>', lambda event: self.draw())
            canvas.bind('<MouseWheel>', lambda event: self.change_zoom(1.25 if event.delta > 0 else .8))
            canvas.bind('<ButtonPress-1>', self.start_pan)
            canvas.bind('<B1-Motion>', self.pan)
            canvas.bind('<Double-Button-1>', lambda event: self.set_zoom(1.0))
            self.canvases.append(canvas)

        tk.Label(self.preview, text='LINKED VIEWS   ·   Mouse wheel to zoom   ·   Drag either image to pan both   ·   Inspect at 100%–400% for micro-detail', bg=BG, fg=MUTED, font=('Segoe UI', 9)).pack(pady=(8, 0))

        # Footer
        footer = tk.Frame(root, bg=PANEL, padx=16, pady=12, highlightbackground=BORDER, highlightthickness=1)
        footer.pack(fill='x', padx=28, pady=(0, 16))
        self.status = tk.StringVar(value='Open an image to begin. Pure local GPU inference.')
        status_row = tk.Frame(footer, bg=PANEL)
        status_row.pack(fill='x')
        tk.Label(status_row, textvariable=self.status, bg=PANEL, fg=FG, anchor='w', wraplength=950).pack(side='left', fill='x', expand=True)
        self.progress_pct = tk.StringVar(value='0%')
        self.progress_label = tk.Label(status_row, textvariable=self.progress_pct, bg=PANEL, fg=ACCENT, font=('Segoe UI', 10, 'bold'), width=6, anchor='e')
        self.progress_label.pack(side='right', padx=(8, 0))

        self.progress = ttk.Progressbar(footer, maximum=1)
        self.progress.pack(fill='x', pady=(6, 8))

        actions = tk.Frame(footer, bg=PANEL)
        actions.pack(fill='x')
        self.run_button = ttk.Button(actions, text='✦  Enhance image', style='Primary.TButton', command=self.run, state='disabled')
        self.run_button.pack(side='left', padx=(0, 8))
        self.cancel_button = ttk.Button(actions, text='Cancel', style='Stop.TButton', command=self.request_cancel, state='disabled')
        self.cancel_button.pack(side='left', padx=(0, 8))
        self.reset_button = ttk.Button(actions, text='Reset', command=self.reset)
        self.reset_button.pack(side='left')
        self.save_button = ttk.Button(actions, text='Export PNG  ↗', command=self.save, state='disabled')
        self.save_button.pack(side='right')

        self.clock = tk.StringVar(value='Ready')
        tk.Label(footer, textvariable=self.clock, bg=PANEL, fg=MUTED, anchor='w').pack(fill='x', pady=(8, 0))

        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(100, self.poll)

    def section(self, parent, number, title):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill='x', pady=(6, 8))
        tk.Label(row, text=number, bg='#23384a', fg=ACCENT, padx=6, pady=2, font=('Segoe UI', 9, 'bold')).pack(side='left')
        tk.Label(row, text=title, bg=PANEL, fg=FG, font=('Segoe UI', 10, 'bold')).pack(side='left', padx=8)

    def update_expected_dims(self):
        scale = self.output_scale.get()
        if self.original:
            tw = self.original.width * scale
            th = self.original.height * scale
            self.target_dim_label.config(text=f'Expected: {tw} × {th}')
        else:
            self.target_dim_label.config(text=f'Scale: {scale}×')

    def set_activity_status(self, key, text, color=FG):
        if key in self.stage_labels:
            lbl_name, lbl_status = self.stage_labels[key]
            lbl_status.config(text=text, fg=color)
            if text == '✓':
                lbl_name.config(fg=FG)
            elif text != '—':
                lbl_name.config(fg=ACCENT)

    def reset_activity(self):
        for key in self.stage_labels:
            lbl_name, lbl_status = self.stage_labels[key]
            lbl_name.config(fg=MUTED)
            lbl_status.config(text='—', fg=MUTED)

    def request_cancel(self):
        if self.busy:
            self.cancel.set()
            self.cancel_button.config(state='disabled')
            self.status.set('Cancelling… waiting for current GPU step.')

    def reset(self):
        if self.busy:
            self.reset_pending = True
            self.request_cancel()
            self.status.set('Reset requested. Cancelling job first…')
            return
        self.reset_pending = False
        self.original = self.result = None
        self.info = {}
        self.source_path = None
        self.zoom = None
        self.center = [.5, .5]
        self.drag_origin = None
        self.output_scale.set(2)
        self.cancel.clear()
        self.progress['value'] = 0
        self.progress_pct.set('0%')
        self.run_button.config(state='disabled')
        self.save_button.config(state='disabled')
        self.cancel_button.config(state='disabled')
        self.source_dim_label.config(text='No photograph loaded')
        self.target_dim_label.config(text='Target: —')
        self.status.set('Ready. Settings restored.')
        self.clock.set('Ready')
        self.reset_activity()
        self.draw()

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[('Images', '*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff')])
        if not path:
            return
        try:
            with Image.open(path) as image:
                if image.width * image.height > 16_000_000:
                    raise ValueError('Use an image of 16 megapixels or less.')
                self.original = ImageOps.exif_transpose(image).convert('RGB')
            self.result = None
            self.zoom = None
            self.center = [.5, .5]
            self.source_path = Path(path)
            self.run_button.config(state='normal')
            self.save_button.config(state='disabled')
            self.source_dim_label.config(text=f'Input: {self.original.width} × {self.original.height}')
            self.update_expected_dims()
            self.status.set(f'{self.source_path.name} loaded.')
            self.reset_activity()
            self.draw()
        except Exception as exc:
            messagebox.showerror('Cannot open image', str(exc))

    def draw(self):
        self.photos = []
        for canvas, source in zip(self.canvases, [self.original, self.result]):
            canvas.delete('all')
            w, h = canvas.winfo_width(), canvas.winfo_height()
            if source is None:
                canvas.create_text(w / 2, h / 2, text='Your image here' if canvas == self.canvases[0] else 'Enhanced preview', fill=MUTED)
            else:
                scale = self.view_scale() * self.original.width / source.width
                cx, cy = self.center[0] * source.width, self.center[1] * source.height
                left, top = max(0, cx - w / (2 * scale)), max(0, cy - h / (2 * scale))
                right, bottom = min(source.width, cx + w / (2 * scale)), min(source.height, cy + h / (2 * scale))
                if right <= left or bottom <= top:
                    continue
                size = (max(1, round((right-left)*scale)), max(1, round((bottom-top)*scale)))
                copy = source.resize(size, Image.Resampling.NEAREST if scale >= 1 else Image.Resampling.LANCZOS, box=(left, top, right, bottom))
                photo = ImageTk.PhotoImage(copy)
                self.photos.append(photo)
                canvas.create_image(w / 2 + (left-cx)*scale, h / 2 + (top-cy)*scale, image=photo, anchor='nw')
        self.zoom_text.set(('Fit · ' if self.zoom is None else '') + f'{self.view_scale()*100:.0f}%')

    def view_scale(self):
        if self.zoom is not None:
            return self.zoom
        if self.original is None or not self.canvases:
            return 1.0
        return max(.001, min(min(max(1, c.winfo_width()-16)/self.original.width, max(1, c.winfo_height()-16)/self.original.height) for c in self.canvases))

    def set_zoom(self, scale):
        self.zoom = scale
        if scale is None:
            self.center = [.5, .5]
        self.draw()

    def change_zoom(self, factor):
        self.set_zoom(max(.05, min(8., self.view_scale()*factor)))

    def start_pan(self, event):
        self.drag_origin = (event.x, event.y)

    def pan(self, event):
        if self.original is None or self.drag_origin is None:
            return
        dx, dy = event.x-self.drag_origin[0], event.y-self.drag_origin[1]
        scale = self.view_scale()
        self.center[0] = min(1., max(0., self.center[0]-dx/(scale*self.original.width)))
        self.center[1] = min(1., max(0., self.center[1]-dy/(scale*self.original.height)))
        self.drag_origin = (event.x, event.y)
        self.draw()

    def run(self):
        if self.busy or self.original is None:
            return
        self.busy = True
        self.cancel.clear()
        self.started = time.perf_counter()
        self.result = None
        self.draw()
        self.save_button.config(state='disabled')
        self.open_button.config(state='disabled')
        self.run_button.config(state='disabled')
        self.r2.config(state='disabled')
        self.r4.config(state='disabled')
        self.cancel_button.config(state='normal')
        self.status.set('Starting Extreme Micro-Detail Reconstruction…')
        self.progress['value'] = 0
        self.progress_pct.set('0%')
        self.reset_activity()

        scale = self.output_scale.get()
        source = self.original.copy()

        def worker():
            try:
                result, info = self.engine.run(
                    source,
                    output_scale=scale,
                    cancel=self.cancel,
                    report=lambda text, progress: self.events.put(('progress', (text, progress)))
                )
                self.events.put(('done', (result, info)))
            except Exception as exc:
                self.events.put(('error', (str(exc), isinstance(exc, Cancelled))))

        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        while True:
            try:
                kind, data = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == 'progress':
                txt, p = data
                if not self.cancel.is_set():
                    self.status.set(txt)
                self.progress['value'] = p
                self.progress_pct.set(f'{int(round(p * 100))}%')

                # Update structured activity checklist
                if 'Analyzing' in txt:
                    self.set_activity_status('analyze', 'Active', ACCENT)
                elif 'Creating neural' in txt:
                    self.set_activity_status('analyze', '✓')
                    self.set_activity_status('sr_base', f'{int(p*100)}%', ACCENT)
                elif 'Generating photographic' in txt:
                    self.set_activity_status('sr_base', '✓')
                    self.set_activity_status('gen_detail', 'Active', ACCENT)
                elif 'Micro-detail tiles' in txt:
                    self.set_activity_status('gen_detail', '✓')
                    parts = txt.split('Micro-detail tiles ')[-1].split(' ·')[0]
                    self.set_activity_status('materials', parts, ACCENT)
                elif 'Ultra-detail' in txt:
                    self.set_activity_status('materials', '✓')
                    self.set_activity_status('refine', 'Active', ACCENT)
                elif 'Artifact' in txt:
                    self.set_activity_status('refine', '✓')
                    self.set_activity_status('safety', 'Active', ACCENT)
                elif 'Finalizing' in txt:
                    self.set_activity_status('safety', '✓')
            else:
                self.busy = False
                self.open_button.config(state='normal')
                self.run_button.config(state='normal')
                self.r2.config(state='normal')
                self.r4.config(state='normal')
                self.cancel_button.config(state='disabled')

                if self.reset_pending:
                    self.reset()
                    continue
                if kind == 'done' and self.cancel.is_set():
                    self.status.set('Cancelled. Result discarded.')
                    self.clock.set(f'Stopped after {time.perf_counter() - self.started:.1f}s')
                    continue
                if kind == 'done':
                    self.result, self.info = data
                    self.draw()
                    self.save_button.config(state='normal')
                    self.progress['value'] = 1.0
                    self.progress_pct.set('100%')
                    for k in self.stage_labels:
                        self.set_activity_status(k, '✓')
                    self.clock.set(f"{self.info['seconds']:.1f}s · {self.info['gpu']} · VRAM {self.info.get('vram_display', '')} · Output {self.info['output_size'][0]} × {self.info['output_size'][1]}")
                else:
                    self.status.set(data[0])
                    self.clock.set(f'Stopped after {time.perf_counter() - self.started:.1f}s')
                    if not data[1]:
                        messagebox.showerror('Enhancement stopped', data[0])
        if self.busy:
            self.clock.set(f'{time.perf_counter() - self.started:.1f}s elapsed · GPU executing…')
        self.root.after(100, self.poll)

    def save(self):
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.png',
            initialfile=self.source_path.stem + f'-{self.output_scale.get()}x-detail.png',
            filetypes=[('PNG image', '*.png')]
        )
        if path:
            try:
                from PIL.PngImagePlugin import PngInfo
                metadata = PngInfo()
                metadata.add_text('Detail Lab', json.dumps(self.info))
                self.result.save(path, pnginfo=metadata)
                self.status.set(f'Saved {path}')
            except Exception as exc:
                messagebox.showerror('Cannot save image', str(exc))

    def close(self):
        self.cancel.set()
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    App(root)
    root.mainloop()

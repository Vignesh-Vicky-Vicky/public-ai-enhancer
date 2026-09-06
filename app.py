"""Local desktop interface for Detail Lab."""
import json
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageOps, ImageTk
from engine import DetailEngine, PRESETS, Cancelled, DETAIL_PROMPT

BG, PANEL, FG, MUTED, ACCENT = '#090e18', '#151f30', '#f1f5ff', '#92a4bf', '#69dfce'
BORDER, WELL = '#2b3c54', '#0c1422'


class App:
    def __init__(self, root):
        self.root = root
        root.title('Detail Lab 9 · Photographic Super-Resolution & Detail')
        root.geometry('1280x880')
        root.minsize(920, 720)
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
        style.configure('Primary.TButton', background=ACCENT, foreground='#082621', font=('Segoe UI', 11, 'bold'), padding=(18, 12), borderwidth=0)
        style.map('Primary.TButton', background=[('disabled', '#243c43'), ('active', '#9aefdf')], foreground=[('disabled', '#758a96')])
        style.configure('Stop.TButton', background='#3a2431', foreground='#ffbbca', padding=(14, 12))
        style.map('Stop.TButton', background=[('disabled', PANEL), ('active', '#573244')])
        style.configure('TCheckbutton', background=PANEL)
        style.configure('TCombobox', fieldbackground=PANEL, background=PANEL, foreground=FG)
        style.map('TCombobox', fieldbackground=[('readonly', PANEL)], foreground=[('readonly', FG)])
        style.configure('Horizontal.TProgressbar', troughcolor=PANEL, background=ACCENT)
        header = tk.Frame(root, bg=BG)
        header.pack(fill='x', padx=28, pady=(22, 14))
        tk.Label(header, text='◈', bg=BG, fg=ACCENT, font=('Segoe UI', 30)).pack(side='left', padx=(0, 12))
        tk.Label(header, text='DETAIL LAB', bg=BG, fg=FG, font=('Segoe UI', 26, 'bold')).pack(side='left')
        tk.Label(header, text='TEXTURE & SUPER-RESOLUTION  /  09', bg=BG, fg=MUTED, font=('Segoe UI', 10)).pack(side='left', padx=18)
        tk.Label(header, text='●  LOCAL GPU', bg='#173331', fg=ACCENT, padx=16, pady=9, font=('Segoe UI', 10, 'bold')).pack(side='right')
        tk.Label(root, text='Photographic super-resolution & micro-detail. Identity and geometry preserved.', bg=BG, fg=MUTED, font=('Segoe UI', 12)).pack(anchor='w', padx=28)
        body = tk.Frame(root, bg=BG)
        body.pack(fill='both', expand=True, padx=28, pady=18)
        sidebar = tk.Frame(body, bg=PANEL, highlightbackground=BORDER, highlightthickness=1, padx=10, pady=12)
        sidebar.pack(side='left', fill='y', padx=(0, 16))
        scroll = tk.Canvas(sidebar, bg=PANEL, width=280, highlightthickness=0)
        scrollbar = ttk.Scrollbar(sidebar, orient='vertical', command=scroll.yview)
        scroll.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        scroll.pack(side='left', fill='y', expand=True)
        controls = tk.Frame(scroll, bg=PANEL)
        scroll.create_window(0, 0, window=controls, anchor='nw', width=280)
        controls.bind('<Configure>', lambda event: scroll.configure(scrollregion=scroll.bbox('all')))
        self.section(controls, '01', 'Source & output')
        self.open_button = ttk.Button(controls, text='+  Open photograph', command=self.open_image)
        self.open_button.pack(fill='x', pady=(0, 16))
        self.mode = tk.StringVar(value='Fast Detail')
        self.preset = tk.StringVar(value='Detail · 896')
        self.creativity = tk.DoubleVar(value=.25)
        self.amount = tk.DoubleVar(value=1.2)
        self.clarity = tk.DoubleVar(value=0.35)
        self.seed = tk.StringVar(value='42')
        self.output_scale = tk.StringVar(value='2× output')
        self.limit_time = tk.BooleanVar(value=True)
        self.inputs = []
        for label, var, options in [('Enhancement', self.mode, ['Fast Detail', 'Maximum Detail']), ('Output size', self.output_scale, ['1× original size', '2× output', '4× output'])]:
            self.label(controls, label)
            box = ttk.Combobox(controls, textvariable=var, values=options, state='readonly')
            box.pack(fill='x', pady=(0, 12))
            self.inputs.append(box)
        budget = ttk.Checkbutton(controls, text='Stop unfinished jobs near 180 seconds', variable=self.limit_time)
        budget.pack(fill='x', pady=(0, 8))
        self.inputs.append(budget)
        tk.Label(controls, text='Fast: Dedicated photographic SR (2×/4× reconstructed in seconds).\nMaximum: SR + generative diffusion micro-detail refinement.', wraplength=270, bg=PANEL, fg=MUTED, justify='left').pack(fill='x', pady=(0, 12))
        self.section(controls, '02', 'Texture & preservation')
        for label, var, low, high, resolution in [('Micro-detail creativity (Maximum mode)', self.creativity, .05, .50, .01), ('Texture blend intensity', self.amount, .1, 2.5, .05), ('Micro-contrast clarity', self.clarity, 0, 2, .05)]:
            self.label(controls, label)
            scale = tk.Scale(controls, variable=var, from_=low, to=high, resolution=resolution, orient='horizontal', bg=PANEL, fg=ACCENT, activebackground=ACCENT, troughcolor=WELL, highlightthickness=0, sliderrelief='flat')
            scale.pack(fill='x', pady=(0, 2))
            self.inputs.append(scale)
        self.section(controls, '03', 'Image guidance')
        self.label(controls, 'Extra image guidance (optional)')
        self.prompt = tk.Text(controls, height=3, bg=WELL, fg=FG, insertbackground=FG, relief='flat', wrap='word', padx=8, pady=8, font=('Segoe UI', 10))
        self.prompt.pack(fill='x', pady=(5, 10))
        self.inputs.append(self.prompt)
        self.label(controls, 'Seed · same settings, repeatable result')
        entry = ttk.Entry(controls, textvariable=self.seed)
        entry.pack(fill='x', pady=(5, 12))
        self.inputs.append(entry)
        self.preview = tk.Frame(body, bg=BG)
        self.preview.pack(side='left', fill='both', expand=True)
        toolbar = tk.Frame(self.preview, bg=PANEL, padx=10, pady=8, highlightbackground=BORDER, highlightthickness=1)
        toolbar.pack(fill='x', pady=(0, 8))
        for label, command in [('−', lambda: self.change_zoom(.8)), ('+', lambda: self.change_zoom(1.25)), ('100%', lambda: self.set_zoom(1.0)), ('200%', lambda: self.set_zoom(2.0)), ('400%', lambda: self.set_zoom(4.0)), ('Fit', lambda: self.set_zoom(None))]:
            ttk.Button(toolbar, text=label, command=command, width=5).pack(side='left', padx=2)
        tk.Label(toolbar, textvariable=self.zoom_text, bg=PANEL, fg=ACCENT).pack(side='left', padx=8)
        image_row = tk.Frame(self.preview, bg=BG)
        image_row.pack(fill='both', expand=True)
        self.canvases = []
        for title in ['ORIGINAL', 'DETAIL BOOST']:
            frame = tk.Frame(image_row, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
            frame.pack(side='left', fill='both', expand=True, padx=4)
            tk.Frame(frame, bg=ACCENT if title == 'DETAIL BOOST' else '#506383', height=3).pack(fill='x')
            tk.Label(frame, text=title, bg=PANEL, fg=ACCENT if title == 'DETAIL BOOST' else MUTED, font=('Segoe UI', 10, 'bold')).pack(pady=12)
            canvas = tk.Canvas(frame, bg=WELL, highlightthickness=0, width=250, cursor='fleur')
            canvas.pack(fill='both', expand=True)
            canvas.bind('<Configure>', lambda event: self.draw())
            canvas.bind('<MouseWheel>', lambda event: self.change_zoom(1.25 if event.delta > 0 else .8))
            canvas.bind('<ButtonPress-1>', self.start_pan)
            canvas.bind('<B1-Motion>', self.pan)
            canvas.bind('<Double-Button-1>', lambda event: self.set_zoom(1.0))
            self.canvases.append(canvas)
        tk.Label(self.preview, text='LINKED VIEWS   ·   Mouse wheel to zoom   ·   Drag either image to pan both', bg=BG, fg=MUTED, font=('Segoe UI', 9)).pack(pady=(10, 0))
        footer = tk.Frame(root, bg=PANEL, padx=16, pady=14, highlightbackground=BORDER, highlightthickness=1)
        footer.pack(fill='x', padx=28, pady=(0, 20))
        self.status = tk.StringVar(value='Open an image to begin. AI model setup is required once; processing stays offline.')
        tk.Label(footer, textvariable=self.status, bg=PANEL, fg=FG, anchor='w', wraplength=1000).pack(fill='x')
        self.progress = ttk.Progressbar(footer, maximum=1)
        self.progress.pack(fill='x', pady=10)
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
        self.clock = tk.StringVar(value='Target: under 180s · actual time depends on settings')
        tk.Label(footer, textvariable=self.clock, bg=PANEL, fg=MUTED, anchor='w').pack(fill='x', pady=(10, 0))
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(100, self.poll)

    def label(self, parent, text):
        tk.Label(parent, text=text, bg=parent.cget('bg'), fg=MUTED, anchor='w').pack(fill='x', pady=(0, 4))

    def section(self, parent, number, title):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill='x', pady=(6, 12))
        tk.Label(row, text=number, bg='#23384a', fg=ACCENT, padx=6, pady=3, font=('Segoe UI', 9, 'bold')).pack(side='left')
        tk.Label(row, text=title, bg=PANEL, fg=FG, font=('Segoe UI', 11, 'bold')).pack(side='left', padx=8)

    def request_cancel(self):
        if self.busy:
            self.cancel.set()
            self.cancel_button.config(state='disabled')
            self.status.set('Cancelling… waiting for the current GPU operation to finish.')

    def reset(self):
        if self.busy:
            self.reset_pending = True
            self.request_cancel()
            self.status.set('Reset requested. Cancelling the current job first…')
            return
        self.reset_pending = False
        self.original = self.result = None
        self.info = {}
        self.source_path = None
        self.zoom = None
        self.center = [.5, .5]
        self.drag_origin = None
        self.mode.set('Fast Detail')
        self.preset.set('Detail · 896')
        self.output_scale.set('2× output')
        self.creativity.set(.25)
        self.amount.set(1.2)
        self.clarity.set(0.35)
        self.limit_time.set(True)
        self.seed.set('42')
        self.prompt.delete('1.0', 'end')
        self.cancel.clear()
        self.progress['value'] = 0
        self.run_button.config(state='disabled')
        self.save_button.config(state='disabled')
        self.cancel_button.config(state='disabled')
        self.status.set('Ready for a new photograph. Settings restored; exported files are untouched.')
        self.clock.set('Target: under 180s · actual time depends on settings')
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
            self.status.set(f'{self.source_path.name} · {self.original.width} × {self.original.height}')
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
                # Display both at the same scene scale, even for a 2× export.
                scale = self.view_scale() * self.original.width / source.width
                # Render only the visible crop; zooming never allocates a giant bitmap.
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
        try:
            seed = int(self.seed.get())
            if not 0 <= seed < 2**32:
                raise ValueError()
        except ValueError:
            messagebox.showerror('Seed', 'Enter a whole number from 0 to 4294967295.')
            return
        settings = dict(mode=self.mode.get(), preset=self.preset.get(), creativity=self.creativity.get(), amount=self.amount.get(), clarity=self.clarity.get(), output_scale=int(self.output_scale.get()[0]), time_budget=170 if self.limit_time.get() else None, prompt=self.prompt.get('1.0', 'end').strip(), seed=seed)
        self.busy = True
        self.cancel.clear()
        self.started = time.perf_counter()
        self.result = None
        self.draw()
        self.save_button.config(state='disabled')
        self.open_button.config(state='disabled')
        self.run_button.config(state='disabled')
        self.cancel_button.config(state='normal')
        for widget in self.inputs:
            widget.config(state='disabled')
        self.status.set('Starting GPU enhancement…')
        self.progress['value'] = 0
        source = self.original.copy()
        def worker():
            try:
                result, info = self.engine.run(source, **settings, cancel=self.cancel, report=lambda text, progress: self.events.put(('progress', (text, progress))))
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
                if not self.cancel.is_set():
                    self.status.set(data[0])
                self.progress['value'] = data[1]
            else:
                self.busy = False
                self.open_button.config(state='normal')
                self.run_button.config(state='normal')
                self.cancel_button.config(state='disabled')
                for widget in self.inputs:
                    widget.config(state='readonly' if isinstance(widget, ttk.Combobox) else 'normal')
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
                    self.clock.set(f"{self.info['seconds']:.1f}s · {self.info['gpu']} · AI work size {self.info['working_size'][0]} × {self.info['working_size'][1]}")
                else:
                    self.status.set(data[0])
                    self.clock.set(f'Stopped after {time.perf_counter() - self.started:.1f}s')
                    if not data[1]:
                        messagebox.showerror('Enhancement stopped', data[0])
        if self.busy:
            self.clock.set(f'{time.perf_counter() - self.started:.1f}s elapsed · cancellation checked between GPU steps')
        self.root.after(100, self.poll)

    def save(self):
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(defaultextension='.png', initialfile=self.source_path.stem + '-detail.png', filetypes=[('PNG image', '*.png')])
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

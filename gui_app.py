"""
gui_app.py - Interactive GUI Application for 4-Class Rice Field State Classification (Dry, Flooded, Planted, Others).
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk, ImageDraw

from src.inference import RiceFieldPredictor
from src.model import CLASSES, CLASS_COLORS


class RiceFieldGUI:
    def __init__(self, root, model_path="rice_field_classifier.pth"):
        self.root = root
        self.root.title("🌾 Rice Field State Identifier (Dry, Flooded, Planted, Others)")
        self.root.geometry("1280x800")
        self.root.minsize(1000, 650)
        
        self.model_path = model_path
        self.predictor = RiceFieldPredictor(model_path=model_path)
        
        self.current_pil_image = None
        self.current_display_image = None
        self.last_result = None
        self.image_path = None
        
        self.status_var = tk.StringVar(value="Ready. Open an image or select a sample from Dataset to classify.")
        
        self._setup_ui()
        self._load_initial_sample()

    def _setup_ui(self):
        # Top toolbar
        toolbar = ttk.Frame(self.root, padding=6)
        toolbar.pack(fill=tk.X, side=tk.TOP)
        
        btn_open = ttk.Button(toolbar, text="📂 Open Image", command=self._on_open_image)
        btn_open.pack(side=tk.LEFT, padx=4)
        
        # Sample Quick Buttons
        ttk.Label(toolbar, text="Quick Samples:").pack(side=tk.LEFT, padx=(10, 4))
        
        samples = [
            ("🏜️ Dry", "Dataset/Dry/dry_01.png"),
            ("💧 Flooded", "Dataset/Flood/flood_01.png"),
            ("🌿 Planted", "Dataset/Planted/planted_01.png"),
            ("🌳 Others", "Dataset/Others/others_01.png")
        ]
        
        for label, path in samples:
            btn = ttk.Button(toolbar, text=label, command=lambda p=path: self._load_image(p))
            btn.pack(side=tk.LEFT, padx=2)
            
        btn_save = ttk.Button(toolbar, text="💾 Save Result", command=self._on_save)
        btn_save.pack(side=tk.RIGHT, padx=4)
        
        # Main Split Frame
        main_frame = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        
        # Left Frame: Canvas
        left_frame = ttk.Frame(main_frame)
        main_frame.add(left_frame, weight=3)
        
        canvas_container = ttk.Frame(left_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_container, bg="#1e222b", cursor="crosshair")
        self.h_scroll = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.v_scroll = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.canvas.yview)
        
        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        
        # Status Bar
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=(6, 4))
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Right Panel: Results & Probability Breakdown
        right_frame = ttk.Frame(main_frame, width=360, padding=10)
        main_frame.add(right_frame, weight=1)
        
        lbl_header = tk.Label(right_frame, text="🌾 Classification Results", font=("Helvetica", 14, "bold"), fg="#2e7d32")
        lbl_header.pack(anchor="w", pady=(0, 10))
        
        # Status Badge Card
        card_group = ttk.LabelFrame(right_frame, text="Identified Rice Field State", padding=12)
        card_group.pack(fill=tk.X, pady=6)
        
        self.lbl_status = tk.Label(card_group, text="--", font=("Helvetica", 16, "bold"), bg="#eceff1", fg="#37474f", padx=12, pady=10, relief=tk.GROOVE)
        self.lbl_status.pack(fill=tk.X, pady=4)
        
        self.lbl_conf = ttk.Label(card_group, text="Confidence: --", font=("Helvetica", 11))
        self.lbl_conf.pack(anchor="w", pady=4)
        
        self.lbl_coord = ttk.Label(card_group, text="Queried Coordinate: Full Image", font=("Helvetica", 10))
        self.lbl_coord.pack(anchor="w", pady=2)
        
        # Probability Bars
        prob_group = ttk.LabelFrame(right_frame, text="4-Class Probability Distribution", padding=12)
        prob_group.pack(fill=tk.BOTH, expand=True, pady=8)
        
        self.prob_widgets = {}
        for cname in CLASSES:
            row = ttk.Frame(prob_group)
            row.pack(fill=tk.X, pady=6)
            
            color_rgb = CLASS_COLORS[cname]
            color_hex = '#{:02x}{:02x}{:02x}'.format(*color_rgb)
            
            dot = tk.Canvas(row, width=14, height=14, bg=color_hex, highlightthickness=0)
            dot.pack(side=tk.LEFT, padx=(0, 6))
            
            lbl = ttk.Label(row, text=f"{cname:8s}", width=10, font=("Helvetica", 10, "bold"))
            lbl.pack(side=tk.LEFT)
            
            pbar = ttk.Progressbar(row, orient=tk.HORIZONTAL, length=120, mode='determinate')
            pbar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
            
            val_lbl = ttk.Label(row, text="0.0%", width=6, anchor="e", font=("Helvetica", 10))
            val_lbl.pack(side=tk.RIGHT)
            
            self.prob_widgets[cname] = (pbar, val_lbl)
            
        # Legend / Guide
        guide_group = ttk.LabelFrame(right_frame, text="Target Output Classes", padding=10)
        guide_group.pack(fill=tk.X, pady=6)
        
        descs = {
            "Dry": "Harvested / dry soil / ripening pale fields",
            "Flooded": "Flooded / wet water paddies",
            "Planted": "Active growing green rice crops",
            "Others": "Trees, obstacles, roads, non-field areas"
        }
        for cname in CLASSES:
            lbl_desc = ttk.Label(guide_group, text=f"• {cname}: {descs[cname]}", font=("Helvetica", 9), wraplength=300)
            lbl_desc.pack(anchor="w", pady=2)

    def _load_initial_sample(self):
        default_paths = ["Dataset/Planted/planted_01.png", "Dataset/Flood/flood_01.png", "Dataset/Dry/dry_01.png"]
        for p in default_paths:
            if os.path.exists(p):
                self._load_image(p)
                break

    def _load_image(self, path):
        try:
            if not os.path.exists(path):
                messagebox.showerror("File Not Found", f"Cannot find image: {path}")
                return
            self.image_path = path
            self.current_pil_image = Image.open(path).convert("RGB")
            self.current_display_image = self.current_pil_image.copy()
            
            # Predict full image
            self.last_result = self.predictor.predict(self.current_pil_image)
            self._update_sidebar(self.last_result)
            self._redraw_canvas()
            
            self.status_var.set(f"Loaded '{path}' ({self.current_pil_image.width}x{self.current_pil_image.height}). Predicted: {self.last_result['status']}")
        except Exception as e:
            messagebox.showerror("Error Loading Image", str(e))

    def _on_canvas_click(self, event):
        if self.current_pil_image is None:
            return
            
        cx = int(self.canvas.canvasx(event.x))
        cy = int(self.canvas.canvasy(event.y))
        
        if 0 <= cx < self.current_pil_image.width and 0 <= cy < self.current_pil_image.height:
            self.status_var.set(f"Analyzing coordinate ({cx}, {cy})...")
            self.root.update_idletasks()
            
            self.last_result = self.predictor.predict_coordinate(self.current_pil_image, (cx, cy))
            self.current_display_image = self.predictor._draw_coordinate_annotation(self.current_pil_image, self.last_result)
            
            self._update_sidebar(self.last_result)
            self._redraw_canvas()
            self.status_var.set(f"Field at ({cx}, {cy}): {self.last_result['status'].upper()} ({self.last_result['confidence']*100:.1f}%)")

    def _on_canvas_motion(self, event):
        if self.current_pil_image:
            cx = int(self.canvas.canvasx(event.x))
            cy = int(self.canvas.canvasy(event.y))
            if 0 <= cx < self.current_pil_image.width and 0 <= cy < self.current_pil_image.height:
                self.root.title(f"🌾 Rice Field Identifier - Cursor: ({cx}, {cy})")

    def _update_sidebar(self, result):
        status = result["status"]
        color_rgb = CLASS_COLORS.get(status, (100, 100, 100))
        color_hex = '#{:02x}{:02x}{:02x}'.format(*color_rgb)
        fg_color = "#ffffff" if status in ["Flooded", "Others"] else "#111111"
        
        self.lbl_status.config(text=status.upper(), bg=color_hex, fg=fg_color)
        self.lbl_conf.config(text=f"Confidence: {result['confidence']*100:.1f}%")
        
        if "coordinate" in result:
            cx, cy = result["coordinate"]
            self.lbl_coord.config(text=f"Queried Coordinate: X={cx}, Y={cy}")
        else:
            self.lbl_coord.config(text="Queried Region: Full Image")
            
        for cname, prob in result["probabilities"].items():
            if cname in self.prob_widgets:
                pbar, val_lbl = self.prob_widgets[cname]
                pbar["value"] = prob * 100.0
                val_lbl.config(text=f"{prob*100:.1f}%")

    def _redraw_canvas(self):
        if self.current_display_image is None:
            return
        self.tk_photo = ImageTk.PhotoImage(self.current_display_image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_photo)
        self.canvas.config(scrollregion=(0, 0, self.current_display_image.width, self.current_display_image.height))

    def _on_open_image(self):
        fpath = filedialog.askopenfilename(
            title="Select Satellite Crop Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif"), ("All Files", "*.*")]
        )
        if fpath:
            self._load_image(fpath)

    def _on_save(self):
        if self.current_display_image is None:
            return
        fpath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")],
            title="Save Classified Image"
        )
        if fpath:
            self.current_display_image.save(fpath)
            messagebox.showinfo("Saved", f"Image saved successfully to:\n{fpath}")


def main():
    root = tk.Tk()
    app = RiceFieldGUI(root, model_path="rice_field_classifier.pth")
    root.mainloop()


if __name__ == "__main__":
    main()

"""
gui_app.py - Interactive GUI Application for 4-Class Rice Field State Classification (Dry, Flooded, Planted, Others).
Supports single image inspection and recursive batch classification of any image placed in Input/ (and subdirectories).
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk

from src.inference import RiceFieldPredictor
from src.model import CLASSES, CLASS_COLORS


class RiceFieldGUI:
    def __init__(self, root, model_path="rice_field_classifier.pth"):
        self.root = root
        self.root.title("🌾 Rice Field State Identifier (Dry, Flooded, Planted, Others)")
        self.root.geometry("1280x820")
        self.root.minsize(1050, 680)
        
        self.model_path = model_path
        self.predictor = RiceFieldPredictor(model_path=model_path)
        
        self.current_pil_image = None
        self.current_display_image = None
        self.last_result = None
        self.image_path = None
        
        self.status_var = tk.StringVar(value="Ready. Open an image or batch classify 'Input/' folder.")
        
        self._setup_ui()
        self._load_initial_sample()

    def _setup_ui(self):
        # Top toolbar
        toolbar = ttk.Frame(self.root, padding=6)
        toolbar.pack(fill=tk.X, side=tk.TOP)
        
        btn_open = ttk.Button(toolbar, text="📂 Open Image", command=self._on_open_image)
        btn_open.pack(side=tk.LEFT, padx=4)
        
        btn_batch_input = ttk.Button(toolbar, text="⚡ Classify 'Input/' Folder", command=lambda: self._on_batch_classify("Input"))
        btn_batch_input.pack(side=tk.LEFT, padx=4)

        btn_batch_folder = ttk.Button(toolbar, text="📁 Batch Custom Folder...", command=self._on_open_folder)
        btn_batch_folder.pack(side=tk.LEFT, padx=4)
        
        # Sample Quick Buttons
        ttk.Label(toolbar, text="Samples:").pack(side=tk.LEFT, padx=(12, 4))
        
        samples = [
            ("🏜️ Dry", "Dataset/Dry/dry_01.png"),
            ("💧 Flooded", "Dataset/Flood/flood_01.png"),
            ("🌿 Planted", "Dataset/Planted/planted_01.png"),
            ("🌳 Others", "Dataset/Others/others_01.png")
        ]
        
        for label, path in samples:
            btn = ttk.Button(toolbar, text=label, command=lambda p=path: self._load_image(p))
            btn.pack(side=tk.LEFT, padx=2)
            
        btn_save = ttk.Button(toolbar, text="💾 Save Image", command=self._on_save)
        btn_save.pack(side=tk.RIGHT, padx=4)
        
        # Main Split Frame
        main_frame = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        
        # Left Frame: Canvas
        left_frame = ttk.Frame(main_frame)
        main_frame.add(left_frame, weight=3)
        
        canvas_container = ttk.Frame(left_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_container, bg="#1e222b", cursor="arrow")
        self.h_scroll = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.v_scroll = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.canvas.yview)
        
        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
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
        
        self.lbl_file = ttk.Label(card_group, text="File: --", font=("Helvetica", 10), wraplength=320)
        self.lbl_file.pack(anchor="w", pady=2)
        
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
            
            self.last_result = self.predictor.predict(self.current_pil_image)
            self._update_sidebar(self.last_result, path)
            self._redraw_canvas()
            
            self.status_var.set(f"Loaded '{os.path.basename(path)}' ({self.current_pil_image.width}x{self.current_pil_image.height}). Predicted: {self.last_result['status']} ({self.last_result['confidence']*100:.1f}%)")
        except Exception as e:
            messagebox.showerror("Error Loading Image", str(e))

    def _update_sidebar(self, result, path=None):
        status = result["status"]
        color_rgb = CLASS_COLORS.get(status, (100, 100, 100))
        color_hex = '#{:02x}{:02x}{:02x}'.format(*color_rgb)
        fg_color = "#ffffff" if status in ["Flooded", "Others"] else "#111111"
        
        self.lbl_status.config(text=status.upper(), bg=color_hex, fg=fg_color)
        self.lbl_conf.config(text=f"Confidence: {result['confidence']*100:.1f}%")
        
        if path:
            self.lbl_file.config(text=f"File: {os.path.basename(path)}")
            
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
            title="Select Rice Field Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.webp"), ("All Files", "*.*")]
        )
        if fpath:
            self._load_image(fpath)

    def _on_open_folder(self):
        folder = filedialog.askdirectory(title="Select Folder to Recursively Classify")
        if folder:
            self._on_batch_classify(folder)

    def _on_batch_classify(self, directory):
        if not os.path.exists(directory):
            messagebox.showerror("Directory Not Found", f"Directory '{directory}' does not exist.")
            return

        self.status_var.set(f"Scanning & classifying all images in '{directory}' (including subdirectories)...")
        self.root.update_idletasks()

        try:
            batch_result = self.predictor.predict_directory(input_dir=directory, recursive=True)
        except Exception as e:
            messagebox.showerror("Classification Error", str(e))
            self.status_var.set("Batch classification failed.")
            return

        total = batch_result["total_images"]
        if total == 0:
            messagebox.showinfo("No Images Found", f"No supported image files found in '{directory}' or its subdirectories.")
            self.status_var.set(f"No images found in '{directory}'.")
            return

        self._show_batch_results_dialog(batch_result, directory)
        self.status_var.set(f"Classified {total} image(s) from '{directory}'.")

    def _show_batch_results_dialog(self, batch_result, directory):
        dlg = tk.Toplevel(self.root)
        dlg.title(f"⚡ Batch Classification Results - '{directory}'")
        dlg.geometry("780x520")
        dlg.minsize(650, 400)

        # Header with summary statistics
        top_frame = ttk.Frame(dlg, padding=10)
        top_frame.pack(fill=tk.X)

        summary = batch_result["summary_counts"]
        summary_str = f"Total Images: {batch_result['total_images']}   |   " + "   |   ".join(
            [f"{c}: {summary[c]}" for c in CLASSES]
        )
        lbl_summary = ttk.Label(top_frame, text=summary_str, font=("Helvetica", 11, "bold"))
        lbl_summary.pack(anchor="w")

        lbl_hint = ttk.Label(top_frame, text="Double-click any item to open it in the main viewer.", font=("Helvetica", 9), foreground="#555555")
        lbl_hint.pack(anchor="w", pady=(2, 0))

        # Table / Treeview
        table_frame = ttk.Frame(dlg, padding=10)
        table_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("rel_path", "status", "confidence")
        tree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="browse")
        tree.heading("rel_path", text="Relative Path (including Subdirectories)")
        tree.heading("status", text="Predicted State")
        tree.heading("confidence", text="Confidence")

        tree.column("rel_path", width=440)
        tree.column("status", width=120, anchor="center")
        tree.column("confidence", width=100, anchor="e")

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        file_map = {}
        for item in batch_result["results"]:
            item_id = tree.insert(
                "",
                tk.END,
                values=(item["relative_path"], item["status"], f"{item['confidence']*100:.1f}%")
            )
            file_map[item_id] = item["full_path"]

        def on_item_select(event=None):
            selected = tree.selection()
            if selected:
                full_path = file_map.get(selected[0])
                if full_path and os.path.exists(full_path):
                    self._load_image(full_path)

        tree.bind("<Double-1>", on_item_select)

        # Action Buttons
        btn_frame = ttk.Frame(dlg, padding=10)
        btn_frame.pack(fill=tk.X)

        def export_csv():
            fpath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV File", "*.csv")],
                initialfile="classification_results.csv",
                title="Export Results to CSV"
            )
            if fpath:
                self.predictor.predict_directory(input_dir=directory, save_csv=fpath)
                messagebox.showinfo("Exported", f"Successfully exported to:\n{fpath}")

        def export_json():
            fpath = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON File", "*.json")],
                initialfile="classification_results.json",
                title="Export Results to JSON"
            )
            if fpath:
                self.predictor.predict_directory(input_dir=directory, save_json=fpath)
                messagebox.showinfo("Exported", f"Successfully exported to:\n{fpath}")

        btn_view = ttk.Button(btn_frame, text="🔍 View Selected in Main", command=on_item_select)
        btn_view.pack(side=tk.LEFT, padx=4)

        btn_csv = ttk.Button(btn_frame, text="📄 Export CSV", command=export_csv)
        btn_csv.pack(side=tk.LEFT, padx=4)

        btn_json = ttk.Button(btn_frame, text="📋 Export JSON", command=export_json)
        btn_json.pack(side=tk.LEFT, padx=4)

        btn_close = ttk.Button(btn_frame, text="Close", command=dlg.destroy)
        btn_close.pack(side=tk.RIGHT, padx=4)

    def _on_save(self):
        if self.current_display_image is None:
            return
        fpath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")],
            title="Save Image"
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


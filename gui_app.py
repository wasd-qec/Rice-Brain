"""
gui_app.py - Interactive Desktop Application & Review Studio for 4-Class Rice Field State Classification.
Classes: ['Dry', 'Flooded', 'Planted', 'Others']

Unified Table & Live Inspection Layout:
- Batch directory classification & single/multi image loading
- Image navigation (Table click, Up/Down arrow keys, Previous/Next buttons, Auto-Advance)
- High-visibility red highlighting for low-confidence (<80%) predictions
- 1-click human operator overrule controls with hotkeys [1-4] & [Space]
- Vertical Tips & Shortcuts guide
- Direct export of human corrections to Dataset/ for continuous training
"""

import os
import sys
import csv
import shutil
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import RiceFieldPredictor, save_reviewed_batch, export_overruled_to_dataset
from src.model import CLASSES, CLASS_COLORS


class RiceFieldGUI:
    def __init__(self, root, model_path="rice_field_classifier.pth"):
        self.root = root
        self.root.title("🌾 Rice Field State Identifier & Operator Review Studio")
        self.root.geometry("1260x840")
        self.root.minsize(1050, 680)

        self.model_path = model_path
        self.predictor = RiceFieldPredictor(model_path=model_path)

        # Batch state tracking
        self.current_batch = {
            "total_images": 0,
            "directory": os.path.abspath("Input"),
            "output_directory": os.path.abspath("Output"),
            "summary_counts": {c: 0 for c in CLASSES},
            "overruled_count": 0,
            "needs_review_count": 0,
            "results": [],
            "subdirectories": [],
            "generated_json_files": []
        }
        self.item_map = {}  # tree_item_id -> result dict
        self.current_inspected_item = None
        self.preview_photo_ref = [None]  # Keep in memory to prevent garbage collection

        # Tkinter variables
        self.status_var = tk.StringVar(value="Ready. Loading initial images...")
        self.summary_var = tk.StringVar(value="Total Images: 0   |   Dry: 0   |   Flooded: 0   |   Planted: 0   |   Others: 0")
        self.review_stat_var = tk.StringVar(value="⚠️ Low Confidence (<80%): 0   |   ✏️ Overruled by Operator: 0")

        self.var_filter_low_conf = tk.BooleanVar(value=False)
        self.var_filter_overruled = tk.BooleanVar(value=False)
        self.var_auto_advance = tk.BooleanVar(value=True)

        self._configure_styles()
        self._setup_ui()
        self._bind_shortcuts()

        # Automatically load and classify 'Input/' if images exist, or ready state
        self._initial_load()

    def _configure_styles(self):
        style = ttk.Style()
        # Ensure Treeview tag foregrounds display cleanly
        if "clam" in style.theme_names():
            try:
                style.theme_use("clam")
            except Exception:
                pass

    def _setup_ui(self):
        # 1. Top Summary Statistics Header
        header_frame = ttk.Frame(self.root, padding=(12, 6))
        header_frame.pack(fill=tk.X, side=tk.TOP)

        lbl_summary = ttk.Label(header_frame, textvariable=self.summary_var, font=("Helvetica", 11, "bold"))
        lbl_summary.pack(anchor="w")

        lbl_rev_summary = ttk.Label(header_frame, textvariable=self.review_stat_var, font=("Helvetica", 10, "bold"), foreground="#c62828")
        lbl_rev_summary.pack(anchor="w", pady=(2, 0))

        # 2. Main Action Toolbar (Open image, batch input, custom folder, prev, next, save image)
        toolbar = ttk.Frame(self.root, padding=(12, 4))
        toolbar.pack(fill=tk.X, side=tk.TOP)

        btn_open = ttk.Button(toolbar, text="📂 Open Image(s)...", command=self._on_open_images)
        btn_open.pack(side=tk.LEFT, padx=(0, 4))

        btn_batch_input = ttk.Button(toolbar, text="⚡ Classify 'Input/' Folder", command=lambda: self.load_directory("Input"))
        btn_batch_input.pack(side=tk.LEFT, padx=4)

        btn_batch_folder = ttk.Button(toolbar, text="📁 Batch Custom Folder...", command=self._on_open_custom_folder)
        btn_batch_folder.pack(side=tk.LEFT, padx=4)

        # Navigation Buttons
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        btn_prev = ttk.Button(toolbar, text="⏮ Previous", command=self._on_prev_image)
        btn_prev.pack(side=tk.LEFT, padx=3)

        btn_next = ttk.Button(toolbar, text="⏭ Next", command=self._on_next_image)
        btn_next.pack(side=tk.LEFT, padx=3)

        btn_save_img = ttk.Button(toolbar, text="💾 Save Image", command=self._on_save_current_image)
        btn_save_img.pack(side=tk.RIGHT, padx=4)

        # 3. Filter & Auto-Advance Options Bar
        filter_frame = ttk.Frame(self.root, padding=(12, 2))
        filter_frame.pack(fill=tk.X, side=tk.TOP)

        chk_low = ttk.Checkbutton(
            filter_frame,
            text="⚠️ Low Confidence (<80%) Only",
            variable=self.var_filter_low_conf,
            command=self._populate_table
        )
        chk_low.pack(side=tk.LEFT, padx=(0, 12))

        chk_over = ttk.Checkbutton(
            filter_frame,
            text="✏️ Overruled Only",
            variable=self.var_filter_overruled,
            command=self._populate_table
        )
        chk_over.pack(side=tk.LEFT, padx=(0, 12))

        chk_advance = ttk.Checkbutton(
            filter_frame,
            text="⚡ Auto-Advance to Next Image on Overrule",
            variable=self.var_auto_advance
        )
        chk_advance.pack(side=tk.LEFT, padx=(0, 12))

        # 4. Operator Overrule Controls Bar
        overrule_bar = ttk.LabelFrame(self.root, text="Operator Overrule Controls (Select Image & Press 1-4)", padding=(8, 4))
        overrule_bar.pack(fill=tk.X, padx=12, pady=4, side=tk.TOP)

        btn_dry = ttk.Button(overrule_bar, text="[1] 🏜️ Dry", command=lambda: self.apply_overrule("Dry"))
        btn_dry.pack(side=tk.LEFT, padx=3)

        btn_flood = ttk.Button(overrule_bar, text="[2] 💧 Flooded", command=lambda: self.apply_overrule("Flooded"))
        btn_flood.pack(side=tk.LEFT, padx=3)

        btn_plant = ttk.Button(overrule_bar, text="[3] 🌿 Planted", command=lambda: self.apply_overrule("Planted"))
        btn_plant.pack(side=tk.LEFT, padx=3)

        btn_other = ttk.Button(overrule_bar, text="[4] 🌳 Others", command=lambda: self.apply_overrule("Others"))
        btn_other.pack(side=tk.LEFT, padx=3)

        btn_reset = ttk.Button(overrule_bar, text="[Space] 🔄 Accept / Reset to AI", command=self.reset_to_ai)
        btn_reset.pack(side=tk.LEFT, padx=(12, 3))

        # 5. Status Bar at Bottom
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=(8, 4))
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        # 6. Footer Action Buttons (Save Reports, Add to Dataset, Open Folder, Export CSV)
        footer_frame = ttk.Frame(self.root, padding=(12, 6))
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)

        btn_save_reports = ttk.Button(footer_frame, text="💾 Save Reviewed Reports", command=self._on_save_reports)
        btn_save_reports.pack(side=tk.LEFT, padx=3)

        btn_export_dataset = ttk.Button(footer_frame, text="📥 Add Overruled to Dataset", command=self._on_add_overruled_to_dataset)
        btn_export_dataset.pack(side=tk.LEFT, padx=3)

        btn_open_folder = ttk.Button(footer_frame, text="📁 Open Output Folder", command=self._on_open_output_folder)
        btn_open_folder.pack(side=tk.LEFT, padx=3)

        btn_csv = ttk.Button(footer_frame, text="📄 Export CSV", command=self._on_export_csv)
        btn_csv.pack(side=tk.LEFT, padx=3)

        # 7. Main Paned Split View: Left = Treeview Table, Right = Live Image Inspector
        center_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        center_pane.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        # Left Side: Table of images
        table_frame = ttk.Frame(center_pane)
        center_pane.add(table_frame, weight=3)

        cols = ("rel_path", "ai_status", "confidence", "final_decision")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="browse")
        self.tree.heading("rel_path", text="Relative Path")
        self.tree.heading("ai_status", text="AI Guess")
        self.tree.heading("confidence", text="Confidence")
        self.tree.heading("final_decision", text="Final Decision")

        self.tree.column("rel_path", width=330, anchor="w")
        self.tree.column("ai_status", width=90, anchor="center")
        self.tree.column("confidence", width=105, anchor="center")
        self.tree.column("final_decision", width=140, anchor="center")

        # Color tags: Red for low confidence (<80%), amber for overruled
        self.tree.tag_configure("tag_low_conf", foreground="#d32f2f", font=("Helvetica", 9, "bold"))
        self.tree.tag_configure("tag_overruled", foreground="#b78103", background="#fff3e0", font=("Helvetica", 9, "bold"))
        self.tree.tag_configure("tag_normal", foreground="#222222")

        vsb = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # Right Side: Live Image Inspection & Breakdown Panel
        preview_panel = ttk.LabelFrame(center_pane, text="🖼️ Live Image Inspection & Breakdown", padding=10)
        center_pane.add(preview_panel, weight=2)

        # Image Display Container
        img_box = tk.Frame(preview_panel, bg="#1e222b", width=360, height=250)
        img_box.pack(fill=tk.X, pady=(0, 6))
        img_box.pack_propagate(False)

        self.lbl_preview_img = tk.Label(img_box, bg="#1e222b", text="Select an image to preview", fg="#888888")
        self.lbl_preview_img.pack(expand=True)

        self.lbl_preview_filename = ttk.Label(preview_panel, text="File: --", font=("Helvetica", 9), wraplength=340)
        self.lbl_preview_filename.pack(anchor="w", pady=(0, 1))

        self.lbl_preview_coords = ttk.Label(preview_panel, text="📍 GPS: --", font=("Helvetica", 9), foreground="#1565c0", wraplength=340)
        self.lbl_preview_coords.pack(anchor="w", pady=(0, 2))

        # Status & Decision Banner
        self.lbl_preview_decision = tk.Label(
            preview_panel, text="--", font=("Helvetica", 14, "bold"),
            bg="#eceff1", fg="#333333", padx=10, pady=5, relief=tk.GROOVE
        )
        self.lbl_preview_decision.pack(fill=tk.X, pady=4)

        self.lbl_preview_ai = tk.Label(preview_panel, text="AI Guess: --", font=("Helvetica", 10), anchor="w")
        self.lbl_preview_ai.pack(fill=tk.X, pady=2)

        # 4-Class Confidence Distribution
        prob_group = ttk.LabelFrame(preview_panel, text="4-Class Confidence Distribution", padding=8)
        prob_group.pack(fill=tk.X, pady=(6, 4))

        self.prob_widgets = {}
        for cname in CLASSES:
            row = ttk.Frame(prob_group)
            row.pack(fill=tk.X, pady=2)

            c_rgb = CLASS_COLORS[cname]
            c_hex = '#{:02x}{:02x}{:02x}'.format(*c_rgb)
            dot = tk.Canvas(row, width=10, height=10, bg=c_hex, highlightthickness=0)
            dot.pack(side=tk.LEFT, padx=(0, 5))

            lbl = ttk.Label(row, text=f"{cname:7s}", width=8, font=("Helvetica", 9, "bold"))
            lbl.pack(side=tk.LEFT)

            pbar = ttk.Progressbar(row, orient=tk.HORIZONTAL, length=110, mode='determinate')
            pbar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

            val_lbl = ttk.Label(row, text="0.0%", width=6, anchor="e", font=("Helvetica", 9))
            val_lbl.pack(side=tk.RIGHT)

            self.prob_widgets[cname] = (pbar, val_lbl)

        # Vertical Tips & Shortcuts Card (Item 5 in user note)
        tips_box = ttk.LabelFrame(preview_panel, text="💡 Tips & Shortcuts", padding=8)
        tips_box.pack(fill=tk.X, pady=(8, 0))

        tips = [
            ("• [1] Dry  |  [2] Flooded  |  [3] Planted  |  [4] Others", "#333333"),
            ("• [Space] Accept AI / Reset Overrule", "#333333"),
            ("• [↑ / ↓] or [Prev / Next] to navigate through images", "#333333"),
            ("• ⚠️ Red text marks low confidence (<80%) for review", "#c62828"),
            ("• ⚡ Auto-Advance jumps to next image on overrule", "#1565c0"),
            ("• Click 'Add Overruled to Dataset' to continuously train AI", "#2e7d32")
        ]
        for tip_text, tip_color in tips:
            lbl_tip = tk.Label(tips_box, text=tip_text, font=("Helvetica", 8, "bold" if tip_color != "#333333" else "normal"), fg=tip_color, anchor="w")
            lbl_tip.pack(fill=tk.X, pady=1)

    def _bind_shortcuts(self):
        self.root.bind("1", lambda e: self.apply_overrule("Dry"))
        self.root.bind("2", lambda e: self.apply_overrule("Flooded"))
        self.root.bind("3", lambda e: self.apply_overrule("Planted"))
        self.root.bind("4", lambda e: self.apply_overrule("Others"))
        self.root.bind("<space>", lambda e: self.reset_to_ai())

    def _initial_load(self):
        """Automatically classify and populate Input/ on startup if images exist."""
        if os.path.exists("Input"):
            valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".webp"}
            has_images = any(
                os.path.splitext(f)[1].lower() in valid_exts
                for _, _, files in os.walk("Input")
                for f in files
            )
            if has_images:
                self.load_directory("Input", silent=True)
                return

        self.status_var.set("Ready. Open an image or batch classify a folder.")

    def load_directory(self, directory="Input", silent=False):
        """Scans and batch classifies all images in directory (and subdirectories)."""
        if not os.path.exists(directory):
            if not silent:
                messagebox.showerror("Directory Not Found", f"Directory '{directory}' does not exist.")
            return

        self.status_var.set(f"Scanning & classifying all images in '{directory}'...")
        self.root.update_idletasks()

        try:
            batch_res = self.predictor.predict_directory(input_dir=directory, recursive=True, output_dir="Output")
            self.current_batch = batch_res
            self._update_header_texts()
            self._populate_table()

            total = batch_res["total_images"]
            self.status_var.set(f"Loaded {total} image(s) from '{directory}'. JSON reports generated in 'Output/'.")
        except Exception as e:
            if not silent:
                messagebox.showerror("Classification Error", str(e))
            self.status_var.set(f"Error scanning '{directory}': {e}")

    def _on_open_images(self):
        """Selects one or more images from file dialog and loads them into the review table."""
        files = filedialog.askopenfilenames(
            title="Select Rice Field Image(s)",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.webp"), ("All Files", "*.*")]
        )
        if not files:
            return

        self.status_var.set(f"Classifying {len(files)} image(s)...")
        self.root.update_idletasks()

        new_results = []
        summary_counts = {c: 0 for c in CLASSES}

        for fpath in files:
            try:
                pred = self.predictor.predict(fpath)
                fname = os.path.basename(fpath)
                conf = pred["confidence"]
                item = {
                    "filename": fname,
                    "relative_path": fname,
                    "relative_directory": ".",
                    "full_path": os.path.abspath(fpath),
                    "ai_status": pred["status"],
                    "status": pred["status"],
                    "confidence": conf,
                    "needs_review": bool(conf < 0.80),
                    "is_overruled": False,
                    "operator_label": None,
                    "reviewed_at": None,
                    "probabilities": pred["probabilities"]
                }
                new_results.append(item)
                summary_counts[pred["status"]] += 1
            except Exception as e:
                print(f"[!] Error classifying '{fpath}': {e}")

        if not new_results:
            messagebox.showerror("Error", "Could not classify any of the selected images.")
            return

        self.current_batch = {
            "total_images": len(new_results),
            "directory": os.path.dirname(files[0]),
            "output_directory": os.path.abspath("Output"),
            "summary_counts": summary_counts,
            "overruled_count": 0,
            "needs_review_count": sum(1 for i in new_results if i["needs_review"]),
            "results": new_results,
            "subdirectories": ["."],
            "generated_json_files": []
        }

        self._update_header_texts()
        self._populate_table()
        self.status_var.set(f"Loaded and classified {len(new_results)} image(s).")

    def _on_open_custom_folder(self):
        folder = filedialog.askdirectory(title="Select Folder to Batch Classify")
        if folder:
            self.load_directory(folder)

    def _update_header_texts(self):
        summary = self.current_batch.get("summary_counts", {c: 0 for c in CLASSES})
        s_str = f"Total Images: {self.current_batch.get('total_images', 0)}   |   " + "   |   ".join(
            [f"{c}: {summary.get(c, 0)}" for c in CLASSES]
        )
        self.summary_var.set(s_str)

        results = self.current_batch.get("results", [])
        overruled = sum(1 for item in results if item.get("is_overruled"))
        needs_rev = sum(1 for item in results if item.get("needs_review") and not item.get("is_overruled"))
        self.current_batch["overruled_count"] = overruled
        self.current_batch["needs_review_count"] = needs_rev
        r_str = f"⚠️ Low Confidence (<80%): {needs_rev}   |   ✏️ Overruled by Operator: {overruled}"
        self.review_stat_var.set(r_str)

    def _populate_table(self):
        self.tree.delete(*self.tree.get_children())
        self.item_map.clear()

        only_low = self.var_filter_low_conf.get()
        only_over = self.var_filter_overruled.get()

        first_id = None
        results = self.current_batch.get("results", [])

        for item in results:
            is_low = item.get("needs_review", False)
            is_over = item.get("is_overruled", False)

            if only_low and not is_low:
                continue
            if only_over and not is_over:
                continue

            conf_val = item["confidence"] * 100.0
            conf_str = f"{conf_val:.1f}%" + (" ⚠️" if is_low else " ✅")

            if is_over:
                decision_str = f"✏️ {item['status']} [OVERRULED]"
                row_tag = "tag_overruled"
            elif is_low:
                decision_str = item["status"]
                row_tag = "tag_low_conf"  # Red text for < 80% confidence
            else:
                decision_str = item["status"]
                row_tag = "tag_normal"

            item_id = self.tree.insert(
                "",
                tk.END,
                values=(item["relative_path"], item["ai_status"], conf_str, decision_str),
                tags=(row_tag,)
            )
            self.item_map[item_id] = item
            if first_id is None:
                first_id = item_id

        if first_id:
            self.tree.selection_set(first_id)
            self.tree.focus(first_id)
            self.tree.see(first_id)
            self._update_inspector(self.item_map[first_id])
        else:
            self._clear_inspector()

    def _on_tree_select(self, event=None):
        selected = self.tree.selection()
        if selected:
            res_item = self.item_map.get(selected[0])
            if res_item:
                self._update_inspector(res_item)

    def _update_inspector(self, res_item):
        if not res_item:
            self._clear_inspector()
            return

        self.current_inspected_item = res_item
        full_path = res_item.get("full_path")

        if full_path and os.path.exists(full_path):
            try:
                pil_im = Image.open(full_path).convert("RGB")
                w, h = pil_im.size
                max_w, max_h = 350, 240
                scale = min(max_w / w, max_h / h, 1.0)
                new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                resized = pil_im.resize((new_w, new_h), Image.Resampling.BILINEAR)
                tk_img = ImageTk.PhotoImage(resized)
                self.preview_photo_ref[0] = tk_img
                self.lbl_preview_img.config(image=tk_img, text="")
                self.lbl_preview_filename.config(text=f"📁 {res_item['relative_path']} ({w}x{h} px)")
            except Exception as err:
                self.lbl_preview_img.config(image="", text=f"Error loading: {err}")
        else:
            self.lbl_preview_img.config(image="", text="File not found")

        gps = res_item.get("gps")
        if gps is None and full_path and os.path.exists(full_path):
            from src.inference import extract_image_gps
            gps = extract_image_gps(full_path)
            res_item["gps"] = gps

        if gps and gps.get("latitude") is not None:
            alt_txt = f" | Alt: {gps['altitude']}m" if gps.get("altitude") is not None else ""
            self.lbl_preview_coords.config(text=f"📍 GPS: {gps['latitude']:.4f}° N, {gps['longitude']:.4f}° E{alt_txt}")
        else:
            self.lbl_preview_coords.config(text="📍 GPS: No coordinates in metadata")

        status = res_item["status"]
        color_rgb = CLASS_COLORS.get(status, (100, 100, 100))
        color_hex = '#{:02x}{:02x}{:02x}'.format(*color_rgb)
        fg_color = "#ffffff" if status in ["Flooded", "Others"] else "#111111"

        is_over = res_item.get("is_overruled", False)
        tag_text = f"✏️ {status.upper()} [OVERRULED]" if is_over else status.upper()
        self.lbl_preview_decision.config(text=tag_text, bg=color_hex, fg=fg_color)

        ai_st = res_item.get("ai_status", status)
        conf_val = res_item["confidence"] * 100.0
        is_low = res_item.get("needs_review")

        # Highlight low confidence in red (Item 3 in user note)
        if is_low:
            self.lbl_preview_ai.config(
                text=f"AI Guess: {ai_st} ({conf_val:.1f}%) ⚠️ [Low Confidence (<80%)]",
                fg="#d32f2f",
                font=("Helvetica", 10, "bold")
            )
        else:
            self.lbl_preview_ai.config(
                text=f"AI Guess: {ai_st} ({conf_val:.1f}%) ✅ [High Confidence]",
                fg="#2e7d32",
                font=("Helvetica", 10, "bold")
            )

        probs = res_item.get("probabilities", {})
        for cname in CLASSES:
            p_val = probs.get(cname, 0.0)
            if cname in self.prob_widgets:
                pbar, val_lbl = self.prob_widgets[cname]
                pbar["value"] = p_val * 100.0
                val_lbl.config(text=f"{p_val*100:.1f}%")

    def _clear_inspector(self):
        self.current_inspected_item = None
        self.lbl_preview_img.config(image="", text="No image selected")
        self.lbl_preview_filename.config(text="File: --")
        self.lbl_preview_decision.config(text="--", bg="#eceff1", fg="#333333")
        self.lbl_preview_ai.config(text="AI Guess: --", fg="#333333", font=("Helvetica", 10))
        for cname in CLASSES:
            if cname in self.prob_widgets:
                pbar, val_lbl = self.prob_widgets[cname]
                pbar["value"] = 0.0
                val_lbl.config(text="0.0%")

    def apply_overrule(self, new_class):
        """Overrules the selected image's classification to new_class."""
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo("No Selection", "Please select an image row from the table first.")
            return

        res_item = self.item_map.get(selected[0])
        if not res_item:
            return

        res_item["is_overruled"] = True
        res_item["operator_label"] = new_class
        res_item["status"] = new_class
        res_item["reviewed_at"] = datetime.now().isoformat()

        conf_val = res_item["confidence"] * 100.0
        conf_str = f"{conf_val:.1f}%" + (" ⚠️" if res_item.get("needs_review") else " ✅")
        decision_str = f"✏️ {new_class} [OVERRULED]"

        self.tree.item(selected[0], values=(res_item["relative_path"], res_item["ai_status"], conf_str, decision_str), tags=("tag_overruled",))

        # Update batch summary counts
        summary = {c: 0 for c in CLASSES}
        for itm in self.current_batch.get("results", []):
            st = itm.get("status")
            if st in summary:
                summary[st] += 1
        self.current_batch["summary_counts"] = summary

        self._update_header_texts()
        self._update_inspector(res_item)
        self.status_var.set(f"Overruled '{res_item['filename']}' -> {new_class}")

        self._advance_to_next()

    def reset_to_ai(self):
        """Resets the selected image's decision back to the original AI prediction."""
        selected = self.tree.selection()
        if not selected:
            return

        res_item = self.item_map.get(selected[0])
        if not res_item:
            return

        res_item["is_overruled"] = False
        res_item["operator_label"] = None
        res_item["status"] = res_item["ai_status"]
        res_item["reviewed_at"] = None

        conf_val = res_item["confidence"] * 100.0
        is_low = res_item.get("needs_review", False)
        conf_str = f"{conf_val:.1f}%" + (" ⚠️" if is_low else " ✅")
        row_tag = "tag_low_conf" if is_low else "tag_normal"

        self.tree.item(selected[0], values=(res_item["relative_path"], res_item["ai_status"], conf_str, res_item["status"]), tags=(row_tag,))

        # Update batch summary counts
        summary = {c: 0 for c in CLASSES}
        for itm in self.current_batch.get("results", []):
            st = itm.get("status")
            if st in summary:
                summary[st] += 1
        self.current_batch["summary_counts"] = summary

        self._update_header_texts()
        self._update_inspector(res_item)
        self.status_var.set(f"Reset '{res_item['filename']}' -> AI: {res_item['ai_status']}")

        self._advance_to_next()

    def _advance_to_next(self):
        if not self.var_auto_advance.get():
            return
        selected = self.tree.selection()
        if selected:
            next_id = self.tree.next(selected[0])
            if next_id:
                self.tree.selection_set(next_id)
                self.tree.focus(next_id)
                self.tree.see(next_id)
                res_item = self.item_map.get(next_id)
                if res_item:
                    self._update_inspector(res_item)

    def _on_prev_image(self):
        selected = self.tree.selection()
        if selected:
            prev_id = self.tree.prev(selected[0])
            if prev_id:
                self.tree.selection_set(prev_id)
                self.tree.focus(prev_id)
                self.tree.see(prev_id)
                res_item = self.item_map.get(prev_id)
                if res_item:
                    self._update_inspector(res_item)
        else:
            children = self.tree.get_children()
            if children:
                self.tree.selection_set(children[0])
                self.tree.focus(children[0])
                self.tree.see(children[0])
                self._update_inspector(self.item_map.get(children[0]))

    def _on_next_image(self):
        selected = self.tree.selection()
        if selected:
            next_id = self.tree.next(selected[0])
            if next_id:
                self.tree.selection_set(next_id)
                self.tree.focus(next_id)
                self.tree.see(next_id)
                res_item = self.item_map.get(next_id)
                if res_item:
                    self._update_inspector(res_item)
        else:
            children = self.tree.get_children()
            if children:
                self.tree.selection_set(children[0])
                self.tree.focus(children[0])
                self.tree.see(children[0])
                self._update_inspector(self.item_map.get(children[0]))

    def _on_save_current_image(self):
        if not self.current_inspected_item:
            messagebox.showinfo("No Image Selected", "Please select an image in the table to save.")
            return
        fpath = self.current_inspected_item.get("full_path")
        if not fpath or not os.path.exists(fpath):
            messagebox.showerror("Error", "Original image file could not be found.")
            return

        dest = filedialog.asksaveasfilename(
            defaultextension=os.path.splitext(fpath)[1],
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"), ("All Files", "*.*")],
            initialfile=self.current_inspected_item["filename"],
            title="Save Inspected Image"
        )
        if dest:
            try:
                shutil.copy2(fpath, dest)
                self.status_var.set(f"Saved image copy to: {dest}")
                messagebox.showinfo("Saved", f"Image saved successfully to:\n{dest}")
            except Exception as e:
                messagebox.showerror("Save Error", str(e))

    def _on_save_reports(self):
        if not self.current_batch or not self.current_batch.get("results"):
            messagebox.showinfo("No Data", "No classification results to save.")
            return
        try:
            saved = save_reviewed_batch(self.current_batch, output_dir="Output")
            self.status_var.set(f"Updated {len(saved)} report(s) in 'Output/'.")
            messagebox.showinfo("Saved", "Successfully updated all JSON reports in 'Output/' with operator review decisions!")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _on_add_overruled_to_dataset(self):
        if not self.current_batch or not self.current_batch.get("results"):
            messagebox.showinfo("No Images", "No images currently loaded.")
            return

        overruled_items = [itm for itm in self.current_batch["results"] if itm.get("is_overruled")]
        if not overruled_items:
            messagebox.showinfo("No Overruled Images", "No images have been overruled yet.\nSelect rows and overrule them with [1-4] first.")
            return

        count = len(overruled_items)
        msg = f"Add {count} overruled image(s) to 'Dataset/' for retraining?\n\nThis will copy corrected photos into Dataset/ so the AI learns from your corrections next time you run 'python train.py'."
        if messagebox.askyesno("Confirm Export to Dataset", msg):
            try:
                copied = export_overruled_to_dataset(overruled_items, dataset_dir="Dataset")
                self.status_var.set(f"Copied {len(copied)} image(s) into 'Dataset/'.")
                messagebox.showinfo(
                    "Dataset Retraining Ready",
                    f"Successfully copied {len(copied)} image(s) into 'Dataset/'!\n\nRun 'python train.py' whenever you're ready to train the AI on these corrections."
                )
            except Exception as e:
                messagebox.showerror("Export Error", str(e))

    def _on_open_output_folder(self):
        out_p = os.path.abspath("Output")
        os.makedirs(out_p, exist_ok=True)
        if hasattr(os, "startfile"):
            os.startfile(out_p)
        else:
            import subprocess
            subprocess.Popen(["explorer", out_p])

    def _on_export_csv(self):
        if not self.current_batch or not self.current_batch.get("results"):
            messagebox.showinfo("No Data", "No results to export.")
            return

        fpath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV File", "*.csv")],
            initialfile="classification_results.csv",
            title="Export Results to CSV"
        )
        if fpath:
            try:
                with open(fpath, "w", newline="", encoding="utf-8") as cf:
                    writer = csv.writer(cf)
                    header = [
                        "filename", "relative_path", "subdirectory",
                        "ai_status", "final_decision", "confidence",
                        "needs_review", "is_overruled", "operator_label"
                    ] + [f"prob_{c}" for c in CLASSES]
                    writer.writerow(header)

                    for item in self.current_batch["results"]:
                        row = [
                            item["filename"],
                            item["relative_path"],
                            item.get("relative_directory", "."),
                            item.get("ai_status", item["status"]),
                            item["status"],
                            f"{item['confidence']*100:.2f}%",
                            item.get("needs_review", False),
                            item.get("is_overruled", False),
                            item.get("operator_label") or ""
                        ] + [f"{item.get('probabilities', {}).get(c, 0.0)*100:.2f}%" for c in CLASSES]
                        writer.writerow(row)

                self.status_var.set(f"Exported CSV report to: {fpath}")
                messagebox.showinfo("Exported", f"Successfully exported to:\n{fpath}")
            except Exception as e:
                messagebox.showerror("Export Error", str(e))


def main():
    root = tk.Tk()
    app = RiceFieldGUI(root, model_path="rice_field_classifier.pth")
    root.mainloop()


if __name__ == "__main__":
    main()

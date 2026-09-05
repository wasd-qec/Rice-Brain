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

from src.inference import RiceFieldPredictor, save_reviewed_batch, export_overruled_to_dataset
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
            batch_result = self.predictor.predict_directory(input_dir=directory, recursive=True, output_dir="Output")
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
        self.status_var.set(f"Classified {total} image(s). JSON reports generated in 'Output/'.")

    def _show_batch_results_dialog(self, batch_result, directory):
        dlg = tk.Toplevel(self.root)
        dlg.title(f"🌾 Operator Review Studio - '{directory}'")
        dlg.geometry("1120x680")
        dlg.minsize(960, 520)

        # Header with summary statistics
        top_frame = ttk.Frame(dlg, padding=(12, 8))
        top_frame.pack(fill=tk.X)

        summary_var = tk.StringVar()
        review_stat_var = tk.StringVar()

        def update_header_texts():
            summary = batch_result["summary_counts"]
            s_str = f"Total Images: {batch_result['total_images']}   |   " + "   |   ".join(
                [f"{c}: {summary[c]}" for c in CLASSES]
            )
            summary_var.set(s_str)

            overruled = sum(1 for item in batch_result["results"] if item.get("is_overruled"))
            needs_rev = sum(1 for item in batch_result["results"] if item.get("needs_review") and not item.get("is_overruled"))
            r_str = f"⚠️ Low Confidence (<80%): {needs_rev}   |   ✏️ Overruled by Operator: {overruled}"
            review_stat_var.set(r_str)

        update_header_texts()

        lbl_summary = ttk.Label(top_frame, textvariable=summary_var, font=("Helvetica", 11, "bold"))
        lbl_summary.pack(anchor="w")

        lbl_rev_summary = ttk.Label(top_frame, textvariable=review_stat_var, font=("Helvetica", 10), foreground="#c62828")
        lbl_rev_summary.pack(anchor="w", pady=(2, 0))

        # Filter & Auto-Advance Bar
        filter_frame = ttk.Frame(dlg, padding=(12, 2))
        filter_frame.pack(fill=tk.X)

        var_filter_low_conf = tk.BooleanVar(value=False)
        var_filter_overruled = tk.BooleanVar(value=False)
        var_auto_advance = tk.BooleanVar(value=True)

        chk_low = ttk.Checkbutton(filter_frame, text="⚠️ Low Confidence (<80%) Only", variable=var_filter_low_conf, command=lambda: populate_table())
        chk_low.pack(side=tk.LEFT, padx=(0, 12))

        chk_over = ttk.Checkbutton(filter_frame, text="✏️ Overruled Only", variable=var_filter_overruled, command=lambda: populate_table())
        chk_over.pack(side=tk.LEFT, padx=(0, 12))

        chk_advance = ttk.Checkbutton(filter_frame, text="⚡ Auto-Advance to Next Image on Overrule", variable=var_auto_advance)
        chk_advance.pack(side=tk.LEFT, padx=(0, 12))

        lbl_hint = ttk.Label(filter_frame, text="Keys: [1] Dry | [2] Flooded | [3] Planted | [4] Others | [Space] Accept AI", font=("Helvetica", 9), foreground="#555555")
        lbl_hint.pack(side=tk.RIGHT)

        # Overrule Action Toolbar
        overrule_bar = ttk.LabelFrame(dlg, text="Operator Overrule Controls (Select Image & Press 1-4)", padding=6)
        overrule_bar.pack(fill=tk.X, padx=12, pady=4)

        # Center PanedWindow: Left = Table, Right = Live Image Inspector
        center_pane = ttk.PanedWindow(dlg, orient=tk.HORIZONTAL)
        center_pane.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        # Left side: Table
        table_frame = ttk.Frame(center_pane)
        center_pane.add(table_frame, weight=3)

        cols = ("rel_path", "ai_status", "confidence", "final_decision")
        tree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="browse")
        tree.heading("rel_path", text="Relative Path")
        tree.heading("ai_status", text="AI Guess")
        tree.heading("confidence", text="Confidence")
        tree.heading("final_decision", text="Final Decision")

        tree.column("rel_path", width=340)
        tree.column("ai_status", width=95, anchor="center")
        tree.column("confidence", width=95, anchor="center")
        tree.column("final_decision", width=140, anchor="center")

        tree.tag_configure("tag_overruled", background="#fff3e0", foreground="#b78103")
        tree.tag_configure("tag_low_conf", background="#fffde7", foreground="#bf360c")
        tree.tag_configure("tag_normal", background="#ffffff")

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Right side: Live Image Inspector Panel
        preview_panel = ttk.LabelFrame(center_pane, text="🖼️ Live Image Inspection & Breakdown", padding=10)
        center_pane.add(preview_panel, weight=2)

        # Preview Image Box
        img_box = tk.Frame(preview_panel, bg="#1e222b", width=340, height=250)
        img_box.pack(fill=tk.X, pady=(0, 4))
        img_box.pack_propagate(False)

        lbl_preview_img = tk.Label(img_box, bg="#1e222b", text="Select an image to preview", fg="#888888")
        lbl_preview_img.pack(expand=True)

        lbl_preview_filename = ttk.Label(preview_panel, text="File: --", font=("Helvetica", 9), wraplength=320)
        lbl_preview_filename.pack(anchor="w", pady=(0, 2))

        # Status & AI banner
        lbl_preview_decision = tk.Label(
            preview_panel, text="--", font=("Helvetica", 13, "bold"),
            bg="#eceff1", fg="#333333", padx=8, pady=4, relief=tk.GROOVE
        )
        lbl_preview_decision.pack(fill=tk.X, pady=2)

        lbl_preview_ai = ttk.Label(preview_panel, text="AI: --", font=("Helvetica", 10))
        lbl_preview_ai.pack(anchor="w", pady=2)

        # Mini Probability Distribution
        prob_group_dlg = ttk.LabelFrame(preview_panel, text="4-Class Confidence Distribution", padding=6)
        prob_group_dlg.pack(fill=tk.X, pady=(6, 0))

        dlg_prob_widgets = {}
        for cname in CLASSES:
            row = ttk.Frame(prob_group_dlg)
            row.pack(fill=tk.X, pady=2)
            c_rgb = CLASS_COLORS[cname]
            c_hex = '#{:02x}{:02x}{:02x}'.format(*c_rgb)
            dot = tk.Canvas(row, width=10, height=10, bg=c_hex, highlightthickness=0)
            dot.pack(side=tk.LEFT, padx=(0, 4))
            lbl = ttk.Label(row, text=f"{cname:7s}", width=8, font=("Helvetica", 9, "bold"))
            lbl.pack(side=tk.LEFT)
            pbar = ttk.Progressbar(row, orient=tk.HORIZONTAL, length=100, mode='determinate')
            pbar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
            val_lbl = ttk.Label(row, text="0.0%", width=6, anchor="e", font=("Helvetica", 9))
            val_lbl.pack(side=tk.RIGHT)
            dlg_prob_widgets[cname] = (pbar, val_lbl)

        # Keep PhotoImage in memory so GC doesn't drop it
        preview_photo_ref = [None]

        def update_inspector(res_item):
            if not res_item:
                return

            full_path = res_item.get("full_path")
            if full_path and os.path.exists(full_path):
                try:
                    pil_im = Image.open(full_path).convert("RGB")
                    w, h = pil_im.size
                    max_w, max_h = 330, 240
                    scale = min(max_w / w, max_h / h, 1.0)
                    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                    resized = pil_im.resize((new_w, new_h), Image.Resampling.BILINEAR)
                    tk_img = ImageTk.PhotoImage(resized)
                    preview_photo_ref[0] = tk_img
                    lbl_preview_img.config(image=tk_img, text="")
                    lbl_preview_filename.config(text=f"📁 {res_item['relative_path']} ({w}x{h} px)")
                except Exception as err:
                    lbl_preview_img.config(image="", text=f"Error loading: {err}")
            else:
                lbl_preview_img.config(image="", text="File not found")

            status = res_item["status"]
            color_rgb = CLASS_COLORS.get(status, (100, 100, 100))
            color_hex = '#{:02x}{:02x}{:02x}'.format(*color_rgb)
            fg_color = "#ffffff" if status in ["Flooded", "Others"] else "#111111"

            is_over = res_item.get("is_overruled", False)
            tag_text = f"✏️ {status.upper()} [OVERRULED]" if is_over else status.upper()
            lbl_preview_decision.config(text=tag_text, bg=color_hex, fg=fg_color)

            ai_st = res_item.get("ai_status", status)
            conf_val = res_item["confidence"] * 100.0
            rev_hint = " ⚠️ [Low Confidence]" if res_item.get("needs_review") else " ✅ [High Confidence]"
            lbl_preview_ai.config(text=f"AI Guess: {ai_st} ({conf_val:.1f}%){rev_hint}")

            probs = res_item.get("probabilities", {})
            for cname in CLASSES:
                p_val = probs.get(cname, 0.0)
                if cname in dlg_prob_widgets:
                    pbar, val_lbl = dlg_prob_widgets[cname]
                    pbar["value"] = p_val * 100.0
                    val_lbl.config(text=f"{p_val*100:.1f}%")

        item_map = {}  # tree_item_id -> result dict

        def populate_table():
            tree.delete(*tree.get_children())
            item_map.clear()
            only_low = var_filter_low_conf.get()
            only_over = var_filter_overruled.get()

            first_id = None
            for item in batch_result["results"]:
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
                    row_tag = "tag_low_conf"
                else:
                    decision_str = item["status"]
                    row_tag = "tag_normal"

                item_id = tree.insert(
                    "",
                    tk.END,
                    values=(item["relative_path"], item["ai_status"], conf_str, decision_str),
                    tags=(row_tag,)
                )
                item_map[item_id] = item
                if first_id is None:
                    first_id = item_id

            if first_id:
                tree.selection_set(first_id)
                tree.focus(first_id)
                update_inspector(item_map[first_id])

        def on_tree_select(event=None):
            selected = tree.selection()
            if selected:
                res_item = item_map.get(selected[0])
                if res_item:
                    update_inspector(res_item)

        tree.bind("<<TreeviewSelect>>", on_tree_select)
        populate_table()

        def advance_to_next():
            if not var_auto_advance.get():
                return
            selected = tree.selection()
            if selected:
                next_id = tree.next(selected[0])
                if next_id:
                    tree.selection_set(next_id)
                    tree.focus(next_id)
                    tree.see(next_id)
                    res_item = item_map.get(next_id)
                    if res_item:
                        update_inspector(res_item)

        # Overrule execution
        def apply_overrule(new_class):
            selected = tree.selection()
            if not selected:
                messagebox.showinfo("No Image Selected", "Please click on an image in the table first.")
                return

            res_item = item_map.get(selected[0])
            if not res_item:
                return

            from datetime import datetime
            res_item["is_overruled"] = True
            res_item["operator_label"] = new_class
            res_item["status"] = new_class
            res_item["reviewed_at"] = datetime.now().isoformat()

            conf_val = res_item["confidence"] * 100.0
            conf_str = f"{conf_val:.1f}%" + (" ⚠️" if res_item.get("needs_review") else " ✅")
            decision_str = f"✏️ {new_class} [OVERRULED]"
            tree.item(selected[0], values=(res_item["relative_path"], res_item["ai_status"], conf_str, decision_str), tags=("tag_overruled",))

            update_header_texts()
            update_inspector(res_item)
            self._update_sidebar(res_item, res_item["full_path"])
            self.status_var.set(f"Overruled '{res_item['filename']}' -> {new_class}")

            advance_to_next()

        def reset_to_ai():
            selected = tree.selection()
            if not selected:
                return

            res_item = item_map.get(selected[0])
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

            tree.item(selected[0], values=(res_item["relative_path"], res_item["ai_status"], conf_str, res_item["status"]), tags=(row_tag,))
            update_header_texts()
            update_inspector(res_item)
            self._update_sidebar(res_item, res_item["full_path"])
            self.status_var.set(f"Reset '{res_item['filename']}' -> AI: {res_item['ai_status']}")

            advance_to_next()

        # Overrule buttons
        btn_over_dry = ttk.Button(overrule_bar, text="[1] 🏜️ Dry", command=lambda: apply_overrule("Dry"))
        btn_over_dry.pack(side=tk.LEFT, padx=3)

        btn_over_flood = ttk.Button(overrule_bar, text="[2] 💧 Flooded", command=lambda: apply_overrule("Flooded"))
        btn_over_flood.pack(side=tk.LEFT, padx=3)

        btn_over_plant = ttk.Button(overrule_bar, text="[3] 🌿 Planted", command=lambda: apply_overrule("Planted"))
        btn_over_plant.pack(side=tk.LEFT, padx=3)

        btn_over_other = ttk.Button(overrule_bar, text="[4] 🌳 Others", command=lambda: apply_overrule("Others"))
        btn_over_other.pack(side=tk.LEFT, padx=3)

        btn_reset = ttk.Button(overrule_bar, text="[Space] 🔄 Accept / Reset to AI", command=reset_to_ai)
        btn_reset.pack(side=tk.LEFT, padx=(12, 3))

        # Hotkeys
        dlg.bind("1", lambda e: apply_overrule("Dry"))
        dlg.bind("2", lambda e: apply_overrule("Flooded"))
        dlg.bind("3", lambda e: apply_overrule("Planted"))
        dlg.bind("4", lambda e: apply_overrule("Others"))
        dlg.bind("<space>", lambda e: reset_to_ai())

        # Action Buttons in Footer
        btn_frame = ttk.Frame(dlg, padding=(12, 10))
        btn_frame.pack(fill=tk.X)

        def save_reviewed_reports():
            save_reviewed_batch(batch_result, output_dir="Output")
            messagebox.showinfo("Saved", "Successfully updated all JSON reports in 'Output/' with operator review decisions!")

        def add_overruled_to_dataset():
            overruled_items = [itm for itm in batch_result["results"] if itm.get("is_overruled")]
            if not overruled_items:
                messagebox.showinfo("No Overruled Images", "No images have been overruled yet.\nSelect rows and overrule them with [1-4] first.")
                return

            count = len(overruled_items)
            msg = f"Add {count} overruled image(s) to 'Dataset/' for retraining?\n\nThis will allow the neural network to learn from your corrections next time you run 'python train.py'."
            if messagebox.askyesno("Confirm Export to Dataset", msg):
                copied = export_overruled_to_dataset(overruled_items, dataset_dir="Dataset")
                messagebox.showinfo("Dataset Updated", f"Successfully copied {len(copied)} image(s) into 'Dataset/'!\nRun 'python train.py' to train the AI on these corrections.")

        def open_output_folder():
            out_p = os.path.abspath("Output")
            os.makedirs(out_p, exist_ok=True)
            if hasattr(os, "startfile"):
                os.startfile(out_p)
            else:
                import subprocess
                subprocess.Popen(["explorer", out_p])

        def export_csv():
            fpath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV File", "*.csv")],
                initialfile="classification_results.csv",
                title="Export Results to CSV"
            )
            if fpath:
                self.predictor.predict_directory(input_dir=directory, save_csv=fpath, output_dir=None)
                messagebox.showinfo("Exported", f"Successfully exported to:\n{fpath}")

        btn_save_reports = ttk.Button(btn_frame, text="💾 Save Reviewed Reports", command=save_reviewed_reports)
        btn_save_reports.pack(side=tk.LEFT, padx=3)

        btn_export_dataset = ttk.Button(btn_frame, text="📥 Add Overruled to Dataset", command=add_overruled_to_dataset)
        btn_export_dataset.pack(side=tk.LEFT, padx=3)

        btn_open_folder = ttk.Button(btn_frame, text="📁 Open Output Folder", command=open_output_folder)
        btn_open_folder.pack(side=tk.LEFT, padx=3)

        btn_csv = ttk.Button(btn_frame, text="📄 Export CSV", command=export_csv)
        btn_csv.pack(side=tk.LEFT, padx=3)

        btn_close = ttk.Button(btn_frame, text="Close", command=dlg.destroy)
        btn_close.pack(side=tk.RIGHT, padx=3)



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


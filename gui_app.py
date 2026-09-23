"""
gui_app.py - Rice Field AI: Dataset Curator & Training Manager.

Allows users to:
1. View and sort images from an 'Unsorted/' pool into 7 categories:
   - Dry -> Dataset/Dry/
   - Water -> Dataset/Water/
   - Wet -> Dataset/Wet/
   - Green rice -> Dataset/Green rice/
   - Green weed -> Dataset/Green weed/
   - Straw -> Dataset/Straw/
   - Others -> Dataset/Others/
2. View and edit sorted images to quickly fix any mis-sorted files.
3. Use keyboard shortcuts [1-7] to sort or re-sort instantly with auto-advance.
4. View image preview, EXIF GPS coordinates, and AI pre-label suggestions.
5. Launch model training (python train.py) directly with live background progress.
"""

import os
import sys
import shutil
import csv
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
from PIL import Image, ImageTk

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.model import CLASSES, CLASS_COLORS
from src.inference import RiceFieldPredictor, extract_image_gps
from src.train import train_classifier

# Mapping from class name to folder in Dataset/
CLASS_TO_FOLDER = {
    "Dry": "Dry",
    "Water": "Water",
    "Wet": "Wet",
    "Green rice": "Green rice",
    "Green weed": "Green weed",
    "Straw": "Straw",
    "Others": "Others",
}

FOLDER_TO_CLASS = {
    "dry": "Dry",
    "water": "Water",
    "wet": "Wet",
    "green rice": "Green rice",
    "green_rice": "Green rice",
    "green weed": "Green weed",
    "green_weed": "Green weed",
    "straw": "Straw",
    "others": "Others",
    "other": "Others",
    # Legacy fallbacks
    "flood": "Water",
    "flooded": "Water",
    "planted": "Green rice",
}

CLASS_ICONS = {
    "Dry": "🏜️",
    "Water": "💧",
    "Wet": "🌧️",
    "Green rice": "🌾",
    "Green weed": "🌿",
    "Straw": "🍂",
    "Others": "🌳",
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class DatasetCuratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🌾 Rice Field AI — Dataset Curator & Training Manager")
        self.root.geometry("1240x840")
        self.root.minsize(1020, 680)

        self.unsorted_dir = os.path.abspath("Unsorted")
        self.dataset_dir = os.path.abspath("Dataset")
        os.makedirs(self.unsorted_dir, exist_ok=True)
        os.makedirs(self.dataset_dir, exist_ok=True)
        for sub in CLASS_TO_FOLDER.values():
            os.makedirs(os.path.join(self.dataset_dir, sub), exist_ok=True)

        # AI Predictor (lazy-loaded or optional)
        self.predictor = None
        self._init_predictor()

        # State data
        self.current_view_mode = "unsorted"  # 'unsorted', 'all_sorted', or any class in CLASS_TO_FOLDER
        self.items_data = []  # List of dicts currently displayed in the table
        self.item_map = {}    # Tree item id -> item dict
        self.current_inspected_item = None
        self.preview_photo_ref = [None]
        self.is_training = False

        # Live Counts
        self.counts = {
            "Unsorted": 0,
            "Total_Sorted": 0,
        }
        for cname in CLASS_TO_FOLDER:
            self.counts[cname] = 0

        # UI Variables
        self.status_var = tk.StringVar(value="Ready. Select an image or press [1-7] to sort.")
        self.var_auto_advance = tk.BooleanVar(value=True)
        self.var_ai_assist = tk.BooleanVar(value=True)

        self._configure_styles()
        self._setup_ui()
        self._bind_shortcuts()

        # Initial refresh
        self.refresh_all_data()

    def _init_predictor(self):
        try:
            if os.path.exists("rice_field_classifier.pth"):
                self.predictor = RiceFieldPredictor("rice_field_classifier.pth")
        except Exception as e:
            print(f"[!] Note: AI Predictor not loaded ({e}). Training can still be run.")
            self.predictor = None

    def _configure_styles(self):
        style = ttk.Style()
        if "clam" in style.theme_names():
            try:
                style.theme_use("clam")
            except Exception:
                pass

        style.configure("Treeview.Heading", font=("Helvetica", 9, "bold"))
        style.configure("Treeview", font=("Helvetica", 9), rowheight=24)

    def _setup_ui(self):
        # 1. Top Dashboard / Summary Header
        header_frame = tk.Frame(self.root, bg="#1a2332", padx=14, pady=10)
        header_frame.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(
            header_frame,
            text="🌾 Rice Field AI — Dataset Curator & Training Manager",
            font=("Helvetica", 14, "bold"),
            bg="#1a2332",
            fg="#ffffff"
        )
        title_lbl.pack(anchor="w")

        # Counts Bar
        self.stats_box = tk.Frame(header_frame, bg="#1a2332")
        self.stats_box.pack(anchor="w", pady=(6, 0), fill=tk.X)

        self.stat_labels = {}
        badge_colors = {
            "Unsorted": "#ff9800",
            "Dry": "#e65100",
            "Water": "#0288d1",
            "Wet": "#3949ab",
            "Green rice": "#2e7d32",
            "Green weed": "#1b5e20",
            "Straw": "#f57f17",
            "Others": "#546e7a",
            "Total_Sorted": "#ffffff",
        }
        badges = [("Unsorted", "📥 Unsorted: 0")]
        badges += [(cname, f"{CLASS_ICONS[cname]} {cname}: 0") for cname in CLASS_TO_FOLDER]
        badges.append(("Total_Sorted", "📊 Total Sorted: 0"))

        for key, text in badges:
            lbl = tk.Label(
                self.stats_box,
                text=text,
                font=("Helvetica", 9, "bold"),
                bg="#263238",
                fg=badge_colors.get(key, "#ffffff"),
                padx=6,
                pady=2,
                relief=tk.FLAT
            )
            lbl.pack(side=tk.LEFT, padx=(0, 5))
            self.stat_labels[key] = lbl

        # 2. View Mode Navigation & Settings Bar
        nav_bar = ttk.Frame(self.root, padding=(12, 6))
        nav_bar.pack(fill=tk.X, side=tk.TOP)

        ttk.Label(nav_bar, text="View Pool:", font=("Helvetica", 9, "bold")).pack(side=tk.LEFT, padx=(0, 6))

        self.btn_views = {}
        modes = [
            ("unsorted", "📥 Unsorted"),
            ("all_sorted", "📁 All Sorted"),
        ]
        for cname in CLASS_TO_FOLDER:
            modes.append((cname, f"{CLASS_ICONS[cname]} {cname}"))

        for m_id, m_text in modes:
            b = ttk.Button(nav_bar, text=m_text, command=lambda m=m_id: self.switch_view(m))
            b.pack(side=tk.LEFT, padx=1)
            self.btn_views[m_id] = b

        ttk.Separator(nav_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=2)

        chk_advance = ttk.Checkbutton(nav_bar, text="⚡ Auto-Advance on Sort", variable=self.var_auto_advance)
        chk_advance.pack(side=tk.LEFT, padx=4)

        chk_ai = ttk.Checkbutton(nav_bar, text="🤖 AI Assist", variable=self.var_ai_assist, command=self._on_toggle_ai_assist)
        chk_ai.pack(side=tk.LEFT, padx=6)

        btn_refresh = ttk.Button(nav_bar, text="🔄 Refresh", command=self.refresh_all_data)
        btn_refresh.pack(side=tk.RIGHT, padx=2)

        btn_import = ttk.Button(nav_bar, text="📥 Import to Unsorted...", command=self._on_import_images)
        btn_import.pack(side=tk.RIGHT, padx=4)

        # 3. Status Bar at Bottom
        self.lbl_status = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=(8, 4))
        self.lbl_status.pack(fill=tk.X, side=tk.BOTTOM)

        # 4. Footer Action Buttons
        footer_frame = ttk.Frame(self.root, padding=(12, 6))
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.btn_train = ttk.Button(
            footer_frame,
            text="🚀 Train AI Model (python train.py)",
            command=self._on_train_model
        )
        self.btn_train.pack(side=tk.LEFT, padx=(0, 6))

        btn_open_unsorted = ttk.Button(footer_frame, text="📂 Open Unsorted Folder", command=self._on_open_unsorted_folder)
        btn_open_unsorted.pack(side=tk.LEFT, padx=4)

        btn_open_dataset = ttk.Button(footer_frame, text="📁 Open Dataset Folder", command=self._on_open_dataset_folder)
        btn_open_dataset.pack(side=tk.LEFT, padx=4)

        btn_csv = ttk.Button(footer_frame, text="📄 Export Dataset CSV", command=self._on_export_csv)
        btn_csv.pack(side=tk.LEFT, padx=4)

        # 5. Main Center Paned Window: Left = Table, Right = Live Inspector & Tips
        center_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        center_pane.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        # Left Table Frame
        table_frame = ttk.Frame(center_pane)
        center_pane.add(table_frame, weight=3)

        cols = ("filename", "category", "ai_guess", "gps", "dimensions")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="browse")
        self.tree.heading("filename", text="Filename")
        self.tree.heading("category", text="Current Category")
        self.tree.heading("ai_guess", text="AI Suggestion")
        self.tree.heading("gps", text="GPS Coordinates")
        self.tree.heading("dimensions", text="Resolution")

        self.tree.column("filename", width=220, anchor="w")
        self.tree.column("category", width=120, anchor="center")
        self.tree.column("ai_guess", width=140, anchor="center")
        self.tree.column("gps", width=170, anchor="center")
        self.tree.column("dimensions", width=100, anchor="center")

        # Color tags
        self.tree.tag_configure("tag_unsorted", foreground="#e65100", font=("Helvetica", 9, "bold"))
        self.tree.tag_configure("tag_dry", foreground="#bf360c")
        self.tree.tag_configure("tag_water", foreground="#0288d1")
        self.tree.tag_configure("tag_wet", foreground="#3949ab")
        self.tree.tag_configure("tag_green_rice", foreground="#2e7d32")
        self.tree.tag_configure("tag_green_weed", foreground="#1b5e20")
        self.tree.tag_configure("tag_straw", foreground="#f57f17")
        self.tree.tag_configure("tag_others", foreground="#455a64")

        vsb = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # Right Inspector Frame
        inspector_frame = ttk.Frame(center_pane, padding=6)
        center_pane.add(inspector_frame, weight=2)

        # Inspector Canvas / Preview Box
        preview_group = ttk.LabelFrame(inspector_frame, text="🖼️ Selected Image Preview", padding=8)
        preview_group.pack(fill=tk.X, side=tk.TOP, pady=(0, 6))

        img_box = tk.Frame(preview_group, bg="#1e222b", width=360, height=240)
        img_box.pack(fill=tk.X, pady=(0, 6))
        img_box.pack_propagate(False)

        self.lbl_preview_img = tk.Label(img_box, bg="#1e222b", text="Select an image to inspect", fg="#888888")
        self.lbl_preview_img.pack(expand=True)

        self.lbl_preview_name = ttk.Label(preview_group, text="File: --", font=("Helvetica", 9, "bold"), wraplength=340)
        self.lbl_preview_name.pack(anchor="w", pady=1)

        self.lbl_preview_cat = tk.Label(preview_group, text="Category: --", font=("Helvetica", 11, "bold"), bg="#eceff1", fg="#37474f", padx=6, pady=3)
        self.lbl_preview_cat.pack(fill=tk.X, pady=3)

        self.lbl_preview_gps = ttk.Label(preview_group, text="📍 GPS: --", font=("Helvetica", 9), foreground="#1565c0", wraplength=340)
        self.lbl_preview_gps.pack(anchor="w", pady=1)

        self.lbl_preview_ai = tk.Label(preview_group, text="AI Guess: --", font=("Helvetica", 9), anchor="w")
        self.lbl_preview_ai.pack(fill=tk.X, pady=1)

        # Fast Move Actions Frame
        action_group = ttk.LabelFrame(inspector_frame, text="⚡ Quick Sort / Re-assign [1-7]", padding=8)
        action_group.pack(fill=tk.X, side=tk.TOP, pady=(0, 6))

        btn_grid = ttk.Frame(action_group)
        btn_grid.pack(fill=tk.X)
        btn_grid.columnconfigure(0, weight=1)
        btn_grid.columnconfigure(1, weight=1)

        class_keys = list(CLASS_TO_FOLDER.keys())
        for idx, cname in enumerate(class_keys, start=1):
            icon = CLASS_ICONS.get(cname, "")
            row = (idx - 1) // 2
            col = (idx - 1) % 2
            colspan = 2 if (idx == len(class_keys) and len(class_keys) % 2 == 1) else 1
            btn = ttk.Button(
                btn_grid,
                text=f"[{idx}] {icon} {cname}",
                command=lambda c=cname: self.move_current_to_class(c)
            )
            btn.grid(row=row, column=col, columnspan=colspan, padx=2, pady=2, sticky="ew")

        ttk.Button(
            action_group,
            text="[U] ↩️ Move back to Unsorted",
            command=self.move_current_to_unsorted
        ).pack(fill=tk.X, pady=(4, 0))

        # Vertical Tips & Controls Card
        tips_group = ttk.LabelFrame(inspector_frame, text="💡 Quick Tips & Controls", padding=8)
        tips_group.pack(fill=tk.BOTH, expand=True, side=tk.TOP, pady=(2, 0))

        tips_lines = [f"• Press [{idx}]  ->  Move to {cname}" for idx, cname in enumerate(class_keys, start=1)]
        tips_lines.extend([
            "• Press [U]  ->  Move to Unsorted",
            "• Press [Space] -> Accept AI Suggestion",
            "• Press [↑ / ↓] -> Navigate Images",
            "• Press [Del]   -> Delete Image"
        ])
        tips_text = "\n".join(tips_lines)

        lbl_tips = tk.Label(
            tips_group,
            text=tips_text,
            font=("Consolas", 9),
            justify=tk.LEFT,
            anchor="nw",
            bg="#f5f7fa",
            fg="#263238",
            padx=8,
            pady=8,
            relief=tk.GROOVE
        )
        lbl_tips.pack(fill=tk.BOTH, expand=True)

    def _bind_shortcuts(self):
        class_keys = list(CLASS_TO_FOLDER.keys())
        for idx, cname in enumerate(class_keys, start=1):
            self.root.bind(f"<Key-{idx}>", lambda e, c=cname: self.move_current_to_class(c))
        self.root.bind("<Key-u>", lambda e: self.move_current_to_unsorted())
        self.root.bind("<Key-U>", lambda e: self.move_current_to_unsorted())
        self.root.bind("<space>", lambda e: self._on_accept_ai_guess())
        self.root.bind("<Delete>", lambda e: self._on_delete_current())

    # --------------------------------------------------------------------------
    # DATA SCANNING & COUNTS
    # --------------------------------------------------------------------------
    def refresh_all_data(self):
        """Scans Unsorted/ and Dataset/ folders, refreshes counts and table."""
        # 1. Update counts
        unsorted_files = self._get_image_files(self.unsorted_dir)
        self.counts["Unsorted"] = len(unsorted_files)

        total_sorted = 0
        for cname, subfolder in CLASS_TO_FOLDER.items():
            fpath = os.path.join(self.dataset_dir, subfolder)
            files = self._get_image_files(fpath)
            self.counts[cname] = len(files)
            total_sorted += len(files)

        self.counts["Total_Sorted"] = total_sorted

        # Update Top Badges
        self._update_stat_badges()

        # 2. Populate table for active view
        self._load_active_view_items()

    def _update_stat_badges(self):
        self.stat_labels["Unsorted"].config(text=f"📥 Unsorted: {self.counts['Unsorted']}")
        for cname in CLASS_TO_FOLDER:
            if cname in self.stat_labels:
                icon = CLASS_ICONS.get(cname, "")
                self.stat_labels[cname].config(text=f"{icon} {cname}: {self.counts[cname]}")
        self.stat_labels["Total_Sorted"].config(text=f"📊 Total Sorted: {self.counts['Total_Sorted']}")

    def switch_view(self, mode):
        self.current_view_mode = mode
        for m_id, b in self.btn_views.items():
            if m_id == mode:
                b.state(["pressed"])
            else:
                b.state(["!pressed"])
        self._load_active_view_items()
        self.status_var.set(f"Switched view to: {mode.replace('_', ' ').title()}")

    def _get_image_files(self, directory):
        if not os.path.exists(directory):
            return []
        files = []
        for f in sorted(os.listdir(directory)):
            if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS:
                files.append(f)
        return files

    def _load_active_view_items(self):
        items = []

        if self.current_view_mode == "unsorted":
            for fname in self._get_image_files(self.unsorted_dir):
                full_p = os.path.join(self.unsorted_dir, fname)
                items.append({
                    "filename": fname,
                    "full_path": full_p,
                    "category": "Unsorted",
                    "folder": self.unsorted_dir
                })
        elif self.current_view_mode == "all_sorted":
            for cname, subfolder in CLASS_TO_FOLDER.items():
                fdir = os.path.join(self.dataset_dir, subfolder)
                for fname in self._get_image_files(fdir):
                    full_p = os.path.join(fdir, fname)
                    items.append({
                        "filename": fname,
                        "full_path": full_p,
                        "category": cname,
                        "folder": fdir
                    })
        elif self.current_view_mode in CLASS_TO_FOLDER:
            cname = self.current_view_mode
            subfolder = CLASS_TO_FOLDER[cname]
            fdir = os.path.join(self.dataset_dir, subfolder)
            for fname in self._get_image_files(fdir):
                full_p = os.path.join(fdir, fname)
                items.append({
                    "filename": fname,
                    "full_path": full_p,
                    "category": cname,
                    "folder": fdir
                })

        self.items_data = items
        self._populate_treeview()

    def _populate_treeview(self):
        # Remember previous selection filename if possible
        prev_filename = self.current_inspected_item["filename"] if self.current_inspected_item else None

        self.tree.delete(*self.tree.get_children())
        self.item_map.clear()

        select_id = None

        for itm in self.items_data:
            full_p = itm["full_path"]
            fname = itm["filename"]
            cat = itm["category"]

            # AI prediction
            ai_str = "--"
            if self.var_ai_assist.get() and self.predictor:
                try:
                    pred = self.predictor.predict(full_p)
                    itm["ai_pred"] = pred
                    ai_str = f"{pred['status']} ({pred['confidence']*100:.1f}%)"
                except Exception:
                    itm["ai_pred"] = None
            else:
                itm["ai_pred"] = None

            # GPS
            gps = extract_image_gps(full_p)
            itm["gps"] = gps
            gps_str = f"{gps['latitude']:.4f}°N, {gps['longitude']:.4f}°E" if gps else "--"

            # Image size
            dim_str = "--"
            try:
                with Image.open(full_p) as im:
                    itm["dimensions"] = im.size
                    dim_str = f"{im.size[0]}x{im.size[1]}"
            except Exception:
                itm["dimensions"] = (0, 0)

            # Row tag
            tag = "tag_unsorted" if cat == "Unsorted" else f"tag_{cat.lower().replace(' ', '_')}"

            node_id = self.tree.insert("", tk.END, values=(fname, cat, ai_str, gps_str, dim_str), tags=(tag,))
            self.item_map[node_id] = itm

            if prev_filename and fname == prev_filename:
                select_id = node_id

        # Select item
        children = self.tree.get_children()
        if children:
            target = select_id if select_id else children[0]
            self.tree.selection_set(target)
            self.tree.focus(target)
            self.tree.see(target)
            self._update_inspector(self.item_map.get(target))
        else:
            self._clear_inspector()

    # --------------------------------------------------------------------------
    # INSPECTION PANEL
    # --------------------------------------------------------------------------
    def _on_tree_select(self, event):
        selected = self.tree.selection()
        if selected:
            itm = self.item_map.get(selected[0])
            if itm:
                self._update_inspector(itm)

    def _update_inspector(self, itm):
        self.current_inspected_item = itm
        full_p = itm["full_path"]

        if os.path.exists(full_p):
            try:
                pil_im = Image.open(full_p).convert("RGB")
                w, h = pil_im.size
                scale = min(350 / w, 230 / h, 1.0)
                nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
                resized = pil_im.resize((nw, nh), Image.Resampling.BILINEAR)
                tk_img = ImageTk.PhotoImage(resized)
                self.preview_photo_ref[0] = tk_img
                self.lbl_preview_img.config(image=tk_img, text="")
                self.lbl_preview_name.config(text=f"📁 {itm['filename']} ({w}x{h} px)")
            except Exception as e:
                self.lbl_preview_img.config(image="", text=f"Error previewing: {e}")
        else:
            self.lbl_preview_img.config(image="", text="File missing")

        # Category pill
        cat = itm.get("category", "Unsorted")
        color_rgb = CLASS_COLORS.get(cat, (120, 120, 120)) if cat != "Unsorted" else (230, 81, 0)
        hex_bg = '#{:02x}{:02x}{:02x}'.format(*color_rgb)
        fg_col = "#ffffff" if cat in ["Water", "Wet", "Others", "Green weed", "Unsorted"] else "#111111"
        self.lbl_preview_cat.config(text=f"Category: {cat.upper()}", bg=hex_bg, fg=fg_col)

        # GPS
        gps = itm.get("gps")
        if gps:
            alt_txt = f" | Alt: {gps['altitude']}m" if gps.get("altitude") is not None else ""
            self.lbl_preview_gps.config(text=f"📍 GPS: {gps['latitude']:.4f}° N, {gps['longitude']:.4f}° E{alt_txt}")
        else:
            self.lbl_preview_gps.config(text="📍 GPS: No coordinates in metadata")

        # AI Guess
        ai_pred = itm.get("ai_pred")
        if ai_pred:
            conf_pct = ai_pred["confidence"] * 100.0
            is_low = conf_pct < 80.0
            col = "#d32f2f" if is_low else "#2e7d32"
            badge = "⚠️ [Low <80%]" if is_low else "✅ [High]"
            self.lbl_preview_ai.config(
                text=f"AI Suggests: {ai_pred['status']} ({conf_pct:.1f}%) {badge}",
                fg=col,
                font=("Helvetica", 9, "bold")
            )
        else:
            self.lbl_preview_ai.config(text="AI Suggests: --", fg="#555555", font=("Helvetica", 9))

    def _clear_inspector(self):
        self.current_inspected_item = None
        self.lbl_preview_img.config(image="", text="No images in current view")
        self.lbl_preview_name.config(text="File: --")
        self.lbl_preview_cat.config(text="Category: --", bg="#eceff1", fg="#37474f")
        self.lbl_preview_gps.config(text="📍 GPS: --")
        self.lbl_preview_ai.config(text="AI Suggests: --")

    # --------------------------------------------------------------------------
    # FAST FILE SORTING / MOVING OPERATIONS
    # --------------------------------------------------------------------------
    def move_current_to_class(self, target_class):
        """Moves the currently selected image to Dataset/<target_class>/."""
        if not self.current_inspected_item:
            return

        itm = self.current_inspected_item
        src_path = itm["full_path"]
        if not os.path.exists(src_path):
            messagebox.showerror("File Error", f"Source file not found:\n{src_path}")
            return

        subfolder = CLASS_TO_FOLDER.get(target_class)
        dest_dir = os.path.join(self.dataset_dir, subfolder)
        os.makedirs(dest_dir, exist_ok=True)

        dest_path = self._generate_unique_dest(dest_dir, itm["filename"])

        try:
            shutil.move(src_path, dest_path)
            new_fname = os.path.basename(dest_path)
            prev_cat = itm["category"]

            self.status_var.set(f"Moved '{itm['filename']}' -> {target_class} ({subfolder}/)")

            # If currently viewing a specific category that no longer matches, remove row and advance
            self._handle_post_move(itm, target_class, dest_path, new_fname)
        except Exception as e:
            messagebox.showerror("Move Error", f"Could not move image: {e}")

    def move_current_to_unsorted(self):
        """Moves the currently selected image back to Unsorted/."""
        if not self.current_inspected_item:
            return

        itm = self.current_inspected_item
        src_path = itm["full_path"]
        if not os.path.exists(src_path):
            messagebox.showerror("File Error", f"Source file not found:\n{src_path}")
            return

        if itm["category"] == "Unsorted":
            self.status_var.set(f"'{itm['filename']}' is already in Unsorted pool.")
            return

        dest_path = self._generate_unique_dest(self.unsorted_dir, itm["filename"])

        try:
            shutil.move(src_path, dest_path)
            new_fname = os.path.basename(dest_path)
            self.status_var.set(f"Returned '{itm['filename']}' -> Unsorted pool")
            self._handle_post_move(itm, "Unsorted", dest_path, new_fname)
        except Exception as e:
            messagebox.showerror("Move Error", f"Could not return image to unsorted: {e}")

    def _generate_unique_dest(self, dest_dir, filename):
        dest_path = os.path.join(dest_dir, filename)
        if not os.path.exists(dest_path):
            return dest_path

        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(dest_path):
            dest_path = os.path.join(dest_dir, f"{base}_{counter}{ext}")
            counter += 1
        return dest_path

    def _handle_post_move(self, itm, new_cat, new_path, new_fname):
        old_cat = itm["category"]

        # Update counts
        if old_cat == "Unsorted":
            self.counts["Unsorted"] = max(0, self.counts["Unsorted"] - 1)
        else:
            self.counts[old_cat] = max(0, self.counts[old_cat] - 1)
            self.counts["Total_Sorted"] = max(0, self.counts["Total_Sorted"] - 1)

        if new_cat == "Unsorted":
            self.counts["Unsorted"] += 1
        else:
            self.counts[new_cat] += 1
            self.counts["Total_Sorted"] += 1

        # Update badge labels
        self._update_stat_badges()

        # Advance or remove row
        selected = self.tree.selection()
        if not selected:
            return
        curr_node = selected[0]

        should_remove_row = False
        if self.current_view_mode == "unsorted" and new_cat != "Unsorted":
            should_remove_row = True
        elif self.current_view_mode in CLASS_TO_FOLDER and new_cat != self.current_view_mode:
            should_remove_row = True

        if should_remove_row:
            next_node = self.tree.next(curr_node) or self.tree.prev(curr_node)
            self.tree.delete(curr_node)
            if curr_node in self.item_map:
                del self.item_map[curr_node]

            if next_node and self.tree.exists(next_node):
                self.tree.selection_set(next_node)
                self.tree.focus(next_node)
                self.tree.see(next_node)
                self._update_inspector(self.item_map.get(next_node))
            else:
                # If tree is empty
                children = self.tree.get_children()
                if children:
                    self.tree.selection_set(children[0])
                    self.tree.focus(children[0])
                    self.tree.see(children[0])
                    self._update_inspector(self.item_map.get(children[0]))
                else:
                    self._clear_inspector()
        else:
            # Row remains in current view (e.g. All Sorted view) -> update row text and tag
            itm["category"] = new_cat
            itm["full_path"] = new_path
            itm["filename"] = new_fname
            tag = "tag_unsorted" if new_cat == "Unsorted" else f"tag_{new_cat.lower().replace(' ', '_')}"

            values = list(self.tree.item(curr_node, "values"))
            values[0] = new_fname
            values[1] = new_cat
            self.tree.item(curr_node, values=values, tags=(tag,))
            self._update_inspector(itm)

            if self.var_auto_advance.get():
                next_node = self.tree.next(curr_node)
                if next_node:
                    self.tree.selection_set(next_node)
                    self.tree.focus(next_node)
                    self.tree.see(next_node)
                    self._update_inspector(self.item_map.get(next_node))

    def _on_accept_ai_guess(self):
        """Accepts AI prediction and moves image to that class."""
        if not self.current_inspected_item:
            return
        ai_pred = self.current_inspected_item.get("ai_pred")
        if ai_pred and ai_pred.get("status") in CLASS_TO_FOLDER:
            self.move_current_to_class(ai_pred["status"])
        else:
            self.status_var.set("No valid AI suggestion available for this image.")

    def _on_delete_current(self):
        """Deletes the current image file with confirmation."""
        if not self.current_inspected_item:
            return
        itm = self.current_inspected_item
        fname = itm["filename"]
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to permanently delete:\n{fname}?"):
            try:
                os.remove(itm["full_path"])
                self.status_var.set(f"Deleted image: {fname}")
                self.refresh_all_data()
            except Exception as e:
                messagebox.showerror("Delete Error", str(e))

    def _on_toggle_ai_assist(self):
        if self.var_ai_assist.get() and not self.predictor:
            self._init_predictor()
        self.refresh_all_data()

    # --------------------------------------------------------------------------
    # IMPORT IMAGES INTO UNSORTED POOL
    # --------------------------------------------------------------------------
    def _on_import_images(self):
        files = filedialog.askopenfilenames(
            title="Select Images to Import into Unsorted Pool",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.tif;*.tiff"), ("All Files", "*.*")]
        )
        if not files:
            return

        imported_count = 0
        for f in files:
            fname = os.path.basename(f)
            dest = self._generate_unique_dest(self.unsorted_dir, fname)
            try:
                shutil.copy2(f, dest)
                imported_count += 1
            except Exception as e:
                print(f"[!] Warning copying {f}: {e}")

        self.status_var.set(f"Imported {imported_count} new image(s) into 'Unsorted/'.")
        messagebox.showinfo("Import Complete", f"Successfully imported {imported_count} image(s) into Unsorted pool!")
        self.switch_view("unsorted")
        self.refresh_all_data()

    # --------------------------------------------------------------------------
    # BACKGROUND MODEL TRAINING
    # --------------------------------------------------------------------------
    def _on_train_model(self):
        if self.is_training:
            messagebox.showinfo("Training Running", "Model training is already running in the background!")
            return

        # Check that we have images in each class
        for cname in CLASS_TO_FOLDER:
            if self.counts.get(cname, 0) == 0:
                messagebox.showwarning(
                    "Missing Class Data",
                    f"Class '{cname}' currently has 0 images!\nPlease sort at least a few images into '{cname}' before training."
                )
                return

        samples_lines = "\n".join(f"  - {c}: {self.counts.get(c, 0)}" for c in CLASS_TO_FOLDER)
        msg = (
            f"Start retraining the Rice Field AI Neural Network now?\n\n"
            f"Current Dataset Samples:\n{samples_lines}\n"
            f"  Total: {self.counts['Total_Sorted']}\n\n"
            f"Training runs in the background. You can continue sorting while it trains."
        )

        if not messagebox.askyesno("Confirm Training", msg):
            return

        self.is_training = True
        self.btn_train.config(text="⏳ Training in Progress...", state=tk.DISABLED)
        self.status_var.set("Training started: epochs=12, batch_size=16... please wait.")

        threading.Thread(target=self._run_training_worker, daemon=True).start()

    def _run_training_worker(self):
        try:
            train_classifier(
                dataset_dir=self.dataset_dir,
                epochs=12,
                batch_size=16,
                learning_rate=1e-3,
                save_path="rice_field_classifier.pth"
            )
            self.root.after(0, self._on_training_finished, True, "Successfully trained and updated 'rice_field_classifier.pth'!")
        except Exception as e:
            self.root.after(0, self._on_training_finished, False, str(e))

    def _on_training_finished(self, success, msg):
        self.is_training = False
        self.btn_train.config(text="🚀 Train AI Model (python train.py)", state=tk.NORMAL)

        if success:
            self.status_var.set("Training finished successfully! Model weights updated.")
            messagebox.showinfo("Training Completed", f"🎉 AI Training Complete!\n\n{msg}")
            # Reload predictor
            self._init_predictor()
            self.refresh_all_data()
        else:
            self.status_var.set(f"Training failed: {msg}")
            messagebox.showerror("Training Error", f"Training encountered an error:\n{msg}")

    # --------------------------------------------------------------------------
    # FOLDER LAUNCHERS & CSV EXPORT
    # --------------------------------------------------------------------------
    def _on_open_unsorted_folder(self):
        self._open_in_explorer(self.unsorted_dir)

    def _on_open_dataset_folder(self):
        self._open_in_explorer(self.dataset_dir)

    def _open_in_explorer(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        if hasattr(os, "startfile"):
            os.startfile(folder_path)
        else:
            import subprocess
            subprocess.Popen(["explorer", folder_path])

    def _on_export_csv(self):
        dest = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            initialfile="dataset_curation_summary.csv",
            title="Export Dataset Curation CSV"
        )
        if not dest:
            return

        try:
            rows = []
            # Gather all sorted images
            for cname, folder in CLASS_TO_FOLDER.items():
                fdir = os.path.join(self.dataset_dir, folder)
                for f in self._get_image_files(fdir):
                    full_p = os.path.join(fdir, f)
                    gps = extract_image_gps(full_p) or {}
                    rows.append({
                        "pool": "Sorted",
                        "category": cname,
                        "filename": f,
                        "full_path": full_p,
                        "latitude": gps.get("latitude", ""),
                        "longitude": gps.get("longitude", ""),
                        "altitude": gps.get("altitude", ""),
                    })

            # Gather unsorted
            for f in self._get_image_files(self.unsorted_dir):
                full_p = os.path.join(self.unsorted_dir, f)
                gps = extract_image_gps(full_p) or {}
                rows.append({
                    "pool": "Unsorted",
                    "category": "Unsorted",
                    "filename": f,
                    "full_path": full_p,
                    "latitude": gps.get("latitude", ""),
                    "longitude": gps.get("longitude", ""),
                    "altitude": gps.get("altitude", ""),
                })

            with open(dest, "w", newline="", encoding="utf-8") as cf:
                writer = csv.DictWriter(cf, fieldnames=["pool", "category", "filename", "full_path", "latitude", "longitude", "altitude"])
                writer.writeheader()
                writer.writerows(rows)

            self.status_var.set(f"Exported {len(rows)} image record(s) to: {dest}")
            messagebox.showinfo("Export Successful", f"Exported dataset summary ({len(rows)} records) to:\n{dest}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))


def main():
    root = tk.Tk()
    app = DatasetCuratorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

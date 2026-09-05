"""
src/inference.py - Inference Engine and API for 4-Class Rice Field State Classification (Dry, Flooded, Planted, Others).
"""

import os
import sys
import argparse
import json
import csv
import shutil
from datetime import datetime
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import build_classifier, CLASSES, CLASS_COLORS


class RiceFieldPredictor:
    """
    Inference Engine that loads the trained PyTorch neural network
    and classifies rice fields into the 4 target states:
    ['Dry', 'Flooded', 'Planted', 'Others']
    """
    def __init__(self, model_path="rice_field_classifier.pth", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = build_classifier(num_classes=len(CLASSES)).to(self.device)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"[+] Loaded model weights from: {model_path}")
        else:
            print(f"[!] Warning: '{model_path}' not found. Using initialized model.")
            
        self.model.eval()

    def _preprocess(self, pil_image, image_size=224):
        """Converts PIL Image to normalized PyTorch tensor [1, 3, H, W]."""
        resized = pil_image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        
        tensor = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

    def predict(self, image_input):
        """
        Classifies an input image or crop into one of the 4 states:
        ['Dry', 'Flooded', 'Planted', 'Others']
        """
        if isinstance(image_input, str):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            pil_img = Image.fromarray(image_input).convert("RGB")
            
        tensor = self._preprocess(pil_img)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            
        class_idx = int(np.argmax(probs))
        status_name = CLASSES[class_idx]
        confidence = float(probs[class_idx])
        
        prob_dict = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
        
        return {
            "status": status_name,
            "class_id": class_idx,
            "confidence": confidence,
            "probabilities": prob_dict
        }

    def predict_directory(self, input_dir="Input", recursive=True, output_dir="Output", save_csv=None, save_json=None):
        """
        Recursively scans input_dir (and all nested subdirectories) for any picture file,
        classifies each into ['Dry', 'Flooded', 'Planted', 'Others'], generates per-subdirectory
        JSON reports inside output_dir, and aggregates results.
        """
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory not found: '{input_dir}'")
            
        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        image_files = []
        
        if recursive:
            for root, _, files in os.walk(input_dir):
                for f in sorted(files):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in valid_extensions:
                        full_p = os.path.join(root, f)
                        rel_p = os.path.relpath(full_p, input_dir)
                        rel_d = os.path.relpath(root, input_dir)
                        image_files.append((full_p, rel_p, rel_d, f))
        else:
            for f in sorted(os.listdir(input_dir)):
                full_p = os.path.join(input_dir, f)
                if os.path.isfile(full_p):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in valid_extensions:
                        image_files.append((full_p, f, ".", f))
                        
        results = []
        summary_counts = {c: 0 for c in CLASSES}
        subdirs_map = {}
        
        for full_p, rel_p, rel_d, fname in image_files:
            try:
                pred = self.predict(full_p)
                norm_rel_p = rel_p.replace("\\", "/")
                norm_rel_d = rel_d.replace("\\", "/")
                conf = pred["confidence"]
                
                item = {
                    "filename": fname,
                    "relative_path": norm_rel_p,
                    "relative_directory": norm_rel_d,
                    "full_path": os.path.abspath(full_p),
                    "ai_status": pred["status"],
                    "status": pred["status"],
                    "confidence": conf,
                    "needs_review": bool(conf < 0.80),
                    "is_overruled": False,
                    "operator_label": None,
                    "reviewed_at": None,
                    "probabilities": pred["probabilities"]
                }
                results.append(item)
                summary_counts[pred["status"]] += 1
                
                if norm_rel_d not in subdirs_map:
                    subdirs_map[norm_rel_d] = []
                subdirs_map[norm_rel_d].append(item)
            except Exception as e:
                print(f"[!] Warning: Failed to classify '{full_p}': {e}")
                
        output = {
            "total_images": len(results),
            "directory": os.path.abspath(input_dir),
            "output_directory": os.path.abspath(output_dir) if output_dir else None,
            "summary_counts": summary_counts,
            "overruled_count": 0,
            "needs_review_count": sum(1 for i in results if i["needs_review"]),
            "results": results,
            "subdirectories": list(subdirs_map.keys()),
            "generated_json_files": []
        }
        
        # Automatically generate JSON reports per subdirectory inside output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for subdir_rel, items in subdirs_map.items():
                if subdir_rel == ".":
                    subdir_json_name = "root.json"
                    subdir_json_path = os.path.join(output_dir, subdir_json_name)
                else:
                    subdir_json_path = os.path.join(output_dir, subdir_rel + ".json")
                    os.makedirs(os.path.dirname(subdir_json_path), exist_ok=True)
                
                subdir_counts = {c: 0 for c in CLASSES}
                for itm in items:
                    subdir_counts[itm["status"]] += 1
                    
                subdir_report = {
                    "directory": os.path.abspath(os.path.join(input_dir, subdir_rel)),
                    "relative_directory": subdir_rel,
                    "total_images": len(items),
                    "summary_counts": subdir_counts,
                    "results": items
                }
                with open(subdir_json_path, "w", encoding="utf-8") as jf:
                    json.dump(subdir_report, jf, indent=2)
                output["generated_json_files"].append(os.path.abspath(subdir_json_path))
            
            # Overall aggregate report
            all_json_path = os.path.join(output_dir, "all_results.json")
            with open(all_json_path, "w", encoding="utf-8") as ajf:
                json.dump(output, ajf, indent=2)
            output["generated_json_files"].append(os.path.abspath(all_json_path))
        
        if save_json:
            with open(save_json, "w", encoding="utf-8") as jf:
                json.dump(output, jf, indent=2)
            print(f"[+] Saved directory classification JSON to: {save_json}")
            
        if save_csv:
            with open(save_csv, "w", newline="", encoding="utf-8") as cf:
                writer = csv.writer(cf)
                header = ["filename", "relative_path", "subdirectory", "predicted_status", "confidence", "needs_review", "is_overruled", "operator_label"] + [f"prob_{c}" for c in CLASSES]
                writer.writerow(header)
                for item in results:
                    row = [
                        item["filename"],
                        item["relative_path"],
                        item["relative_directory"],
                        item["status"],
                        f"{item['confidence']*100:.2f}%",
                        item["needs_review"],
                        item["is_overruled"],
                        item["operator_label"] or ""
                    ] + [f"{item['probabilities'].get(c, 0.0)*100:.2f}%" for c in CLASSES]
                    writer.writerow(row)
            print(f"[+] Saved directory classification CSV to: {save_csv}")
            
        return output


def save_reviewed_batch(batch_result, output_dir=None):
    """
    Saves an updated/reviewed batch dictionary with operator overrules to disk.
    Updates each subdirectory JSON and all_results.json.
    """
    out_dir = output_dir or batch_result.get("output_directory") or "Output"
    os.makedirs(out_dir, exist_ok=True)
    
    # Recalculate summary counts based on active status
    summary_counts = {c: 0 for c in CLASSES}
    overruled_count = 0
    needs_review_count = 0
    subdirs_map = {}
    
    for item in batch_result["results"]:
        status = item["status"]
        if status in summary_counts:
            summary_counts[status] += 1
        if item.get("is_overruled"):
            overruled_count += 1
        if item.get("needs_review") and not item.get("is_overruled"):
            needs_review_count += 1
            
        rel_d = item.get("relative_directory", ".")
        if rel_d not in subdirs_map:
            subdirs_map[rel_d] = []
        subdirs_map[rel_d].append(item)
        
    batch_result["summary_counts"] = summary_counts
    batch_result["overruled_count"] = overruled_count
    batch_result["needs_review_count"] = needs_review_count
    
    generated_files = []
    for subdir_rel, items in subdirs_map.items():
        if subdir_rel == ".":
            subdir_json_path = os.path.join(out_dir, "root.json")
        else:
            subdir_json_path = os.path.join(out_dir, subdir_rel + ".json")
            os.makedirs(os.path.dirname(subdir_json_path), exist_ok=True)
            
        subdir_counts = {c: 0 for c in CLASSES}
        for itm in items:
            s = itm["status"]
            if s in subdir_counts:
                subdir_counts[s] += 1
                
        subdir_report = {
            "directory": items[0]["full_path"] if items else "",
            "relative_directory": subdir_rel,
            "total_images": len(items),
            "summary_counts": subdir_counts,
            "results": items
        }
        with open(subdir_json_path, "w", encoding="utf-8") as jf:
            json.dump(subdir_report, jf, indent=2)
        generated_files.append(os.path.abspath(subdir_json_path))
        
    all_json_path = os.path.join(out_dir, "all_results.json")
    with open(all_json_path, "w", encoding="utf-8") as ajf:
        json.dump(batch_result, ajf, indent=2)
    generated_files.append(os.path.abspath(all_json_path))
    
    batch_result["generated_json_files"] = generated_files
    return generated_files


def export_overruled_to_dataset(overruled_items, dataset_dir="Dataset"):
    """
    Copies human-overruled images into the appropriate Dataset/<class> folder
    so they can be trained on during the next python train.py run.
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")
        
    class_dir_map = {
        "Dry": "Dry",
        "Flooded": "Flood",  # Dataset folder is named Flood
        "Planted": "Planted",
        "Others": "Others"
    }
    
    copied = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for idx, item in enumerate(overruled_items):
        target_label = item.get("operator_label") or item.get("status")
        folder_name = class_dir_map.get(target_label, target_label)
        target_dir = os.path.join(dataset_dir, folder_name)
        os.makedirs(target_dir, exist_ok=True)
        
        src_path = item["full_path"]
        base_name = os.path.basename(src_path)
        name_part, ext_part = os.path.splitext(base_name)
        new_filename = f"overruled_{timestamp}_{idx:03d}_{name_part}{ext_part}"
        dest_path = os.path.join(target_dir, new_filename)
        
        shutil.copy2(src_path, dest_path)
        copied.append((src_path, dest_path))
        
    return copied


def predict_field_state(image_path, model_path="rice_field_classifier.pth"):
    predictor = RiceFieldPredictor(model_path=model_path)
    return predictor.predict(image_path)


def predict_input_directory(input_dir="Input", model_path="rice_field_classifier.pth", **kwargs):
    predictor = RiceFieldPredictor(model_path=model_path)
    return predictor.predict_directory(input_dir=input_dir, **kwargs)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Rice Field State (Dry, Flooded, Planted, Others).")
    parser.add_argument("--image", type=str, default=None, help="Path to single input image")
    parser.add_argument("--dir", "--input_dir", dest="input_dir", type=str, default=None, help="Directory to scan recursively (default: Input/)")
    parser.add_argument("--output_dir", "-o", type=str, default="Output", help="Directory to save per-subdirectory JSON reports (default: Output/)")
    parser.add_argument("--model", type=str, default="rice_field_classifier.pth", help="Model checkpoint path")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path for batch results")
    parser.add_argument("--json", type=str, default=None, help="Optional JSON output path for batch results")
    
    args = parser.parse_args()
    predictor = RiceFieldPredictor(model_path=args.model)
    
    if args.image:
        print(f"[*] Analyzing single image '{args.image}'...")
        res = predictor.predict(args.image)
        print("\n" + "="*50)
        print(f"[+] Predicted State:   {res['status']}")
        print(f"[+] Confidence:        {res['confidence']*100:.2f}%")
        print(f"[+] 4-Class Probabilities:")
        for cname, prob in res['probabilities'].items():
            print(f"    - {cname:10s}: {prob*100:5.1f}%")
        print("="*50 + "\n")
    else:
        target_dir = args.input_dir if args.input_dir else "Input"
        print(f"[*] Scanning & classifying all images in '{target_dir}' (including all subdirectories)...")
        batch_res = predictor.predict_directory(
            input_dir=target_dir,
            output_dir=args.output_dir,
            save_csv=args.csv,
            save_json=args.json
        )
        
        print("\n" + "="*75)
        print(f"{'RELATIVE PATH':<45} | {'STATUS':<10} | {'CONFIDENCE':<10}")
        print("="*75)
        for item in batch_res["results"]:
            rel = item["relative_path"]
            if len(rel) > 43:
                rel = "..." + rel[-40:]
            print(f"{rel:<45} | {item['status']:<10} | {item['confidence']*100:6.1f}%")
        print("="*75)
        print(f"Total Images Classified: {batch_res['total_images']}")
        print("Summary Breakdown:", ", ".join([f"{k}: {v}" for k, v in batch_res["summary_counts"].items()]))
        
        if batch_res.get("generated_json_files"):
            print(f"[+] Subdirectory JSON reports saved to '{args.output_dir}/':")
            for jpath in batch_res["generated_json_files"]:
                rel_jpath = os.path.relpath(jpath, os.getcwd()).replace("\\", "/")
                print(f"    - {rel_jpath}")
        print("="*75 + "\n")


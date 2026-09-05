"""
src/inference.py - Inference Engine and API for 4-Class Rice Field State Classification (Dry, Flooded, Planted, Others).
"""

import os
import sys
import argparse
import json
import csv
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

    def predict_directory(self, input_dir="Input", recursive=True, save_csv=None, save_json=None):
        """
        Recursively scans input_dir (and all nested subdirectories) for any picture file,
        classifies each into ['Dry', 'Flooded', 'Planted', 'Others'], and aggregates results.
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
                        image_files.append((full_p, rel_p, f))
        else:
            for f in sorted(os.listdir(input_dir)):
                full_p = os.path.join(input_dir, f)
                if os.path.isfile(full_p):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in valid_extensions:
                        image_files.append((full_p, f, f))
                        
        results = []
        summary_counts = {c: 0 for c in CLASSES}
        
        for full_p, rel_p, fname in image_files:
            try:
                pred = self.predict(full_p)
                item = {
                    "filename": fname,
                    "relative_path": rel_p.replace("\\", "/"),
                    "full_path": os.path.abspath(full_p),
                    "status": pred["status"],
                    "confidence": pred["confidence"],
                    "probabilities": pred["probabilities"]
                }
                results.append(item)
                summary_counts[pred["status"]] += 1
            except Exception as e:
                print(f"[!] Warning: Failed to classify '{full_p}': {e}")
                
        output = {
            "total_images": len(results),
            "directory": os.path.abspath(input_dir),
            "summary_counts": summary_counts,
            "results": results
        }
        
        if save_json:
            with open(save_json, "w", encoding="utf-8") as jf:
                json.dump(output, jf, indent=2)
            print(f"[+] Saved directory classification JSON to: {save_json}")
            
        if save_csv:
            with open(save_csv, "w", newline="", encoding="utf-8") as cf:
                writer = csv.writer(cf)
                header = ["filename", "relative_path", "predicted_status", "confidence"] + [f"prob_{c}" for c in CLASSES]
                writer.writerow(header)
                for item in results:
                    row = [
                        item["filename"],
                        item["relative_path"],
                        item["status"],
                        f"{item['confidence']*100:.2f}%"
                    ] + [f"{item['probabilities'].get(c, 0.0)*100:.2f}%" for c in CLASSES]
                    writer.writerow(row)
            print(f"[+] Saved directory classification CSV to: {save_csv}")
            
        return output


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
        batch_res = predictor.predict_directory(input_dir=target_dir, save_csv=args.csv, save_json=args.json)
        
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
        print("="*75 + "\n")

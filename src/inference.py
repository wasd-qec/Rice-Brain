"""
src/inference.py - Inference Engine and API for 4-Class Rice Field State Classification (Dry, Flooded, Planted, Others).
"""

import os
import sys
import argparse
from urllib.parse import quote
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import build_classifier, CLASSES, CLASS_COLORS


def _clickable_file_label(path):
    """Return a clickable file label for terminal environments that support OSC 8 links."""
    if not path:
        return ""

    abs_path = os.path.abspath(path)
    encoded_path = quote(abs_path)
    file_uri = f"file://{encoded_path}"
    label = os.path.basename(path)
    return f"\033]8;;{file_uri}\a{label}\033]8;;\a"


def confirm_others_classification(image_paths, prompt_message=None):
    """Print the review-warning block for images classified as 'Others'."""
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    image_paths = [path for path in image_paths if path]
    if not image_paths:
        return True

    print(f"\033[31m[WARNING WARNING!!!]\033[0m {len(image_paths)} image(s) are classified as OTHERS operator need to confirm the classification status :")
    for idx, path in enumerate(image_paths, start=1):
        print(f"{idx}, {_clickable_file_label(path)}")

    if prompt_message is None:
        return True

    print(prompt_message, end="")
    response = input().strip().lower()
    return response in ("", "y", "yes")


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
        result = {
            "status": status_name,
            "class_id": class_idx,
            "confidence": confidence,
            "probabilities": prob_dict,
            "requires_human_confirmation": status_name == "Others",
            "operator_confirmation_required": status_name == "Others",
        }

        return result

    def predict_coordinate(self, full_image_path, coordinate, crop_size=180, output_annotated_path=None):
        """
        Given a full satellite image and an (x, y) coordinate, crops the field
        centered at (x, y) and classifies its status among ['Dry', 'Flooded', 'Planted', 'Others'].
        """
        if isinstance(full_image_path, str):
            full_img = Image.open(full_image_path).convert("RGB")
        else:
            full_img = full_image_path.convert("RGB")
            
        w, h = full_img.size
        cx, cy = int(coordinate[0]), int(coordinate[1])
        
        # Clamp center coordinate to image bounds
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))
        
        half = min(crop_size, min(w, h)) // 2
        left = max(0, min(w - 2 * half, cx - half))
        top = max(0, min(h - 2 * half, cy - half))
        right = min(w, left + 2 * half)
        bottom = min(h, top + 2 * half)
        
        if right <= left:
            left, right = 0, w
        if bottom <= top:
            top, bottom = 0, h
            
        crop = full_img.crop((left, top, right, bottom))
        result = self.predict(crop)
        
        result["coordinate"] = (cx, cy)
        result["bounding_box"] = [left, top, right, bottom]
        
        if output_annotated_path is not None:
            annotated = self._draw_coordinate_annotation(full_img, result)
            annotated.save(output_annotated_path)
            result["annotated_image_path"] = output_annotated_path
            
        return result

    def _draw_coordinate_annotation(self, pil_img, result):
        """Draws visual HUD overlay on the satellite image."""
        overlay = pil_img.copy().convert("RGBA")
        draw = ImageDraw.Draw(overlay)
        
        cx, cy = result["coordinate"]
        bbox = result["bounding_box"]
        status = result["status"]
        color = CLASS_COLORS.get(status, (120, 80, 70))
        
        draw.rectangle(bbox, outline=(*color, 255), width=3)
        
        fill_layer = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        fill_draw = ImageDraw.Draw(fill_layer)
        fill_draw.rectangle(bbox, fill=(*color, 75))
        overlay = Image.alpha_composite(overlay, fill_layer)
        
        draw_final = ImageDraw.Draw(overlay)
        r = 6
        draw_final.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 30, 30, 255), outline=(255, 255, 255, 255), width=2)
        
        card_w, card_h = 360, 140
        margin = 15
        card_x = margin if cx > pil_img.width // 2 else pil_img.width - card_w - margin
        card_y = margin
        
        draw_final.rectangle([card_x, card_y, card_x + card_w, card_y + card_h], fill=(20, 25, 35, 230), outline=(*color, 255), width=2)
        draw_final.rectangle([card_x + 10, card_y + 12, card_x + 20, card_y + card_h - 12], fill=(*color, 255))
        
        tx = card_x + 30
        draw_final.text((tx, card_y + 12), f"Status: {status.upper()}", fill=(255, 255, 255, 255))
        draw_final.text((tx, card_y + 36), f"Confidence: {result['confidence']*100:.1f}%", fill=(180, 230, 255, 255))
        draw_final.text((tx, card_y + 60), f"Coordinate: ({cx}, {cy})", fill=(200, 200, 200, 255))
        
        prob_str = " | ".join([f"{k[:3]}:{v*100:.0f}%" for k, v in result["probabilities"].items()])
        draw_final.text((tx, card_y + 88), f"Probs: {prob_str}", fill=(180, 220, 180, 255))
        
        return overlay.convert("RGB")


def predict_field_state(image_path, model_path="rice_field_classifier.pth"):
    predictor = RiceFieldPredictor(model_path=model_path)
    return predictor.predict(image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Rice Field State (Dry, Flooded, Planted, Others).")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="rice_field_classifier.pth", help="Model checkpoint path")
    parser.add_argument("--x", type=int, default=None, help="Optional X coordinate on full map")
    parser.add_argument("--y", type=int, default=None, help="Optional Y coordinate on full map")
    parser.add_argument("--output", type=str, default="prediction_result.png", help="Output annotated image path")
    
    args = parser.parse_args()
    predictor = RiceFieldPredictor(model_path=args.model)
    
    if args.x is not None and args.y is not None:
        print(f"[*] Analyzing field at coordinate ({args.x}, {args.y}) in '{args.image}'...")
        res = predictor.predict_coordinate(args.image, (args.x, args.y), output_annotated_path=args.output)
    else:
        print(f"[*] Analyzing image '{args.image}'...")
        res = predictor.predict(args.image)
        
    print("\n" + "="*50)
    print(f"[+] Predicted State:   {res['status']}")
    print(f"[+] Confidence:        {res['confidence']*100:.2f}%")
    print(f"[+] 4-Class Probabilities:")
    for cname, prob in res['probabilities'].items():
        print(f"    - {cname:10s}: {prob*100:5.1f}%")
    print("="*50 + "\n")

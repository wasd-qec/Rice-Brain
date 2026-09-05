"""
inference.py - Root inference script for predicting the 4 rice field states (Dry, Flooded, Planted, Others).
"""

import sys
import argparse
from src.inference import RiceFieldPredictor, confirm_others_classification, predict_field_state
from src.model import CLASSES

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Rice Field State (Dry, Flooded, Planted, Others).")
    parser.add_argument("--image", type=str, required=True, help="Path to input image/crop")
    parser.add_argument("--model", type=str, default="rice_field_classifier.pth", help="Model checkpoint path")
    parser.add_argument("--x", type=int, default=None, help="Optional X coordinate on full satellite map")
    parser.add_argument("--y", type=int, default=None, help="Optional Y coordinate on full satellite map")
    parser.add_argument("--output", type=str, default="prediction_result.png", help="Output annotated image path")
    
    args = parser.parse_args()
    predictor = RiceFieldPredictor(model_path=args.model)
    
    if args.x is not None and args.y is not None:
        print(f"[*] Analyzing field at coordinate ({args.x}, {args.y}) in '{args.image}'...")
        res = predictor.predict_coordinate(args.image, (args.x, args.y), output_annotated_path=args.output)
    else:
        print(f"[*] Analyzing image '{args.image}'...")
        res = predictor.predict(args.image)

    if res.get("status") == "Others" or res.get("requires_human_confirmation"):
        confirm_others_classification([args.image])
        response = input("Are you sure that the classification status is correct?[Y/n]").strip().lower()
        if response not in ("", "y", "yes"):
            print("[!] Operator rejected the classification. Please review the image manually and re-run the prediction.")

    print("\n" + "="*50)
    print(f"[+] Predicted State:   {res['status']}")
    print(f"[+] Confidence:        {res['confidence']*100:.2f}%")
    print(f"[+] 4-Class Probabilities:")
    for cname, prob in res['probabilities'].items():
        print(f"    - {cname:10s}: {prob*100:5.1f}%")
    print("="*50 + "\n")

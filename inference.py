"""
inference.py - Root inference script for predicting the 7 rice field states (Dry, Water, Wet, Green rice, Green weed, Straw, Others).
Supports single image prediction or recursive batch classification of any image placed in Input/ (and subdirectories).
"""

import os
import sys
import argparse
from src.inference import RiceFieldPredictor, predict_field_state, predict_input_directory
from src.model import CLASSES

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Rice Field State (Dry, Water, Wet, Green rice, Green weed, Straw, Others).")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image to classify")
    parser.add_argument("--dir", "--input_dir", dest="input_dir", type=str, default=None, help="Directory to scan recursively (default: Input/)")
    parser.add_argument("--model", type=str, default="rice_field_classifier.pth", help="Model checkpoint path")
    parser.add_argument("--csv", type=str, default=None, help="Optional path to export batch results as CSV")
    
    args = parser.parse_args()
    predictor = RiceFieldPredictor(model_path=args.model)
    
    if args.image:
        print(f"[*] Analyzing image '{args.image}'...")
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
            save_csv=args.csv
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
        print("="*75 + "\n")



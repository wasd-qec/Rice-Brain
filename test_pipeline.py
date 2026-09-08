"""
test_pipeline.py - Automated verification and test suite for the 4-Class Rice Field Neural Network.
"""

import os
import unittest
import numpy as np
import torch
from PIL import Image

from src.model import build_classifier, CLASSES
from src.inference import RiceFieldPredictor, predict_field_state


class TestRiceFieldClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = "rice_field_classifier.pth"
        cls.predictor = RiceFieldPredictor(model_path=cls.model_path)

    def test_01_model_forward_shape(self):
        model = build_classifier(num_classes=len(CLASSES))
        dummy = torch.randn(2, 3, 224, 224)
        out = model(dummy)
        self.assertEqual(out.shape, (2, 4))
        print("[PASS] Test 1: Core model forward pass valid. Output shape (2, 4).")

    def test_02_dry_field_prediction(self):
        dry_path = "Dataset/Dry/dry_01.png"
        if os.path.exists(dry_path):
            res = self.predictor.predict(dry_path)
            self.assertEqual(res["status"], "Dry")
            self.assertGreater(res["confidence"], 0.70)
            print(f"[PASS] Test 2: Dry image '{dry_path}' -> {res['status']} ({res['confidence']*100:.1f}%)")

    def test_03_flood_field_prediction(self):
        flood_path = "Dataset/Flood/flood_01.png"
        if os.path.exists(flood_path):
            res = self.predictor.predict(flood_path)
            self.assertEqual(res["status"], "Flooded")
            self.assertGreater(res["confidence"], 0.70)
            print(f"[PASS] Test 3: Flooded image '{flood_path}' -> {res['status']} ({res['confidence']*100:.1f}%)")

    def test_04_planted_field_prediction(self):
        planted_path = "Dataset/Planted/planted_01.png"
        if os.path.exists(planted_path):
            res = self.predictor.predict(planted_path)
            self.assertEqual(res["status"], "Planted")
            self.assertGreater(res["confidence"], 0.70)
            print(f"[PASS] Test 4: Planted image '{planted_path}' -> {res['status']} ({res['confidence']*100:.1f}%)")

    def test_05_others_detection(self):
        others_path = "Dataset/Others/others_01.png"
        if os.path.exists(others_path):
            res = self.predictor.predict(others_path)
            self.assertEqual(res["status"], "Others")
            print(f"[PASS] Test 5: Others image '{others_path}' -> {res['status']} ({res['confidence']*100:.1f}%)")

    def test_06_recursive_input_classification(self):
        # Create a nested subdirectory in Input/ to test recursive discovery and JSON output
        nested_dir = os.path.join("Input", "_test_temp_sub", "inner")
        os.makedirs(nested_dir, exist_ok=True)
        temp_img_path = os.path.join(nested_dir, "nested_sample.png")
        test_output_dir = os.path.join("Output", "_test_out")
        
        try:
            # Create a simple test RGB image
            test_img = Image.new("RGB", (224, 224), color=(34, 180, 76))
            test_img.save(temp_img_path)
            
            res = self.predictor.predict_directory("Input", recursive=True, output_dir=test_output_dir)
            self.assertGreater(res["total_images"], 0)
            
            # Verify that the nested file was found
            found_nested = any("_test_temp_sub" in item["relative_path"] for item in res["results"])
            self.assertTrue(found_nested, "Failed to discover nested image in Input subdirectory!")
            
            # Verify that per-subdirectory JSON files were created
            nested_json_path = os.path.join(test_output_dir, "_test_temp_sub", "inner.json")
            self.assertTrue(os.path.exists(nested_json_path), f"Per-subdirectory JSON not created at '{nested_json_path}'!")
            
            root_json_path = os.path.join(test_output_dir, "root.json")
            self.assertTrue(os.path.exists(root_json_path), f"Root JSON not created at '{root_json_path}'!")
            
            all_json_path = os.path.join(test_output_dir, "all_results.json")
            self.assertTrue(os.path.exists(all_json_path), f"All results JSON not created at '{all_json_path}'!")
            
            # Verify contents of the nested JSON report
            import json
            with open(nested_json_path, "r", encoding="utf-8") as jf:
                nested_data = json.load(jf)
            self.assertEqual(nested_data["total_images"], 1)
            self.assertIn("results", nested_data)
            self.assertEqual(nested_data["results"][0]["filename"], "nested_sample.png")
            
            nested_item = next(item for item in res["results"] if "_test_temp_sub" in item["relative_path"])
            self.assertIn(nested_item["status"], CLASSES)
            self.assertGreater(nested_item["confidence"], 0.0)
            print(f"[PASS] Test 6: Recursive Input discovery & per-subdirectory JSON -> {len(res['generated_json_files'])} JSON reports generated in '{test_output_dir}'")
        finally:
            # Clean up temporary test files
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
            temp_parent = os.path.join("Input", "_test_temp_sub")
            if os.path.exists(temp_parent):
                import shutil
                shutil.rmtree(temp_parent, ignore_errors=True)
            if os.path.exists(test_output_dir):
                import shutil
                shutil.rmtree(test_output_dir, ignore_errors=True)

    def test_07_operator_overrule_and_dataset_export(self):
        from src.inference import save_reviewed_batch, export_overruled_to_dataset
        import shutil

        # Create temporary directories for test
        temp_dataset_dir = os.path.join("Dataset", "_test_retrain_dataset")
        os.makedirs(temp_dataset_dir, exist_ok=True)
        temp_out_dir = os.path.join("Output", "_test_reviewed_out")
        os.makedirs(temp_out_dir, exist_ok=True)

        sample_img_path = "Dataset/Dry/dry_01.jpg" if os.path.exists("Dataset/Dry/dry_01.jpg") else "Dataset/Dry/dry_01.png"
        if not os.path.exists(sample_img_path):
            return

        sample_name = os.path.basename(sample_img_path)
        try:
            # Fake a batch result with 1 overruled item
            fake_item = {
                "filename": sample_name,
                "relative_path": sample_name,
                "relative_directory": ".",
                "full_path": os.path.abspath(sample_img_path),
                "ai_status": "Dry",
                "status": "Flooded",  # Overruled by operator to Flooded
                "confidence": 0.65,
                "needs_review": True,
                "is_overruled": True,
                "operator_label": "Flooded",
                "reviewed_at": "2026-09-05T14:15:00",
                "probabilities": {"Dry": 0.35, "Flooded": 0.65, "Planted": 0.0, "Others": 0.0}
            }
            fake_batch = {
                "total_images": 1,
                "directory": os.path.abspath("Dataset/Dry"),
                "output_directory": os.path.abspath(temp_out_dir),
                "summary_counts": {"Dry": 0, "Flooded": 1, "Planted": 0, "Others": 0},
                "overruled_count": 1,
                "needs_review_count": 0,
                "results": [fake_item],
                "subdirectories": ["."],
                "generated_json_files": []
            }

            # 1. Test saving reviewed batch
            saved_files = save_reviewed_batch(fake_batch, output_dir=temp_out_dir)
            self.assertGreater(len(saved_files), 0)
            
            all_path = os.path.join(temp_out_dir, "all_results.json")
            self.assertTrue(os.path.exists(all_path))
            import json
            with open(all_path, "r", encoding="utf-8") as jf:
                loaded = json.load(jf)
            self.assertEqual(loaded["overruled_count"], 1)
            self.assertEqual(loaded["results"][0]["status"], "Flooded")

            # 2. Test export to dataset
            copied = export_overruled_to_dataset([fake_item], dataset_dir=temp_dataset_dir)
            self.assertEqual(len(copied), 1)
            copied_dest = copied[0][1]
            self.assertTrue(os.path.exists(copied_dest))
            self.assertIn("Flood", copied_dest)  # Target directory should be Flood
            print(f"[PASS] Test 7: Operator overrule & continuous training export verified -> Copied to '{copied_dest}'")
        finally:
            if os.path.exists(temp_dataset_dir):
                shutil.rmtree(temp_dataset_dir, ignore_errors=True)
            if os.path.exists(temp_out_dir):
                shutil.rmtree(temp_out_dir, ignore_errors=True)


if __name__ == "__main__":
    print("="*60)
    print("[*] Running 4-Class Rice Field Neural Network Test Suite...")
    print("="*60)
    unittest.main()


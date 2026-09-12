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
        # Create a nested subdirectory in Input/ to test recursive discovery
        nested_dir = os.path.join("Input", "_test_temp_sub", "inner")
        os.makedirs(nested_dir, exist_ok=True)
        temp_img_path = os.path.join(nested_dir, "nested_sample.png")
        
        try:
            # Create a simple test RGB image
            test_img = Image.new("RGB", (224, 224), color=(34, 180, 76))
            test_img.save(temp_img_path)
            
            res = self.predictor.predict_directory("Input", recursive=True)
            self.assertGreater(res["total_images"], 0)
            
            # Verify that the nested file was found
            found_nested = any("_test_temp_sub" in item["relative_path"] for item in res["results"])
            self.assertTrue(found_nested, "Failed to discover nested image in Input subdirectory!")
            
            nested_item = next(item for item in res["results"] if "_test_temp_sub" in item["relative_path"])
            self.assertIn(nested_item["status"], CLASSES)
            self.assertGreater(nested_item["confidence"], 0.0)
            print(f"[PASS] Test 6: Recursive Input discovery -> Discovered and classified nested image successfully")
        finally:
            # Clean up temporary test files
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
            temp_parent = os.path.join("Input", "_test_temp_sub")
            if os.path.exists(temp_parent):
                import shutil
                shutil.rmtree(temp_parent, ignore_errors=True)

    def test_07_operator_overrule_and_dataset_export(self):
        from src.inference import export_overruled_to_dataset
        import shutil

        # Create temporary directory for test
        temp_dataset_dir = os.path.join("Dataset", "_test_retrain_dataset")
        os.makedirs(temp_dataset_dir, exist_ok=True)

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

            # Test export to dataset
            copied = export_overruled_to_dataset([fake_item], dataset_dir=temp_dataset_dir)
            self.assertEqual(len(copied), 1)
            copied_dest = copied[0][1]
            self.assertTrue(os.path.exists(copied_dest))
            self.assertIn("Flood", copied_dest)  # Target directory should be Flood
            print(f"[PASS] Test 7: Operator overrule & continuous training export verified -> Copied to '{copied_dest}'")
        finally:
            if os.path.exists(temp_dataset_dir):
                shutil.rmtree(temp_dataset_dir, ignore_errors=True)


if __name__ == "__main__":
    print("="*60)
    print("[*] Running 4-Class Rice Field Neural Network Test Suite...")
    print("="*60)
    unittest.main()


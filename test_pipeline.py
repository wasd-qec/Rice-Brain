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

    def test_06_coordinate_prediction(self):
        sample_path = "Dataset/Planted/planted_01.png"
        if os.path.exists(sample_path):
            res = self.predictor.predict_coordinate(sample_path, (200, 200), output_annotated_path="test_coord_output.png")
            self.assertIn("status", res)
            self.assertIn(res["status"], CLASSES)
            print(f"[PASS] Test 6: Coordinate crop at (200, 200) -> {res['status']} ({res['confidence']*100:.1f}%)")


if __name__ == "__main__":
    print("="*60)
    print("[*] Running 4-Class Rice Field Neural Network Test Suite...")
    print("="*60)
    unittest.main()

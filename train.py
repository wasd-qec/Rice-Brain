"""
train.py - Root training runner for the 4-Class Rice Field State Neural Network (Dry, Flooded, Planted, Others).
"""

from src.train import train_classifier

if __name__ == "__main__":
    train_classifier(
        dataset_dir="Dataset",
        epochs=25,
        batch_size=16,
        learning_rate=1e-3,
        save_path="rice_field_classifier.pth"
    )

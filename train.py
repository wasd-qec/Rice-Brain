"""
train.py - Root training runner for the 7-Class Rice Field State Neural Network (Dry, Water, Wet, Green rice, Green weed, Straw, Others).
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

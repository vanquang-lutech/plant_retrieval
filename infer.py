from PIL import Image
import numpy as np
import torch
from src.config import Config
from src.model import OrgansClassifier 
from src.transforms import BuildTransforms
import json

class Inferencer:
    def __init__(self, cfg: Config, checkpoint_path: str):
        self.cfg = cfg 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.num_classes = cfg.model.num_classes
        self.pretrained = cfg.model.pretrained
        self.backbone = cfg.model.backbone

        self.model = OrgansClassifier(self.backbone, self.pretrained, self.num_classes)
        self.model.load_state_dict(self.checkpoint['model'])
        self.model.to(self.device)
        self.model.eval()

    def get_probs(self, probs):
        return max(0.01, min(0.99, float(f"{probs:.4f}")))

    def infer(self, image: Image.Image):
        test_transforms = BuildTransforms(self.cfg).build_val_test_transforms()
        image = test_transforms(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
            label = np.argmax(probabilities, axis=0)
        return self.get_probs(probabilities[label]), label


if __name__ == "__main__":
    cfg = Config.load("checkpoint/config.json")

    if isinstance(cfg.data.padding_color, list):
        cfg.data.padding_color = tuple(cfg.data.padding_color)

    class_names = {v: k for k, v in json.load(open("checkpoint/class_to_idx.json")).items()}
    inferencer = Inferencer(
        cfg=cfg,
        checkpoint_path="checkpoint/model.pt",
    )
    image_test_path = "path/to/test/image.jpg"
    image = Image.open(image_test_path).convert("RGB")
    probabilities, label = inferencer.infer(image)
    print(f"Predicted class: {class_names[label]}")
    print(f"Probabilities: {probabilities}")
from torch.utils.data import Dataset
import os
from PIL import Image

class PlantDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels, self.image_paths = self.image_path_and_label()


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"[skip] {self.image_paths[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        
    def image_path_and_label(self):
        image_paths, labels_str = [], []
        for label in sorted(os.listdir(self.root_dir)):
            label_dir = os.path.join(self.root_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for image in sorted(os.listdir(label_dir)):
                if os.path.splitext(image)[1].lower() in  {".jpg", ".jpeg", ".png", ".webp"}:
                    image_paths.append(os.path.join(label_dir, image))
                    labels_str.append(label)
        self.class_to_idx = {c: i for i, c in enumerate(sorted(set(labels_str)))}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        labels = [self.class_to_idx[c] for c in labels_str]
        return labels, image_paths

    @property
    def num_classes(self):
        return len(self.class_to_idx)
    
    

    
        
    

    
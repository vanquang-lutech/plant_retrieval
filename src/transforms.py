from functools import partial
import timm
from torchvision import transforms

from src.utils import resize_pwd, UnshaprMask
    

class BuildTransforms:
    def __init__(self, cfg):
        self.cfg = cfg

    def build_train_transforms(self):
        pad_resize = partial(
            resize_pwd,
            padding_color=self.cfg.data.padding_color,
            target_size=self.cfg.data.image_size,
        )

        sharpen = UnshaprMask(radius=1.0, amount=1.0)

        train_transform = transforms.Compose([
            pad_resize,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomApply([transforms.RandomRotation(30)], p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
            ], p=0.2),
            sharpen,
            transforms.ToTensor(),
            transforms.RandomApply([
                transforms.RandomErasing(
                    p=1.0,
                    scale=(0.02, 0.2),
                    ratio=(0.3, 3.3),
                    value=0,
                )
            ], p=0.4),

            transforms.Normalize(self.cfg.data.mean, self.cfg.data.std),
        ])

        return train_transform
    
    def build_val_test_transforms(self):
        pad_resize = partial(
            resize_pwd,
            padding_color=self.cfg.data.padding_color,
            target_size=self.cfg.data.image_size,
        )

        sharpen = UnshaprMask(radius=1.0, amount=1.0)

        val_test_transform = transforms.Compose([
            pad_resize,
            sharpen,
            transforms.ToTensor(),
            transforms.Normalize(self.cfg.data.mean, self.cfg.data.std),
        ])

        return val_test_transform
    
    def __call__(self):
        train_transform = self.build_train_transforms()
        val_test_transform = self.build_val_test_transforms()
        return train_transform, val_test_transform


import torch.nn as nn
import timm

class OrgansClassifier(nn.Module):
    def __init__(self, backbone, pretrained, num_classes):
        super().__init__()
        self.backbone = backbone
        self.pretrained = pretrained
        self.num_classes = num_classes
    
        self.base_model = timm.create_model(backbone, pretrained=pretrained, drop_path_rate=0.1, num_classes=0)

        for param in self.base_model.parameters():
            param.requires_grad = False

        for block in self.base_model.blocks[-4:]:
            for param in block.parameters():
                param.requires_grad = True
        
        for param in self.base_model.norm.parameters():
            param.requires_grad = True

        self.in_features = self.base_model.num_features
        self.classifier = nn.Sequential(
            nn.Linear(self.in_features, self.in_features // 2),
            nn.BatchNorm1d(self.in_features // 2),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(self.in_features // 2, self.in_features // 4),
            nn.BatchNorm1d(self.in_features // 4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.in_features // 4, num_classes),
        )
    
    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self
    
    def forward(self, x):
        features = self.base_model(x)
        return self.classifier(features)




        
        

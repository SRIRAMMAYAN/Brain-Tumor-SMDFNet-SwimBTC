import torch
import torch.nn as nn
import timm

class HybridModel(nn.Module):
    def __init__(self, num_classes=4):
        super(HybridModel, self).__init__()

        # 🔹 Swin Tiny (lightweight)
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            num_classes=0
        )

        # 🔹 EfficientNet B0 (lightweight)
        self.efficient = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0
        )

        # Get output dimensions by passing a dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        swin_out = self.swin(dummy_input).shape[1]
        eff_out = self.efficient(dummy_input).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(swin_out + eff_out, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        swin_feat = self.swin(x)
        eff_feat = self.efficient(x)

        combined = torch.cat((swin_feat, eff_feat), dim=1)
        output = self.classifier(combined)
        return output
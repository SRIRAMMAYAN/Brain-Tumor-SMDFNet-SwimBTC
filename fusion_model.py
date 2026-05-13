import torch
import torch.nn as nn

class SimpleFusionNet(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleFusionNet, self).__init__()

        # MRI branch
        self.mri_branch = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # CT branch
        self.ct_branch = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 56 * 56 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, mri, ct):

        mri_feat = self.mri_branch(mri)
        ct_feat = self.ct_branch(ct)

        mri_feat = torch.flatten(mri_feat, 1)
        ct_feat = torch.flatten(ct_feat, 1)

        fused = torch.cat((mri_feat, ct_feat), dim=1)

        return self.fc(fused)
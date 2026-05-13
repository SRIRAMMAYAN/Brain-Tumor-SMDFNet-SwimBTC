import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pydicom
import numpy as np
import cv2

from fusion_model import SimpleFusionNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# FUNCTION TO COLLECT ALL DICOM FILES RECURSIVELY
# =====================================================

def collect_dicom_files(root_dir):
    dicom_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".dcm"):
                dicom_files.append(os.path.join(root, file))
    return sorted(dicom_files)

# =====================================================
# DATASET
# =====================================================

class FusionDataset(Dataset):
    def __init__(self, mri_dir, ct_dir):

        self.mri_paths = collect_dicom_files(mri_dir)
        self.ct_paths = collect_dicom_files(ct_dir)

        min_len = min(len(self.mri_paths), len(self.ct_paths))
        self.mri_paths = self.mri_paths[:min_len]
        self.ct_paths = self.ct_paths[:min_len]

        # TEMPORARY LABELS (replace later)
        self.labels = [0] * min_len

    def load_dicom(self, path):
        dicom = pydicom.dcmread(path, force=True)
        img = dicom.pixel_array.astype(np.float32)

        if img.ndim == 3:
            img = img[img.shape[0] // 2]

        img = cv2.resize(img, (224, 224))
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        return torch.tensor(img).unsqueeze(0)

    def __len__(self):
        return len(self.mri_paths)

    def __getitem__(self, idx):
        mri = self.load_dicom(self.mri_paths[idx])
        ct = self.load_dicom(self.ct_paths[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mri, ct, label

# =====================================================
# CORRECT PATHS FOR YOUR STRUCTURE
# =====================================================

mri_folder = "fused_dataset/T1-MRI/DICOM"
ct_folder = "fused_dataset/CT/DICOM"

dataset = FusionDataset(mri_folder, ct_folder)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

print("Total MRI DICOM files:", len(dataset.mri_paths))
print("Total CT DICOM files:", len(dataset.ct_paths))

# =====================================================
# MODEL
# =====================================================

model = SimpleFusionNet(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5

# =====================================================
# TRAIN LOOP
# =====================================================

for epoch in range(epochs):
    total_loss = 0

    for mri, ct, label in loader:
        mri, ct, label = mri.to(device), ct.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(mri, ct)

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "fusion_model.pth")
print("Fusion model saved successfully ✅")
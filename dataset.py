import os
import glob
import numpy as np
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class OmamaDataset(Dataset):
    def __init__(self, data_dir, crop_size=128):
        self.data_dir = data_dir
        self.crop_size = crop_size
        self.samples = []
        image_dir = os.path.join(data_dir, "images")
        patient_folders = [f for f in os.listdir(image_dir) if not f.startswith('.')]
        
        for patient in patient_folders:
            patient_image_path = os.path.join(image_dir, patient)
            patient_meta_path = os.path.join(data_dir, "metadata", patient)
            npz_files = glob.glob(os.path.join(patient_image_path, "*.npz"))
            
            for npz_file in npz_files:
                filename = os.path.basename(npz_file).replace('.npz', '.json')
                json_file = os.path.join(patient_meta_path, filename)
                if os.path.exists(json_file):
                    self.samples.append((npz_file, json_file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, meta_path = self.samples[idx]
        data = np.load(img_path)
        image = data['data']
        
        image = image.astype(np.float32) / 255.0 
        
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
        label_str = meta.get('label', 'NonCancer')
        label = 1.0 if label_str == 'IndexCancer' else 0.0
        
        # --- NEW: ROI CROPPING ---
        image_tensor = torch.from_numpy(image).unsqueeze(0) # Add channel dim for torchvision
        
        if label == 1.0 and 'coords' in meta:
            # Crop around the cancer coordinates
            x_min, y_min, x_max, y_max = meta['coords']
            # Add a margin so we see the tissue around the cancer
            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(256, x_max + margin)
            y_max = min(256, y_max + margin)
            
            # Crop and then resize to a standard size (128x128)
            image_tensor = TF.crop(image_tensor, y_min, x_min, y_max - y_min, x_max - x_min)
            image_tensor = TF.resize(image_tensor, [self.crop_size, self.crop_size])
        else:
            # For normal images, just crop the center of the breast
            image_tensor = TF.center_crop(image_tensor, [self.crop_size, self.crop_size])
            
        image = image_tensor.squeeze(0).numpy()
        
        # Normalize brightness
        mean = np.mean(image)
        std = np.std(image)
        if std > 0: 
            image = (image - mean) / std
            
        return image, label

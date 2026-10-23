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
        # Loop through the Study folders directly (e.g., 2.25.12345...)
        study_folders = [f for f in os.listdir(image_dir) if not f.startswith('.')]
        
        for study in study_folders:
            study_image_path = os.path.join(image_dir, study)
            study_meta_path = os.path.join(data_dir, "metadata", study)
            
            npz_files = glob.glob(os.path.join(study_image_path, "*.npz"))
            
            # We only want studies that have exactly 4 views
            if len(npz_files) == 4:
                self.samples.append((study, npz_files, study_meta_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        study_id, npz_files, meta_dir = self.samples[idx]
        
        images = []
        study_label = 0.0
        
        # Loop through all 4 views for this study
        for npz_file in npz_files:
            # 1. Load Image
            data = np.load(npz_file)
            img = data['data'].astype(np.float32) / 255.0
            
            # 2. Load Metadata
            filename = os.path.basename(npz_file).replace('.npz', '.json')
            json_file = os.path.join(meta_dir, filename)
            with open(json_file, 'r') as f:
                meta = json.load(f)
            
            # 3. If ANY view has cancer, the whole study is positive
            if meta.get('label') == 'IndexCancer':
                study_label = 1.0
                
            # 4. Crop and Normalize
            img_tensor = torch.from_numpy(img).unsqueeze(0)
            if meta.get('label') == 'IndexCancer' and 'coords' in meta:
                x_min, y_min, x_max, y_max = meta['coords']
                margin = 20
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(256, x_max + margin)
                y_max = min(256, y_max + margin)
                img_tensor = TF.crop(img_tensor, y_min, x_min, y_max - y_min, x_max - x_min)
                img_tensor = TF.resize(img_tensor, [self.crop_size, self.crop_size])
            else:
                img_tensor = TF.center_crop(img_tensor, [self.crop_size, self.crop_size])
                
            img = img_tensor.squeeze(0).numpy()
            
            mean = np.mean(img)
            std = np.std(img)
            if std > 0:
                img = (img - mean) / std
                
            images.append(img)
            
        # Stack the 4 images into a single tensor: Shape becomes (4, 128, 128)
        images = np.stack(images)
            
        return images, study_label

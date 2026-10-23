import numpy as np
import json
import matplotlib.pyplot as plt

# Data paths
patient_id = "2.25.100019467787541382690455566629544103893"
image_id = "2.25.124381528605702917763164197238060295247"

img_path = f"/Users/ab/Downloads/256x256_studylevel/images/{patient_id}/{image_id}.npz"
meta_path = f"/Users/ab/Downloads/256x256_studylevel/metadata/{patient_id}/{image_id}.json"

# Load metadata
with open(meta_path, 'r') as f:
    metadata = json.load(f)
    print("--- Metadata ---")
    print(metadata)

# Load image
data = np.load(img_path)
print("\n--- NPZ Keys ---")
print(data.files) 

image_array = data[data.files[0]]
print("\n--- Image Shape ---")
print(image_array.shape)

# Display image
plt.imshow(image_array, cmap='gray')
plt.title(f"Image ID: {image_id[-5:]}")
plt.axis('off')
plt.show()

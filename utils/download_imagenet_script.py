from datasets import load_dataset
import os
from tqdm import tqdm

# Set HF Token
HF_TOKEN = 'example_token' # replace with your actual token when running
print("Script starting...")
print("Loading ImageNet validation set from HuggingFace...")
try:
    imagenet_val = load_dataset('imagenet-1k', split='validation', token=HF_TOKEN)
    print(f"Dataset loaded successfully! Found {len(imagenet_val)} images.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)



# Set the root directory
root_dir = './imagenet_data'
val_dir = os.path.join(root_dir, 'val')
os.makedirs(val_dir, exist_ok=True)

# Get class names mapping (synset IDs)
# ImageNet uses WordNet IDs like 'n01440764'
class_names = imagenet_val.features['label'].names

print(f"Saving {len(imagenet_val)} images to disk in ImageNet format...")

# Iterate through dataset and save images
for idx, item in enumerate(tqdm(imagenet_val)):
    label = item['label']
    image = item['image']
    
    # Get the synset ID (WordNet ID) for this class
    # ImageNet-1k classes are mapped to their synset IDs
    synset_id = class_names[label]
    
    # Create class directory
    class_dir = os.path.join(val_dir, synset_id)
    os.makedirs(class_dir, exist_ok=True)
    
    # Save image with original filename or sequential name
    image_path = os.path.join(class_dir, f'ILSVRC2012_val_{idx:08d}.JPEG')
    image.save(image_path)

print(f"Done! Images saved to {val_dir}")
print(f"You can now use: ImageNet(root='{root_dir}', split='val', transform=preprocess)")
import os
from PIL import Image
import matplotlib.pyplot as plt

def show_image_all_captions(img_name, captions_dict, images_dir):
    img_path = os.path.join(images_dir, img_name)
    image = Image.open(img_path).convert("RGB")
    
    plt.figure(figsize=(6,6))
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    
    print("Captions de", img_name)
    for c in captions_dict[img_name]:
        print(" •", c)



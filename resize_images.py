from PIL import Image
import os

# size example: (64, 64)

def resize_images(input, output, size):

    # Tworzy folder wyjściowy, jeśli nie istnieje
    os.makedirs(output, exist_ok=True)
    
    for filename in os.listdir(input):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(input, filename)
            img = Image.open(img_path)
            img_resized = img.resize(size, Image.Resampling.LANCZOS)

            output_path = os.path.join(output, filename)
            img_resized.save(output_path)

            print(f"Przeskalowano: {filename}")

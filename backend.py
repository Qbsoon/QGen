import torch
from torchvision import transforms
from imagen_pytorch.data import Dataset, DataLoader
from PIL import Image
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
import json
from transformers import T5Tokenizer, T5EncoderModel

def create_captions (source_directory, captions_creating_batch_size = 8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    # --- List Image Files ---
    # Ensure consistent order by sorting
    image_files = sorted([
        os.path.join(source_directory, f)
        for f in os.listdir(source_directory)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    print(f"Found {len(image_files)} images in {source_directory}")

    # --- Generate Captions ---
    captions_data = [] # List to store captions
    model.eval() # Set model to evaluation mode

    with torch.no_grad(): # Disable gradient calculation for inference
        for i in range(0, len(image_files), captions_creating_batch_size):
            batch_files = image_files[i:i+captions_creating_batch_size]
            raw_images = []
            valid_files_in_batch = [] # Keep track of files successfully loaded in this batch

            # Load images in the batch
            for img_path in batch_files:
                try:
                    image = Image.open(img_path).convert('RGB')
                    raw_images.append(image)
                    valid_files_in_batch.append(img_path)
                except Exception as e:
                    print(f"Warning: Could not load image {img_path}. Skipping. Error: {e}")

            if not raw_images: # Skip if no images were loaded in the batch
                print(f"Skipping empty batch starting at index {i}")
                continue

            # Preprocess images
            inputs = processor(images=raw_images, return_tensors="pt").to(device)

            # Generate captions
            outputs = model.generate(**inputs, max_length=75, num_beams=4)

            # Decode captions
            decoded_captions = processor.batch_decode(outputs, skip_special_tokens=True)

            # Store results for this batch
            for img_path, caption in zip(valid_files_in_batch, decoded_captions):
                captions_data.append({'image_path': img_path, 'caption': caption.strip()})

            print(f"Processed batch {i//captions_creating_batch_size + 1}/{(len(image_files) + captions_creating_batch_size - 1)//captions_creating_batch_size}, Captions generated: {len(captions_data)}")


    # --- Save Captions ---
    try:
        with open(os.path.join(source_directory, 'captions.json'), 'w') as f:
            json.dump(captions_data, f, indent=4)
        print("Captions saved successfully.")
    except Exception as e:
        print(f"Error saving captions: {e}")

def encode_captions(source_directory):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(source_directory, 'captions.json'), 'r') as f:
        captions_data = json.load(f)

    image_files = sorted([
        os.path.join(source_directory, f)
        for f in os.listdir(source_directory)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    captions = [image['caption'] for image in captions_data if image['image_path'] in image_files]

    t5_model_name = 't5-large'
    print(f"Loading T5 Tokenizer and Encoder Model ({t5_model_name})...")
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
    t5_model = T5EncoderModel.from_pretrained(t5_model_name).cuda() # Move model to GPU
    t5_model.eval() # Set T5 model to evaluation mode
    print("T5 models loaded.")

    text_encoding_batch_size = 16
    max_text_seq_len = 256 
    all_text_embeds = []

    with torch.no_grad(): # No need to track gradients for encoding
        for i in range(0, len(captions), text_encoding_batch_size):
            batch_texts = captions[i:i+text_encoding_batch_size]
            tokenized_inputs = t5_tokenizer(
                batch_texts,
                padding='max_length',       # Pad to max_length
                truncation=True,            # Truncate sequences longer than max_length
                max_length=max_text_seq_len,# The maximum sequence length
                return_tensors='pt'         # Return PyTorch tensors
            ).to(device)

            # Get embeddings from T5 model
            # T5EncoderModel output is a BaseModelOutput, embeddings are in last_hidden_state
            outputs = t5_model(input_ids=tokenized_inputs.input_ids, attention_mask=tokenized_inputs.attention_mask)
            text_embeds_batch = outputs.last_hidden_state # Shape: [batch_size, max_text_seq_len, 1024]

            # Append to list (move to CPU to save GPU memory during accumulation)
            all_text_embeds.append(text_embeds_batch.cpu())

    # Concatenate all embedding batches
    text_embeddings_tensor = torch.cat(all_text_embeds, dim=0)
    return text_embeddings_tensor

# Handles None returns from failed image loads
def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch)) # Filter out samples where image loading failed
    if not batch: return torch.Tensor(), torch.Tensor() # Return empty tensors if batch is empty
    return torch.utils.data.dataloader.default_collate(batch)

class ImageDataset(Dataset):
    def __init__(self, root_dir, text_embeds, transform=None):
        self.root_dir = root_dir
        self.files = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.text_embeds = text_embeds
        self.transform = transform

        assert len(self.files) == len(self.text_embeds), "Number of images is different from number of text encodings"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f'Error loading image {img_path}: {e}')
            return None, None

        text_embedding = self.text_embeds[idx]

        if self.transform:
            image_t = self.transform(image)
        return image_t, text_embedding

def create_dataloader(source_directory, text_embeddings_tensor, batch_size=4, unet2_image_size=256):
    transform = transforms.Compose([
        transforms.Resize(unet2_image_size),       # Resize smaller edge to image_size
        transforms.CenterCrop(unet2_image_size),   # Crop center to image_size x image_size
        transforms.ToTensor(),               # Convert PIL image [0, 255] to tensor [0, 1]
        # Optional: Add normalization if required by the model, e.g., for [-1, 1] range
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
        
    dataset = ImageDataset(root_dir=source_directory, text_embeds=text_embeddings_tensor, transform=transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn) # num_workers sets parallel data loading threads

    return dataloader

def workflow(source_dir, batch_size=4, unet2_image_size=256):
    text_embeds = encode_captions(source_dir)
    dataloader = create_dataloader(source_dir, text_embeds, batch_size, unet2_image_size)

    return dataloader
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os, os.path

# Load CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define grasp type prompts
grasp_prompts = [
    "An object that requires a **cylindrical grasp**, because it is held cylindrically with the fingers wrapped around it, like a bottle, a hammer handle, or a flashlight.",
    "An object that requires a **hook grasp**, where the fingers form a hook-like shape without the thumb, like carrying a bag, a bucket, or a suitcase.",
    "An object that requires a **palmar grasp**, because it rests against the palm with fingers wrapped around it, like a notebook, a smartphone, or a book.",
    "An object that requires a **spherical grasp**, because it is round and gripped with the fingers and thumb forming a curve, like an apple, a small ball, or an orange.",
    "An object that requires a **tripod grasp**, because it is held between the thumb, index, and middle fingers, like a pen, a piece of chalk, or a small screwdriver.",
    "An object that requires a **pinch grasp**, because it is small and held between the thumb and index finger, like a coin, a needle, or a small button.",
    "An object that requires an **open grasp**, because it is large and flat, requiring a wide grip, like a box, a book, or a tablet.",
    "An object that requires an **adaptive grasp**, because it has an uneven shape, requiring fingers to adjust dynamically, like a stuffed toy, a rock, or a crumpled piece of paper.",
    "An object that requires a **lateral grasp**, because it is held between the thumb and the side of the index finger, like a key, a credit card, or a thin piece of paper."
]



text_inputs = processor(
    text=grasp_prompts,  # List of prompts
    return_tensors="pt",  # Return PyTorch tensors
    padding=True,  # Ensure prompts are padded to the same length
    truncation=True  # Truncate prompts if they exceed the max length
).to(device)

def predict_image_grasp_type(image_path):
    # Preprocess query image
    # image_path = '/home/kpi_anna/data/test_grasp_dataset/query_set/image1.png'
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Get image and text features
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        text_features = model.get_text_features(**text_inputs)

    # Calculate similarity scores
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = torch.matmul(image_features, text_features.T)

    # Predict the grasp type
    predicted_index = similarity.argmax().item()
    predicted_prompt = grasp_prompts[predicted_index]
    print(f"Predicted grasp type for {os.path.basename(image_path)}: {predicted_prompt}")
    return predicted_prompt

# model evaluation
query_set_directory = '/home/kpi_anna/data/test_grasp_dataset/query_set'
num_images = len([name for name in os.listdir(query_set_directory) if os.path.isfile(name)])
num_correct_predictions = 0
set_directory = '/home/kpi_anna/data/test_grasp_dataset/set_1'

# Load ground truth labels (example as a dictionary)
ground_truth = {
    "image.png": "cylindrical grasp",
    "image1.png": "cylindrical grasp",
    "image2.png": "hook grasp",
    "image3.png": "hook grasp",
    "image4.png": "palmar grasp",
    "image5.png": "palmar grasp",
    "image6.png": "spherical grasp",
    "image7.png": "pinch grasp",
    "image8.png": "pinch grasp",
    "image9.png": "tripod grasp",
    "image10.png": "cylindrical grasp",
    "image11.png": "open grasp",
    "image12.png": "adaptive grasp",
}

# Initialize counters
num_correct_predictions = 0
num_images = len(ground_truth)

# Iterate through all images in the query set
for image_name, true_label in ground_truth.items():
    image_path = os.path.join(query_set_directory, image_name)
    predicted_label = predict_image_grasp_type(image_path)
     # Compare prediction with ground truth
    if true_label in predicted_label:
        num_correct_predictions += 1

# Calculate accuracy
accuracy = (num_correct_predictions / num_images) * 100
print(f"Accuracy: {accuracy:.2f}%")

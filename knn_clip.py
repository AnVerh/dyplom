import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import numpy as np

# Initialize CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
# clip_model, clip_preprocess = clip.load("ViT-B/32", device)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Directory where the support images are located
support_dir = "/home/kpi_anna/data/test_grasp_dataset/set_1"  # Modify with your actual directory path
grasp_types = os.listdir(support_dir)  # List of grasp types (folders)

# Extract image features for the support set
support_features = {}  # Dictionary to hold features for each class
for grasp_type in grasp_types:
    grasp_images = os.listdir(os.path.join(support_dir, grasp_type))
    image_features = []
    for image_name in grasp_images:
        image_path = os.path.join(support_dir, grasp_type, image_name)
        image = Image.open(image_path)
        inputs = clip_processor(images=image, return_tensors='pt').to(device)
        with torch.no_grad():
            image_feature = clip_model.get_image_features(**inputs)
        image_features.append(image_feature.cpu().numpy())
    # Store mean feature for each grasp type
    support_features[grasp_type] = np.mean(image_features, axis=0)

# Convert the dictionary to numpy arrays for later use in KNN
support_features = {grasp: np.array(features) for grasp, features in support_features.items()}

# Define multiple prompts for each grasp type
prompts = {
    "cylindrical": ["An object that requires a cylindrical grasp, like a bottle or a handle.",
                    "An object that is held cylindrically with all fingers wrapped around it."],
    "hook": ["An object that requires a hook grasp, like a bag or a bucket.",
             "An object held with a hook-like shape using 4 fingers, without the thumb."],
    "palmar": ["An object that requires a palmar grasp, can fit in a hand, like a notebook or smartphone.",
               "An object that is held in the palm of the hand."],
    "spherical": ["An object that requires a spherical grasp, like an apple or a small ball.",
                  "An object that is grasped with fingers and palm forming a sphere."],
    "tripod": ["An object that requires a tripod grasp, like a pen or a pencil.",
               "An object held between three fingers."],
    "pinch": ["An object that requires a pinch grasp, like a coin or needle.",
              "An object held with the thumb and one finger."],
    "open": ["An object that requires an open grasp, like a flat box.",
             "An object that is grasped with an open hand."],
    "adaptive": ["An object that requires an adaptive grasp, has an uneven form.",
                 "An object that needs an adaptable grip based on its shape."]
}

# Function to encode multiple prompts and average text embeddings
def encode_text_prompts(prompts, clip_model, device):
    text_features = {}
    for grasp_type, prompt_list in prompts.items():
        # Tokenize and encode all prompts for this grasp type
        text_inputs = clip_processor(text=prompt_list, return_tensors='pt', padding=True).to(device)
        with torch.no_grad():
            text_embedding = clip_model.get_text_features(**text_inputs)
        # Average the text embeddings
        text_features[grasp_type] = text_embedding.mean(dim=0).cpu().numpy()
    return text_features

# Get text features for each grasp class
text_features = encode_text_prompts(prompts, clip_model, device)
from sklearn.neighbors import NearestNeighbors

def predict_image_grasp_type(image_path):
    query_image = Image.open(image_path)
    query_image_input = clip_processor(images=query_image, return_tensors='pt').to(device)

    # Extract image feature for query image
    with torch.no_grad():
        query_image_feature = clip_model.get_image_features(**query_image_input)

    # Reshape support features to fit into KNN
    support_labels = list(support_features.keys())
    support_vectors = np.array([support_features[label] for label in support_labels])
    support_vectors = support_vectors.squeeze(1)

    # Perform KNN to find nearest neighbors
    knn = NearestNeighbors(n_neighbors=7, metric="cosine")  # Use cosine similarity
    knn.fit(support_vectors)

    # Find the nearest neighbor to the query image
    distances, indices = knn.kneighbors(query_image_feature.cpu().numpy().reshape(1, -1))

    # Get predicted label from the nearest neighbor
    predicted_label = support_labels[indices[0][0]]
    print(f"Predicted grasp type for {os.path.basename(image_path)}: {predicted_label}")
    predicted_label = str(predicted_label).replace('_', ' ')
    return predicted_label

# Example query image
# query_image_path = "/home/kpi_anna/data/test_grasp_dataset/query_set/image5.png"  # Modify with your query image path
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
# print(f"Predicted Grasp: {predicted_label}")

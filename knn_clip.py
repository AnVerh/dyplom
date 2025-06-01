import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from collections import Counter


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
# Reshape support features to fit into KNN
support_labels = list(support_features.keys())
support_vectors = np.array([support_features[label] for label in support_labels])
support_vectors = support_vectors.squeeze(1)

# Define multiple prompts for each grasp type
prompts = {
    "cylindrical_grasp": ["An object that requires a cylindrical grasp, like a bottle or a handle.",
                    "An object that is held cylindrically with all fingers wrapped around it.",
                    "An object that is heavy and requires power to lift it"],
    "hook_grasp": ["An object that requires a hook grasp, like a bag or a bucket.",
             "An object held with a hook-like shape using 4 fingers, without the thumb."],
    "palmar_grasp": ["An object that requires a palmar grasp, can fit in a hand, like a notebook or smartphone.",
               "An object that is held in the palm of the hand."],
    "spherical_grasp": ["An object that requires a spherical grasp, like an apple or a small ball.",
                  "An object that is grasped with fingers and palm forming a sphere."],
    "tripod_grasp": ["An object that requires a tripod grasp, like a pen or a pencil.",
               "An object held between three fingers."],
    "pinch_grasp": ["An object that requires a pinch grasp, like a coin or needle.",
              "An object held with the thumb and one other finger.", 
              "An object that is tiny, light, and delicate"],
    "open_grasp": ["An object that requires an open grasp, like a flat box.",
             "An object that is grasped with an open hand."],
    #"adaptive_grasp": ["An object that requires an adaptive grasp, has an uneven form.",
    #             "An object that needs an adaptable grip based on its shape."],
    "lateral_grasp": ["An object that irequires a lateral grasp, like a key or a credit card.",
                      "An object that is held between the thumb and the side of the index finger"]
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

def predict_image_grasp_type_with_knn(image_path, top_k=3):
    query_image = Image.open(image_path)
    query_image_input = clip_processor(images=query_image, return_tensors='pt').to(device)

    # Extract image feature for query image
    with torch.no_grad():
        query_image_feature = clip_model.get_image_features(**query_image_input)
        query_vector = query_image_feature[0].cpu().numpy()
    # Normalize
    query_vector = query_vector / np.linalg.norm(query_vector)

    # Perform KNN to find nearest neighbors
    knn = NearestNeighbors(n_neighbors=7, metric="cosine")  # Use cosine similarity
    knn.fit(support_vectors)

    # Find the nearest neighbor to the query image
    _, indices = knn.kneighbors(query_image_feature.cpu().numpy().reshape(1, -1))
    label_scores = defaultdict(float)
    for idx in indices[0]:
        label = support_labels[idx]
        img_sim = cosine_similarity([query_vector], [support_vectors[idx]])[0][0]
        txt_sim = cosine_similarity([query_vector], [text_features[label]])[0][0]
        combined = 0.5 * img_sim + 0.5 * txt_sim
        label_scores[label] += combined

    top_predicted_labels = sorted(label_scores.items(), key=lambda x:x[1], reverse=True)[:top_k]
    top_predicted_labels = [str(label).replace('_', ' ') for label in top_predicted_labels]
    parsed_predictions = [eval(str(pred)) for pred in top_predicted_labels]
    # Extract just the classes
    predicted_classes = [pred for (pred, score) in parsed_predictions]
    return predicted_classes

def evaluate():
    # Example query image
    # query_image_path = "/home/kpi_anna/data/test_grasp_dataset/query_set/image5.png"  # Modify with your query image path
    query_set_directory = '/home/kpi_anna/data/test_grasp_dataset/query_set'
    query_labels_path = "/home/kpi_anna/grasp_scripts/query_labels.json"
    # Load ground truth labels (example as a dictionary)
    import json

    with open(query_labels_path, "r") as f:
        ground_truth = json.load(f)

    # Initialize counters
    class_total = 0
    correct=0
    num_images = len(ground_truth)
    class_correct = Counter()
    class_total = Counter()
    # Iterate through all images in the query set
    for image_name, true_labels in ground_truth.items():
        image_path = os.path.join(query_set_directory, image_name)
        
        top_k_predictions = predict_image_grasp_type_with_knn(image_path, top_k=1)  # returns list of top 3 predictions
        print(f'Top k predictions are: {top_k_predictions}')
        # parsed_predictions = [eval(str(pred)) for pred in top_k_predictions]
        # Extract just the classes
        # predicted_classes = [pred.replace('_', ' ') for (pred, score) in parsed_predictions]
        predicted_classes = [pred.replace('_', ' ') for pred in top_k_predictions]

        # Count total instances for each ground truth class
        for grasp_type in true_labels:
            class_total[grasp_type] += 1  # Tracks how many times this grasp type appears in ground truth

        # Check which predictions match the ground truth
        if any(pred in true_labels for pred in predicted_classes):
            correct += 1
            for pred in predicted_classes:
                if pred in true_labels:  # If prediction is correct for any grasp type
                    class_correct[pred] += 1  # Increment correct count for that grasp type

            print(f"[✔] {image_name} | Predicted: {predicted_classes} | GT: {true_labels}")
        else:
            print(f"[✘] {image_name} | Predicted: {predicted_classes} | GT: {true_labels}")

    # Compute per-class accuracy
    class_accuracy = {cls: class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0 for cls in class_total}
    
    # Print class-wise accuracy
    print("\nClass-Wise Accuracy:")
    for grasp_type, acc in class_accuracy.items():
        print(f"{grasp_type}: {acc*100:.2f}%")

    # Final Accuracy
    overall_accuracy = 100 * correct / len(ground_truth)
    print(f"\nFinal Accuracy: {overall_accuracy:.2f}%")

if __name__ == "__main__":
    evaluate()
    # im_path = '/home/kpi_anna/data/image3.png'
    # preds = predict_image_grasp_type_with_knn(im_path)
    # print(preds)

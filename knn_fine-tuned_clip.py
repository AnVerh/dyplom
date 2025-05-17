import torch
from transformers import CLIPProcessor
from PIL import Image
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import json
from clip_classifier import CLIPWithProjectionHead  # Your custom model

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPWithProjectionHead(freeze_base=True).to(device)
clip_model.load_state_dict(torch.load("/home/kpi_anna/clip_with_projection_head.pth"))
clip_model.eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Dataset paths
support_dir = "/home/kpi_anna/data/test_grasp_dataset/set_1"
query_dir = "/home/kpi_anna/data/test_grasp_dataset/query_set"
query_labels_path = "/home/kpi_anna/grasp_scripts/query_labels.json"

# Load ground truth
with open(query_labels_path, "r") as f:
    ground_truth = json.load(f)

# Prompts per class
prompts = {
    "cylindrical_grasp": ["An object that requires a cylindrical grasp, like a bottle or a handle.",
                          "An object that is held cylindrically with all fingers wrapped around it."],
    "hook_grasp": ["An object that requires a hook grasp, like a bag or a bucket.",
                   "An object held with a hook-like shape using 4 fingers, without the thumb."],
    "palmar_grasp": ["An object that requires a palmar grasp, like a notebook or smartphone.",
                     "An object held in the palm of the hand."],
    "spherical_grasp": ["An object that requires a spherical grasp, like an apple or a small ball.",
                        "An object grasped with fingers and palm forming a sphere."],
    "tripod_grasp": ["An object that requires a tripod grasp, like a pen or a pencil.",
                     "An object held between three fingers."],
    "pinch_grasp": ["An object that requires a pinch grasp, like a coin or needle.",
                    "An object held with the thumb and one other finger."],
    "open_grasp": ["An object that requires an open grasp, like a flat box.",
                   "An object that is grasped with an open hand."],
    "lateral_grasp": ["An object that requires a lateral grasp, like a key or a credit card.",
                      "An object held between the thumb and the side of the index finger."]
}

# Get text embeddings
def encode_text_prompts(prompts_dict):
    text_features = {}
    for label, prompt_list in prompts_dict.items():
        inputs = clip_processor(text=prompt_list, return_tensors='pt', padding=True).to(device)
        with torch.no_grad():
            features = clip_model.get_text_features(**inputs)
        text_features[label] = features.mean(dim=0).cpu().numpy()
    return text_features

text_features = encode_text_prompts(prompts)

# Get averaged image features per class
support_features = {}
for label in os.listdir(support_dir):
    img_features = []
    for fname in os.listdir(os.path.join(support_dir, label)):
        img_path = os.path.join(support_dir, label, fname)
        img = Image.open(img_path)
        inputs = clip_processor(images=img, return_tensors='pt').to(device)
        with torch.no_grad():
            feat = clip_model.get_image_features(**inputs)
        img_features.append(feat.cpu().numpy())
    support_features[label] = np.mean(img_features, axis=0)

support_labels = list(support_features.keys())
support_vectors = np.array([support_features[label] for label in support_labels]).squeeze(1)

# Build KNN
knn = NearestNeighbors(n_neighbors=7, metric="cosine")
knn.fit(support_vectors)

# Inference
correct = 0
top_k = 3

for fname, true_labels in ground_truth.items():
    image_path = os.path.join(query_dir, fname)
    image = Image.open(image_path)
    # inputs = clip_processor(images=image, return_tensors='pt').to(device)

    # # Get query image feature
    # with torch.no_grad():
    #     query_feat = clip_model.get_image_features(**inputs)[0].cpu().numpy()
    query_feat = clip_model.get_image_features(image)[0].cpu().detach().numpy()
    norm_query_feat = query_feat / np.linalg.norm(query_feat)

    # --- (1) Similarity to Text Prompts ---
    text_similarities = {}
    for label, text_feat in text_features.items():
        sim = cosine_similarity([norm_query_feat], [text_feat])[0][0]
        text_similarities[label] = sim

    # --- (2) Similarity to Support Images (KNN) ---
    _, indices = knn.kneighbors([query_feat])
    image_similarities = defaultdict(float)
    for idx in indices[0]:
        label = support_labels[idx]
        sim = cosine_similarity([norm_query_feat], [support_vectors[idx]])[0][0]
        image_similarities[label] += sim

    # --- (3) Combine Scores ---
    combined_scores = {}
    for label in support_labels:
        text_score = text_similarities.get(label, 0)
        image_score = image_similarities.get(label, 0)
        combined_scores[label] = 0.5 * text_score + 0.5 * image_score

    # Top-k prediction
    top_preds = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    predicted_classes = [label.replace('_', ' ') for label, _ in top_preds]

    # Evaluation
    if any(pred in true_labels for pred in predicted_classes):
        correct += 1
        print(f"[✔] {fname} | Predicted: {predicted_classes} | GT: {true_labels}")
    else:
        print(f"[✘] {fname} | Predicted: {predicted_classes} | GT: {true_labels}")

# Final Accuracy
accuracy = 100 * correct / len(ground_truth)
print(f"\nFinal Accuracy: {accuracy:.2f}%")

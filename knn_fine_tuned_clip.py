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
from PIL import Image, ImageEnhance, ImageOps
import random
from torchvision import transforms
from collections import Counter


def augment(img):
    return [
        img,                              # original
        img.transpose(Image.FLIP_LEFT_RIGHT),
        ImageEnhance.Brightness(img).enhance(1.2),
        ImageOps.autocontrast(img),
    ]

# augment = transforms.Compose([
#     transforms.RandomRotation(degrees=10), 
#     transforms.ColorJitter(brightness=0.1, contrast=0.1),
#     transforms.RandomHorizontalFlip(p=0.5)
# ])

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPWithProjectionHead(freeze_base=True).to(device)
clip_model.load_state_dict(torch.load("/home/kpi_anna/clip_with_projection_head_75_2.pth"))
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
        "cylindrical_grasp": [
            "An object that requires a cylindrical grasp, like a bottle or a handle, with fingers wrapping completely around it.",
            "An object that is held in a power grip, where all fingers enclose the object and the thumb provides support.",
            "A rounded object that is lifted by encircling it with the hand, ensuring a strong grip and stability."
        ],
        "hook_grasp": [
            "An object that requires a hook grasp, like a bag or a bucket, where four fingers curl around it while the thumb stays out.",
            "An object held using a hook-like hand shape, without using the thumb, enabling a secure grip while carrying weight.",
            "A handle-like object grasped with flexed fingers without direct palm contact, commonly used for lifting heavy loads."
        ],
        "palmar_grasp": [
            "An object that requires a palmar grasp, fitting completely against the palm with the fingers and thumb supporting it.",
            "An object that is flat and fits against the palm with support from the thumb, like holding a book or a tablet.",
            "A broad, flat object grasped with full palm contact, using finger pressure to maintain stability while carrying or holding."
        ],
        "spherical_grasp": [
            "An object that requires a spherical grasp, like an apple or a ball, held with fingers and palm forming a rounded shape.",
            "A curved object that is grasped by spreading the fingers around its surface and using the palm for support.",
            "A grasp used to hold round objects securely, using an arched hand shape and distributed finger pressure."
        ],
        "tripod_grasp": [
            "An object that requires a tripod grasp, like a pencil or a small tool, held between the thumb, index, and middle finger.",
            "A small object grasped using three fingers, where precision and stability are required for control and fine movements.",
            "A writing tool or delicate item that is balanced using a three-finger grip for precise handling."
        ],
        "pinch_grasp": [
            "An object that requires a pinch grasp, like a coin or needle, held firmly between the thumb and a single finger.",
            "A tiny object grasped delicately between the tips of two fingers, requiring fine control and precise positioning.",
            "A lightweight item that is picked up using only the fingertips, ideal for handling small, detailed objects."
        ],
        "open_grasp": [
            "An object that requires an open grasp, like a wide box or a folder, where the fingers spread out for stability.",
            "A flat or large object held with an open palm and extended fingers to maintain grip and control.",
            "A grasp technique where the hand remains open while supporting a broad or irregularly shaped item."
        ],
        "lateral_grasp": [
            "An object that requires a lateral grasp, like a key or a credit card, held securely between the thumb and the side of the index finger.",
            "A flat object grasped using lateral pressure between two fingers, often for precision handling.",
            "A grasp where the thumb and index finger work together to grip an object from the side, commonly used for small, flat items."
        ]
    }
prompts_2 = {
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

def get_support_labels_vectors():
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
            # augmented_feats = []
            # for aug_img in augment(img):  # e.g., list of augmented versions
            #     inputs = clip_processor(images=aug_img, return_tensors='pt').to(device)
            #     with torch.no_grad():
            #         aug_feat = clip_model.get_image_features(**inputs)
            #     augmented_feats.append(aug_feat.cpu().numpy())
            # img_features.extend(augmented_feats)

        support_features[label] = np.mean(img_features, axis=0)

    support_labels = list(support_features.keys())
    support_vectors = np.array([support_features[label] for label in support_labels]).squeeze(1)
    return support_labels, support_vectors

def get_image_predictions(image_path, top_k):
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
    return predicted_classes

support_labels, support_vectors = get_support_labels_vectors()
# Build KNN
knn = NearestNeighbors(n_neighbors=7, metric="cosine")
knn.fit(support_vectors)

def evaluate():
    # Inference
    correct = 0
    top_k = 1
    class_correct = Counter()
    class_total = Counter()
    for fname, true_labels in ground_truth.items():
        image_path = os.path.join(query_dir, fname)
        predicted_classes = get_image_predictions(image_path, top_k)

        # Count total instances for each ground truth class
        for grasp_type in true_labels:
            class_total[grasp_type] += 1  # Tracks how many times this grasp type appears in ground truth

        # Check which predictions match the ground truth
        if any(pred in true_labels for pred in predicted_classes):
            correct += 1
            for pred in predicted_classes:
                if pred in true_labels:  # If prediction is correct for any grasp type
                    class_correct[pred] += 1  # Increment correct count for that grasp type

            print(f"[✔] {fname} | Predicted: {predicted_classes} | GT: {true_labels}")
        else:
            print(f"[✘] {fname} | Predicted: {predicted_classes} | GT: {true_labels}")

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
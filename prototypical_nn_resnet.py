import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
from matching_networks_clip import save_loss_plot

# Перетворення для зображень
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

device = "cuda" if torch.cuda.is_available() else "cpu"
# Завантаження зображень (припустимо, що в тебе є dataset у папках за класами)
dataset_path = "/home/kpi_anna/data/test_grasp_dataset/set_1"
support_dataset = ImageFolder(root=dataset_path, transform=transform)
query_set_directory = '/home/kpi_anna/data/test_grasp_dataset/query_set'
query_json_path = "grasp_scripts/query_labels.json"

# Вивід класів і їх індексів
class_to_idx = support_dataset.class_to_idx
# print("Class to Index Mapping:", class_to_idx)

# Розділення на support set (один приклад на клас) і query set (інші зображення)
def load_support_set(dataset):
    support_set = []
    class_dict = {}
    for img, label in dataset:
        class_dict.setdefault(label, []).append(img)
    for label, images in class_dict.items():
        support_set.extend([(img, label) for img in images])
    return support_set

def load_query_set_from_json(query_json_path, transform, class_to_idx):
    with open(query_json_path, 'r') as f:
        query_labels = json.load(f)
    query_set = []
    for filename, labels in query_labels.items():
        image_path = os.path.join(query_set_directory, filename)
        if not os.path.exists(image_path):
            print(f"⚠️ Image {filename} not found, skipping.")
            continue
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image)
        # main_label = labels[0].replace(" ", "_")
        # if main_label not in class_to_idx:
        #     print(f"⚠️ Label {main_label} not in support set mapping.")
        #     continue
        label_indexes = [class_to_idx[label.replace(' ', '_')] for label in labels]
        # label_idx = class_to_idx[main_label]
        query_set.append((filename, image_tensor, label_indexes))
    return query_set

support_set = load_support_set(support_dataset)
query_set = load_query_set_from_json(query_json_path, transform, class_to_idx)

import torchvision.models as models
# Завантажуємо предтреновану модель (без останнього шару)
resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Видаляємо останній шар
resnet.eval()

# Функція для витягування фіч
def extract_features(model, images):
    with torch.no_grad():
        if images.dim() == 3:
            images = images.unsqueeze(0)
        features = model(images)
        # features = model(images.unsqueeze(0))  # Додаємо batch dimension
        return features.flatten()

# Витягуємо ознаки для support set
support_features = {label: extract_features(resnet, img) for img, label in support_set}

# Витягуємо ознаки для query set
# query_features = [(extract_features(resnet, img), label) for img, label in query_set]

import torch.nn.functional as F

# Обчислення прототипів класів (усереднення фіч support set)
def compute_prototypes(support_features):
    prototypes = {}
    for label, features in support_features.items():
        prototypes[label] = features.mean(dim=0)  # Усереднюємо
    return prototypes

prototypes = compute_prototypes(support_features)

# Функція для класифікації нового зображення
def classify_image_top_1(query_feature, prototypes):
    min_distance = float("inf")

    best_label = None

    for label, proto in prototypes.items():
        distance = F.pairwise_distance(query_feature.unsqueeze(0), proto.unsqueeze(0))
        if distance < min_distance:
            min_distance = distance
            best_label = label
    
    return best_label

def classify_image(query_feature, prototypes):
    distances = []
    for label, proto in prototypes.items():
        distance = F.pairwise_distance(query_feature.unsqueeze(0), proto.unsqueeze(0))
        distances.append((label, distance.item()))  # Convert distance to float
    # Sort by distance (ascending)
    distances.sort(key=lambda x: x[1])
    # Return top 3 labels
    top_3_labels = [label for label, dist in distances]

    return top_3_labels

def get_image_predictions_pnn_resnet(image_path, top_k=3):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)
    image_feature = extract_features(resnet, image_tensor)
    image_feature = F.normalize(image_feature, dim=-1)
    predictions = classify_image(image_feature, prototypes)[:top_k]
    return predictions

def evaluate():
    # Тест на query set
    top_k=1
    correct = 0
    losses = []
    total = len(query_set)
    for filename, query_image, true_labels in query_set:
        query_image = query_image
        query_feature = extract_features(resnet, query_image)
        query_feature = F.normalize(query_feature, dim=-1)
        topk_indices = classify_image(query_feature, prototypes)[:top_k]
        match_found = any(pred in true_labels for pred in topk_indices)
        
        if match_found:
            correct += 1
            print(f"Prediction for {filename} - {topk_indices} - correct - when correct are {true_labels}")
        else:
            print(f"Prediction for {filename} - {topk_indices} - INCORRECT - when correct are {true_labels}")
        
        all_labels = list(prototypes.keys())
        distances = []
        for label in all_labels:
            proto = prototypes[label]
            distance = F.pairwise_distance(query_feature.unsqueeze(0), proto.unsqueeze(0))
            distances.append(distance.item())
        # Convert distances to similarity logits (negative distance = higher similarity)
        logits = torch.tensor([-d for d in distances]).unsqueeze(0)

        # Create multi-hot ground truth vector
        target = torch.zeros(len(all_labels)).unsqueeze(0)
        for label in true_labels:
            target[0][label] = 1.0

        loss = F.binary_cross_entropy_with_logits(logits, target)
        losses.append(loss.item())
    # Final results
    accuracy = correct / total * 100
    avg_loss = sum(losses) / len(losses)
    save_loss_plot(losses, "Prototypical NN with ResNet")

    print(f"\n✅ Evaluation Results:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average Loss: {avg_loss:.4f}")
if __name__ == "__main__":
    evaluate()
    # print(class_to_idx)
    # preds = get_image_predictions_pnn_resnet("/home/kpi_anna/data/test_grasp_dataset/query_set/image2.png")
    # print(preds)
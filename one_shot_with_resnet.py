import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np

# Перетворення для зображень
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Завантаження зображень (припустимо, що в тебе є dataset у папках за класами)
dataset_path = "/home/kpi_anna/data/test_grasp_dataset/set_1"
dataset = ImageFolder(root=dataset_path, transform=transform)

# Вивід класів і їх індексів
class_to_idx = dataset.class_to_idx
print("Class to Index Mapping:", class_to_idx)

# Розділення на support set (один приклад на клас) і query set (інші зображення)
def split_dataset(dataset, num_shots=4):
    support_set = []
    query_set = []
    class_dict = {}

    for img, label in dataset:
        if label not in class_dict:
            class_dict[label] = []
        class_dict[label].append(img)
    
    for label, images in class_dict.items():
        support_set.extend([(img, label) for img in images[:num_shots]])  # Один приклад на клас
        query_set.extend([(img, label) for img in images[num_shots:]])  # Решта йде в query set
    
    return support_set, query_set

support_set, query_set = split_dataset(dataset)

import torchvision.models as models
# Завантажуємо предтреновану модель (без останнього шару)
resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Видаляємо останній шар
resnet.eval()

# Функція для витягування фіч
def extract_features(model, images):
    with torch.no_grad():
        features = model(images.unsqueeze(0))  # Додаємо batch dimension
        return features.flatten()

# Витягуємо ознаки для support set
support_features = {label: extract_features(resnet, img) for img, label in support_set}

# Витягуємо ознаки для query set
query_features = [(extract_features(resnet, img), label) for img, label in query_set]

import torch.nn.functional as F

# Обчислення прототипів класів (усереднення фіч support set)
def compute_prototypes(support_features):
    prototypes = {}
    for label, features in support_features.items():
        prototypes[label] = features.mean(dim=0)  # Усереднюємо
    return prototypes

prototypes = compute_prototypes(support_features)

# Функція для класифікації нового зображення
def classify_image(query_feature, prototypes):
    min_distance = float("inf")

    best_label = None

    for label, proto in prototypes.items():
        distance = F.pairwise_distance(query_feature.unsqueeze(0), proto.unsqueeze(0))
        if distance < min_distance:
            min_distance = distance
            best_label = label
    
    return best_label

# Тест на query set
correct = 0
for query_feature, label in query_features:
    predicted_label = classify_image(query_feature, prototypes)
    print(f"predicted {predicted_label}, when correct is {label}")
    if predicted_label == label:
        correct += 1

accuracy = correct / len(query_features)
print(f"One-Shot Learning Accuracy: {accuracy * 100:.2f}%")

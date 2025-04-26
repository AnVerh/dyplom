import os
import json
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Завантажуємо CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. Вказуємо шляхи до даних
support_dir = "/home/kpi_anna/data/test_grasp_dataset/set_1"  # Папки по класах
query_dir = '/home/kpi_anna/data/test_grasp_dataset/query_set'  # Всі зображення в одній папці
query_labels_path = '/home/kpi_anna/grasp_scripts/query_labels.json'  # JSON з ground truth

# 3. Функції завантаження даних
def load_support_set(folder_path):
    features, labels = [], []
    for class_name in os.listdir(folder_path):
        class_folder = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_folder):
            continue
        for img_file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_file)
            image = Image.open(img_path).convert("RGB")
            inputs = clip_processor(images=image, return_tensors='pt').to(device)
            with torch.no_grad():
                img_features = clip_model.get_image_features(**inputs)
            features.append(img_features.cpu().numpy().squeeze(0))  # squeeze бо (1, 512)
            labels.append([class_name])  # Важливо! Список списків
    return np.array(features), labels

def load_query_set(query_folder, labels_json_path):
    with open(labels_json_path, 'r') as f:
        label_dict = json.load(f)  # { "img1.jpg": ["power_grasp", "precision_grasp"], ... }

    features, labels = [], []
    for img_file in os.listdir(query_folder):
        img_path = os.path.join(query_folder, img_file)
        if not img_file.lower().endswith(('png', 'jpg', 'jpeg')):
            continue  # пропустити не-зображення
        image = Image.open(img_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors='pt').to(device)
        with torch.no_grad():
            img_features = clip_model.get_image_features(**inputs)
        features.append(img_features.cpu().numpy().squeeze(0))  # (512,)
        # Важливо: отримуємо список лейблів для цього зображення
        labels.append(label_dict.get(img_file, []))  # Якщо немає лейблу - пустий список
    return np.array(features), labels

# 4. Завантажуємо дані
X_support, y_support = load_support_set(support_dir)
X_query, y_query = load_query_set(query_dir, query_labels_path)

# 5. Кодуємо лейбли для мультилейбл задачі
mlb = MultiLabelBinarizer()
y_support_encoded = mlb.fit_transform(y_support)
y_query_encoded = mlb.transform(y_query)

# 6. Навчання класифікатора
clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
clf.fit(X_support, y_support_encoded)

# 7. Передбачення
y_pred = clf.predict(X_query)

# 8. Оцінка
print("🎯 Accuracy (subset accuracy):", accuracy_score(y_query_encoded, y_pred))  # дуже строге мірило
print("📋 Classification Report:")
print(classification_report(y_query_encoded, y_pred, target_names=mlb.classes_, zero_division=0))

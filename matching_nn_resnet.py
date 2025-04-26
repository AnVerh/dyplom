import json
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.datasets import ImageFolder

# ---------- Matching Network ----------
class MatchingNetwork(nn.Module):
    def __init__(self, num_classes):
        super(MatchingNetwork, self).__init__()
        self.num_classes = num_classes

    def forward(self, support_embeddings, support_labels, query_embedding):
        similarities = F.cosine_similarity(query_embedding, support_embeddings)
        attention_weights = F.softmax(similarities, dim=0)

        support_labels_one_hot = torch.zeros(len(support_labels), self.num_classes).to(device)
        support_labels_one_hot.scatter_(1, support_labels.unsqueeze(1), 1)

        predictions = torch.matmul(attention_weights, support_labels_one_hot)
        return predictions

# ---------- Feature Extractor ----------
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

# ---------- Image Processing ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_image_tensor(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image)

# ---------- Custom Dataset for Queries ----------
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

# ---------- Paths ----------
device = "cuda" if torch.cuda.is_available() else "cpu"

support_dir = "/home/kpi_anna/data/test_grasp_dataset/set_1/"
query_json_path = "/home/kpi_anna/grasp_scripts/query_labels.json"
query_set_directory = '/home/kpi_anna/data/test_grasp_dataset/query_set'
support_dataset = ImageFolder(root=support_dir, transform=transform)
class_to_idx = support_dataset.class_to_idx
# ---------- Load Support Set ----------
support_images = []
support_labels = []

def load_support_set(dataset):
    support_set = []
    class_dict = {}
    for img, label in dataset:
        class_dict.setdefault(label, []).append(img)
    for label, images in class_dict.items():
        support_set.extend([(img, label) for img in images])
    return support_set

support_set = load_support_set(support_dataset)
support_images = torch.stack([img for img, _ in support_set]).to(device)
support_labels = torch.tensor([label for _, label in support_set]).to(device)
print("Support set labels:", support_labels)
print("Unique labels in support:", torch.unique(torch.tensor(support_labels)))
# support_embeddings = encode_images(support_images)
# ---------- Load Query Set ----------
query_set = load_query_set_from_json(query_json_path, transform, class_to_idx)


# ---------- Device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
support_images = support_images.to(device)
support_labels = support_labels.to(device)

# ---------- Initialize Models ----------
num_classes = len(class_to_idx)
feature_extractor = FeatureExtractor().to(device)
matching_network = MatchingNetwork(num_classes=num_classes).to(device)

feature_extractor.eval()
matching_network.eval()

# ---------- Precompute Support Features ----------
with torch.no_grad():
    support_features = feature_extractor(support_images)
    support_features = F.normalize(support_features, dim=-1)

# ---------- Evaluation ----------
# Accuracy counter
correct = 0
total = len(query_set)
losses = []

# Evaluation loop
for filename, query_image, true_labels in query_set:
    query_image = query_image.unsqueeze(0).to(device)
    query_feature = feature_extractor(query_image)
    query_feature = F.normalize(query_feature, dim=-1)
    predictions = matching_network(support_features, support_labels, query_feature)
    print(f"Prediction for {filename}:")
    print(predictions)
    predicted_class = torch.argmax(predictions).item()
    if predicted_class in true_labels:
        correct += 1
        print(f"Prediction for {filename} - {predicted_class} - correct - when correct are {true_labels}")
    else:
        print(f"Prediction for {filename} - {predicted_class} - INCORRECT - when correct are {true_labels}")

    # Convert list of true labels into multi-hot vector
    target = torch.zeros(predictions.shape[-1], dtype=torch.float).to(device)
    target[true_labels] = 1.0

    # BCE loss expects raw logits, so don't apply softmax/sigmoid beforehand
    loss = F.binary_cross_entropy_with_logits(predictions, target)
    losses.append(loss.item())


# Final results
accuracy = correct / total * 100
avg_loss = sum(losses) / len(losses)

print(f"\n✅ Evaluation Results:")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Average Loss: {avg_loss:.4f}")

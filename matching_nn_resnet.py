from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os


class MatchingNetwork(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, query_features, support_features, support_labels):
        # Compute cosine similarity
        similarities = F.cosine_similarity(query_features.unsqueeze(1), support_features, dim=-1)  # (N_query, N_support)

        # Softmax over support set
        attention = F.softmax(similarities, dim=-1)  # (N_query, N_support)

        # Compute weighted sum of support labels
        unique_labels = torch.unique(support_labels)
        label_probs = torch.zeros(len(query_features), len(unique_labels)).to(query_features.device)

        for i, label in enumerate(unique_labels):
            mask = (support_labels == label).float()  # Mask for label
            label_probs[:, i] = torch.sum(attention * mask, dim=-1)  # Sum attention scores

        return unique_labels[torch.argmax(label_probs, dim=1)]  # Return predicted labels


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)  # Use ResNet-18 as the backbone
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer

    def forward(self, x):
        x = self.encoder(x)  # Extract features
        return x.view(x.size(0), -1)  # Flatten to (batch_size, feature_dim)


# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

def load_image(image_path):
    """Load an image from a file and preprocess it."""
    image = Image.open(image_path).convert("RGB")  # Ensure 3 channels
    image = transform(image)  # Apply transformations
    return image


# Example file paths
query_dir = "/home/kpi_anna/data/test_grasp_dataset/query_set/"
support_dir = "/home/kpi_anna/data/test_grasp_dataset/set_1/"

# Load support images
support_images = []
support_labels = []

for label in os.listdir(support_dir):  # Label folders: "0/", "1/", "2/"
    label_path = os.path.join(support_dir, label)
    
    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        img_tensor = load_image(img_path)
        support_images.append(img_tensor)
        support_labels.append(label)  # Convert folder name to int label

# Convert lists to tensors
support_images = torch.stack(support_images)  # Shape: (N_support, 3, 224, 224)
label_to_idx = {label: idx for idx, label in enumerate(set(support_labels))}
support_labels_numeric = [label_to_idx[label] for label in support_labels]

support_labels = torch.tensor(support_labels_numeric, dtype=torch.long)  # Shape: (N_support,)

# Load a query image
# query_images = load_image(query_image_path).unsqueeze(0)  # Add batch dimension
query_images = os.listdir(query_dir)  
query_images = [transform(Image.open(os.path.join(query_dir, img_path)).convert("RGB")) for img_path in query_images]
query_images = torch.stack(query_images)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
support_images, support_labels, query_images = support_images.to(device), support_labels.to(device), query_images.to(device)

# Initialize models
feature_extractor = FeatureExtractor().to(device)
matching_net = MatchingNetwork(feature_dim=512).to(device)
matching_net.eval()  # Set to evaluation mode

# Extract features
query_features = feature_extractor(query_images)  # (1, feature_dim)
support_features = feature_extractor(support_images)  # (N_support, feature_dim)

#normalizing feature - is it necessary?
query_features = F.normalize(query_features, dim=-1)
support_features = F.normalize(support_features, dim=-1)

# Classify the query image
predicted_label = matching_net(query_features, support_features, support_labels)

correct_labels = [0, 1, 1, 2, 2, 3]
correct_predictions = 0
# print(f"Predicted Label: {predicted_label.item()}")
for i, label in enumerate(predicted_label):
    print(f"Query Image {i}: Predicted Label: {label.item()+1}")
    if int(label.item()) == correct_labels[i]:
        correct_predictions+=1

accuracy = correct_predictions/len(correct_labels) * 100
print(f'Predictions accuracy: {accuracy}')

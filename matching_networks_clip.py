import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import Image
import os, json

# Load CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Image encoding
def encode_images(images):
    inputs = processor(images=images, return_tensors="pt", padding=True, do_rescale=False).to(device)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    return embeddings / embeddings.norm(dim=-1, keepdim=True)

# Matching Network
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

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset_path = "/home/kpi_anna/data/test_grasp_dataset/set_1"
query_set_directory = '/home/kpi_anna/data/test_grasp_dataset/query_set'
query_json_path = "grasp_scripts/query_labels.json"

support_dataset = ImageFolder(root=dataset_path, transform=transform)
class_to_idx = support_dataset.class_to_idx
num_classes = len(class_to_idx)

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

# Load support and query sets
support_set = load_support_set(support_dataset)
query_set = load_query_set_from_json(query_json_path, transform, class_to_idx)

# Prepare support embeddings
support_images = torch.stack([img for img, _ in support_set]).to(device)
support_labels = torch.tensor([label for _, label in support_set]).to(device)
print("Support set labels:", support_labels)
print("Unique labels in support:", torch.unique(torch.tensor(support_labels)))
support_embeddings = encode_images(support_images)

# Initialize model
matching_network = MatchingNetwork(num_classes=num_classes).to(device)
# Accuracy counter
correct = 0
total = len(query_set)
losses = []

# Evaluation loop
for filename, query_image, true_labels in query_set:
    query_image = query_image.unsqueeze(0).to(device)
    query_embedding = encode_images([query_image.squeeze(0)])
    predictions = matching_network(support_embeddings, support_labels, query_embedding)
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

    # Cross-entropy loss if there's only 1 true label
    # target = torch.tensor([true_label], dtype=torch.long).to(device)
    # loss = F.cross_entropy(predictions.unsqueeze(0), target)
    # losses.append(loss.item())

# Final results
accuracy = correct / total * 100
avg_loss = sum(losses) / len(losses)

print(f"\n✅ Evaluation Results:")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Average Loss: {avg_loss:.4f}")

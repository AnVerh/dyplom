import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# Load CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def encode_images(images):
    # preprocessed_images = [transform(img) for img in images] 
    inputs = processor(images=images, return_tensors="pt", padding=True, do_rescale=False).to(device)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    return embeddings / embeddings.norm(dim=-1, keepdim=True)  # Normalize embeddings

class MatchingNetwork(nn.Module):
    def __init__(self):
        super(MatchingNetwork, self).__init__()
        
    def forward(self, support_embeddings, support_labels, query_embedding):
        """
        support_embeddings: Tensor of shape (num_support, embedding_dim)
        support_labels: Tensor of shape (num_support, num_classes)
        query_embedding: Tensor of shape (embedding_dim)
        """
        # Compute cosine similarities between query and support embeddings
        similarities = F.cosine_similarity(query_embedding.unsqueeze(0), support_embeddings, dim=-1)
        
        # Compute attention weights
        attention_weights = F.softmax(similarities, dim=0)  # Shape: (num_support,)

        support_labels = support_labels.to(torch.int64)
        num_classes = 4  # Replace with the actual number of classes
        support_labels_one_hot = torch.zeros(len(support_labels), num_classes).to(device)
        support_labels_one_hot.scatter_(1, support_labels.unsqueeze(1), 1)

        # Predict class probabilities as weighted sum of support labels
        predictions = torch.matmul(attention_weights, support_labels_one_hot)  # Shape: (num_classes,)
        
        return predictions

# Example support data (images and labels)
support_images = [...]  # List of PIL images
support_labels = torch.tensor([
    [1, 0, 0],  # Class 0
    [0, 1, 0],  # Class 1
    [0, 0, 1]   # Class 2
])  # One-hot encoded labels

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure compatibility with CLIP
    transforms.ToTensor()
])

dataset_path = "/home/kpi_anna/data/test_grasp_dataset/set_1"
dataset = ImageFolder(root=dataset_path, transform=transform)

# Print class mapping
class_to_idx = dataset.class_to_idx
print("Class to Index Mapping:", class_to_idx)

# Split dataset into support & query sets
def split_dataset(dataset, num_shots=4):
    support_set, query_set = [], []
    class_dict = {}

    for img, label in dataset:
        if label not in class_dict:
            class_dict[label] = []
        class_dict[label].append(img)

    for label, images in class_dict.items():
        support_set.extend([(img, label) for img in images[:num_shots]])
        query_set.extend([(img, label) for img in images[num_shots:]])
    
    return support_set, query_set

support_set, query_set = split_dataset(dataset)

support_images = torch.stack([img for img, _ in support_set]).to(device)  # Tensor of shape (num_support, channels, height, width)
support_labels = torch.tensor([label for _, label in support_set]).to(device)  # Tensor of shape (num_support,)

# Example query image
query_image = query_set[0][0].to(device)  # Single PIL image
support_embeddings = encode_images(support_images)
query_embedding = encode_images([query_image])#.squeeze(0)  # Remove batch dimension
# Remove any detach() or no_grad() contexts where support_embeddings are created
support_embeddings.requires_grad = True  # Ensure gradients are tracked
query_embedding.requires_grad = True

# Example Matching Network usage
matching_network = MatchingNetwork().to(device)

# Forward pass
predictions = matching_network(support_embeddings, support_labels.float(), query_embedding).to(device)

# Compute loss
query_label = torch.tensor([0, 1, 0, 0]).to(device)  # Ground truth one-hot label for query
# Convert one-hot encoded query label to a class index
query_label_index = torch.argmax(query_label).to(device)
# Debug information
print(f"Query label index: {query_label_index.item()}, Predictions shape: {predictions.shape}")

# Check range
assert 0 <= query_label_index.item() < predictions.size(1), "query_label_index is out of range!"

print(f"query label index should be between 0 and 3, got: {query_label_index}")
# Compute cross-entropy loss
loss = F.cross_entropy(predictions.unsqueeze(0), query_label_index)
# loss = F.cross_entropy(predictions.unsqueeze(0), query_label.unsqueeze(0).float())
loss.backward()

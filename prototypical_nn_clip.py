import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F
from ultralytics import YOLO
import torchvision.models.segmentation as segmentation
import json, os
from matching_networks_clip import save_loss_plot

# Prototypical NN + CLIP, with object detection and segmentation
# Завантаження моделей
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
od_model = YOLO("yolov8m.pt")  # Object Detection (YOLO)
seg_model = segmentation.deeplabv3_resnet50(pretrained=True).eval().to(device)  # Segmentation Model

# Preprocessing function
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure compatibility with CLIP
    transforms.ToTensor()
])

# Image encoding
def encode_images(images):
    inputs = clip_processor(images=images, return_tensors="pt", padding=True, do_rescale=False).to(device)
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
    return embeddings / embeddings.norm(dim=-1, keepdim=True)

# Convert tensor to PIL image
def tensor_to_pil(image_tensor):
    image_np = image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(image_np)

# Detect objects with YOLO and crop the main object
def detect_and_crop(image):
    results = od_model(image)

    for r in results:
        for box in r.boxes.xyxy:  # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box)
            cropped = image.crop((x1, y1, x2, y2))  # Crop object
            return cropped  # Return cropped object
    
    print("No objects detected.")
    return None  # No object detected

# Segment the object in a cropped image
def segment_image(image):
    transform_seg = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])

    image_tensor = transform_seg(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = seg_model(image_tensor)['out'][0]  # Get segmentation mask
    mask = output.argmax(0).cpu().numpy()

    # Convert mask to binary (object=1, background=0)
    binary_mask = (mask > 0.1).astype(np.uint8)

    # Apply mask to image
    image_np = image_tensor.cpu().numpy().squeeze(0).transpose(1, 2, 0)
    segmented_image = (image_np * binary_mask[:, :, None]).squeeze()#.to(device)  # Keep object, remove background
    segmented_image = (segmented_image * 255).astype(np.uint8)  # Convert float32 to uint8

    print("Segmented Image Shape:", segmented_image.shape)
    segmented_image = Image.fromarray(segmented_image)

    return segmented_image

# Extract CLIP features from segmented images
def extract_features_with_clip(model, processor, image_tensor):
    image = tensor_to_pil(image_tensor)
    # image = detect_and_crop(image_orig)  # Apply Object Detection

    # if image is None:
        # print("Images skipped because no objects were detected!")
        # return None  # Skip images where no object is detected
        # image=image_orig

    # image = segment_image(image)  # Apply Segmentation
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    
    return features.cpu().squeeze(0)  # Remove batch dimension and move to CPU

# Load Dataset and Apply Detection + Segmentation
dataset_path = "/home/kpi_anna/data/test_grasp_dataset/set_1"
support_dataset = ImageFolder(root=dataset_path, transform=transform)
query_set_directory = '/home/kpi_anna/data/test_grasp_dataset/query_set'
query_json_path = "grasp_scripts/query_labels.json"

# Print class mapping
class_to_idx = support_dataset.class_to_idx
# print("Class to Index Mapping:", class_to_idx)

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
        label_indexes = [class_to_idx[label.replace(' ', '_')] for label in labels]
        query_set.append((filename, image_tensor, label_indexes))
    return query_set

support_set = load_support_set(support_dataset)
query_set = load_query_set_from_json(query_json_path, transform, class_to_idx)

# Prepare support embeddings
support_images = torch.stack([img for img, _ in support_set]).to(device)
support_labels = torch.tensor([label for _, label in support_set]).to(device)
# print("Support set labels:", support_labels)
# print("Unique labels in support:", torch.unique(torch.tensor(support_labels)))
support_embeddings = encode_images(support_images)

# Compute Prototypes
from collections import defaultdict

def compute_prototypes(support_set):
    feature_dict = defaultdict(list)
    for img, label in support_set:
        features = extract_features_with_clip(clip_model, clip_processor, img)
        if features is not None:
            feature_dict[label].append(features)

    prototypes = {}
    for label, feats in feature_dict.items():
        stacked_feats = torch.stack(feats)
        prototypes[label] = stacked_feats.mean(dim=0)

    return prototypes


prototypes = compute_prototypes(support_set)

# Classification using Nearest Prototype
def classify_image(query_feature, prototypes):
    min_distance = float("inf")
    best_label = None

    for label, proto in prototypes.items():
        distance = F.pairwise_distance(query_feature.unsqueeze(0).to(device), proto.unsqueeze(0).to(device))
        if distance < min_distance:
            min_distance = distance
            best_label = label
    
    return best_label  # Always return the closest class

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

def get_image_predictions_pnn_clip(image_path, top_k=3):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)
    image_feature = extract_features_with_clip(clip_model, clip_processor, image_tensor)
    image_feature = F.normalize(image_feature, dim=-1)
    predictions = classify_image(image_feature, prototypes)[:top_k]
    return predictions

def evaluate():
    # Test on Query Set
    top_k=1
    correct = 0
    total = len(query_set)
    losses = []
    for filename, image_tensor, true_labels in query_set:
        query_feature = extract_features_with_clip(clip_model, clip_processor, image_tensor)
        top3_indices = classify_image(query_feature, prototypes)[:top_k]
        match_found = any(pred in true_labels for pred in top3_indices)
        
        if match_found:
            correct += 1
            print(f"Prediction for {filename} - {top3_indices} - correct - when correct are {true_labels}")
        else:
            print(f"Prediction for {filename} - {top3_indices} - INCORRECT - when correct are {true_labels}")

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

    accuracy = correct / total * 100
    avg_loss = sum(losses) / len(losses)
    save_loss_plot(losses, "Prototypical NN with CLIP")

    print(f"\n✅ Evaluation Results:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    evaluate()
    # print(class_to_idx)
    # preds = get_image_predictions_pnn_clip("/home/kpi_anna/data/test_grasp_dataset/query_set/image2.png")
    # print(preds)
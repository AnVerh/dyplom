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

# Convert tensor to PIL image
def tensor_to_pil(image_tensor):
    image_np = image_tensor.numpy().transpose(1, 2, 0)
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

# Extract Features for Support and Query Sets
support_features = {
    label: extract_features_with_clip(clip_model, clip_processor, img)
    for img, label in support_set
}
query_features = [
    (extract_features_with_clip(clip_model, clip_processor, img), label)
    for img, label in query_set
]

# Compute Prototypes
def compute_prototypes(support_features):
    prototypes = {}
    for label, features in support_features.items():
        if features is not None:
            prototypes[label] = features.mean(dim=0)  # Average feature vectors
    return prototypes

prototypes = compute_prototypes(support_features)

# Classification using Nearest Prototype
def classify_image(query_feature, prototypes):
    min_distance = float("inf")
    best_label = None

    for label, proto in prototypes.items():
        if query_feature is None or proto is None:
            if query_feature is None:
                print("Query feature is None!")
            if proto is None:
                print("Prototype is None!")
            continue
        distance = F.pairwise_distance(query_feature.unsqueeze(0), proto.unsqueeze(0))
        # cos_sim = F.cosine_similarity(query_feature, proto, dim=0)

        if distance < min_distance:
            min_distance = distance
            best_label = label if label else "Unknown, try 0"
    
    return best_label

# Test on Query Set
correct = 0
for query_feature, label in query_features:
    predicted_label = classify_image(query_feature, prototypes)
    print(f"Predicted {predicted_label}, when correct is {label}")
    if predicted_label == label:
        correct += 1

# Print Accuracy
accuracy = correct / len(query_features)
print(f"Accuracy: {accuracy * 100:.2f}%")

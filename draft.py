import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.models.detection import maskrcnn_resnet50_fpn
import ultralytics
from ultralytics import YOLO

# Load DeepLabV3 model
# model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
# model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set model to evaluation mode

od_model = YOLO("yolov8s.pt")

def segment_image_deeplab(image_path):
    # Load and preprocess image
    # image_path = "your_image.jpg"
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        # transforms.Resize((512, 512)),  # Resize to match model input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)["out"]
        mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    # Convert mask to binary (1 = object, 0 = background)
    binary_mask = (mask > 0).astype(np.uint8)  # Keep only the main object
    if np.sum(binary_mask) == 0:
        print("⚠️ Warning: Segmentation mask is empty! Image might be all black.")

    plt.imshow(binary_mask, cmap="gray")
    plt.title("Segmentation Mask")
    plt.show()


    # Convert PIL image to NumPy
    # image_np = np.array(image)
    image_np = input_tensor.squeeze(0).numpy()  # Shape: (3, 224, 224)

    # If you need to change the order from (C, H, W) → (H, W, C) for OpenCV/Matplotlib
    image_np = np.transpose(image_np, (1, 2, 0))
    image_np = (image_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    image_np = np.clip(image_np, 0, 1)
    # Apply mask to image (remove background)
    segmented_image = image_np * binary_mask[:, :, None]  # Keep object, remove background

    # Show the segmented image
    plt.imshow(segmented_image)
    plt.axis("off")
    plt.show()

    if np.max(segmented_image) == 0:
        print("⚠️ Warning: Segmented image is completely black!")

    # Optional: Save the segmented image
    segmented_image_pil = Image.fromarray((segmented_image * 255).astype(np.uint8))  # Перетворення в uint8

    segmented_image_pil.save("segmented_output1.png")

def detect_and_crop(image_path):
    image = Image.open(image_path).convert("RGB")
    results = od_model(image)

    for r in results:
        for box in r.boxes.xyxy:  # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box)
            cropped = image.crop((x1, y1, x2, y2))  # Crop object
            cropped.save('yolo_image1.jpg')
            print("cropped image saved")
            return cropped  # Return cropped object
        
    print("no objects detected")
    return None  # No object detected

# Приклад:
dataset_path = "/home/kpi_anna/data/test_grasp_dataset/set_1"
# segmented = segment_image_deeplab(f"{dataset_path}/palmar_grasp/image2.png")
cropped_img = detect_and_crop(f"{dataset_path}/hook_grasp/image2.png")
# cv2.imwrite("./segmented_image1.jpg", segmented)

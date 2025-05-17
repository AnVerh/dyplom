import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from torchvision import transforms
import os
import json
import numpy as np
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------- Projection Head ---------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, mid_dim=512, out_dim=512):
        super().__init__()
        # self.fc = nn.Linear(in_dim, out_dim)
        dropout_prob=0.1
        self.fc = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mid_dim, out_dim)
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(in_dim, out_dim),
        #     nn.ReLU(),
        #     nn.Linear(out_dim, out_dim)
        # )

    def forward(self, x):
        return self.fc(x)

# --------- Custom CLIP with Projection Head ---------
class CLIPWithProjectionHead(nn.Module):
    def __init__(self, base_model_name="openai/clip-vit-base-patch32", freeze_base=True, device='cuda'):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(base_model_name)
        self.image_proj = ProjectionHead(in_dim=512, out_dim=512)
        self.text_proj = ProjectionHead(in_dim=512, out_dim=512)
        self.device = device
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        if freeze_base:
            for param in self.clip.parameters():
                param.requires_grad = False
            for param in self.clip.vision_model.encoder.layers[-6:].parameters():
                param.requires_grad = True 

        # Learnable logit scale (temperature)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image=None, text=None, processor=None):
        image_proj = text_proj = None

        if image is not None:
            image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            image_outputs = self.clip.vision_model(pixel_values=image_inputs["pixel_values"])
            image_embeds = self.clip.visual_projection(image_outputs.pooler_output)
            image_proj = self.image_proj(image_embeds)

        if text is not None:
            text_inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
            text_outputs = self.clip.text_model(input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"])
            text_embeds = self.clip.text_projection(text_outputs.pooler_output)
            text_proj = self.text_proj(text_embeds)

        # image_proj = F.normalize(image_proj, dim=-1)
        # text_proj = F.normalize(text_proj, dim=-1)

        return image_proj, text_proj
    
    def get_image_features(self, images=None, pixel_values=None):
        """
        Extract image features and apply projection head.
        Accepts either raw images or preprocessed pixel_values.
        """
        if images is not None:
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"]

        elif pixel_values is None:
            raise ValueError("Must provide either `images` or `pixel_values`")

        # Ensure the tensor is on the same device as the model
        device = next(self.clip.parameters()).device
        pixel_values = pixel_values.to(device)

        outputs = self.clip.vision_model(pixel_values=pixel_values)
        image_embeds = self.clip.visual_projection(outputs.pooler_output)
        return self.image_proj(image_embeds)


    def get_text_features(self, text=None, input_ids=None, attention_mask=None):
        """
        Extract text features and apply projection head.
        Accepts either raw text or tokenized input_ids + attention_mask.
        """
        if text is not None:
            inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

        if input_ids is None or attention_mask is None:
            raise ValueError("Must provide either `text` or (`input_ids`, `attention_mask`)")

        outputs = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = self.clip.text_projection(outputs.pooler_output)
        return self.text_proj(text_embeds)


def main():
    # --------- Load model and processor ---------
    model = CLIPWithProjectionHead(freeze_base=True).to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # --------- Data loading ---------
    support_dir = "/home/kpi_anna/data/test_grasp_dataset/set_1"
    query_labels_path = "/home/kpi_anna/grasp_scripts/query_labels.json"
    query_set_directory = "/home/kpi_anna/data/test_grasp_dataset/query_set"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    prompts = {
        "cylindrical_grasp": [
            "An object that requires a cylindrical grasp, like a bottle or a handle.",
            "An object that is held cylindrically with all fingers wrapped around it.",
            "An object that is heavy and requires power to lift it"
        ],
        "hook_grasp": [
            "An object that requires a hook grasp, like a bag or a bucket.",
            "An object held with a hook-like shape using 4 fingers, without the thumb."
        ],
        "palmar_grasp": [
            "An object that requires a palmar grasp, can fit in a hand, like a notebook or smartphone.",
            "An object that is held in the palm of the hand."
        ],
        "spherical_grasp": [
            "An object that requires a spherical grasp, like an apple or a small ball.",
            "An object that is grasped with fingers and palm forming a sphere."
        ],
        "tripod_grasp": [
            "An object that requires a tripod grasp, like a pen or a pencil.",
            "An object held between three fingers."
        ],
        "pinch_grasp": [
            "An object that requires a pinch grasp, like a coin or needle.",
            "An object held with the thumb and one other finger.",
            "An object that is tiny, light, and delicate"
        ],
        "open_grasp": [
            "An object that requires an open grasp, like a flat box.",
            "An object that is grasped with an open hand."
        ],
        "lateral_grasp": [
            "An object that requires a lateral grasp, like a key or a credit card.",
            "An object that is held between the thumb and the side of the index finger"
        ]
    }

    support_images, support_texts = [], []

    for class_name in os.listdir(support_dir):
        class_path = os.path.join(support_dir, class_name)
        class_prompts = prompts.get(class_name, [f"An object that requires a {class_name.replace('_', ' ')}."])

        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = Image.open(image_path).convert("RGB")

            for prompt in class_prompts:
                support_images.append(image)
                support_texts.append(prompt)

    # --------- Training ---------
    epochs = 20
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    batch_size = 16
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        
        total_loss = 0

        for i in range(0, len(support_images), batch_size):
            batch_images = support_images[i:i+batch_size]
            batch_texts = support_texts[i:i+batch_size]

            image_proj, text_proj = model(image=batch_images, text=batch_texts, processor=processor)
            
            logits = image_proj @ text_proj.T
            targets = torch.arange(len(logits)).to(device)

            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    # --------- Evaluation ---------
    with torch.no_grad():
        model.eval()

    with open(query_labels_path, "r") as f:
        ground_truth = json.load(f)

    query_images = []
    query_filenames = []

    for img_name in os.listdir(query_set_directory):
        img_path = os.path.join(query_set_directory, img_name)
        img = Image.open(img_path).convert("RGB")
        query_images.append(img)
        query_filenames.append(img_name)

    unique_labels = sorted(os.listdir(support_dir))
    class_prompts = [f"An object that requires a {label.replace('_', ' ')}." for label in unique_labels]

    correct = 0
    top_k = 3
    total = len(query_images)

    _, text_proj = model(image=None, text=class_prompts, processor=processor)

    for idx, image in enumerate(query_images):
        
        # image_proj, _ = model(image=[image], text=[class_prompts[0]], processor=processor)
        image_proj, _ = model(image=[image], processor=processor)
        image_proj = image_proj[0]

        logits = (image_proj @ text_proj.T).squeeze()
        top_indices = torch.topk(logits, k=top_k).indices.cpu().numpy()
        predicted_classes = [unique_labels[i] for i in top_indices]

        true_labels = ground_truth[query_filenames[idx]]

        print(f"Prediction: {predicted_classes}, Ground Truth: {true_labels}")
        if any(pred.replace('_', ' ') in true_labels for pred in predicted_classes):
            print("✓ Correct")
            correct += 1
        else:
            print("✗ Incorrect")

    accuracy = correct / total * 100
    print(f"\nTop-{top_k} Accuracy with Projection Head: {accuracy:.2f}%")
    torch.save(model.state_dict(), "clip_with_projection_head.pth")

if __name__ == "__main__":
    main()

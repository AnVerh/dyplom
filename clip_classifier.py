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
import random
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from matching_networks_clip import save_loss_plot

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# --------- Projection Head ---------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, mid_dim=512, out_dim=512):
        super().__init__()
        dropout_prob = 0.1
        self.fc = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mid_dim, out_dim)
        )

    def forward(self, x):
        # return self.fc(x)
        return F.normalize(self.fc(x), p=2, dim=-1)  # Normalize to improve retrieval

# --------- Custom CLIP with Projection Head ---------
class CLIPWithProjectionHead(nn.Module):
    def __init__(self, base_model_name="openai/clip-vit-base-patch32", freeze_base=True, device='cuda'):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(base_model_name)
        self.image_proj = ProjectionHead(in_dim=512, out_dim=512)
        self.text_proj = ProjectionHead(in_dim=512, out_dim=512)
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(base_model_name)

        if freeze_base:

            for param in self.clip.parameters():
                param.requires_grad = False
            for param in self.clip.vision_model.encoder.layers[-4:].parameters():
                param.requires_grad = True
            for param in self.clip.text_model.encoder.layers[-6:].parameters():
                param.requires_grad = True

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image=None, text=None, processor=None):
        image_proj = text_proj = None

        if image is not None:
            image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            image_outputs = self.clip.vision_model(pixel_values=image_inputs["pixel_values"])
            image_embeds = self.clip.visual_projection(image_outputs.pooler_output)
            image_proj = self.image_proj(image_embeds)
            # image_proj = F.normalize(self.image_proj(image_embeds), p=2, dim=-1)  # Normalization here

        if text is not None:
            text_inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
            text_outputs = self.clip.text_model(input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"])
            text_embeds = self.clip.text_projection(text_outputs.pooler_output)
            text_proj = self.text_proj(text_embeds)
            # text_proj = F.normalize(self.text_proj(text_embeds), p=2, dim=-1)  # Normalization here

        return image_proj, text_proj

    def get_image_features(self, images=None, pixel_values=None):
        if images is not None:
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"]
        elif pixel_values is None:
            raise ValueError("Must provide either `images` or `pixel_values`")
        pixel_values = pixel_values.to(self.device)
        outputs = self.clip.vision_model(pixel_values=pixel_values)
        image_embeds = self.clip.visual_projection(outputs.pooler_output)
        return self.image_proj(image_embeds)

    def get_text_features(self, text=None, input_ids=None, attention_mask=None):
        if text is not None:
            inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
        if input_ids is None or attention_mask is None:
            raise ValueError("Must provide either `text` or (`input_ids`, `attention_mask`)")
        outputs = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = self.clip.text_projection(outputs.pooler_output)
        return self.text_proj(text_embeds)

def select_hard_negatives(true_class, image_proj, support_features, support_labels, num_negatives=2):
    true_feat = image_proj.mean(dim=0).detach().cpu().numpy()  # Aggregate batch embeddings

    similarities = {}
    for label in support_labels:
        if label != true_class:
            support_embedding = support_features[label].mean(dim=0).squeeze().cpu().detach().numpy()  # Ensure correct shape
            similarities[label] = cosine_similarity([true_feat], [support_embedding])[0][0]

    hard_negative_classes = sorted(similarities, key=similarities.get, reverse=True)[:num_negatives]
    
    hard_negative_tensors = torch.stack([support_features[label].mean(dim=0) for label in hard_negative_classes])
    hard_negative_tensors = hard_negative_tensors.mean(dim=0).squeeze(dim=0).to(device)  # Ensure proper dimensions

    return hard_negative_tensors 



def main():
    
    model = CLIPWithProjectionHead(freeze_base=True).to(device)
    processor = model.processor

    support_dir = "/home/kpi_anna/data/test_grasp_dataset/set_1"
    query_labels_path = "/home/kpi_anna/grasp_scripts/query_labels.json"
    query_set_directory = "/home/kpi_anna/data/test_grasp_dataset/query_set"

    prompts = {
        "cylindrical_grasp": [
            "An object that requires a cylindrical grasp, like a bottle or a handle, with fingers wrapping completely around it.",
            "An object that is held in a power grip, where all fingers enclose the object and the thumb provides support.",
            "A rounded object that is lifted by encircling it with the hand, ensuring a strong grip and stability."
        ],
        "hook_grasp": [
            "An object that requires a hook grasp, like a bag or a bucket, where four fingers curl around it while the thumb stays out.",
            "An object held using a hook-like hand shape, without using the thumb, enabling a secure grip while carrying weight.",
            "A handle-like object grasped with flexed fingers without direct palm contact, commonly used for lifting heavy loads."
        ],
        "palmar_grasp": [
            "An object that requires a palmar grasp, fitting completely against the palm with the fingers and thumb supporting it.",
            "An object that is flat and fits against the palm with support from the thumb, like holding a book or a tablet.",
            "A broad, flat object grasped with full palm contact, using finger pressure to maintain stability while carrying or holding."
        ],
        "spherical_grasp": [
            "An object that requires a spherical grasp, like an apple or a ball, held with fingers and palm forming a rounded shape.",
            "A curved object that is grasped by spreading the fingers around its surface and using the palm for support.",
            "A grasp used to hold round objects securely, using an arched hand shape and distributed finger pressure."
        ],
        "tripod_grasp": [
            "An object that requires a tripod grasp, like a pencil or a small tool, held between the thumb, index, and middle finger.",
            "A small object grasped using three fingers, where precision and stability are required for control and fine movements.",
            "A writing tool or delicate item that is balanced using a three-finger grip for precise handling."
        ],
        "pinch_grasp": [
            "An object that requires a pinch grasp, like a coin or needle, held firmly between the thumb and a single finger.",
            "A tiny object grasped delicately between the tips of two fingers, requiring fine control and precise positioning.",
            "A lightweight item that is picked up using only the fingertips, ideal for handling small, detailed objects."
        ],
        "open_grasp": [
            "An object that requires an open grasp, like a wide box or a folder, where the fingers spread out for stability.",
            "A flat or large object held with an open palm and extended fingers to maintain grip and control.",
            "A grasp technique where the hand remains open while supporting a broad or irregularly shaped item."
        ],
        "lateral_grasp": [
            "An object that requires a lateral grasp, like a key or a credit card, held securely between the thumb and the side of the index finger.",
            "A flat object grasped using lateral pressure between two fingers, often for precision handling.",
            "A grasp where the thumb and index finger work together to grip an object from the side, commonly used for small, flat items."
        ]
    }


    support_images, support_texts, support_labels = [], [], [] 
    support_features = {}

    for class_name in os.listdir(support_dir):
        class_path = os.path.join(support_dir, class_name)
        class_prompts = prompts.get(class_name, [f"An object that requires a {class_name.replace('_', ' ')}."])
        embeddings_list = []
        
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = Image.open(image_path).convert("RGB")

            for prompt in class_prompts:
                support_images.append(image)
                support_texts.append(prompt)
                support_labels.append(class_name)

                image_proj, _ = model(image=image, text=prompt, processor=processor)  # Extract feature embedding
                embeddings_list.append(image_proj.detach())

        support_features[class_name] = torch.stack(embeddings_list).mean(dim=0)  # Aggregate embeddings per class

    # Training
    epochs = 30
    batch_size = 16
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2)
    contrastive_loss = nn.TripletMarginLoss(margin=0.3)
    loss_weights = {
        "cylindrical_grasp": 1.3, "open_grasp": 1.3,  # Increase weighting
        "hook_grasp": 1.2, "lateral_grasp": 1.2, "pinch_grasp": 1.2,
        "palmar_grasp": 1.1, "spherical_grasp": 1.0, "tripod_grasp":1.0
        }   
    grasp_types = list(prompts.keys())
    loss_history = []

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        epoch_loss = 0.0
        for i in range(0, len(support_images), batch_size):
            batch_images = support_images[i:i+batch_size]
            batch_texts = support_texts[i:i+batch_size]
            batch_grasp_types = support_labels[i:i+batch_size]

            image_proj, text_proj = model(image=batch_images, text=batch_texts, processor=processor)

            logits = image_proj @ text_proj.T
            targets = torch.arange(len(logits)).to(device)

            # Compute base loss
            base_loss = loss_fn(logits, targets)

            # Hard Negatives Selection & Shape Fixes
            hard_neg_samples = torch.stack([select_hard_negatives(grasp, image_proj, support_features, support_labels, num_negatives=2) for grasp in batch_grasp_types]).to(device)
            # hard_neg_samples = hard_neg_samples.squeeze(dim=3)  # Remove extra dimensions
            hard_neg_samples = hard_neg_samples.view(hard_neg_samples.shape[0], hard_neg_samples.shape[-1])  # Ensure proper shape
            if hard_neg_samples.dim() > 2:  # Only squeeze if extra dims exist
                hard_neg_samples = hard_neg_samples.squeeze()

            contrastive_loss_weight = max(0.05, 0.2 * (1 - epoch / epochs))  # Reduce contrastive effect gradually
            contrastive_loss_value = contrastive_loss(image_proj, text_proj, hard_neg_samples)
            scaled_contrastive_loss = contrastive_loss_weight * contrastive_loss_value

            batch_loss_weight = torch.tensor([loss_weights[grasp] for grasp in batch_grasp_types]).to(device) 
            normalized_weights = batch_loss_weight / batch_loss_weight.sum()
            weighted_base_loss = (base_loss * batch_loss_weight).mean() # Scale loss by class weights

            # Weighted total loss
            total_loss = weighted_base_loss + scaled_contrastive_loss
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += total_loss.item()

            pred_probs = F.softmax(logits, dim=-1)

        scheduler.step()
        epoch_loss_avg = epoch_loss / batch_size
        loss_history.append(epoch_loss_avg)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

        if epoch % 10 == 0:  # Check every 10 epochs
            print("Logits Mean:", logits.mean().item())
            print("Logits Std Dev:", logits.std().item())
            print("Max Prediction Confidence:", pred_probs.max().item())

    save_loss_plot(loss_history, "CLIP fine-tuning")
    # Evaluation
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
    _, text_proj = model(image=None, text=class_prompts, processor=processor)

    correct = 0
    top_k = 3
    for idx, image in enumerate(query_images):
        image_proj, _ = model(image=[image], processor=processor)
        image_proj = image_proj[0]
        logits = (image_proj @ text_proj.T).squeeze()
        top_indices = torch.topk(logits, k=top_k).indices.cpu().numpy()
        predicted_classes = [unique_labels[i] for i in top_indices]
        true_labels = ground_truth[query_filenames[idx]]
        print(f"Prediction for {query_filenames[idx]}: {predicted_classes}, Ground Truth: {true_labels}")
        if any(pred.replace('_', ' ') in true_labels for pred in predicted_classes):
            print("\u2713 Correct")
            correct += 1
        else:
            print("\u2717 Incorrect")

    accuracy = correct / len(query_images) * 100
    print(f"\nTop-{top_k} Accuracy with Projection Head: {accuracy:.2f}%")
    print("Validation Logits Mean:", logits.mean().item())
    print("Validation Max Prediction Confidence:", pred_probs.max().item())
    torch.save(model.state_dict(), "clip_with_projection_head.pth")

if __name__ == "__main__":
    main()

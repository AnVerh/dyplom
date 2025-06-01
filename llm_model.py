import requests
from huggingface_hub import login
import base64
import json
import os
import time

OPENROUTER_API_KEY = "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"
image_path = "/home/kpi_anna/data/test_grasp_dataset/query_set/image10.png"
QUERY_DIR = "/home/kpi_anna/data/test_grasp_dataset/query_set/"
LABELS_PATH = "/home/kpi_anna/grasp_scripts/query_labels.json"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL = "google/gemma-3-27b-it:free"
# Load and encode image to base64
with open(image_path, "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")


GRASP_CLASSES = [
    "open grasp",
    "spherical grasp",
    "pinch grasp",
    "tripod grasp",
    "lateral grasp",
    "palmar grasp",
    "hook grasp",
    "cylindrical grasp"
]

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

QUESTION = ("What kind of grasp should be used to pick up the object in this image?"
            # "Only choose one of the following: cylindrical grasp, hook grasp, pinch grasp, tripod grasp, spherical grasp, palmar grasp, lateral grasp, open grasp."
            "Only answer with the grasp type you think is correct.")

prompts = {
    "cylindrical_grasp": ["An object that requires a cylindrical grasp, like a bottle or a handle.",
                    "An object that is held cylindrically with all fingers wrapped around it.",
                    "An object that is heavy and requires power to lift it"],
    "hook_grasp": ["An object that requires a hook grasp, like a bag or a bucket.",
             "An object held with a hook-like shape using 4 fingers, without the thumb."],
    "palmar_grasp": ["An object that requires a palmar grasp, can fit in a hand, like a notebook or smartphone.",
               "An object that is held in the palm of the hand."],
    "spherical_grasp": ["An object that requires a spherical grasp, like an apple or a small ball.",
                  "An object that is grasped with fingers and palm forming a sphere."],
    "tripod_grasp": ["An object that requires a tripod grasp, like a pen or a pencil.",
               "An object held between three fingers."],
    "pinch_grasp": ["An object that requires a pinch grasp, like a coin or needle.",
              "An object held with the thumb and one other finger.", 
              "An object that is tiny, light, and delicate"],
    "open_grasp": ["An object that requires an open grasp, like a flat box.",
             "An object that is grasped with an open hand."],
    "lateral_grasp": ["An object that irequires a lateral grasp, like a key or a credit card.",
                      "An object that is held between the thumb and the side of the index finger"]
}

def get_text_prompt_for_classes(prompts_dict):
    prompt_parts = ["You are an expert in robotic grasp classification. Given the image of an object, classify the most suitable grasp types for it.",
                    "Here are the possible classes:\n"]
    for grasp_type, descriptions in prompts_dict.items():
        prompt_text = " ".join(descriptions)
        prompt_parts.append(f"{grasp_type.replace('_', ' ').title()}: {prompt_text}")
    prompt_parts.append("\nGiven the following image, what type of grasp should be used? Your answer must contain three different possible grasp types, separated by commas. A single grasp type is insufficient")
    return "\n".join(prompt_parts)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def ask_model_with_image(image_path, text_prompts=get_text_prompt_for_classes(prompts)):
    image_b64 = encode_image(image_path)
    
    # Replace this with your desired multimodal-compatible model
    payload = {
        "model": MODEL,
        "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text_prompts
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    }
                },
            ]
        }
    ]
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    result = response.json()
    print("Raw model response:", result)
    return result

def get_predicted_class(response_text):
    response_text = response_text.lower()
    predicted_grasps = [grasp for grasp in GRASP_CLASSES if grasp in response_text]
    return predicted_grasps if predicted_grasps else ["unknown"]

def predict_image_grasp(image, text_prompts=get_text_prompt_for_classes(prompts)):
    image_b64 = base64.b64encode(image.decode("utf-8"))
    payload = {
        "model": MODEL,
        "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text_prompts
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    }
                },
            ]
        }
    ]
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    result = response.json()
    if "choices" in result:
        result_content = result["choices"][0]["message"]["content"]
    predicted_label = get_predicted_class(response)
    return result_content

def evaluate():
    with open(LABELS_PATH, "r") as f:
        ground_truth = json.load(f)

    total = len(ground_truth)
    correct = 0

    text_prompt = get_text_prompt_for_classes(prompts)
    for filename in os.listdir(QUERY_DIR):
        if not filename.endswith(".png"):
            continue

        image_path = os.path.join(QUERY_DIR, filename)
        true_labels = ground_truth.get(filename, "") 

        print(f"Processing {filename}...")

        result_text = None
        while True:
            # response = ask_model_with_image(image_path, text_prompt)
            result = ask_model_with_image(image_path)
            if "choices" in result:
                result_text = result["choices"][0]["message"]["content"]
                break
            else:
                print("API Error:", result)
                result_text = result['error']['message']
                time.sleep(60)
                continue

        predicted_labels = get_predicted_class(result_text)

        print(f"Response: {predicted_labels} | Ground Truth: {true_labels}")
        if any(tl in predicted_labels for tl in true_labels):
            correct += 1
            print("CORRECT!")
        else:
            print("INCORRECT!")
    accuracy = correct / total if total > 0 else 0
    print(f"\n✅ Accuracy: {accuracy:.2%} ({correct}/{total})")

if __name__ == "__main__":
    evaluate()
    
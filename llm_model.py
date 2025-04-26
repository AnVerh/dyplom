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

MODEL = "google/gemma-3-12b-it:free"
# Load and encode image to base64
with open(image_path, "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

'''headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": "google/gemma-3-4b-it:free",  # or "openai/gpt-4-vision-preview"
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data}"
                    }
                },
                {
                    "type": "text",
                    "text": ("What kind of grasp should be used to pick up the object in this image?"
                             "Only choose one of the following: cylindrical grasp, hook grasp, pinch grasp, tripod grasp, spherical grasp, palmar grasp, lateral grasp, open grasp.")
                }
            ]
        }
    ]
}

response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
print(response.json())'''

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
    prompt_parts = ["You are an expert in robotic grasp classification. Given the image of an object, classify the grasp type for it.",
                    "Here are the possible classes:\n"]
    for grasp_type, descriptions in prompts_dict.items():
        prompt_text = " ".join(descriptions)
        prompt_parts.append(f"{grasp_type.replace('_', ' ').title()}: {prompt_text}")
    prompt_parts.append("\nGiven the following image, what type of grasp should be used? Only answer with the grasp type you think is correct.")
    return "\n".join(prompt_parts)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def ask_model_with_image(image_path, text_prompts):
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
                # {
                #     "type": "text",
                #     "text": QUESTION
                # }
            ]
        }
    ]
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    result = response.json()
    if "choices" in result:
        return result["choices"][0]["message"]["content"]
    else:
        print("API Error:", result)
        return ""
    
def get_predicted_class(response_text):
    response_text = response_text.lower()
    for grasp in GRASP_CLASSES:
        if grasp in response_text:
            return grasp
    return "unknown"

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
        response = None
        while True:
            response = ask_model_with_image(image_path, text_prompt)
            if response.strip() == "":
                time.sleep(60)
                continue
            # if isinstance(response, dict) and 'error' in response:
                # error = response['error']
                # if error.get('code') == 429:
                #     print("Rate limit hit.")
                #     reset_ts = error.get('metadata', {}).get('headers', {}).get('X-RateLimit-Reset')
                #     if reset_ts:
                #         wait_time = int(reset_ts) / 1000 - time.time()
                #         wait_time = max(5, wait_time)  # at least wait a few seconds
                #         print(f"Waiting for {wait_time:.1f} seconds until rate limit resets...")
                #         time.sleep(wait_time)
                #     else:
                #         print("No reset time found, waiting default 60s...")
                #         time.sleep(60)
                #     continue
            else:
                break

        predicted_label = get_predicted_class(response)

        print(f"Response: {response} | Ground Truth: {true_labels}")
        if predicted_label in true_labels:
            correct += 1
            print("CORRECT!")
        else:
            print("INCORRECT!")
    accuracy = correct / total if total > 0 else 0
    print(f"\n✅ Accuracy: {accuracy:.2%} ({correct}/{total})")

if __name__ == "__main__":
    evaluate()

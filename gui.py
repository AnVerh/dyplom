import gradio as gr
from PIL import Image
from llm_model import ask_model_with_image
from knn_fine_tuned_clip import get_image_predictions
from knn_clip import predict_image_grasp_type_with_knn
from matching_nn_resnet import get_image_predictions_mnn_resnet
from matching_networks_clip import get_image_predictions_mnn_clip
from prototypical_nn_clip import get_image_predictions_pnn_clip
from prototypical_nn_resnet import get_image_predictions_pnn_resnet
from zero_shot_clip import predict_image_grasp_type
import tempfile

def classify_grasp(image, method):
    prediction=None
    top_predictions = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        image_path = temp_file.name
    match method:
        case "Gemma":
            prediction = ask_model_with_image(image_path)
        case "KNN with fine-tuned CLIP":
            top_predictions = get_image_predictions(image_path, 3)
        case "KNN with CLIP":
            top_predictions = predict_image_grasp_type_with_knn(image_path, 3)
        case "matching network with ResNet":
            top_predictions = get_image_predictions_mnn_resnet(image_path, 3)
        case "matching network with CLIP":
            top_predictions = get_image_predictions_mnn_clip(image_path, 3)
        case "prototypical network with CLIP":
            top_predictions = get_image_predictions_pnn_clip(image_path, 3)
        case "prototypical network with ResNet":
            top_predictions = get_image_predictions_pnn_resnet(image_path, 3)
        case "zero-shot CLIP":
            top_predictions = predict_image_grasp_type(image_path)

    print(f"given prediction: {prediction}")
    if method=="Gemma":
        # grasp_type_image_path = ""
        if 'grasp' in prediction.lower():
            pred_key = prediction.lower().replace(" ", "_")
            ukr_name = GRASP_TRANSLATIONS.get(pred_key, prediction)
            grasp_type_image_path = f"/home/kpi_anna/data/test_grasp_dataset/grasp_types/{pred_key}.jpg"
            return ukr_name, [grasp_type_image_path] #Image.open(grasp_type_image_path)
        else:
            prediction += '\nНе було отримано захвату, але можна спробувати:'
            grasp_type_image_path = "/home/kpi_anna/data/test_grasp_dataset/grasp_types/cylindrical_grasp.jpg"
            return prediction, [grasp_type_image_path] #Image.open(grasp_type_image_path)
    else:
        image_paths = []
        display_names = []

        for pred in top_predictions:
            pred_key = pred.lower().replace(" ", "_")
            ukr_name = GRASP_TRANSLATIONS.get(pred_key, pred) 
            display_names.append(ukr_name)
            grasp_img_path = f"/home/kpi_anna/data/test_grasp_dataset/grasp_types/{pred_key}.jpg"
            try:
                img = Image.open(grasp_img_path)
            except:
                img = Image.new("RGB", (224, 224), (128, 128, 128))  # fallback gray image
            image_paths.append(img)

        return ", ".join(display_names), image_paths


classification_methods = ["matching network та CLIP", "matching network та ResNet",
                          "prototypical network та CLIP", "prototypical network та ResNet",
                          "zero-shot CLIP", "Gemma", "KNN та CLIP", "KNN та fine-tuned CLIP"]

GRASP_TRANSLATIONS = {
    "hook_grasp": "гачковий захват",
    "lateral_grasp": "латеральний захват",
    "open_grasp": "відкритий захват",
    "cylindrical_grasp": "циліндричний захват",
    "tripod_grasp": "трьохпальцевий захват",
    "pinch_grasp": "щипковий захват",
    "palmar_grasp": "долонний захват",
    "spherical_grasp": "сферичний захват"
}


with gr.Blocks() as demo:
    gr.Markdown("## Визначення типу захвату")

    with gr.Row():
        with gr.Column(scale=1):  # Ліва колонка
            img_input = gr.Image(type="pil", label="Зображення")
            method = gr.Dropdown(classification_methods, label="Метод")
            predict_btn = gr.Button("Дізнатися захват")

        with gr.Column(scale=1):  # Права колонка
            output_text = gr.Textbox(label="Передбачений захват:")
            gallery = gr.Gallery(label="Вигляд захвату:", height=500)

    def wrapped_classify(image, method):
        text, imgs = classify_grasp(image, method)
        return text, imgs

    predict_btn.click(fn=wrapped_classify, inputs=[img_input, method], outputs=[output_text, gallery])

demo.launch()



# app/inference.py
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.77148203, 0.55764165, 0.58345652],
                         std=[0.12655577, 0.14245141, 0.15189891]),
])

def predict_image(image_bytes, model_b7, model_xcp):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs_b7 = F.softmax(model_b7(input_tensor), dim=1)
        probs_xcp = F.softmax(model_xcp(input_tensor), dim=1)
        probs_ensemble = 0.5 * probs_b7 + 0.5 * probs_xcp

    confidences = probs_ensemble.cpu().numpy()[0]
    pred_index = np.argmax(confidences)
    pred_label = class_names[pred_index]

    return {
        'predicted_label': pred_label,
        'class_probabilities': {class_names[i]: float(prob) for i, prob in enumerate(confidences)}
    }

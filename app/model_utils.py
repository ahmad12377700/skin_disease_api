# app/model_utils.py
import torch
import timm
import gdown
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec']

def download_model(url, output_path):
    if not os.path.exists(output_path):
        print(f"Downloading model to {output_path}...")
        gdown.download(url, output_path, quiet=False)

def load_models():
    eff_url = "https://drive.google.com/uc?id=1fSgY2pLgx4AeL5UqOZG1Kgqfj6koRX1H"
    xcp_url = "https://drive.google.com/uc?id=1AvP8KXiNH6KrPEAUy08Fmp5GOC468NvF"
    
    eff_path = "efficientnet_b7.pth"
    xcp_path = "xception.pth"

    download_model(eff_url, eff_path)
    download_model(xcp_url, xcp_path)

    model_b7 = timm.create_model('efficientnet_b7', pretrained=False, num_classes=len(class_names))
    model_b7.load_state_dict(torch.load(eff_path, map_location=device))
    model_b7.to(device).eval()

    model_xcp = timm.create_model('xception', pretrained=False, num_classes=len(class_names))
    model_xcp.load_state_dict(torch.load(xcp_path, map_location=device))
    model_xcp.to(device).eval()

    return model_b7, model_xcp

# app.py
import io
from typing import Dict, Any, Optional

import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# ---------------- Config ----------------
# Your exact mapping (index -> label)
IDX_TO_LABEL: Dict[int, str] = {
    2: 'Benign keratosis-like lesions ',
    4: 'Melanocytic nevi',
    3: 'Dermatofibroma',
    6: 'Dermatofibroma',
    5: 'Vascular lesions',
    1: 'Basal cell carcinoma',
    0: 'Actinic keratoses',
}
NUM_CLASSES = len(IDX_TO_LABEL)

# Default checkpoint path (you can change in the sidebar or upload a file)
DEFAULT_MODEL_PATH = "best_model_weights.pt"

DEVICE = torch.device("cpu")  # set to torch.device("cuda") if running with GPU

# Must match your train-time transforms
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ---------------- Model helpers ----------------
def initialize_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    m = models.densenet121(weights=None)
    in_feats = m.classifier.in_features
    m.classifier = nn.Linear(in_feats, num_classes)
    return m

@st.cache_resource(show_spinner=True)
def load_model_from_source(ckpt_bytes: Optional[bytes], ckpt_path: Optional[str]) -> nn.Module:
    model = initialize_model()
    map_location = DEVICE

    # Load checkpoint (support plain state_dict or wrapped formats)
    state: Any = None
    if ckpt_bytes:
        state = torch.load(io.BytesIO(ckpt_bytes), map_location=map_location)
    elif ckpt_path:
        state = torch.load(ckpt_path, map_location=map_location)
    else:
        raise FileNotFoundError("No checkpoint provided.")

    # Unwrap common keys
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    if not isinstance(state, dict):
        raise RuntimeError("Unexpected checkpoint format; expected a state_dict-like object.")

    model.load_state_dict(state)
    model.eval()
    model.to(DEVICE)
    return model

def idx_to_label(idx: int) -> str:
    # .strip() just to remove the trailing space in 'Benign keratosis-like lesions '
    return IDX_TO_LABEL.get(idx, f"Unknown ({idx})").strip()

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="DenseNet Image Classifier", page_icon="🧪", layout="centered")
st.title("🧪 DenseNet Skin Image Classifier")
st.caption("Demo app — shows only the predicted label. Not a medical device or diagnosis.")

with st.sidebar:
    st.header("Model")
    ckpt_file = st.file_uploader("Upload checkpoint (.pt / .bin)", type=["pt", "bin"])
    ckpt_path = st.text_input("Or use local checkpoint path", value=DEFAULT_MODEL_PATH)
    st.caption("If both are provided, the uploaded file is used.")

st.subheader("Upload an image")
img_file = st.file_uploader("Choose a JPG/PNG", type=["jpg", "jpeg", "png"])

if img_file is not None:
    # Preview image
    try:
        pil_img = Image.open(img_file).convert("RGB")
    except Exception as e:
        st.error(f"Invalid image: {e}")
        st.stop()

    st.image(pil_img, caption="Uploaded image", use_container_width=True)

    # Load model
    ckpt_bytes = ckpt_file.read() if ckpt_file is not None else None
    try:
        with st.spinner("Loading model…"):
            model = load_model_from_source(ckpt_bytes, ckpt_path if ckpt_bytes is None else None)
    except Exception as e:
        st.error(f"Checkpoint load error: {e}")
        st.stop()

    # Predict (no confidence, no top-k)
    tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        pred_idx = int(torch.argmax(logits, dim=1).item())

    label = idx_to_label(pred_idx)

    st.markdown("### Prediction")
    st.markdown(f"**{label}**")
    # (Intentionally not showing confidence or top-k)

st.markdown("---")
st.caption("Tip: run with `python -m streamlit run app.py`. If you see a blank page, check the terminal for errors or try another browser.")

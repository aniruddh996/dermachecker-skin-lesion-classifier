# server.py -- DermaChecker inference API
# Serves the real DenseNet121 checkpoint trained on HAM10000.
#
#   pip install flask torch torchvision pillow
#   python server.py

import io
import os
from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model_weights_fp16.pt")
DEVICE = torch.device("cpu")

# Matches the categorical encoding produced during training
# (df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes).
#
# Index 6 was originally deployed as "Dermatofibroma" -- but tracing the training
# notebook shows the label-mapping dict used to build df_original had a copy-paste
# typo ('mel': 'dermatofibroma' instead of 'melanoma'), so class 6 was actually
# trained on melanoma images under a lowercase 'dermatofibroma' category that
# pandas treated as distinct from the real (capitalized) Dermatofibroma class at
# index 3. Corrected here to the label the training images actually represent.
IDX_TO_LABEL = {
    0: "Actinic keratoses",
    1: "Basal cell carcinoma",
    2: "Benign keratosis-like lesions",
    3: "Dermatofibroma",
    4: "Melanocytic nevi",
    5: "Vascular lesions",
    6: "Melanoma",
}
NUM_CLASSES = len(IDX_TO_LABEL)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


print("Loading DermaChecker model...")
MODEL = load_model()
print("Model ready.")

app = Flask(__name__)


@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "classes": NUM_CLASSES})


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204

    if "file" not in request.files:
        return jsonify({"error": "No file part named 'file'"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = MODEL(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    ranked = sorted(
        ((IDX_TO_LABEL[i], float(probs[i])) for i in range(NUM_CLASSES)),
        key=lambda x: x[1],
        reverse=True,
    )
    top_label, top_prob = ranked[0]

    return jsonify({
        "label": top_label,
        "confidence": round(top_prob, 4),
        "top3": [{"label": l, "probability": round(p, 4)} for l, p in ranked[:3]],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8790, debug=False)

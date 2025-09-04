# app.py
import io, base64
from PIL import Image
from flask import Flask, request, jsonify, render_template_string
import torch
import torch.nn as nn
from torchvision import models, transforms

# ------- config -------
# Your exact mapping (index -> label)
IDX_TO_LABEL = {
    2: 'Benign keratosis-like lesions ',
    4: 'Melanocytic nevi',
    3: 'Dermatofibroma',
    6: 'Dermatofibroma',
    5: 'Vascular lesions',
    1: 'Basal cell carcinoma',
    0: 'Actinic keratoses',
}
NUM_CLASSES = len(IDX_TO_LABEL)
MODEL_PATH = "best_model_weights.pt"   # put your trained weights here
DEVICE = torch.device("cpu")           # set to torch.device("cuda") if GPU is available

# Must match your train-time inference transforms
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),      
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],    
        std=[0.229, 0.224, 0.225]
    ),
])

# ------- model load -------
def initialize_model(num_classes=NUM_CLASSES):
    m = models.densenet121(weights=None)
    in_feats = m.classifier.in_features
    m.classifier = nn.Linear(in_feats, num_classes)
    return m

def load_model():
    model = initialize_model()
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    model.to(DEVICE)
    return model

MODEL = load_model()
torch.set_num_threads(1)

# ------- HTML Template -------
PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>DenseNet Inference</title>
  <style>
    body { font-family: system-ui, Arial; max-width: 720px; margin: 40px auto; }
    .card { border: 1px solid #e5e7eb; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.04); }
    .row { display: flex; gap: 24px; align-items: flex-start; }
    img { max-width: 320px; height: auto; border-radius: 10px; border: 1px solid #eee; }
    .pred { font-size: 1.1rem; }
    button, input[type=file] { padding: 8px 12px; }
    code { background: #f3f4f6; padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <div class="card">
    <h2>DenseNet Image Classifier</h2>
    <form action="/predict" method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept="image/*" required />
      <button type="submit">Predict</button>
    </form>
  </div>

  {% if predicted %}
  <div class="card" style="margin-top:20px;">
    <div class="row">
      <div>
        <img src="data:image/png;base64,{{ img_b64 }}" alt="uploaded image" />
      </div>
      <div>
        <div class="pred"><strong>Prediction:</strong> {{ label }}</div>
        <p class="muted">Want JSON? POST to <code>/predict</code> with <code>Accept: application/json</code>.</p>
      </div>
    </div>
  </div>
  {% endif %}
</body>
</html>
"""

app = Flask(__name__)

def idx_to_label(idx: int) -> str:
    return IDX_TO_LABEL.get(idx, f"Unknown ({idx})").strip()

@app.route("/", methods=["GET"])
def index():
    return render_template_string(PAGE, predicted=False)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/labels", methods=["GET"])
def labels():
    return jsonify({int(k): v for k, v in IDX_TO_LABEL.items()})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part named 'file'"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = MODEL(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    # Final prediction
    pred_idx = int(torch.argmax(probs).item())
    label = idx_to_label(pred_idx)



    # Render HTML with image preview
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return render_template_string(
        PAGE,
        predicted=True,
        label=label,
        img_b64=img_b64
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

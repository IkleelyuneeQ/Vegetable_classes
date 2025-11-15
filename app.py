# app.py  (upload-only version)
import time
from pathlib import Path
import numpy as np
import streamlit as st
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from PIL import Image
from torchvision import datasets, models, transforms

# Config
st.set_page_config(page_title="Vegetable Classifier", page_icon="🥦", layout="centered")
DEFAULT_TRAIN_DIR = Path("./Vegetable_Images/Vegetable Images/train")
DEFAULT_MODEL_PATH = Path("best_model.pth")
STATS_PATH = Path("stats.pt")  # optional {"mean":[...], "std":[...]}

# Device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Utils
class ConvertToRGB:
    def __call__(self, img):
        return img.convert("RGB") if img.mode != "RGB" else img

@st.cache_resource(show_spinner=False)
def load_class_names(train_dir: Path):
    ds = datasets.ImageFolder(root=train_dir, transform=transforms.ToTensor())
    return list(ds.class_to_idx.keys())

def compute_mean_std(train_dir: Path):
    ds = datasets.ImageFolder(
        root=train_dir,
        transform=transforms.Compose([ConvertToRGB(), transforms.Resize((224, 224)), transforms.ToTensor()])
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    channel_sum, channel_squared_sum, n = 0, 0, 0
    for x, _ in loader:
        channel_sum += x.mean(dim=[0, 2, 3])
        channel_squared_sum += (x**2).mean(dim=[0, 2, 3])
        n += 1
    mean = (channel_sum/n).tolist()
    std = (channel_squared_sum/n - torch.tensor(mean)**2).sqrt().tolist()
    return mean, std

@st.cache_resource(show_spinner=False)
def get_norm_stats(train_dir: Path):
    if STATS_PATH.exists():
        d = torch.load(STATS_PATH, map_location="cpu")
        return d["mean"], d["std"]
    mean, std = compute_mean_std(train_dir)
    torch.save({"mean": mean, "std": std}, STATS_PATH)
    return mean, std

@st.cache_resource(show_spinner=True)
def build_and_load_model(num_classes: int, model_path: Path):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(model_path, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval().to(DEVICE)
    return model

def preprocess(img: Image.Image, mean, std):
    tfm = transforms.Compose([
        ConvertToRGB(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return tfm(img).unsqueeze(0)

def infer(model, x):
    with torch.no_grad():
        p = torch.softmax(model(x.to(DEVICE)), dim=1).squeeze(0).cpu().numpy()
    return p

def topk(probs, classes, k=3):
    idx = np.argsort(probs)[::-1][:k]
    return [(classes[i], float(probs[i])) for i in idx]

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    train_dir = Path(st.text_input("Train directory (for classes/stats)", str(DEFAULT_TRAIN_DIR)))
    model_path = Path(st.text_input("Model checkpoint", str(DEFAULT_MODEL_PATH)))
    conf_thresh = st.slider("Unknown threshold (max softmax)", 0.10, 0.99, 0.60, 0.01)
    show_topk = st.slider("Show top-k", 1, 5, 3)
    st.caption(f"Device: `{DEVICE}`")

# Load assets
classes = load_class_names(train_dir)
mean, std = get_norm_stats(train_dir)
model = build_and_load_model(len(classes), model_path)

st.title("🥦 Vegetable Classifier — Upload Only")
st.markdown("Upload a photo of a vegetable. The app predicts its class and returns **Unknown** "
            "when confidence is below the chosen threshold.")

st.markdown(f"**Classes:** {', '.join(classes)}")
st.markdown(f"**Normalization:** mean={np.round(mean,4)} | std={np.round(std,4)}")

# Upload
uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"])
if not uploaded:
    st.info("Choose an image to begin.")
    st.stop()

img = Image.open(uploaded)
st.image(img, caption="Input image", use_column_width=True)

# Predict
x = preprocess(img, mean, std)
t0 = time.time()
probs = infer(model, x)
dt = (time.time() - t0) * 1000

mx = float(np.max(probs))
pred_idx = int(np.argmax(probs))
pred_class = classes[pred_idx]

if mx < conf_thresh:
    st.error(f"Prediction: **Unknown / Out-of-scope**  \n(Confidence {mx:.2%} < {conf_thresh:.2%})")
else:
    st.success(f"Prediction: **{pred_class}**  \nConfidence: **{mx:.2%}**")

st.write("**Top-k probabilities:**")
top = topk(probs, classes, k=show_topk)
st.dataframe({"class": [c for c, _ in top], "probability": [round(p, 4) for _, p in top]}, use_container_width=True)
st.caption(f"Inference: {dt:.1f} ms on {DEVICE.upper()}  |  OOD rule: max-softmax < threshold → Unknown")

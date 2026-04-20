import os, io, base64, cv2, torch, numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Model Architecture (exact match to drcnnrb_cbam_best__2_.pth) ─────────────
class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        mid = max(1, ch // r)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc       = nn.Sequential(
            nn.Linear(ch, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, ch, bias=False),
        )
        self.sigmoid  = nn.Sigmoid()
    def forward(self, x):
        b, c = x.shape[:2]
        avg  = self.fc(self.avg_pool(x).view(b, c))
        mx   = self.fc(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg + mx).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.bn      = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        return x * self.sigmoid(self.bn(self.conv(torch.cat([avg, mx], dim=1))))

class CBAM(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.channel = ChannelAttention(ch, r)
        self.spatial = SpatialAttention()
    def forward(self, x): return self.spatial(self.channel(x))

class IdentityBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)
        self.conv3 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(ch)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        return F.relu(out + x, inplace=True)

class ConvBlock(nn.Module):
    def __init__(self, ic, oc, s=2):
        super().__init__()
        self.conv1   = nn.Conv2d(ic, oc, 3, stride=s, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(2, stride=1, padding=1)
        self.bn1     = nn.BatchNorm2d(oc)
        self.conv2   = nn.Conv2d(oc, oc, 3, padding=1, bias=False)
        self.bn2     = nn.BatchNorm2d(oc)
        self.conv3   = nn.Conv2d(oc, oc, 3, padding=1, bias=False)
        self.bn3     = nn.BatchNorm2d(oc)
        self.sc_conv = nn.Conv2d(ic, oc, 1, stride=s, bias=False)
        self.sc_pool = nn.MaxPool2d(2, stride=1, padding=1)
        self.sc_bn   = nn.BatchNorm2d(oc)
    def forward(self, x):
        out = F.relu(self.bn1(self.maxpool(self.conv1(x))), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        skip = self.sc_bn(self.sc_pool(self.sc_conv(x)))
        return F.relu(out + skip, inplace=True)

class ResBlock(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.conv_block = ConvBlock(ic, oc)
        self.identity1  = IdentityBlock(oc)
        self.identity2  = IdentityBlock(oc)
    def forward(self, x):
        return self.identity2(self.identity1(self.conv_block(x)))

class DRCNNRB(nn.Module):
    def __init__(self, num_classes=5, dropout=0.6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.ZeroPad2d(3), nn.Conv2d(3, 64, 7, stride=2, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.resblock1 = ResBlock(64, 128);  self.cbam1 = CBAM(128)
        self.resblock2 = ResBlock(128, 256); self.cbam2 = CBAM(256)
        self.resblock3 = ResBlock(256, 512); self.cbam3 = CBAM(512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.stem(x)
        x = self.cbam1(self.resblock1(x))
        x = self.cbam2(self.resblock2(x))
        x = self.cbam3(self.resblock3(x))
        return self.fc(self.drop(self.pool(x).flatten(1)))

# ── Load model ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = DRCNNRB(num_classes=5, dropout=0.6).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
print(f"Model loaded on {device} ✓")

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"]
SEVERITY = {
    "No_DR":          {"label": "No Diabetic Retinopathy", "level": 0,
                       "color": "#22c55e", "bg": "#052e16",
                       "advice": "No signs of DR detected. Continue annual eye screenings and maintain healthy blood sugar levels.",
                       "action": "Annual screening recommended"},
    "Mild":           {"label": "Mild NPDR",               "level": 1,
                       "color": "#eab308", "bg": "#1c1917",
                       "advice": "Early stage non-proliferative DR detected. Monitor blood sugar closely and schedule follow-up in 12 months.",
                       "action": "Monitor every 12 months"},
    "Moderate":       {"label": "Moderate NPDR",            "level": 2,
                       "color": "#f97316", "bg": "#1c0a00",
                       "advice": "Moderate non-proliferative DR detected. Ophthalmologist referral is advised within 6 months.",
                       "action": "Ophthalmologist referral in 6 months"},
    "Severe":         {"label": "Severe NPDR",              "level": 3,
                       "color": "#ef4444", "bg": "#1c0000",
                       "advice": "Severe non-proliferative DR detected. Urgent referral to ophthalmologist required within 3 months.",
                       "action": "Urgent referral within 3 months"},
    "Proliferate_DR": {"label": "Proliferative DR",         "level": 4,
                       "color": "#a855f7", "bg": "#1a0030",
                       "advice": "Proliferative DR detected — highest severity. Immediate ophthalmologist referral required. Risk of blindness without treatment.",
                       "action": "IMMEDIATE referral required"},
}

INFER_TF = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_retinal_image(img: Image.Image) -> Image.Image:
    img_np = np.array(img.convert("RGB"))
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        pad    = 5
        img_np = img_np[max(0,y-pad):min(img_np.shape[0],y+h+pad),
                        max(0,x-pad):min(img_np.shape[1],x+w+pad)]
    img_np = cv2.resize(img_np, (256, 256))
    blur   = cv2.GaussianBlur(img_np, (0, 0), sigmaX=10)
    img_np = cv2.addWeighted(img_np, 4, blur, -4, 128)
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    table  = np.array([((i/255.0)**(1/1.2))*255 for i in range(256)]).astype("uint8")
    img_np = cv2.LUT(img_np, table)
    lab    = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab    = cv2.merge((clahe.apply(l), a, b))
    img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(img_np)

def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        raw_img = Image.open(file.stream).convert("RGB")
        pre_img = preprocess_retinal_image(raw_img)

        with torch.no_grad():
            t         = INFER_TF(pre_img).unsqueeze(0).to(device)
            probs_avg = torch.softmax(model(t), dim=1).squeeze(0).cpu().numpy()

        pred_idx   = int(probs_avg.argmax())
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs_avg[pred_idx]) * 100
        info       = SEVERITY[pred_class]

        return jsonify({
            "prediction":  pred_class,
            "label":       info["label"],
            "confidence":  round(confidence, 2),
            "color":       info["color"],
            "bg":          info["bg"],
            "advice":      info["advice"],
            "action":      info["action"],
            "level":       info["level"],
            "probabilities": {
                CLASS_NAMES[i]: round(float(probs_avg[i]) * 100, 2)
                for i in range(len(CLASS_NAMES))
            },
            "original_img":     img_to_b64(raw_img.resize((300, 300))),
            "preprocessed_img": img_to_b64(pre_img.resize((300, 300))),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#if __name__ == "__main__":
   # app.run(debug=True, port=5000)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
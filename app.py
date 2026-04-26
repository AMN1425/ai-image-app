import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import os
import json

# =========================
# إعداد الصفحة
# =========================
st.set_page_config(
    page_title="نظام تصنيف الصور الذكي",
    layout="centered"
)

# =========================
# 🎨 واجهة احترافية
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');

.stApp {
    background-color: #0b1220;
    font-family: 'Cairo', sans-serif;
}

/* العنوان */
h1 {
    color: #38bdf8;
    text-align: center;
    font-weight: 800;
}

/* النص */
p, span, label {
    color: #e2e8f0;
}

/* الأزرار */
.stButton>button {
    background: #2563eb;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
    font-weight: 600;
}

.stButton>button:hover {
    background: #1d4ed8;
}

/* صندوق النتائج */
.result-box {
    background: #111827;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #1f2937;
    box-shadow: 0 10px 25px rgba(0,0,0,0.4);
}

/* شريط التقدم */
.stProgress > div > div {
    background-color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

# =========================
# العنوان + الأسماء
# =========================
st.markdown("""
<h1>🤖 نظام تصنيف الصور الذكي</h1>

<h4 style="text-align:center; color:#38bdf8;">
تم تطويره بواسطة: <b>أيمن اليزيدي</b> و <b>خالد القحطاني</b>
</h4>
""", unsafe_allow_html=True)

st.write("ارفع صورة أو استخدم الكاميرا وسيتم التعرف على محتواها")

# =========================
# قاموس عربي
# =========================
AR_DICT = {
    "cat": "قطة",
    "dog": "كلب",
    "car": "سيارة",
    "person": "شخص",
    "bottle": "زجاجة",
    "cup": "كوب",
    "chair": "كرسي",
    "table": "طاولة",
    "bird": "طائر",
    "pizza": "بيتزا",
    "phone": "هاتف",
    "laptop": "حاسوب محمول",
    "bus": "حافلة",
    "truck": "شاحنة",
    "horse": "حصان"
}

def translate(label):
    label = label.lower()
    return AR_DICT.get(label, f"غير معروف ({label})")

# =========================
# تحميل labels
# =========================
def load_labels():
    if os.path.exists("labels.json"):
        with open("labels.json", "r") as f:
            return json.load(f)
    return []

labels = load_labels()

# =========================
# تحميل النموذج
# =========================
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()
    return model

model = load_model()

# =========================
# تجهيز الصورة
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# إدخال الصورة
# =========================
option = st.radio("اختر طريقة إدخال الصورة:", ["رفع صورة", "كاميرا"])

image = None

if option == "رفع صورة":
    file = st.file_uploader("ارفع صورة", type=["jpg", "jpeg", "png"])
    if file:
        image = Image.open(file).convert("RGB")

elif option == "كاميرا":
    cam = st.camera_input("التقط صورة")
    if cam:
        image = Image.open(cam).convert("RGB")

# =========================
# التحليل
# =========================
if image:
    st.image(image, caption="📷 الصورة المدخلة", use_container_width=True)

    with st.spinner("🔍 جاري التحليل..."):
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0]

        best_idx = torch.argmax(probs).item()
        confidence = probs[best_idx].item()

        label_en = labels[best_idx] if best_idx < len(labels) else "unknown"
        label_ar = translate(label_en)

    # =========================
    # النتيجة
    # =========================
    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    st.subheader("📊 النتيجة النهائية")
    st.success(f"🎯 التصنيف: {label_ar}")
    st.write(f"📈 نسبة الثقة: {round(confidence * 100, 2)}%")

    st.progress(float(confidence))

    if confidence > 0.7:
        st.success("✔️ دقة عالية")
    elif confidence > 0.4:
        st.warning("⚠️ دقة متوسطة")
    else:
        st.error("❌ دقة ضعيفة")

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("⬆️ الرجاء رفع صورة أو استخدام الكاميرا")

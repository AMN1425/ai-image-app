st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');

/* خلفية عامة */
html, body, [class*="css"] {
    font-family: 'Cairo', sans-serif;
    direction: rtl;
    text-align: right;
    background: linear-gradient(135deg, #eef2ff, #ffffff);
}

/* عنوان رئيسي */
h1 {
    text-align: center;
    color: #1f2937;
    font-weight: 700;
}

/* صندوق التطبيق */
.block-container {
    padding: 2rem 2rem 2rem 2rem;
}

/* صندوق رفع الملف */
.stFileUploader {
    background: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

/* زر */
.stButton>button {
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
    font-weight: 600;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #4338ca, #4f46e5);
    transform: scale(1.02);
}

/* صندوق النتائج */
.result-box {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin-top: 20px;
}

/* شريط التقدم */
.stProgress > div > div {
    background-color: #4f46e5;
}

/* تحسين النص */
p {
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)


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
# واجهة احترافية
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Cairo', sans-serif;
    direction: rtl;
    text-align: right;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 نظام تصنيف الصور باستخدام الذكاء الاصطناعي")
st.write("يقوم هذا النظام بتحليل الصور وتحديد محتواها باستخدام نموذج تعلم عميق")

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
    "laptop": "حاسوب محمول"
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
option = st.radio("اختر طريقة إدخال الصورة:", ["رفع صورة", "التقاط بالكاميرا"])

image = None

if option == "رفع صورة":
    file = st.file_uploader("ارفع صورة", type=["jpg", "jpeg", "png"])
    if file:
        image = Image.open(file).convert("RGB")

elif option == "التقاط بالكاميرا":
    cam = st.camera_input("التقط صورة")
    if cam:
        image = Image.open(cam).convert("RGB")

# =========================
# التحليل
# =========================
if image:
    st.image(image, caption="الصورة المدخلة", use_container_width=True)

    with st.spinner("جاري تحليل الصورة..."):
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
    st.subheader("📊 النتيجة النهائية")

    st.success(f"🎯 التصنيف: {label_ar}")
    st.write(f"📈 نسبة الثقة: {round(confidence * 100, 2)}%")

    st.progress(float(confidence))

    # تقييم الثقة
    if confidence > 0.7:
        st.success("✔️ دقة عالية في النتيجة")
    elif confidence > 0.4:
        st.warning("⚠️ دقة متوسطة")
    else:
        st.error("❌ دقة منخفضة - حاول صورة أوضح")

else:
    st.info("⬆️ الرجاء رفع صورة أو استخدام الكاميرا")

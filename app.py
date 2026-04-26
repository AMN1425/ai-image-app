import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import json
import os

# =========================
# إعداد الصفحة
# =========================
st.set_page_config(page_title="كاشف الصور الذكي", layout="centered")

# =========================
# واجهة عربية
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

st.title("🤖 كاشف الصور الذكي")
st.write("ارفع صورة أو استخدم الكاميرا وسيتم التعرف على الصورة")

# =========================
# قاموس ترجمة بسيط
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
    "bus": "حافلة",
    "truck": "شاحنة",
    "horse": "حصان",
    "sheep": "خروف",
    "cow": "بقرة",
    "tv": "تلفاز",
    "keyboard": "لوحة مفاتيح",
    "laptop": "حاسوب محمول",
    "phone": "هاتف"
}

def translate(label):
    label = label.lower()
    return AR_DICT.get(label, label)

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
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# اختيار الإدخال
# =========================
option = st.radio("اختر طريقة الإدخال:", ["رفع صورة", "الكاميرا"])

image = None

if option == "رفع صورة":
    uploaded = st.file_uploader("ارفع صورة", type=["jpg", "png", "jpeg"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")

elif option == "الكاميرا":
    camera = st.camera_input("التقط صورة")
    if camera:
        image = Image.open(camera).convert("RGB")

# =========================
# تحليل الصورة
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

        label_en = labels[best_idx] if labels else "unknown"
        label_ar = translate(label_en)

        st.markdown("## 🎯 النتيجة النهائية")
        st.success(f"التصنيف: {label_ar}")
        st.write(f"📊 نسبة الثقة: {round(confidence * 100, 2)}%")

        st.progress(float(confidence))

        if confidence < 0.5:
            st.warning("⚠️ النموذج غير متأكد من النتيجة")

else:
    st.info("⬆️ الرجاء رفع صورة أو استخدام الكاميرا")

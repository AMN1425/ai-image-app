import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import json
import requests

# إعداد الصفحة
st.set_page_config(page_title="🤖 التعرّف على الصور (AI)", layout="centered")

# تنسيق بسيط لدعم اللغة العربية
st.markdown("""
    <style>
    .main { direction: rtl; text-align: right; }
    .stAlert { direction: rtl; text-align: right; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 نظام التعرّف على الصور الذكي")
st.caption("ارفع صورة وسيقوم الذكاء الاصطناعي بتحليلها وإخبارك بمحتواها.")

# -----------------------------
# تحميل الفئات (Labels) من الإنترنت تلقائياً
# -----------------------------
@st.cache_data
def get_labels():
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    try:
        return requests.get(url).json()
    except:
        return None

labels = get_labels()

# -----------------------------
# قاموس الترجمة العربية (لأشهر الفئات)
# -----------------------------
AR_TRANSLATIONS = {
    "tench": "سمكة التنش", "goldfish": "سمكة ذهبية", "great white shark": "القرش الأبيض الكبير",
    "tiger shark": "قرش النمر", "hammerhead shark": "قرش المطرقة", "hen": "دجاجة",
    "ostrich": "نعامة", "goose": "إوزة", "koala": "كوالا", "tiger": "نمر", "lion": "أسد",
    "cheetah": "فهد", "dalmatian": "كلب دالميشن", "beagle": "كلب بيغل", "poodle": "كلب بودل",
    "mountain bike": "دراجة جبلية", "car": "سيارة", "sports car": "سيارة رياضية",
    "jeep": "جيب", "ambulance": "سيارة إسعاف", "bus": "حافلة", "laptop": "حاسوب محمول",
    "cell phone": "هاتف ذكي", "keyboard": "لوحة مفاتيح", "mouse": "فأرة", "coffee mug": "كوب قهوة",
    "pizza": "بيتزا", "burger": "برجر", "ice cream": "آيس كريم", "table": "طاولة", "chair": "كرسي"
}

def translate_to_ar(en_label):
    return AR_TRANSLATIONS.get(en_label.lower(), en_label)

# -----------------------------
# تحميل النموذج المعالج
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# -----------------------------
# واجهة المستخدم ورفع الصور
# -----------------------------
uploaded_file = st.file_uploader("اختر صورة (JPG أو PNG)...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="الصورة المرفوعة", use_container_width=True)
    
    with st.spinner('جاري التحليل...'):
        # معالجة الصورة
        img_t = transform(image).unsqueeze(0)
        
        # التوقع
        with torch.no_grad():
            outputs = model(img_t)
            _, predicted = outputs.max(1)
            idx = predicted.item()

    # عرض النتيجة
    st.subheader("النتيجة المستخلصة:")
    if labels:
        res_en = labels[idx]
        res_ar = translate_to_ar(res_en)
        st.success(f"✅ التوقع: **{res_ar}**")
        st.info(f"الاسم بالإنجليزية: {res_en}")
    else:
        st.error("فشل في تحميل مسميات الفئات، الرقم التعريفي هو: " + str(idx))

else:
    st.info("يرجى رفع صورة للبدء.")

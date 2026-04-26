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

# تنسيق عربي

# =========================

st.markdown("""

<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Cairo', sans-serif;
    direction: rtl;
    text-align: right;
}

.result-box {
    background: #f1f8ff;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    border: 1px solid #d0e3ff;
}

.center {
    text-align: center;
}
</style>

""", unsafe_allow_html=True)

# =========================

# العنوان

# =========================

st.title("🤖 مستكشف الصور الذكي")
st.write("ارفع صورة أو التقط صورة وسأعطيك النتيجة مباشرة بالعربي")

# =========================

# قاموس الترجمة

# =========================

AR_DICT = {
"cat": "قطة",
"dog": "كلب",
"car": "سيارة",
"person": "شخص",
"laptop": "كمبيوتر محمول",
"mobile": "هاتف",
"phone": "هاتف",
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
else:
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

# إدخال الصورة

# =========================

option = st.radio("اختر طريقة الإدخال:", ["رفع صورة", "التقاط بالكاميرا"])

image = None

if option == "رفع صورة":
uploaded = st.file_uploader("ارفع صورة", type=["jpg", "png", "jpeg"])
if uploaded:
image = Image.open(uploaded).convert("RGB")

else:
camera = st.camera_input("التقط صورة")
if camera:
image = Image.open(camera).convert("RGB")

# =========================

# التحليل

# =========================

if image:
st.image(image, caption="📷 الصورة المدخلة", use_container_width=True)

```
with st.spinner("🔍 جاري تحليل الصورة..."):
    try:
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)

        probs = torch.nn.functional.softmax(output, dim=1)[0]

        # 🎯 أفضل نتيجة فقط
        best_idx = torch.argmax(probs).item()
        confidence = probs[best_idx].item()

        label_en = labels[best_idx] if labels else "unknown"
        label_ar = translate(label_en)

        conf_percent = round(confidence * 100, 2)

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.subheader("✅ النتيجة النهائية:")

        st.write(f"🎯 **{label_ar}**")
        st.progress(float(confidence))
        st.caption(f"نسبة الثقة: {conf_percent}%")

        if conf_percent < 50:
            st.warning("⚠️ النموذج غير متأكد من النتيجة")

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception:
        st.error("❌ حدث خطأ أثناء تحليل الصورة")
```

else:
st.warning("⬆️ الرجاء إدخال صورة للبدء")

# =========================

# تذييل

# =========================

st.markdown("---")
st.markdown("<div class='center'>💡 استخدم صورة واضحة لتحصل على نتيجة أدق</div>", unsafe_allow_html=True)

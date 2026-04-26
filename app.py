import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import requests

# إعداد الصفحة
st.set_page_config(page_title="كاشف الصور الذكي", layout="centered")

# 🎨 تنسيق احترافي عربي
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Cairo', sans-serif;
    direction: rtl;
    text-align: right;
}

.result-box {
    background: linear-gradient(135deg, #d4edda, #c3e6cb);
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    border: 1px solid #b1dfbb;
}

.conf-high { color: #155724; }
.conf-mid { color: #856404; }
.conf-low { color: #721c24; }

.center { text-align: center; }
</style>
""", unsafe_allow_html=True)

# العنوان
st.title("🤖 مستكشف الصور الذكي")
st.write("ارفع صورة أو التقط صورة وسأحاول التعرف عليها باستخدام الذكاء الاصطناعي.")

# 📚 قاموس محسن
AR_DICT = {
    "cat": "قطة",
    "dog": "كلب",
    "car": "سيارة",
    "person": "شخص",
    "laptop": "كمبيوتر محمول",
    "phone": "هاتف",
    "bottle": "زجاجة",
    "cup": "كوب",
    "chair": "كرسي",
    "table": "طاولة",
    "bird": "طائر",
    "pizza": "بيتزا",
}

def translate(label):
    label = label.lower()
    for key in AR_DICT:
        if key in label:
            return AR_DICT[key]
    return f"({label})"

# تحميل البيانات
@st.cache_data
def load_labels():
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    return requests.get(url).json()

@st.cache_resource
def load_model():
    return models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).eval()

labels = load_labels()
model = load_model()

# تحويل الصورة
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# اختيار مصدر الصورة
option = st.radio("اختر طريقة الإدخال:", ["رفع صورة", "التقاط بالكاميرا"])

image = None

if option == "رفع صورة":
    uploaded = st.file_uploader("ارفع صورة", type=["jpg", "png", "jpeg"])
    if uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
        except:
            st.error("❌ فشل في قراءة الصورة")

else:
    camera = st.camera_input("التقط صورة")
    if camera:
        image = Image.open(camera).convert("RGB")

# المعالجة
if image:
    st.image(image, caption="📷 الصورة", use_container_width=True)

    with st.spinner("🔍 جاري التحليل..."):
        try:
            tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(tensor)

            probs = torch.nn.functional.softmax(output, dim=1)[0]
            top3 = torch.topk(probs, 3)

            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown("### 📊 النتائج:")

            for i in range(3):
                idx = top3.indices[i].item()
                conf = top3.values[i].item()
                label_en = labels[idx]
                label_ar = translate(label_en)

                conf_percent = round(conf * 100, 2)

                if conf_percent > 70:
                    cls = "conf-high"
                elif conf_percent > 40:
                    cls = "conf-mid"
                else:
                    cls = "conf-low"

                st.markdown(
                    f"<p class='{cls}'>🔹 {label_ar} — {conf_percent}%</p>",
                    unsafe_allow_html=True
                )

            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error("حدث خطأ أثناء التحليل")

else:
    st.warning("⬆️ الرجاء إدخال صورة للبدء")

# تذييل
st.markdown("---")
st.markdown("<div class='center'>💡 للحصول على أفضل النتائج استخدم صور واضحة</div>", unsafe_allow_html=True)


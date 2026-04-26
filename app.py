import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms

# إعداد الصفحة لتكون باللغة العربية ومن اليمين لليسار
st.set_page_config(page_title="التعرف على الصور بالذكاء الاصطناعي", layout="centered")

# تنسيق CSS لجعل الواجهة تدعم اللغة العربية (RTL)
st.markdown("""
    <style>
    body, div, p, h1, h2, h3, h4, h5, h6 {
        direction: rtl;
        text-align: right;
    }
    .stFileUploader label {
        display: block;
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 نظام التعرف على الصور (AI)")
st.write("قم برفع صورة وسأحاول التعرف على محتواها باستخدام نموذج ResNet50.")

# تحميل النموذج
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

model = load_model()

# التحويلات المطلوبة للصورة
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# أداة رفع الملفات
uploaded_file = st.file_uploader("اختر صورة من جهازك...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # عرض الصورة المرفوعة
    image = Image.open(uploaded_file)
    st.image(image, caption='الصورة التي تم رفعها', use_column_width=True)
    
    st.info("جاري تحليل الصورة... انتظر قليلاً")
    
    # المعالجة والتوقع
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = outputs.max(1)
    
    # عرض النتيجة
    st.success(f"النتيجة (كود الفئة): {predicted.item()}")
    st.warning("ملاحظة: النتيجة تظهر حالياً كرقم فئة (Class ID)، يمكن تطويرها لاحقاً لعرض أسماء الكائنات بالعربي.")

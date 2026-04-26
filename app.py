import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import requests

# 1. إعداد واجهة التطبيق (عربي بالكامل)
st.set_page_config(page_title="كاشف الصور الذكي", layout="centered")

# تنسيق CSS احترافي لدعم اللغة العربية وتحسين الخط
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"], .stMarkdown, .stText {
        font-family: 'Cairo', sans-serif;
        direction: rtl;
        text-align: right;
    }
    .stAlert { direction: rtl; text-align: right; }
    div[data-testid="stFileUploader"] section { text-align: center; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 مستكشف الصور الذكي")
st.write("ارفع صورة أي شيء وسأخبرك ما هو باللغة العربية.")

# 2. قاموس ترجمة ذكي وشامل (عربي بحت)
AR_DICT = {
    "book": "كتاب", "notebook": "دفتر", "comic": "قصة مصورة",
    "couch": "أريكة / كنبة", "sofa": "كنبة", "desk": "مكتب",
    "laptop": "كمبيوتر محمول", "mouse": "فأرة حاسوب", "keyboard": "لوحة مفاتيح",
    "phone": "هاتف جوال", "screen": "شاشة", "monitor": "شاشة حاسوب",
    "cat": "قطة", "dog": "كلب", "golden retriever": "كلب جولدن",
    "coffee": "قهوة", "mug": "كوب", "cup": "فنجان", "pizza": "بيتزا",
    "water bottle": "قارورة ماء", "pen": "قلم", "pencil": "قلم رصاص",
    "remote": "جهاز تحكم (ريموت)", "spectacles": "نظارات", "glasses": "نظارات",
    "clock": "ساعة", "watch": "ساعة يد", "car": "سيارة", "bicycle": "دراجة"
}

def translate_to_arabic(en_label):
    en_label = en_label.lower().replace('_', ' ')
    # البحث عن الكلمة في القاموس
    for key, val in AR_DICT.items():
        if key in en_label:
            return val
    # إذا لم توجد في القاموس، نحاول تنظيف الكلمة وتقديمها
    return f"شيء يشبه ({en_label})"

# 3. تحميل الفئات والنموذج
@st.cache_data
def load_labels():
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    return requests.get(url).json()

@st.cache_resource
def load_ai_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

labels = load_labels()
model = load_ai_model()

# 4. تحضير الصورة (المعايير العالمية للدقة)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 5. رفع الصورة والتحليل
uploaded_file = st.file_uploader("اضغط هنا لرفع صورة من جهازك", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="الصورة التي تم رفعها", use_container_width=True)
    
    with st.spinner('جاري التحليل بالذكاء الاصطناعي...'):
        input_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            index = predicted.item()
            
        result_en = labels[index]
        result_ar = translate_to_arabic(result_en)
        
    # عرض النتيجة بالعربي فقط وبشكل بارز
    st.markdown(f"""
        <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; border: 1px solid #c3e6cb;">
            <h2 style="color: #155724; text-align: center; margin: 0;">النتيجة: {result_ar}</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("") # مساحة فارغة
    st.info("💡 نصيحة: للحصول على أفضل دقة، استخدم صوراً حقيقية وواضحة.")

else:
    st.warning("بانتظار رفع صورة للبدء في تحليلها...")

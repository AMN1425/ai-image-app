import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import json

st.set_page_config(page_title="🤖 التعرّف على الصور (AI)", layout="center")

st.title("🤖 التعرّف على الصور (AI)")
st.caption("ارفع صورة وسيعطيك التطبيق أقرب فئة (مع ترجمة عربية تقريبة).")

# -----------------------------
# النموذج والتحويلات
# -----------------------------
model = models.resnet50(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

uploaded_file = st.file_uploader("ارفع صورة (JPG / PNG)", type=["jpg", "png"])

# -----------------------------
# أسماء الفئات (ImageNet)
# -----------------------------
# نحاول نجيب أسماء الفئات من مصدر محلي داخل الكود.
# لو فشل التحميل، بنرجع نعطي اسم إنجليزي بدون ترجمة.
def load_imagenet_labels():
    # بديل بسيط: محاولة قراءة labels من ملف لو موجود
    # (ممكن ما يكون عندك—وقتها نعرض index فقط)
    try:
        with open("imagenet_labels.json", "r", encoding="utf-8") as f:
            labels = json.load(f)
            # labels المفروض تكون list بطول 1000 أو dict index->name
            if isinstance(labels, list) and len(labels) == 1000:
                return labels
            if isinstance(labels, dict):
                # تحويل dict إلى list
                out = [""] * 1000
                for k, v in labels.items():
                    out[int(k)] = v
                return out
    except Exception:
        pass
    return None

labels = load_imagenet_labels()

# ترجمة عربية تقريبة (لن تغطي كل 1000 فئة حرفيًا، لكن تعطي مظهر عربي)
# الفكرة: إذا ما كانت التسمية موجودة نقول: "فئة غير محددة".
AR_TRANSLATIONS = {
    "tench": "سمكة التنش",
    "goldfish": "سمكة ذهبية",
    "great white shark": "قرش أبيض عظيم",
    "tiger shark": "قرش نمر",
    "hammerhead shark": "قرش المطرقة",
    "electric ray": "قرش/سمكة كهربائية",
    "stingray": "راي لدغ/شفنين",
    "cock": "ديك",
    "hen": "دجاجة",
    "ostrich": "نعامة",
    "brambling": "عصفور صغير (بَرامبِلنج)",
    "goldfinch": "طائر ذهبية (جولدفنچ)",
    "house finch": "عصفور منزل (هاوس فنش)",
    "gray wolf": "ذئب رمادي",
    "red fox": "ثعلب أحمر",
    "wildebeest": "حيوان وحشي (ويليدبِست)",
    "cat": "قط",
    "dog": "كلب",
    "horse": "حصان",
    "sheep": "خروف",
    "cow": "بقرة",
    "elephant": "فيل",
    "bear": "دب",
    "zebra": "حمار مخطط (زِبرا)",
    "giraffe": "زرافة",
    "backpack": "حقيبة ظهر",
    "umbrella": "مظلة",
    "handbag": "شنطة يد",
    "suitcase": "شنطة سفر",
    "tie": "ربطة عنق",
    "shirt": "قميص",
    "sneaker": "حذاء رياضي",
    "sandal": "صندل",
    "bow tie": "فراشة/ربطة رسمية",
    "scarf": "وشاح",
    "book": "كتاب",
    "clock": "ساعة",
    "keyboard": "لوحة مفاتيح",
    "mouse": "فأرة (حاسوب)",
    "remote control": "جهاز تحكم عن بعد",
    "cell phone": "هاتف خلوي",
    "laptop": "حاسوب محمول",
    "microwave": "ميكروويف",
    "oven": "فرن",
    "toaster": "محماة/توستر",
    "refrigerator": "ثلاجة",
    "chair": "كرسي",
    "sofa": "كنبة",
    "bed": "سرير",
    "dining table": "طاولة طعام",
    "toilet": "مرحاض",
    "tv": "تلفاز",
}

def translate_ar(label_en: str) -> str:
    if not label_en:
        return "—"
    key = label_en.strip().lower()
    return AR_TRANSLATIONS.get(key, f"فئة: {label_en}")

# -----------------------------
# المعالجة + العرض
# -----------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="الصورة المُرفوعة", use_container_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = outputs.max(1)

    idx = predicted.item()

    st.subheader("النتيجة")
    if labels is not None:
        label_en = labels[idx]
        label_ar = translate_ar(label_en)
        st.success(f"✅ التوقع: {label_ar}")
        st.info(f"English: {label_en} (index: {idx})")
    else:
        st.warning(f"⚠️ حصلت نتيجة لكن ما عندي أسماء الفئات. index = {idx}")
        st.success(f"✅ التوقع (تقريبي بالعربي): فئة غير محددة (index: {idx})")

import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms

st.title("🤖 AI Image Recognition")

model = models.resnet50(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = outputs.max(1)

    st.success(f"Prediction: {predicted.item()}")

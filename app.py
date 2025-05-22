import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from models.unet import UNet

st.set_page_config(page_title="Eco-Smart Route Predictor", layout="wide")
st.title("🌱 Eco-Smart Route Predictor")
st.markdown("Upload a satellite image to detect eco-friendly routes using AI-powered segmentation.")

uploaded_file = st.file_uploader("Upload a satellite image (JPG, PNG, TIF)", type=["jpg", "png", "tif"])

@st.cache_resource
def load_model():
    model = UNet(in_channels=3, out_channels=1)
    # In real scenario, load state_dict from trained model
    model.eval()
    return model

def preprocess_image(image):
    image = image.resize((256, 256))
    img_array = np.array(image).astype(np.float32) / 255.0
    if img_array.ndim == 2:
        img_array = np.stack([img_array]*3, axis=-1)
    img_tensor = torch.tensor(img_array.transpose(2, 0, 1)).unsqueeze(0)
    return img_tensor

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Segment Eco-Friendly Route"):
        with st.spinner("Segmenting image..."):
            model = load_model()
            input_tensor = preprocess_image(image)
            with torch.no_grad():
                output = model(input_tensor)[0][0].numpy()

        st.subheader("🧭 Segmentation Output (Simulated)")
        fig, ax = plt.subplots()
        ax.imshow(output, cmap='Greens')
        ax.axis('off')
        st.pyplot(fig)

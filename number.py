import streamlit as st
from PIL import Image
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier

# TRAIN MODEL 
@st.cache_resource
def load_model():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist["data"][:10000]
    y = mnist["target"][:10000].astype(np.uint8)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler

model, scaler = load_model()

# streamlit
st.title("Handwritten Digit Recognition")

uploaded_file = st.file_uploader("Upload an image of a digit", type=["jpg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")

    # preprocessing
    img_array = np.array(img)
    img_array = 255 - img_array
    img_array = (img_array > 100) * 255

    # crop
    coords = np.argwhere(img_array > 0)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    img_array = img_array[y_min:y_max+1, x_min:x_max+1]

    # resize
    img = Image.fromarray(img_array.astype(np.uint8))
    img = img.resize((28, 28))
    img_array = np.array(img)

    # flatten + scale
    img_flat = img_array.reshape(1, -1)
    img_scaled = scaler.transform(img_flat)

    # predict
    prediction = model.predict(img_scaled)

    st.image(img, caption="Processed Image", width=150)
    st.write(f"### Predicted digit: {prediction[0]}")
    

import os
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import psutil
import requests
from PIL import Image

# Disable GPU and suppress TensorFlow logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Monitor memory usage
#st.sidebar.write(f"Memory Usage: {psutil.virtual_memory().percent}%")

# Hugging Face API Configuration


API_TOKEN = st.secrets["default"]["API_TOKEN"]

headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/e1a6050"  # Replace <your-model-id>

def preprocess(text):
    # Example preprocessing function
    return text.strip()

def query(payload):
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise error for HTTP codes >= 400
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error querying API: {e}")
        return None

# Placeholder function for verdict message
def print_verdict_message(message):
    st.write(f"API Verdict: {message}")

# Load and preprocess the image
def model_predict(image_path, model):
    img = cv2.imread(image_path)  # Read the file and convert into array
    H, W, C = 224, 224, 3
    img = cv2.resize(img, (H, W))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0  # Rescaling
    img = img.reshape(1, H, W, C)  # Reshaping
    prediction = np.argmax(model.predict(img), axis=-1)[0]
    return prediction

# Sidebar
st.sidebar.title("🌱 Plant Disease Detection System")
st.sidebar.markdown("<p style='font-size: 16px;'>Empowering Sustainable Agriculture</p>", unsafe_allow_html=True)

app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center; color: green;'>Plant Disease Detection System 🌿</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>Leveraging AI to revolutionize agriculture with early disease detection.</p>", unsafe_allow_html=True)
    img = Image.open("pic.jpg")
    st.image(img, use_container_width=True)

elif app_mode == "DISEASE RECOGNITION":
    st.header("🌾 Plant Disease Detection 🌾")
    test_image = st.file_uploader("📂 Upload an Image:", type=["jpg", "jpeg", "png"])
    
    if test_image:
        # Define the save path
        save_path = os.path.join(os.getcwd(), test_image.name)
        with open(save_path, "wb") as f:
            f.write(test_image.getbuffer())

        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)

        # Predict button
        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction")

            # Add your model loading logic here
            try:
                # Replace with your actual TensorFlow model path
                model = tf.keras.models.load_model("best_vgg16_plant_disease_model.keras")
                result_index = model_predict(save_path, model)

                class_name = ['Apple scab', 'Apple Black rot', 'Apple rust', 'Apple healthy',
                              'Blueberry healthy', 'Cherry Powdery mildew', 'Cherry healthy',
                              'Corn Cercospora leaf spot Gray leaf spot', 'Corn Common rust',
                              'Corn Northern Leaf Blight', 'Corn healthy', 'Grape Black rot',
                              'Grape Esca', 'Grape Leaf blight', 'Grape healthy', 'Orange Haunglongbing',
                              'Peach Bacterial spot', 'Peach healthy', 'Pepper bell Bacterial spot',
                              'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight',
                              'Potato healthy', 'Raspberry healthy', 'Soybean healthy',
                              'Squash Powdery mildew', 'Strawberry Leaf scorch', 'Strawberry healthy',
                              'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight',
                              'Tomato Leaf Mold', 'Tomato Septoria leaf spot', 'Tomato Spidermites Two-spotted spider mite',
                              'Tomato Target Spot', 'Tomato Tomato Yellow Leaf Curl Virus',
                              'Tomato Tomato mosaic virus', 'Tomato healthy']
                
                st.success(f"Model is Predicting it's a {class_name[result_index]}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

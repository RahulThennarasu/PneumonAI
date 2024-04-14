import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import base64

# Function to set background image
def set_background(image_file, width, height):
    try:
        with open(image_file, "rb") as f:
            img_data = f.read()
        b64_encoded = base64.b64encode(img_data).decode()
        style = f"""
            <style>
            .stApp {{
                background-image: url('data:image/png;base64,{b64_encoded}');
                background-size: {width}px {height}px;  /* Adjust background size */
                background-repeat: no-repeat;
                background-position: center;
            }}
            </style>
        """
        st.markdown(style, unsafe_allow_html=True)
    except FileNotFoundError:
        st.write(f"Error: Background image file '{image_file}' not found.")
    except Exception as e:
        st.write(f"Error: An unexpected error occurred - {e}")

# Function to classify image
def classify(image, model, class_names):
    # Resize image to (224, 224)
    image_resized = ImageOps.fit(image, (224, 224), Image.LANCZOS)

    # Convert image to numpy array
    image_array = np.asarray(image_resized)

    # Normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Set model input
    data = np.expand_dims(normalized_image_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(data)
    index = 0 if prediction[0][0] > 0.95 else 1
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

def main():
    # Set title and header
    st.title('PneumonAI')
    st.header('Upload Image of Chest X-ray')

    # Upload file
    file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

    # Load classifier model and class names
    model = load_model('./model/pneumonia_classifier.h5')
    with open('./model/labels.txt', 'r') as f:
        class_names = [a[:-1].split(' ')[1] for a in f.readlines()]

    # Display background image (modify path and dimensions as needed)
    set_background('./images/logo.png', width=800, height=200)

    # Display uploaded image and perform classification
    if file is not None:
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)

        # Classify image
        class_name, conf_score = classify(image, model, class_names)

        # Display classification result
        st.write("## Predicted Class: {}".format(class_name))
        st.write("### Confidence Score: {}".format(conf_score))

if __name__ == '__main__':
    main()

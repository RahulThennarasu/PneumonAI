import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import base64
import matplotlib.pyplot as plt

# Function to set background image and overall styling
def set_background(image_file, width, height):
    try:
        with open(image_file, "rb") as f:
            img_data = f.read()
        b64_encoded = base64.b64encode(img_data).decode()
        style = f"""
            <style>
            body {{
                background-image: url('data:image/png;base64,{b64_encoded}');
                background-size: cover;
                font-family: 'Arial', sans-serif;
                color: #333;  /* Text color */
            }}
            .stApp {{
                max-width: 800px;  /* Limit content width */
                margin: auto;  /* Center content */
                padding: 20px;  /* Add spacing around content */
                background-color: rgba(255, 255, 255, 0.9);  /* Semi-transparent background for content */
                border-radius: 10px;  /* Rounded corners for content */
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);  /* Box shadow for container */
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
    st.subheader('Upload Image of Chest X-ray')

    # Upload file
    file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

    # Load classifier model and class names
    model = load_model('./model/pneumonia_classifier.h5')
    with open('./model/labels.txt', 'r') as f:
        class_names = [a[:-1].split(' ')[1] for a in f.readlines()]

    # Display background image and overall styling
    set_background('./images/logo.png', width=1200, height=800)

    # Display uploaded image and perform classification
    if file is not None:
        image = Image.open(file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Classify image
        class_name, conf_score = classify(image, model, class_names)

        # Display classification result
        st.markdown(f"**Predicted Class:** {class_name}")
        st.markdown(f"**Confidence Score:** {conf_score:.2f}")

        # Visualize confidence scores with a horizontal bar chart
        fig, ax = plt.subplots()
        ax.barh(class_names, [conf_score, 1 - conf_score], color=['skyblue', 'lightgray'])
        ax.set_xlabel('Confidence Score')
        ax.set_title('Prediction Confidence')
        st.pyplot(fig)

        # Display health risk information related to smoking
        st.subheader("Health Risks of Smoking")
        st.markdown("""
            Smoking damages the respiratory system and weakens the immune system, increasing the risk of 
            respiratory infections such as pneumonia. The harmful chemicals in tobacco smoke impair the 
            ability of the lungs to fight off infections, leading to more severe pneumonia and longer recovery 
            times. Quitting smoking can significantly reduce the risk of pneumonia and improve lung health.
        """)

if __name__ == '__main__':
    main()

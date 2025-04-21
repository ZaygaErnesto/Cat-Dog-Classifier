import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure Tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

def load_model():
    """Load the saved Keras model"""
    return tf.keras.models.load_model('intermediate_amcc.keras')

def preprocess_image(image):
    """
    Preprocess the image for model prediction
    Args:
        image: PIL Image object
    Returns:
        Preprocessed image as numpy array
    """
    # Resize image to expected dimensions
    target_size = (64, 64)  # Adjust these dimensions to match your model's expected input
    image = image.resize(target_size)
    
    # Convert to array and preprocess
    img_array = np.array(image)
    
    # Expand dimensions to create batch
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize pixel values
    img_array = img_array.astype('float32') / 255.0
    
    return img_array

def interpret_prediction(prediction):
    """
    Interpret the model's prediction output
    Args:
        prediction: Model prediction array
    Returns:
        predicted_class: Index of predicted class
        confidence_scores: List of confidence scores
    """
    # Check if prediction is a single number (binary classification)
    if prediction.shape[-1] == 1:
        # Binary classification case
        score = prediction[0][0]
        predicted_class = 0 if score >= 0.5 else 1
        confidence_scores = [score, 1 - score, 0]  # [cat_score, dog_score, other_score]
    else:
        # Multi-class classification case
        confidence_scores = prediction[0]
        predicted_class = np.argmax(confidence_scores)
    
    return predicted_class, confidence_scores

def predict(model, image):
    """
    Make prediction using the model
    Args:
        model: Loaded tensorflow model
        image: Preprocessed image array
    Returns:
        Prediction array
    """
    return model.predict(image, batch_size=1)

def create_sidebar():
    """Create sidebar with information about the app"""
    with st.sidebar:
        st.header("About")
        st.write("""
        This app classifies images of pets into three categories:
        - 🐱 Cats
        - 🐶 Dogs
        - 🤔 Other
        
        Upload your image and click 'Classify' to see the results!
        """)
        
        st.header("Instructions")
        st.write("""
        1. Use the file uploader to select an image
        2. Wait for the image to load
        3. Click the 'Classify Image' button
        4. View the classification result
        """)

def main():
    # Page configuration
    st.set_page_config(
        page_title="Pet Image Classifier",
        page_icon="🐾",
        layout="centered"
    )
    
    # Create sidebar
    create_sidebar()
    
    # Main content
    st.title("🐱 Pet Image Classifier 🐶")
    st.write("Upload an image to check if it's a cat, dog, or something else!")

    # Load model
    try:
        model = load_model()
        # Display model summary for debugging
        st.sidebar.write("Model Output Shape:", model.output_shape)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        try:
            # Create columns for layout
            col1, col2 = st.columns([2, 1])
            
            # Display uploaded image
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)

            # Add classification button and results
            with col2:
                if st.button('Classify Image', use_container_width=True):
                    with st.spinner('Classifying image...'):
                        # Preprocess image
                        processed_image = preprocess_image(image)
                        
                        # Make prediction
                        prediction = predict(model, processed_image)
                        
                        # Interpret prediction
                        predicted_class, confidence_scores = interpret_prediction(prediction)
                        
                        # Get class name
                        class_names = ['dog', 'cat', 'other']
                        class_emojis = {'cat': '🐱', 'dog': '🐶', 'other': '❓'}
                        result = class_names[predicted_class]
                        
                        # Display result with emoji
                        st.success(f"Prediction: {class_emojis[result]} {result.capitalize()}")
                        
                        # Display confidence scores
                        st.write("Confidence Scores:")
                        for class_name, score in zip(class_names, confidence_scores):
                            st.progress(float(score))
                            st.write(f"{class_name.capitalize()}: {score:.2%}")
                            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please make sure you uploaded a valid image file.")
            # Print full error details in debug mode
            st.write("Debug - Error details:", str(e))

if __name__ == '__main__':
    main()
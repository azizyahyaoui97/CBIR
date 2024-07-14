import streamlit as st
import os
import numpy as np
from descriptor import glcm, bitdesc
from app_distance import calculate_similarity

# Load signatures based on selected descriptor
def load_signatures(desc_type):
    if desc_type == 'GLCM':
        return np.load('signatures_glcm.npy', allow_pickle=True)
    elif desc_type == 'BiT':
        return np.load('signatures_bit.npy', allow_pickle=True)
    else:
        return None

# Configuration of the Streamlit app
st.title('Content-Based Image Retrieval')
st.write('This App retrieves images based on their content using GLCM and BiT descriptors.')

# Interface parameters
num_images = st.sidebar.number_input('Number of similar images to display', min_value=1, max_value=10, value=5)
distance_measure = st.sidebar.selectbox('Select distance measure', ['Euclidean', 'Manhattan', 'Chebyshev', 'Canberra'])
descriptor_selected = st.sidebar.selectbox('Select descriptor', ['GLCM', 'BiT'])

# Load signatures from the database
signatures = load_signatures(descriptor_selected)

# Image upload
uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'png'])

if uploaded_file is not None:
    # Save the uploaded image temporarily
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    # Extract features from the uploaded image
    features = glcm("temp_image.png") if descriptor_selected == 'GLCM' else bitdesc("temp_image.png")
    
    if len(signatures) == 0:
        st.error("No features found. Please ensure the dataset has been processed and features are stored in the appropriate signatures file.")
    else:
        # Calculate similarities
        similar_images = calculate_similarity(signatures, features, distance_measure, num_images)
        
         # Affichage des résultats
        for img_path in similar_images:
            img_abs_path = os.path.abspath(os.path.join('./Projet1_Dataset/Projet1_Dataset', img_path))
            if os.path.isfile(img_abs_path):
                st.image(img_abs_path, caption=os.path.basename(img_abs_path), use_column_width=True)
            else:
                st.warning(f"Cannot open image {img_path}. File not found.")
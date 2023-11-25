import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.sequence import pad_sequences



# Load MobileNetV2 model
inception_model = InceptionV3(weights="imagenet")
inception_model = Model(inputs=inception_model.inputs, outputs=inception_model.layers[-2].output)

# Load your trained model
model = tf.keras.models.load_model('C:/Users/fatim/Desktop/My first Folder/model_9.h5')

with open('wordtoix.pkl', 'rb') as pkl_file:
    wordtoix = pickle.load(pkl_file)

with open('ixtoword.pkl', 'rb') as pkl_file:
    ixtoword = pickle.load(pkl_file)

    
# Set custom web page title
st.set_page_config(page_title="Caption Generator App", page_icon="📷")

# Streamlit app
st.title("Image Caption Generator")
st.markdown(
    "Upload an image, and this app will generate a caption for it using a trained LSTM model."
)

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Process uploaded image
if uploaded_image is not None:
    st.subheader("Uploaded Image")
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Generated Caption")
    # Display loading spinner while processing
    with st.spinner("Generating caption..."):
        # Load image
        image = load_img(uploaded_image, target_size=(299, 299))
        image = img_to_array(image)
        image =  np.expand_dims(image, axis=0)
        image = preprocess_input(image)

    




        # Extract features using VGG16
        image_features = inception_model.predict(image, verbose=0)
        image_features = np.reshape(image_features, image_features.shape[1]) # reshape from (1, 2048) to (2048, )

        # Max caption length
        max_length = 34


        print('Description Length: %d' % max_length)
# Function to generate a description for an image
        def predict_caption(model, image_features, wordtoix,max_length):
            # Initialize the input sequence
            input_seq = 'startseq'
            # Generate the description word by word
            for i in range(max_length):
                sequence = [wordtoix[word] for word in input_seq.split() if word in wordtoix]
                sequence = pad_sequences([sequence], maxlen=max_length)

                # Predict the next word
                image_features = image_features.reshape((1,2048))
                yhat = model.predict([image_features, sequence], verbose=0)
                yhat = np.argmax(yhat)
                
                # Convert the index to a word
                word = ixtoword[yhat]

                # Break if we predict the end of the sequence
                if word == 'endseq':
                    break
                # Append the predicted word to the input sequence
                input_seq += ' ' + word

            return input_seq.replace('startseq', '').replace('endseq', '').strip()
        
        generated_caption = predict_caption(model,image_features, wordtoix, max_length)
    # Display the generated caption with custom styling
    st.markdown(
        f'<div style="border-left: 6px solid #ccc; padding: 5px 20px; margin-top: 20px;">'
        f'<p style="font-style: italic;">“{generated_caption}”</p>'
        f'</div>',
        unsafe_allow_html=True
    )
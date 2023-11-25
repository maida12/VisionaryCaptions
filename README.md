# VisionaryCaptions
The web Application To generate the image captions

# Image Caption Generator
## Overview
This project involves the development of an image caption generator using the Flickr8k dataset. The model combines a pre-trained Inception v3 model for image feature extraction with a Long Short-Term Memory (LSTM) network for text generation.

## Data Collection and Preprocessing
Data Collection
I utilized the Flickr8k dataset for image captioning. This dataset contains images along with corresponding captions.

###  Data Preprocessing
#### Text Data Cleaning:

Lowercased all captions.
Removed redundant words and punctuation.
#### Image Data Cleaning:

Resized each image to 299x299 pixels.
Extracted feature vectors using the pre-trained Inception v3 model.
### Dataset Splitting:

Divided the dataset into 6,000 images for training and 1,000 images for testing.
### Data Analysis and Visualization
I performed exploratory data analysis and visualization to gain insights, including:

Maximum length of captions.
Most used words in captions.
## Deliverable 2: Model Development and Training
### Model Selection
The chosen model architecture involves using the Inception v3 model for image features and an LSTM network for caption generation.

### Model Training
#### Implementation:

Implemented the image caption generator model.
### Training:

Trained the model on the 6,000 images using the collected dataset.
### Fine-Tuning:

Experimented with different layers in the model, focusing on LSTM layers.
### Model Evaluation
Metrics
Evaluated the model using BLEU scores to measure the quality of generated captions.

### Model Deployment
Deployment
Deployed the complete model locally.

### User Interface and Integration
User Interface
Created a user-friendly interface for the AI system using Streamlit.

Integration
Integrated the AI model into the user interface for practical applications.

## How to Use
### Installation:
Use following command
```
pip install package name

```
or if you are using ANACONDA write this 
```
conda install package name 

```
### Run the Interface:

Execute the Streamlit script to run the user interface locally.
For GUI, simply run this command :
```
streamlit run main.py

```


### Generate Captions:

Upload an image to the interface to generate captions.
### Conclusion
This project successfully developed an image caption generator using a combination of pre-trained image models and LSTM networks. The user interface provides a convenient way to interact with the model and generate captions for input images.
The accuracy maybe approved by using larger dataset.

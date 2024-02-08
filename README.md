# Fire Detection from CCTV 


This project aims to detect fires from CCTV images using deep learning models. 
The implemented solution includes data preprocessing, model creation, training,
 and evaluation. Three different models are explored: a custom CNN, VGG16, and ResNet50.

Prerequisites
Kaggle API key for dataset download
Python environment with necessary dependencies (TensorFlow, Matplotlib, etc.)
Dataset
The dataset is obtained from Kaggle using the Kaggle API. It contains images 
categorized into 'fire,' 'default' (no fire), and 'smoke' classes.

Project Structure
Data Preparation

Downloading and extracting the dataset
Organizing data into training and validation sets
Data Exploration

Displaying counts of images in training and validation sets for each class
Computing class weights for imbalanced datasets
Data Augmentation

Implementing data augmentation using ImageDataGenerator
Model Architecture

Custom CNN architecture
VGG16 architecture with transfer learning
ResNet50 architecture with transfer learning
Model Training

Generating data for training and validation
Calculating class weights
Training each model using early stopping and model checkpoint callbacks
Model Evaluation

Plotting training and validation loss/accuracy over epochs for each model
Model Prediction

Loading trained models
Predicting classes for sample images and displaying results
Custom Model
Architecture

Convolutional layers with max-pooling and dropout
Global Average Pooling layer
Dense layers for classification
Training

Optimizer: Adam
Loss function: Categorical Crossentropy
Early stopping with model checkpoint callback
VGG16 Model
Architecture

VGG16 with transfer learning
Additional layers for classification
Training

Optimizer: Adam
Loss function: Categorical Crossentropy
Early stopping with model checkpoint callback
ResNet50 Model
Architecture

ResNet50 with transfer learning
Dense layer for classification
Training

Optimizer: Adam
Loss function: Categorical Crossentropy
Early stopping with model checkpoint callback
Conclusion
This documentation provides an overview of the fire detection project, 
including data preparation, model creation, training, evaluation, and predictions using custom CNN, VGG16, and ResNet50 models.


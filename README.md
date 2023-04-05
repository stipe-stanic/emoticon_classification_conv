# Emoticon Classification Model using Tensorflow and Keras
- This repository contains the code for a deep learning model that classifies emoticons using TensorFlow and Keras. The model is built using a Convolutional Neural Network (CNN) architecture and is trained on a dataset of emoticon images.

## Dataset
- The dataset consists of emoticon images in PNG format, with each image representing a different emoticon. The images are preprocessed and resized to a fixed size of 120x120 pixels using the rescaling layer from Keras. The dataset is split into training and test sets with a ratio of 80:20.

- <img alt="Training dataset" height=50% src="/plots_img/training_data_00.png" width=50%/>

## Model architecture
- The model architecture consists of augmentation layer, three convolutional layers, each followed by a max pooling layer. The output from the final max pooling layer is flattened and passed through two fully connected (dense) layers, with a final output layer that predicts the class of the emoticon. The model is compiled with the categorical cross-entropy loss function and the Adam optimizer.

## Model metrics
<img alt="Training val and acc" height=50% src="/plots_img/model_metrics_01.png" width=50%/>

## Evaluation
<img alt="Training val and acc" height=50% src="/plots_img/confusion_matrix_00_model_01.png" width=50%/>
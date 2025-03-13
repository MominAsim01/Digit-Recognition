# Digits Recognition using CNN

## Overview
This project implements a Convolutional Neural Network (CNN) for digit recognition using the MNIST dataset. The model is trained to classify handwritten digits (0-9) accurately.

## Features
- Uses Convolutional Neural Networks (CNN) for efficient digit classification.
- Trained on the MNIST dataset, a widely used dataset for handwritten digit recognition.
- Includes data preprocessing, model building, training, and evaluation.
- Implements key deep learning concepts like convolutional layers, pooling layers, and fully connected layers.

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install tensorflow keras numpy matplotlib
```

## Dataset
The project uses the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits.

## Usage
1. Clone the repository:

```bash
git clone https://github.com/yourusername/digits-recognition-cnn.git
cd digits-recognition-cnn
```

2. Run the Jupyter Notebook:

```bash
jupyter notebook digits_recognition_cnn_hands_on.ipynb
```

3. Follow the steps in the notebook to train and evaluate the model.

## Model Architecture
The CNN model consists of:
- **Convolutional layers** to extract features from images.
- **Pooling layers** to reduce dimensionality.
- **Fully connected layers** to classify digits.
- **Softmax activation** for multi-class classification.

## Evaluation Metrics
- **Accuracy**: Measures how well the model predicts digits.
- **Loss Function**: Categorical Crossentropy used for optimization.

## Results
The trained model achieves a high accuracy on the MNIST dataset, demonstrating its effectiveness in digit recognition.

## Contributions
Feel free to contribute by submitting pull requests or reporting issues.




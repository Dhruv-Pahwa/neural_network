# Neural Network from Scratch using NumPy

## Overview
This project demonstrates how to build a neural network from the ground up using only NumPy, without relying on high-level machine learning frameworks like TensorFlow or Keras. The focus is on understanding the fundamental mathematical concepts behind neural networks, such as forward propagation, backpropagation, and weight optimization.

## Dataset
The model is trained on the **MNIST dataset**, which consists of 28x28 grayscale images of handwritten digits (0-9). This dataset is commonly used for benchmarking image classification algorithms.

## Neural Network Architecture
The network consists of:
- **Input Layer**: 784 neurons (corresponding to 28x28 flattened pixel values)
- **Hidden Layer**: 128 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation (representing probabilities for each digit)

## Key Features
- **Implemented from Scratch**: The entire neural network, including weight initialization, activation functions, forward propagation, and backpropagation, is implemented using NumPy.
- **Activation Functions**: Uses ReLU for non-linearity and Softmax for output probability distribution.
- **Gradient Descent Optimization**: Weights and biases are updated iteratively to minimize error.
- **Training & Validation**: The dataset is split into training and validation sets to assess model performance and prevent overfitting.

## Installation
To run this project, ensure you have Python and NumPy installed:

```bash
pip install numpy
```

## Usage
Clone the repository and navigate to the project folder:

```bash
git clone https://github.com/yourusername/neural-net-numpy.git
cd neural-net-numpy
```

Run the script to train the neural network:

```bash
python train.py
```

## Results
After training, the model achieves a validation accuracy of approximately **82.7%**, showcasing its effectiveness despite being implemented without deep learning libraries.

## Insights & Learnings
- Understanding the **mathematical foundations** of neural networks is crucial before using high-level frameworks.
- Proper **data preprocessing** and weight initialization significantly impact model performance.
- Debugging and optimizing a neural network require careful attention to **gradient updates** and activation functions.

## Future Improvements
- Implementing additional hidden layers for deeper learning.
- Experimenting with different weight initialization techniques.
- Optimizing training with adaptive learning rate strategies (e.g., Adam optimizer).

## Contributing
If you'd like to contribute, feel free to fork the repository and submit a pull request with your enhancements.

## License
This project is open-source and available under the MIT License.


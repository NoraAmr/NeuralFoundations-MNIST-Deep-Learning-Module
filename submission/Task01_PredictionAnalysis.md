# Task01_PredictionAnalysis

## 1. Objective

Explain how the forward pass works in the neural network, the role of activation functions, and the optimizer used.

## 2. Code Used

```python
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

## 3. Forward Pass Explanation

The forward pass moves the input image through the neural network layers.

* The **Flatten layer** converts the 2D image into a 1D vector.
* The **Dense (hidden) layer** applies a weighted sum followed by an activation function.
* The **output layer** returns probabilities for each class, and the class with the highest probability is selected as the prediction.

## 4. Short Analysis

The forward pass describes how input data flows through the neural network to generate an output.

* The process begins at the input layer, where the model receives a 2D image.
* The Flatten layer reshapes the image into a 1D vector so it can be processed by dense layers.
* Each Dense layer computes a weighted sum of its inputs and applies an activation function.

The **ReLU (Rectified Linear Unit)** activation function is used in the hidden layer. It keeps positive values and sets negative values to zero, which helps reduce the vanishing gradient problem and improves learning efficiency.

The **Softmax** activation function is used in the output layer to convert raw scores into probabilities. This is suitable for multiclass classification tasks, such as recognizing handwritten digits.

The **Adam optimizer** is used to update the model weights during training. It combines momentum and adaptive learning rates, allowing faster and more stable convergence.

## 5. Key Takeaway

The forward pass demonstrates how the model processes an input image step by step, applies activation functions, and produces class probabilities while relying on the Adam optimizer to optimize learning.

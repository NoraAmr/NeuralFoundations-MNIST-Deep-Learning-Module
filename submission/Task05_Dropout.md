# Task05_Dropout

## 1. Objective
The objective of this task is to analyze how different **dropout rates** affect overfitting, training stability, and the model’s ability to generalize.

---

## 2. Code Used
```python
from tensorflow.keras.layers import Dropout

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    Dropout(0.1),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
## 3. Results
Dropout = 0
The model becomes too smart for its own good.

It starts to overfit.

The gap between training accuracy and validation accuracy is slightly larger than the other dropout cases.

Dropout = 0.1
The model randomly drops 10% of the neurons during training.

The gap between training accuracy and validation accuracy becomes very small and almost reaches 0%.

This is the best case, as the validation loss keeps decreasing with no signs of overfitting.

Dropout = 0.3
The validation accuracy becomes higher than the training accuracy.

This happens because dropout is removed during validation, which makes training harder but evaluation easier.

The model performs better when dropout is turned off during validation.

## 4. Short Analysis
Neuron co-dependency happens when the network trains neurons that depend on each other to work correctly. In this case, some neurons only function well if other neurons provide specific signals. This leads to overfitting, because the model is no longer learning real features, but instead learning a complex path that only exists in the training data.

Dropout solves this problem by randomly removing neurons during training. Since any neuron can disappear at any time, neurons cannot rely on each other and are forced to learn features in multiple different ways across the network. This makes the model more robust and improves its ability to generalize to unseen data.

## 5. Key Takeaway
Using a small dropout value, especially 0.1, gives the best balance between learning and generalization by reducing neuron co-dependency and preventing overfitting
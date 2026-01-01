# Task06_L2

## 1. Objective
The objective of this task is to explain how **L2 regularization** affects the loss function, weight magnitude, and the gap between training and validation performance.

---

## 2. Code Used
```python
from tensorflow.keras import regularizers

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu',
                 kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
## 3. Results
L2 regularization adds the square of the weights to the loss function.

When the sum of the weights becomes large, the total loss increases.

Comparing Different Lambda Values
At λ = 0.01, the loss starts very high (0.9991).

At λ = 0.0001, the loss starts much lower (0.4618).

To minimize the loss, the optimizer is forced to keep the weights small.

Accuracy Gap
At λ = 0.0001, the training accuracy (98.3%) and validation accuracy (97.8%) have a small gap of 0.5%.

At λ = 0.01, the training accuracy (94.6%) is lower than the validation accuracy (96.0%), creating a negative gap. This means the model is strongly regularized and generalizes very well.

<img width="612" height="455" alt="image" src="https://github.com/user-attachments/assets/e394e90d-db32-4907-9fe0-af57967a8117" />

<img width="598" height="455" alt="image" src="https://github.com/user-attachments/assets/f57dd869-e46e-42d7-a074-b36725d1992a" />

<img width="592" height="455" alt="image" src="https://github.com/user-attachments/assets/9fd0429e-d3fb-4353-a367-2405098eeb5b" />






## 4. Short Analysis
Large weights can make the model very sensitive, where a tiny change in the input causes a massive change in the output prediction. L2 regularization prevents this by penalizing large weights and keeping them within a smaller range.

Without L2 regularization, the validation loss may reach a minimum and then start creeping back up as the weights grow too large, which is a sign of overfitting. With L2 regularization, the validation loss becomes more predictable and flatter, since the weights are controlled and cannot grow freely.

## 5. Key Takeaway

L2 regularization improves generalization by keeping weights small, stabilizing validation loss, and reducing the gap between training and validation performance.

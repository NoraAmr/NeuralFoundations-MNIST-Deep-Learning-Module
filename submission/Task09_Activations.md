# Task09_Activations

## 1. Objective
The goal of this task is to compare different activation functions and understand how they affect gradient flow, learning speed, and overall model performance during training.

## 2. Code Used
```python
import tensorflow as tf

# Activation functions used
tanh = tf.keras.layers.Activation('tanh')
softsign = tf.keras.layers.Activation('softsign')
gelu = tf.keras.layers.Activation(tf.nn.gelu)
relu = tf.keras.layers.Activation('relu')
```
## 3. Results
Tanh outputs values in the range [-1, 1] and is zero-centered.

Softsign behaves similarly to tanh but changes more gradually.

GELU keeps positive values while smoothly reducing negative values.

ReLU keeps positive values and sets all negative values to zero.

Softsign and GELU showed better gradient behavior compared to tanh when inputs became large.


<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/d1faed44-ed43-469d-ae1c-0ae1735f4881" />


<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/91cb8e92-ef4c-4732-9bbb-1ce657b4dd1b" />


<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/55f82fbe-490b-4f7c-9cb2-ff109b97ceb7" />




## 4. Short Analysis
Tanh is an activation function that transfers inputs to a range between -1 and 1. Because it is zero-centered, the average output is close to zero, which helps the model learn faster. However, when the input values become very large (such as -10 or 10), the slope of tanh becomes almost zero. During training, this causes the gradient to vanish, which slows or even stops learning.

Softsign is similar to tanh but much smoother. Instead of squashing values aggressively, it does so gradually. Because of this behavior, Softsign does not kill the gradient as strongly as tanh, making training more stable.

GELU is considered one of the best activation functions because it keeps positive values mostly unchanged while smoothly pushing negative values close to zero. Unlike ReLU, it does not sharply cut off negative values but keeps a small curve. This means the gradient does not suddenly vanish, especially for positive values, and neurons remain active even on the negative side.

GELU performs very well in transformer architectures because it converges faster and avoids vanishing gradients on the positive side while keeping neurons alive on the negative side.

Even though GELU performs better, ReLU is still preferred in many MLP and CNN models. This is because ReLU is very simple and computationally efficient, allowing larger models or more data to be trained in the same amount of time. By turning all negative inputs to exactly zero, ReLU naturally shuts off some neurons. This makes the model more efficient and the learned representations more robust and less sensitive to noise.

## 5. Key Takeaway
Activation functions strongly affect gradient flow and learning stability, and while GELU provides smoother and faster convergence, ReLU remains popular due to its simplicity and efficiency.

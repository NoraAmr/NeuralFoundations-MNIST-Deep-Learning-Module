# Task 02 — Prediction Analysis

## 1. Objective

Explain why the image was recognized correctly as the number **5** based on its similarity to the dataset used to train the model.

## 2. Code Used

```python
from PIL import Image
import numpy as np
img = Image.open("/content/Screenshot 2025-12-14 202354.png").convert("L")
img = img.resize((28,28))
img_array = np.array(img) / 255.0
img_array = img_array.reshape(1, 28, 28)

plt.imshow(img_array[0], cmap="gray")
plt.show()
pred = model.predict(img_array)
predicted_class = np.argmax(pred)
print("Predicted digit:", predicted_class)
```

## 3. Results

* **Prediction output:** The image was recognized correctly as the number **5**.
![alt text](image-1.png)
## 4. Short Analysis

The image was recognized correctly as the number **5** because it has the same features as the dataset the model was trained on.

These features include:

* A **black background**.
* A **white-colored number**.
* The image was **resized to 28 × 28 pixels**, just like the dataset images.
* **Normalized pixel values**, which help the neural network process the image correctly.

Because the input image matches the training data format, the activations inside the network behave as expected, leading to a correct prediction.

Some features may make the prediction process incorrect, such as:

* Shifting the number away from the center of the image.
* Using a background color other than black.

These changes can cause the model to misinterpret the image and produce an incorrect result.

## 5. Key Takeaway

When the input image follows the same features and preprocessing steps as the training dataset, the model is more likely to make a correct prediction.

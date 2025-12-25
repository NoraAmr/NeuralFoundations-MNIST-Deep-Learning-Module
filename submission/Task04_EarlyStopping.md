# Task 04 — Early Stopping 

## 1. Objective
The objective of this task is to understand how early stopping works with different numbers of epochs and how changing the optimizer affects overfitting and model generalization.

---

## 2. Code Used
```python
from tensorflow.keras.callbacks import EarlyStopping

model10= keras.models.clone_model(model)
callbacks = [
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
]
model10.compile(
    optimizer="sgd",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```
## 3. Results
## Using Early Stopping (patience = 3)

5 epochs: Training did not stop and continued normally.

10 epochs: Training did not stop and continued normally.

20 epochs: Training stopped at epoch 8 because the validation loss started to increase after epoch 4, so the model did not reach the rest of the epochs.

<img width="576" height="432" alt="image-5" src="https://github.com/user-attachments/assets/4a6afa17-ce5f-4036-84d6-3f43373c1f2f" />


<img width="576" height="432" alt="image-6" src="https://github.com/user-attachments/assets/a6ad6a52-a6d9-4cdd-aefc-7c7ead03ecc2" />


<img width="576" height="432" alt="image-7" src="https://github.com/user-attachments/assets/d3c73c4f-9dd1-4b16-9a78-e5c56071368d" />


## Increasing Patience to 5

5 epochs: Training did not stop.

10 epochs: Training stopped at epoch 7

20 epochs: Training stopped at epoch 8.

<img width="576" height="432" alt="image-8" src="https://github.com/user-attachments/assets/e32cdc17-b334-4af2-9a7c-602c7eb08eb3" />

<img width="576" height="432" alt="image-9" src="https://github.com/user-attachments/assets/32233615-59e8-43eb-bb5b-880b181dfcf5" />


<img width="576" height="432" alt="image-10" src="https://github.com/user-attachments/assets/bab23acc-f583-46aa-9f1d-2f2ee1a2fe30" />


## Using SGD Optimizer

When using SGD with 5, 10, and 20 epochs, the training did not stop at all.

The validation loss did not increase during training.

The gap between training accuracy and validation accuracy was very small.

## 4. Short Analysis

Early stopping is used to prevent overfitting. The validation loss works like a compass that shows how well the model generalizes, while the training loss shows how the model performs on the data it has already seen.

If the training loss keeps going down but the validation loss starts to rise, this means the model is no longer learning and is instead memorizing the training data.

With the Adam optimizer, the model learns faster, which can cause overfitting when the number of epochs is large. This is why early stopping was triggered at higher epochs.

Using SGD, the optimizer moves slower, so the learning process is more stable. This is why there were no clear signs of overfitting and the difference between training and validation accuracy stayed small. However, using only 5 epochs with SGD caused underfitting, since the accuracy was low compared to using Adam.

Early stopping acts as a type of regularization by stopping the training at the best point between underfitting and overfitting.

## 5. Key Takeaway

Early stopping helps the model stop at the point of best generalization and prevents it from training too much or too little.

# Task03_Epochs

## 1. Objective
The objective of this task is to explain overfitting by observing how the training loss and validation loss change with different numbers of epochs.

---

## 2. Code Used
```python
history1= model.fit(
    x_tr,y_tr,
    epochs=5,
    batch_size=64,
    validation_data=(x_val,y_val)
history2= model2.fit(
    x_tr,y_tr,
    epochs=10,
    batch_size=64,
    validation_data=(x_val,y_val))
history3= model3.fit(
    x_tr,y_tr,
    epochs=20,
    batch_size=64,
    validation_data=(x_val,y_val))
```
## 3. Results
### Epoch 5

Training Loss: Decreased from 0.5251 to 0.0542

Validation Loss: Decreased from 0.1309 to 0.0809

The training and validation loss curves are close and both are decreasing.

### Epoch 10

Training loss keeps decreasing.

Validation loss starts to increase from epoch 7 to epoch 10.

This shows the beginning of overfitting.

<img width="576" height="432" alt="image-2" src="https://github.com/user-attachments/assets/94fd6a08-bff4-47f1-b0e0-f1078bd496aa" />


### Epoch 20

Training loss continues to decrease.

Validation loss keeps increasing after epoch 7.

There is a clear gap between training accuracy and validation accuracy.
<img width="576" height="432" alt="image-3" src="https://github.com/user-attachments/assets/edd69fb2-14a7-466a-b665-371b1cf7c97b" />


## 4. Short Analysis
Overfitting happens when the model keeps improving on the training data (memorizing the data ), but the validation performance keeps getting worse.
In epoch 5, there are no signs of overfitting because both the training loss and validation loss are decreasing and are close to each other.
In 10 epochs, the model starts to overfit, since the validation loss begins to increase while the training loss continues to decrease. This is a sign that the model is starting to memorize the data instead of learning from it.
In epoch 20, the model is clearly overfitting. The training loss keeps decreasing, but the validation loss increases, and there is a large gap between training and validation accuracy. This shows that the model is memorizing the data rather than generalizing.
The Adam optimizer helps the model learn faster and reach high accuracy in the first few epochs, but it does not stop overfitting when more epochs are added.

## 5. Key Takeaway
Overfitting increases when training for many epochs without control, even if the model uses the Adam optimizer.

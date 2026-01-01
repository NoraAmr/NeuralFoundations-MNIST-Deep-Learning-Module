# Task07_Optimizers

## 1. Objective
The goal of this task is to compare different optimization algorithms (SGD, SGD with Momentum, Adam, and AdamW) in terms of convergence speed, stability, and validation performance during training.

---

## 2. Code Used
```python
model.compile(
    optimizer=optimizer_name,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=32
)
```
## 3. Results

Adam and AdamW achieved the highest training and validation accuracy starting from epoch 1.

By epoch 5, Adam and AdamW continued to outperform SGD and SGD with Momentum in accuracy.

SGD showed very stable behavior but had the slowest convergence.

SGD with Momentum trained faster than plain SGD but showed instability in validation loss.

A noticeable jump in validation loss was observed with SGD with Momentum (from 0.0788 to 0.0835 at epoch 5), indicating overshooting.
![alt text](image-11.png)
![alt text](image-12.png)
![alt text](image-13.png)
![alt text](image-14.png)
## 4. Short Analysis

SGD is very stable but slow because it uses a single learning rate and is sensitive to local noise. When the slope is small, updates become tiny, which can cause the optimizer to move slowly or get stuck in shallow local minima.

SGD with Momentum improves speed by accumulating past gradients, behaving like a heavy ball rolling downhill. While this helps accelerate training, it can cause overshooting in narrow valleys, leading to oscillations and sudden increases in validation loss.

Adam adapts the learning rate for each parameter individually using gradient history. This allows it to move quickly in flat regions and slow down in steep ones, resulting in fast convergence. However, this adaptability can make Adam jittery and sometimes lead it to sharp minima that do not generalize well.

AdamW builds on Adam by separating weight decay from the gradient update. This encourages simpler models with smaller weights, improving generalization and stability. AdamW matches Adam’s speed while maintaining more controlled and stable validation behavior.

Overall, Adam often outperforms classical optimizers because it combines momentum and RMSProp concepts, treating each parameter independently instead of updating all weights equally.

## 5. Key Takeaway

AdamW provides the best balance between fast convergence and stable generalization, making it a strong choice compared to SGD, SGD with Momentum, and Adam.
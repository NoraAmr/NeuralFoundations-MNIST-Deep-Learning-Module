## Task 08 — Batch Size Impact Analysis

### 1. Objective
The objective of this task is to analyze how batch size influences training stability, convergence speed, and the model’s ability to generalize to unseen data.

---

### 2. Code Used
    model.fit(
        X_train, y_train,
        batch_size=32,   # changed to larger values for comparison
        epochs=10,
        validation_data=(X_val, y_val)
    )

---

### 3. Results
- Small batch sizes produced noisy and unstable loss curves.
- Large batch sizes resulted in smooth and stable loss curves.
- Large batches reached lower training loss faster.
- Small batches often showed better validation performance.

---

### 4. Short Analysis
Smaller batch sizes introduce gradient noise because each update is computed from a narrow and incomplete view of the data. This increases the variance of the gradient, making the training path look messy and unstable. However, this noise is actually beneficial. It gives the optimizer enough randomness to escape local minima and avoid getting stuck in narrow or sharp traps. In this sense, noise acts as a form of regularization and helps reduce overfitting, which improves generalization on validation data.

Large batch sizes, on the other hand, reduce gradient variance and produce very clean updates. Thanks to GPU parallelism, large batches make better use of hardware and converge faster with fewer updates per epoch. The loss curve becomes very smooth, making it easy to detect when the model stops improving.

Despite faster convergence, large batches often generalize worse. The clean gradient may cause the optimizer to fall into the nearest sharp minimum with no noise to push it out. Because outliers are averaged out, the model may miss flatter minima that usually lead to better performance on unseen data.

---

### 5. Key Takeaway
Small batch sizes improve generalization by adding helpful noise, while large batch sizes train faster but are more likely to overfit and get stuck in sharp minima.

$$ \mathbf{Custom\ PELU\ Activation\ Function\ in\ TensorFlow/Keras} $$  
### Parametric ELU, Analytic Gradients, and MNIST Classification

---

$$ \mathbf{1.\ Overview} $$

This project implements and tests a **custom activation layer** called **PELU (Parametric ELU)** in TensorFlow/Keras, and then applies it to a simple classification task using a subset of the **MNIST** dataset.

The main steps are:

- Implement a **Keras Layer** for PELU with trainable parameters \( \alpha \) and \( \beta \)  
- Verify **analytic derivatives** w.r.t. both parameters using `tf.GradientTape`  
- Integrate PELU into a fully-connected network  
- Train on MNIST and **compare performance** with a standard ReLU-based model  

---

$$ \mathbf{2.\ Custom\ PELU\ Layer} $$

### $$ \mathbf{2.1\ Definition} $$

The PELU layer is implemented as:

$$
\text{PELU}(x;\ \alpha,\ \beta) = \alpha \cdot \text{ELU}\left(\frac{x}{\beta}\right)
$$

where ELU is:

$$
\text{ELU}(z) =
\begin{cases}
z, & z \ge 0 \\
e^{z} - 1, & z < 0
\end{cases}
$$

With this definition, PELU is equivalent to:

- For \( x \ge 0 \):  
  $$
  \text{PELU}(x) = \alpha \cdot \frac{x}{\beta}
  $$
- For \( x < 0 \):  
  $$
  \text{PELU}(x) = \alpha \cdot \left(e^{x / \beta} - 1\right)
  $$

Both parameters \( \alpha \) and \( \beta \) are:

- **Trainable**  
- **Constrained to be non-negative** (`NonNeg`)  
- Initialized from a uniform distribution in \([0.1, 0.9]\)  

### $$ \mathbf{2.2\ Implementation\ Outline} $$

Key points of the implementation:

- Subclass `tf.keras.layers.Layer`
- Define `self.paramater_A` and `self.paramater_B` as trainable weights
- Use `elu` on `inputs / beta`, then scale by `alpha`

Usage example:

```python
pelu = PELU(units=1)
x_range = tf.linspace(-5.0, 5.0, 200)
y_range = pelu(x_range)
```

The resulting curve illustrates a smooth, parametric activation with shape controlled by \( \alpha \) and \( \beta \).

---

$$ \mathbf{3.\ Analytic\ Derivatives\ vs\ Automatic\ Differentiation} $$

To validate the implementation, we compute **closed-form derivatives** of PELU w.r.t. its parameters, and compare them with gradients obtained via `tf.GradientTape`.

### $$ \mathbf{3.1\ Derivative\ w.r.t.\ \alpha} $$

Given:

$$
f(x;\ \alpha,\ \beta) = \alpha \cdot \text{ELU}\left(\frac{x}{\beta}\right)
$$

The derivative w.r.t. \( \alpha \) is:

$$
\frac{\partial f}{\partial \alpha} =
\begin{cases}
\frac{x}{\beta}, & x \ge 0 \\
e^{x / \beta} - 1, & x < 0
\end{cases}
$$

This is implemented as a piecewise function using boolean masks, and then compared to:

```python
with tf.GradientTape() as tape:
    y = pelu(x_range)
dy_dalpha = tape.jacobian(y, pelu.paramater_A)
```

We verify:

```python
tf.reduce_all(tf.abs(analytic - dy_dalpha) < 1e-4)
```

which returns `True` up to numerical precision.

---

### $$ \mathbf{3.2\ Derivative\ w.r.t.\ \beta} $$

For \( x \ge 0 \):

$$
f(x) = \alpha \cdot \frac{x}{\beta}
\quad\Rightarrow\quad
\frac{\partial f}{\partial \beta}
= -\alpha \cdot \frac{x}{\beta^2}
$$

For \( x < 0 \):

$$
f(x) = \alpha \left(e^{x / \beta} - 1\right)
\Rightarrow
\frac{\partial f}{\partial \beta}
= -\alpha \cdot \frac{x}{\beta^2} \cdot e^{x / \beta}
$$

So overall:

$$
\frac{\partial f}{\partial \beta} =
\begin{cases}
-\alpha \dfrac{x}{\beta^2}, & x \ge 0 \\
-\alpha \dfrac{x}{\beta^2} e^{x / \beta}, & x < 0
\end{cases}
$$

This is implemented in `der_pelu_b(...)` and compared to:

```python
with tf.GradientTape() as tape:
    y = pelu(x_range)
dy_dbeta = tape.jacobian(y, pelu.paramater_B)
```

Again, we check:

```python
tf.reduce_all(tf.abs(analytic - dy_dbeta) < 1e-4)
```

which confirms correctness of the gradient implementation.

---

$$ \mathbf{4.\ Applying\ PELU\ to\ MNIST\ Classification} $$

### $$ \mathbf{4.1\ Dataset\ and\ Preprocessing} $$

We use the sample MNIST CSVs from:

- `mnist_train_small.csv`  
- `mnist_test.csv`  

Steps:

- Split into features and labels:
  - \( X \): pixel values (columns 1:)  
  - \( y \): digit labels (column 0)  
- Convert to `float32`
- Standardize features with `StandardScaler`
- Wrap into `tf.data.Dataset` objects with batch size 32

---

### $$ \mathbf{4.2\ Model\ Architecture\ with\ PELU} $$

We build a simple feed-forward network:

```python
model = tf.keras.Sequential(layers=[
    tf.keras.layers.Dense(50),
    PELU(50),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

Training setup:

- Loss: `SparseCategoricalCrossentropy`
- Metric: `SparseCategoricalAccuracy`
- Optimizer: SGD with learning rate \( 10^{-3} \)
- Training:
  ```python
  model.compile(optimizer=sgd,
                loss=cross_entropy,
                metrics=[acc])

  model.fit(train_dataset, epochs=25, validation_data=test_dataset)
  ```

The PELU activation learns its own shape during training via \( \alpha \) and \( \beta \), potentially offering more flexibility than a fixed nonlinearity.

---

$$ \mathbf{5.\ Comparison\ with\ ReLU} $$

To compare PELU with a standard activation, we can train a **baseline model**:

```python
baseline_model = tf.keras.Sequential(layers=[
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

Under the same:

- Optimizer  
- Learning rate  
- Batch size  
- Number of epochs  

we can compare:

- Training and validation accuracy  
- Convergence speed  
- Stability of loss curves  

### **Typical observations:**

- ReLU is strong and simple, but fixed  
- PELU adds **learnable shape parameters**, which:
  - Can adjust slopes for positive and negative inputs  
  - May capture more nuanced feature transformations  
  - Sometimes improves performance or convergence smoothness  

---

$$ \mathbf{6.\ Summary} $$

This project demonstrates:

- How to **implement a custom Keras layer** with trainable parameters and constraints  
- How to **derive and validate analytic gradients** using `tf.GradientTape`  
- How to **integrate custom activations** into a standard training workflow  
- How to **benchmark** a custom activation (PELU) against classic ReLU on MNIST  

PELU provides a flexible, learnable activation that generalizes ELU and can adapt its shape to the data, potentially improving model expressiveness and performance.

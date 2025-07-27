Read file: logistic_regression.py
Here’s a step-by-step explanation of what happens in `logistic_regression.py` using the Iris dataset:

1. **Load the Iris dataset**  
   - The code reads the data from `iris/iris.data` using pandas, with no header row.
   - It assigns column names: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`, and `class`.

2. **Convert to a binary classification problem**  
   - The original Iris dataset has three classes. For simplicity, the code creates a binary label:  
     - `1` if the class is `Iris-setosa`
     - `0` otherwise (not-Setosa)

3. **Select features and labels**  
   - `X` contains all four features (sepal and petal measurements).
   - `y` contains the binary labels, reshaped as a column vector.

4. **Feature normalization**  
   - Each feature in `X` is normalized to have mean 0 and standard deviation 1.  
   - This helps gradient descent converge faster and prevents features with larger scales from dominating.

5. **Add bias term**  
   - A column of ones is added to `X` as the first column.  
   - This allows the model to learn an intercept (bias) term.

6. **Sigmoid function**  
   - The sigmoid function is defined:  
     \( \sigma(z) = \frac{1}{1 + e^{-z}} \)  
   - It maps any real value to a probability between 0 and 1.

7. **Logistic regression training (gradient descent)**  
   - The function `train_logistic_regression` initializes weights to zeros.
   - For a set number of epochs (iterations):
     - Computes the linear combination of inputs and weights.
     - Applies the sigmoid to get predicted probabilities.
     - Calculates the gradient of the loss with respect to the weights.
     - Updates the weights using the learning rate (`lr`).

8. **Train the model**  
   - The model is trained on the normalized data for 1000 epochs with a learning rate of 0.1.

9. **Prediction function**  
   - The `predict` function computes probabilities for each sample.
   - It assigns a label of 1 if the probability is at least 0.5, otherwise 0.

10. **Evaluate accuracy**  
    - The code compares predictions to the true labels and prints the accuracy as a percentage.

If you want this explanation in a more formal or detailed format for your `iris_logistic_regression_readme.txt`, let me know!
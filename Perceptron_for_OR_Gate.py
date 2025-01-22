import numpy as np

# Define the step activation function
def step_function(x):
    return 1 if x >= 0 else 0

# Perceptron training function
def train_perceptron(X, y, lr=0.1, epochs=10):
    # Initialize weights and bias
    weights = np.zeros(X.shape[1])
    bias = 0
    
    # Training process
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        for i in range(len(X)):
            # Compute weighted sum
            linear_output = np.dot(weights, X[i]) + bias
            # Apply activation function
            y_pred = step_function(linear_output)
            # Update weights and bias if prediction is incorrect
            error = y[i] - y_pred
            if error != 0:
                weights += lr * error * X[i]
                bias += lr * error
            print(f"Input: {X[i]}, Predicted: {y_pred}, True: {y[i]}, Weights: {weights}, Bias: {bias}")
        print("-" * 50)
    
    return weights, bias

# Inputs and outputs for the OR gate
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 1])

# Train the perceptron
learning_rate = 0.1
epochs = 10
weights, bias = train_perceptron(X, y, lr=learning_rate, epochs=epochs)

print("Final Weights:", weights)
print("Final Bias:", bias)

# Test the perceptron
print("\nTesting the trained perceptron:")
for i in range(len(X)):
    linear_output = np.dot(weights, X[i]) + bias
    prediction = step_function(linear_output)
    print(f"Input: {X[i]}, Predicted Output: {prediction}, True Output: {y[i]}")

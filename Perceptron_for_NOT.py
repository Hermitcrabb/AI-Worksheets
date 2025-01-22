import numpy as np

# Define the step activation function
def step_function(x):
    return 1 if x >= 0 else 0

# Perceptron training function for NOT gate
def train_not_gate(X, y, lr=0.1, epochs=10):
    # Initialize weight and bias
    weight = np.zeros(1)
    bias = 0

    # Training process
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        for i in range(len(X)):
            # Compute weighted sum
            linear_output = np.dot(weight, X[i]) + bias
            # Apply activation function
            y_pred = step_function(linear_output)
            # Update weight and bias if prediction is incorrect
            error = y[i] - y_pred
            if error != 0:
                weight += lr * error * X[i]
                bias += lr * error
            print(f"Input: {X[i]}, Predicted: {y_pred}, True: {y[i]}, Weight: {weight}, Bias: {bias}")
        print("-" * 50)

    return weight, bias

# Inputs and outputs for the NOT gate
X = np.array([0, 1])  # Single input
y = np.array([1, 0])  # NOT gate outputs

# Train the perceptron
learning_rate = 0.1
epochs = 10
weight, bias = train_not_gate(X, y, lr=learning_rate, epochs=epochs)

print("Final Weight:", weight)
print("Final Bias:", bias)

# Test the perceptron
print("\nTesting the trained perceptron:")
for i in range(len(X)):
    linear_output = np.dot(weight, X[i]) + bias
    prediction = step_function(linear_output)
    print(f"Input: {X[i]}, Predicted Output: {prediction}, True Output: {y[i]}")

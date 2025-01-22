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

# Generate data for odd/even classification
def generate_data(n, bit_length=8):
    X = []
    y = []
    for i in range(n):
        binary_representation = [int(b) for b in format(i, f"0{bit_length}b")]
        X.append(binary_representation)
        y.append(1 if i % 2 != 0 else 0)  # 1 for odd, 0 for even
    return np.array(X), np.array(y)

# Generate training data
num_samples = 16  # Train on numbers 0 to 15
bit_length = 4    # Use 4-bit binary representation
X, y = generate_data(num_samples, bit_length)

# Train the perceptron
learning_rate = 0.1
epochs = 10
weights, bias = train_perceptron(X, y, lr=learning_rate, epochs=epochs)

print("Final Weights:", weights)
print("Final Bias:", bias)

# Test the perceptron
print("\nTesting the trained perceptron:")
for i in range(num_samples):
    linear_output = np.dot(weights, X[i]) + bias
    prediction = step_function(linear_output)
    print(f"Number: {i}, Binary: {X[i]}, Predicted: {'Odd' if prediction == 1 else 'Even'}, True: {'Odd' if y[i] == 1 else 'Even'}")

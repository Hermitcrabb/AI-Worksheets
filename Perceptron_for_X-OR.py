import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Training function for a simple MLP
def train_xor(X, y, lr=0.1, epochs=10000):
    # Initialize weights for the input to hidden layer and hidden to output layer
    np.random.seed(1)
    weights_input_hidden = np.random.uniform(-1, 1, (2, 2))  # 2 inputs, 2 hidden neurons
    weights_hidden_output = np.random.uniform(-1, 1, (2, 1))  # 2 hidden neurons, 1 output
    
    bias_hidden = np.random.uniform(-1, 1, (1, 2))  # Bias for the hidden layer
    bias_output = np.random.uniform(-1, 1, (1, 1))  # Bias for the output layer

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
        hidden_layer_output = sigmoid(hidden_layer_input)

        final_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
        final_layer_output = sigmoid(final_layer_input)

        # Compute error
        error = y - final_layer_output

        # Backpropagation
        output_gradient = error * sigmoid_derivative(final_layer_output)
        hidden_gradient = output_gradient.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_layer_output)

        # Update weights and biases
        weights_hidden_output += hidden_layer_output.T.dot(output_gradient) * lr
        bias_output += np.sum(output_gradient, axis=0, keepdims=True) * lr

        weights_input_hidden += X.T.dot(hidden_gradient) * lr
        bias_hidden += np.sum(hidden_gradient, axis=0, keepdims=True) * lr

    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

# Test the trained network
def predict_xor(X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    final_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    final_layer_output = sigmoid(final_layer_input)

    return np.round(final_layer_output)

# XOR Gate inputs and outputs
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Train the XOR gate network
learning_rate = 0.1
epochs = 10000
weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = train_xor(X, y, lr=learning_rate, epochs=epochs)

# Test the trained network
print("Testing the trained XOR network:")
predictions = predict_xor(X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
for i in range(len(X)):
    print(f"Input: {X[i]}, Predicted Output: {predictions[i][0]}, True Output: {y[i][0]}")

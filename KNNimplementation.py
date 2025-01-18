import pandas as pd
import numpy as np
from google.colab import drive
drive.mount('/content/drive')
# calculate the Euclidean distance between two vectors
# Euclidean Distance
def euclidean_distance(p, q):
    return np.sqrt(np.sum((p - q) ** 2))

# Manhattan Distance (L1 Norm)
def manhattan_distance(p, q):
    return np.sum(np.abs(p - q))

# Minkowski Distance
def minkowski_distance(p, q, p_value):
    return np.sum(np.abs(p - q) ** p_value) ** (1 / p_value)

data_set = pd.read_excel('/content/drive/MyDrive/AIKNN/student_dataset.xlsx')


# Define two vectors
vector1 = data_set['Previous Semester CGPA'].to_numpy()
vector2 = data_set['CGPA'].to_numpy()

# Call the function
distance = euclidean_distance(vector1, vector2)

print("Euclidean distance between the two vectors:", distance)
import numpy as np
# Define the data
data = [
    [2.7810836, 2.550537003, 0],
    [1.465489372, 2.362125076, 0],
    [3.396561688, 4.400293529, 0],
    [1.38807019, 1.850220317, 0],
    [3.06407232, 3.005305973, 0],
    [7.627531214, 2.759262235, 1],
    [5.332441248, 2.088626775, 1],
    [6.922596716, 1.77106367, 1],
    [8.675418651, -0.242068655, 1],
    [7.673756466, 3.508563011, 1]
]
# Convert the data to a numpy array
data_np = np.array(data)

# Split the data into X and Y
X = data_np[:, :2]  # Select all rows and the first two columns
Y = data_np[:, 2]   # Select all rows and the third column

# Display the X and Y arrays to verify
print("X array:")
print(X)
print("\nY array:")
print(Y)

import matplotlib.pyplot as plt
# Plot the data
plt.figure(figsize=(10, 6))

# Separate the data points based on their labels
for label in np.unique(Y):
    plt.scatter(X[Y == label][:, 0], X[Y == label][:, 1], label=f'Class {int(label)}')

# Add labels and title
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Scatter Plot of Data Points')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
from collections import Counter
# kNN algorithm
def k_nearest_neighbors(X, Y, query_point, k):
    distances = []

    # Calculate distances from the query point to all other points
    for i, point in enumerate(X):
        distance = euclidean_distance(point, query_point)
        distances.append((distance, Y[i]))

    # Sort distances in ascending order
    distances.sort(key=lambda x: x[0])

    # Select the k nearest neighbors
    k_nearest = distances[:k]

    # Get the labels of the k nearest neighbors
    k_nearest_labels = [label for _, label in k_nearest]

    # Determine the most common label (majority vote)
    majority_vote = Counter(k_nearest_labels).most_common(1)

    return majority_vote[0][0]

# Define a query point
query_point = np.array([7.0, 3.0])

# Set the value of k
k = 3

# Predict the class for the query point
predicted_class = k_nearest_neighbors(X, Y, query_point, k)

print(f"The predicted class for the query point {query_point} is: {predicted_class}")
# Method to plot the data
def plot_data(X, Y, query_point=None):
    plt.figure(figsize=(10, 6))

    # Separate the data points based on their labels
    for label in np.unique(Y):
        plt.scatter(X[Y == label][:, 0], X[Y == label][:, 1], label=f'Class {int(label)}')

    # Plot the query point if provided
    if query_point is not None:
        plt.scatter(query_point[0], query_point[1], c='red', marker='x', s=100, label='Query Point')

    # Add labels and title
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Scatter Plot of Data Points')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

# Define the dataset
X = np.array([
    [2.7810836, 2.550537003],
    [1.465489372, 2.362125076],
    [3.396561688, 4.400293529],
    [1.38807019, 1.850220317],
    [3.06407232, 3.005305973],
    [7.627531214, 2.759262235],
    [5.332441248, 2.088626775],
    [6.922596716, 1.77106367],
    [8.675418651, -0.242068655],
    [7.673756466, 3.508563011]
])

Y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Define a query point
query_point = np.array([3.0, 3.0])

# Set the value of k
k = 3

# Predict the class for the query point
predicted_class = k_nearest_neighbors(X, Y, query_point, k)
print(f"The predicted class for the query point {query_point} is: {predicted_class}")

# Plot the data with the query point
plot_data(X, Y, query_point)
import pandas as pd

# Specify the path to the CSV file
file_path = r'/content/drive/MyDrive/AIKNN/student_dataset.xlsx'

# Read the CSV file into a DataFrame
data_2C = pd.read_excel(file_path)

# Display the first few rows of the DataFrame to verify the data is loaded correctly
print(data_2C.head())
data_2C.describe().transpose()
data_2C.dtypes
colnames_numeric = data_2C.columns[1:6]
#Scaling a data in always a good idea while using KNN
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data_2C[colnames_numeric] = scaler.fit_transform(data_2C[colnames_numeric])

data_2C.head()
data_2C.shape
df = data_2C.values.tolist()
#Breaking the data into training and test set
import random
def train_test_split(data, split, trainingSet = [], testSet = []):
    for x in range(len(data)):
        if random.random() < split:
            trainingSet.append(data[x])
        else:
            testSet.append(data[x])

trainingSet = []
testSet = []
split = 0.66
train_test_split(df, split, trainingSet, testSet)
len(trainingSet)
len(testSet)
#Define Euclidean distances
import math
def Euclideandist(x,xi, length, start_index=1):
    d = 0.0
    for i in range(start_index, length + start_index):
        d += pow(float(x[i])- float(xi[i]),2)
    return math.sqrt(d)
#Getting the K neighbours having the closest Euclidean distance to the test instance
import operator
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-2
    for x in range(len(trainingSet)):
        dist = Euclideandist(testInstance, trainingSet[x], length, start_index=1)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors
#After sorting the neighbours based on their respective classes, max voting to give the final class of the test instance
import operator
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)#Sorting it based on votes
	return sortedVotes[0][0] #Please note we need the class for the top voted class, hence [0][0]#
#Getting the accuracy
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
	# generate predictions
predictions=[]
k = 3
for x in range(len(testSet)):
    neighbors = getNeighbors(trainingSet, testSet[x], k)
    result = getResponse(neighbors)
    predictions.append(result)
    print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))

accuracy = getAccuracy(testSet, predictions)
print('Accuracy: ' + repr(accuracy) + '%')
#Implementing Naive Bayes using scikitlearn
trainingSet2 = pd.DataFrame(np.array(trainingSet).reshape(len(trainingSet),7), columns = data_2C.columns)
testSet2 = pd.DataFrame(np.array(testSet).reshape(len(testSet),7), columns = data_2C.columns)
trainingSet2.head()

trainingSet2.dtypes
#Even the numeric terms have been converted into an object. Hence need to reconvert
trainingSet2[colnames_numeric] = trainingSet2[colnames_numeric].apply(pd.to_numeric, errors = 'coerce', axis = 0)
trainingSet2.dtypes

testSet2[colnames_numeric] = testSet2[colnames_numeric].apply(pd.to_numeric, errors = 'coerce', axis = 0)
testSet2.dtypes
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
num_bins = 3
x_train = trainingSet2.select_dtypes(include=['number']).drop('CGPA', axis=1)
x_test = testSet2.select_dtypes(include=['number']).drop('CGPA', axis=1)
y_train = pd.cut(trainingSet2['CGPA'], bins=num_bins, labels=False)
y_test = pd.cut(testSet2['CGPA'], bins=num_bins, labels=False)
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print('Prediction: {}'.format(prediction))
print('With KNN (K=3) accuracy is: ',knn.score(x_test,y_test)) # accuracy

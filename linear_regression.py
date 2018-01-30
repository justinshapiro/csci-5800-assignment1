import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random as rand
import os


# Custom linear regression function which produces a model (function), parameter w, and accuracy reported as SSE
# Input: training_data (used to train a model, refining w)
#        normalize (whether or not we normalize attributes through standardization)
#        regularize (whether or not we perform regularization on w. L_2 regularization by default)
# Output: h(x, w): the linear regression hypothesis function
#         w: the value of w that we obtained from the training data, in other words: this is the model parameter
#         sse: the accuracy of the model being returned
def linear_regression_model(training_data, normalize=False, regularize=False):
    def h(x, w, normalize=normalize):
        if normalize:
            x = StandardScaler().fit_transform(x)

        # this will always evaluate to true with the current implementation
        # but the goal is this prediction function should generalize to batch and single predictions
        if x.shape[0] != w.shape[0]:
            w = np.tile(w, (x.shape[0], 1))

        # obtaining thousands of predictions with a for loop takes HOURS on CPU
        # this is a vectorized implementation of batch prediction that completes in only seconds
        # swap the operands before multiply to ensure the prediction results exit in the resultant matrix's diagonal
        # although the operands are swapped, this will also work for a single prediction
        return np.diagonal(np.matmul(x, np.transpose(w)))  # must have 64-bit Python for this to work

    X_train = training_data[0]
    y_train = training_data[1]

    if normalize:
        X_train = pd.DataFrame(StandardScaler().fit_transform(X_train))

    if regularize:
        pass  # unimplemented, should perform L_2 regularization when implemented

    w = np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(X_train), X_train)), np.transpose(X_train)), y_train)[np.newaxis, :]

    # produce three random training samples/results to calculate SSE for
    sse_inputs = []
    for _ in range(0, 3, 1):
        rand_sample = rand.randint(0, X_train.shape[0])
        sse_inputs.append([X_train.iloc[rand_sample, :][np.newaxis, :], y_train.iloc[rand_sample]])

    sse = 0
    for sse_input in sse_inputs:
        sse += np.power(h(sse_input[0], w) - sse_input[1], 2)

    sse = 0.5 * sse

    return h, w, sse


##################
# Problems 1 - 6 #
##################

DATA_PATH = str(os.path.dirname(os.path.realpath(__file__))) + "\\Dataset\\Training\\Features_Variant_1.csv"

# 1. Load the data into memory
data = pd.read_csv(DATA_PATH, header=None)

# "Delete the 38th column having all zero entries"
assert(sum(list(data.iloc[:, 37])) == 0)
data = data.drop(data.columns[37], axis=1)

# How many samples are there in the dataset? How many attributes per sample did you see?
m, n = data.shape
n -= 1  # last column is the target/label, not the attribute

print("Problem #1: \n---------------")
print("How many samples are there in the dataset? " + str(m))
print("How many attributes per sample did you see? " + str(n) + "\n")

# 2. Assume we have m samples and n attributes  per sample. Then make an appropriate X matrix and y vector
X = data.drop(data.columns[n], axis=1)
y = data.iloc[:, n]

# add x_0 = 1 to account for the bias term
X['bias'] = np.ones(m)
X = X[[X.columns.tolist()[-1]] + X.columns.tolist()[:-1]]

# The dimension of X should be m x n and dimension of y vector would be m x 1.
# It is a standard practice to denote the dimensionalities by X e R^(m x n) and the column vector y e R^m
assert(X.shape == (m, n + 1))
assert(y.shape == (m,))

# Note: Shape (m, 1) corresponds to a numpy shape of (m,).
#       This simplifies things for now, but when we want to multiply with it, we need to cast it with y[:, np.newaxis].

print("Problem #2: \n---------------")
print("Given that m = " + str(m) + " and n = " + str(n) + ", ")
print("Dimension of X (should be m x n+1): " + str(X.shape[0]) + " x " + str(X.shape[1]))
print("Dimension of y (should be m x 1): " + str(y.shape[0]) + " x " + (str(y.shape[1]) if len(y.shape) == 2 else str(1)) + "\n")

# 3. Prepare 3 datasets from the data you loaded into memory

# Set the seed for the random number generator to be 123456
rng_seed = 123456

# Split the data at random into set A: (X_train, y_train) containing 80% of the samples which will be used for training
X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=rng_seed)
assert(X_train_A.shape[0] == int(m * 0.8) and y_train_A.shape[0] == int(m * 0.8))

# Split the data at random into set B: (X_train, y_train) containing 50% of the samples which will be used for training
X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X, y, train_size=0.5, test_size=0.5, random_state=rng_seed)
assert(X_train_B.shape[0] == int(m * 0.5) and y_train_B.shape[0] == int(m * 0.5))

# Split the data at random into set C: (X_train, y_train) containing 20% of the samples which will be used for training
X_train_C, X_test_C, y_train_C, y_test_C = train_test_split(X, y, train_size=0.2, test_size=0.8, random_state=rng_seed)
assert(X_train_C.shape[0] == int(m * 0.2) and y_train_C.shape[0] == int(m * 0.2))

print("Problem #3: \n---------------")
print("Set A has " + str(X_train_A.shape[0]) + " (~" + str(int(np.ceil((X_train_A.shape[0] / m) * 100))) + "%) training samples and " + str(y_test_A.shape[0]) + " (~" + str(int(np.floor((y_test_A.shape[0] / m) * 100))) + "%) testing samples")
print("Set B has " + str(X_train_B.shape[0]) + " (~" + str(int(np.ceil((X_train_B.shape[0] / m) * 100))) + "%) training samples and " + str(y_test_B.shape[0]) + " (~" + str(int(np.floor((y_test_B.shape[0] / m) * 100))) + "%) testing samples")
print("Set C has " + str(X_train_C.shape[0]) + " (~" + str(int(np.ceil((X_train_C.shape[0] / m) * 100))) + "%) training samples and " + str(y_test_C.shape[0]) + " (~" + str(int(np.floor((y_test_C.shape[0] / m) * 100))) + "%) testing samples\n")

# 4. For each of the A, B, C train datasets above, solve w for the linear regression hypothesis (without a regularizer)
#    and predict the target values using that
train_datasets = [[X_train_A, y_train_A], [X_train_B, y_train_B], [X_train_C, y_train_C]]
test_datasets = [[X_test_A, y_test_A], [X_test_B, y_test_B], [X_test_C, y_test_C]]

# produce three random inputs to predict and to calculate SSE for
prediction_samples = []
for dataset in test_datasets:
    prediction_sample = []
    for _ in range(0, 3, 1):
        prediction_sample.append(rand.randint(0, dataset[0].shape[0]))
    prediction_samples.append(prediction_sample)

results = []
for i, training_data in enumerate(train_datasets):
    # obtain our random samples
    prediction_inputs = []
    y_actual = []
    for sample in prediction_samples[i]:
        prediction_inputs.append(test_datasets[i][0].iloc[sample, :].as_matrix())
        y_actual.append(test_datasets[i][1].iloc[sample])

    # produce our ML model
    h, w, sse = linear_regression_model(training_data)

    # use the ML model to get a prediction
    prediction = h(np.stack(prediction_inputs), w)

    results.append([prediction, y_actual, sse])

# Plot/report the results of running linear regression on the three sets
print("Problem #4: \n---------------")
print("*** Three samples from the data have been picked randomly for testing ***")
sets = ['A', 'B', 'C']
for i, result in enumerate(results):
    print("The model for Set " + sets[i] + " has an SSE of " + str(result[2][0]))
    index = np.arange(3)
    plt.bar(index, result[0], 0.35, alpha=0.8, color='orange', label="Set " + sets[i] + " Prediction")
    plt.bar(index + 0.35, result[1], 0.35, alpha=0.8, color='black', label="Ground Truth")

    for a, b in zip(index, result[0]):
        plt.text(a, b, "~" + str(int(b)))

    for a, b in zip(index, result[1]):
        plt.text(a + 0.35, b, str(int(b)))

    plt.xlabel("Random Test Sample")
    plt.ylabel("Target/Label")
    plt.title("Prediction vs. Actual Value for Set " + sets[i])
    plt.xticks(index + 0.35, ("Sample 1", "Sample 2", "Sample 3"))
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot the SSE value for each set
index = np.arange(3)
plt.bar(index, [result[2][0] for result in results], 0.35, alpha=0.8, color='red')
for a, b in zip(index, [result[2][0] for result in results]):
    plt.text(a, b, "~" + str(int(b)))

plt.xlabel("Data Sets")
plt.ylabel("SSE")
plt.title("SSEs for Sets A, B and C")
plt.xticks(index, ("Set A", "Set B", "Set C"))
plt.tight_layout()
plt.show()
print("")

# 6. For each of the three training datasets A, B, C, perform normalization on each of the attributes
#    Repeat problem 4 and 5
results_norm = []
for i, training_data in enumerate(train_datasets):
    # obtain our random samples
    prediction_inputs = []
    y_actual = []
    for sample in prediction_samples[i]:
        prediction_inputs.append(test_datasets[i][0].iloc[sample, :].as_matrix())
        y_actual.append(test_datasets[i][1].iloc[sample])

    # produce our ML normalized model
    h, w, sse = linear_regression_model(training_data, normalize=True)

    # use the normalized ML model to get a prediction
    prediction = h(np.stack(prediction_inputs), w)

    results_norm.append([prediction, y_actual, sse])

# Plot/report the results of running linear regression on the three sets
print("Problem #6: \n---------------")
print("*** Three samples from the data have been picked randomly for testing ***")
sets = ['A', 'B', 'C']
for i, result in enumerate(results_norm):
    print("The normalized model for Set " + sets[i] + " has an SSE of " + str(result[2][0]))
    index = np.arange(3)
    plt.bar(index, result[0], 0.35, alpha=0.8, color='orange', label="Set " + sets[i] + " Prediction")
    plt.bar(index + 0.35, result[1], 0.35, alpha=0.8, color='black', label="Ground Truth")

    for a, b in zip(index, result[0]):
        plt.text(a, b, "~" + str(int(b)))

    for a, b in zip(index, result[1]):
        plt.text(a + 0.35, b, str(int(b)))

    plt.xlabel("Random Test Sample")
    plt.ylabel("Target/Label")
    plt.title("Normalized Model: Prediction vs. Actual Value for Set " + sets[i])
    plt.xticks(index + 0.35, ("Sample 1", "Sample 2", "Sample 3"))
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot the SSE value for each set
index = np.arange(3)
plt.bar(index, [result[2][0] for result in results_norm], 0.35, alpha=0.8, color='red')
for a, b in zip(index, [result[2][0] for result in results_norm]):
    plt.text(a, b, "~" + str(int(b)))

plt.xlabel("Data Sets")
plt.ylabel("SSE")
plt.title("Normalized Model: SSEs for Sets A, B and C")
plt.xticks(index, ("Set A", "Set B", "Set C"))
plt.tight_layout()
plt.show()
print("")

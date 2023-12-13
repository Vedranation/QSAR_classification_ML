'''Written in Python 3.8
THIS CODE OPENS QSAR AS .csv, NOT .mat'''
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor


np.random.seed(0)  # For reproducibility
def generate_dataset(n_rows, n_features, test_ratio, theta_initial_value):
    '''Generates random dataset, input is n_rows, n_features, test ratio, and start value for theta'''
    # Check if the number of features is at least 1
    if n_features < 1:
        raise ValueError("Number of features must be at least 1")

    # Create a dictionary to hold the data
    data = {}

    # Generate independent variables
    for i in range(n_features):
        if i == 0:
            data[f'X{i+1}'] = np.random.randn(n_rows)  # Normally distributed values
        elif i == 1:
            data[f'X{i+1}'] = np.random.rand(n_rows) * 10  # Uniformly distributed values between 0 and 10
        elif i == 2:
            data[f'X{i+1}'] = np.random.randint(0, 2, n_rows)  # Random 0s and 1s
        elif i == 3:
            data[f'X{i+1}'] = np.random.rand(n_rows) * -10
        elif i == 4:
            data[f'X{i+1}'] = np.random.randint(-1, 1, n_rows)  # Random negative 0s and 1s
        else:
            # Additional features can be added here with different distributions or rules
            if random.randint(0, 4) == 0:
                data[f'X{i+1}'] = np.random.normal(5, 2, n_rows)  # Normally distributed around 5 with std dev 2
            elif random.randint(0, 4) == 1:
                data[f'X{i+1}'] = np.random.normal(-7.2, 2, n_rows)  # Normally distributed around -5 with std dev 2
            elif random.randint(0, 4) == 1:
                data[f'X{i+1}'] = np.random.randint(0, 2, n_rows)  # Random 0s and 1s
            else:
                data[f'X{i + 1}'] = np.random.randint(-1, 1, n_rows)  # Random negative 0s and 1s

    # Construct the target variable with a non-linear relationship and some noise
    noise = np.random.normal(0, 0.5, n_rows)
    # For simplicity, using a linear combination of the features to construct Y
    Y = np.sum([data[f'X{i+1}'] for i in range(n_features)], axis=0) + noise
    data['Y'] = (Y > 0).astype(int)

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Split the DataFrame into features and target variable
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)

    # Initialize Theta
    Theta = np.full((n_features, 1), theta_initial_value)

    return X_train, X_test, y_train, y_test, Theta


def process_and_split_dataset(csv_file, test_ratio, theta_initial_value=None, drop_high_vif=False, columns_to_drop=[]):
    '''Splits QSAR into test and train, generates theta, and optionally drops high VIF columns'''
    df = pd.read_csv(csv_file, header=None)

    # Optionally drop columns with high VIF
    if drop_high_vif:
        df.drop(df.columns[columns_to_drop], axis=1, inplace=True)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=43) #random seed, change to get different k-fold

    n_features = X.shape[1]
    Theta = np.full((n_features, 1), theta_initial_value)

    return X_train, X_test, y_train, y_test, Theta


def VIF(csv_file, drop=False):
    '''computes VFIF and correlation matrix'''
    df = pd.read_csv(csv_file, header=None)



    '''find VIF'''

    features = df.iloc[:, :-1]  # Excludes the last column which is 'Y'
    vif_data = pd.DataFrame()
    vif_data["feature"] = features.columns
    vif_data["VIF"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]

    print(vif_data.sort_values('VIF', ascending=False))

    '''make correlation matrix with VIF > 10'''
    high_vif_features = vif_data[vif_data['VIF'] > 10]['feature']
    filtered_df = df[high_vif_features]  # DataFrame containing only high VIF features

    # Compute the correlation matrix
    corr_matrix_high_vif = filtered_df.corr()

    '''display full result, not recommend cause it lags'''
    if drop == False:
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix_high_vif, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap between QSAR features')
        plt.xlabel('QSAR features index')
        plt.ylabel('QSAR features index')
        plt.show()

    else:
        '''Drop highly correlated features'''
        # Add a line to remove a column with a specific index 'col_index_to_remove'
        col_index_to_remove = [38, 14, 16, 17, 26, 12, 0, 15],  # Replace with the index of the column you want to remove
        df.drop(df.columns[col_index_to_remove], axis=1, inplace=True)

        features = df.iloc[:, :-1]  # Excludes the last column which is 'Y'
        vif_data = pd.DataFrame()
        vif_data["leftover feature"] = features.columns
        vif_data["VIF"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]

        print(vif_data.sort_values('VIF', ascending=False))

        high_vif_features = vif_data[vif_data['VIF'] > 10]['leftover feature']
        filtered_df = df[high_vif_features]  # DataFrame containing only high VIF features

        # Compute the correlation matrix
        corr_matrix_high_vif = filtered_df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix_high_vif, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap between QSAR features')
        plt.xlabel('QSAR features index')
        plt.ylabel('QSAR features index')
        plt.show()
'''Initialise algorithm'''
n_rows = 1000
n_features = 40
test_ratio = 0.3
lambda_reg = 10 #for L2
theta_initial_value = 0.001
runtime = 8 #run it until Log-likelihood stop improving

'''Replace with your actual file name or path
NOTE: this code opens QSAR as .csv, NOT .mat'''
csv_file = "QSAR.csv"

'''Generator of random data, I used it to develope model - you dont need it'''
#X, X_test, Y, Y_test, Theta = generate_dataset(n_rows, n_features, test_ratio, theta_initial_value)

'''read the .CSV of QSAR and process it - setting bool to True makes the model drop redudant features'''
X, X_test, Y, Y_test, Theta = process_and_split_dataset(csv_file, test_ratio, theta_initial_value, True, [38, 14, 16, 17, 26, 12, 0, 15])

'''Generate a visual VIF report and correlation matrix - this does NOT drop features from the dataset'''
VIF(csv_file, True)

X = X.to_numpy()
Y = Y.to_numpy().reshape(-1, 1)
X_test = X_test.to_numpy()
Y_test = Y_test.to_numpy().reshape(-1, 1)
print(type(X))
print(X.shape[1])
print(type(Y))
print(type(Theta))
print(X)
print(Y)
log_likelihood_array = []
def sigmoid(z):
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))

'''train loop'''
for j in range(runtime):

    Z = X @ Theta
    P = sigmoid(Z) #(6, 1) computes sigmoid, had to make custom case cause of numerical overflow
    # Constructing the weight matrix W
    W = (P * (1 - P))

    W = np.diag(W.flatten()) #6x6

    X_W_X = X.T @ W @ X
    L2_matrix = lambda_reg * np.eye(X.shape[1])
    L2_matrix[0, 0] = 0 #(4, 4)
    X_W_X = X_W_X + L2_matrix

    X_W_X = np.linalg.inv(X_W_X)  # (4, 4)
    P = np.array(P).flatten()
    Y = np.array(Y).flatten()
    X_Y_P = (Y - P).reshape(-1, 1) #(6, 1)
    X_Y_P = X.T @ X_Y_P #(4, 1)

    Theta = Theta + (X_W_X @ X_Y_P) #(4, 1)

    # Log-likelihood calculation
    log_likelihood = np.sum(np.multiply(Y, np.log(P)) + np.multiply((1 - Y), np.log(1 - P)))
    log_likelihood_array.append(float(round(log_likelihood, 2)))
    print(f"\nLog-likelihood at iteration {j}: {log_likelihood}")
    print("\nTheta update:")
    print(Theta)

print("\n\n\n")

'''Test loop'''
# Initialize counters for True Positives, True Negatives, False Positives, False Negatives
TP = 0
TN = 0
FP = 0
FN = 0

for row in range(len(Y_test)):
    prediction = (1 / (1 + np.exp(-X_test[row, :] @ Theta)))
    prediction = round(float(prediction[0]), 3)
    print(f"Checking row: {row}, prediction is: {prediction}, result is {Y_test[row][0]}")

    if prediction >= 0.5:  # Predicted as 1
        if Y_test[row][0] == 1:
            TP += 1  # True Positive
        else:
            FP += 1  # False Positive
    else:  # Predicted as 0
        if Y_test[row][0] == 0:
            TN += 1  # True Negative
        else:
            FN += 1  # False Negative
print("\n\n\n")
# Constructing the confusion matrix
confusion_matrix = np.array([[TP, FP], [FN, TN]])

confusion_matrix_df = pd.DataFrame(confusion_matrix,
                                   columns=['Predicted Positive', 'Predicted Negative'],
                                   index=['Actual Positive', 'Actual Negative'])
# Calculating accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)

def plot_log_likelihood(log_likelihood_array):
    # Create a list of iteration numbers
    iterations = list(range(len(log_likelihood_array)))

    # Create a plot of log likelihood values per iteration
    plt.figure(figsize=(8, 6))
    plt.plot(iterations, log_likelihood_array, marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('Log Likelihood')
    plt.title('Log Likelihood per Iteration')
    plt.grid(True)
    plt.show()
plot_log_likelihood(log_likelihood_array)
print("Confusion Matrix:")
print(confusion_matrix_df)
print(f"Accuracy: {round(100 * accuracy, 2)}%, {TP + TN} out of {TP + TN + FP + FN}")

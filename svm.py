import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def svmCalculation(input_coordinates,y):
    # Step 1: Split the dataset into training and testing subsets. In this example, we allocate 80% of  the data for training and 20% for testing
    st_svm_train_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(input_coordinates, y, test_size=0.2,    random_state=42)

    # Step 2: Create and train the SVM model
    svm_model = svm.SVC(kernel='linear')  # Linear kernel
    svm_model.fit(X_train, y_train)

    # Step 3: Evaluate the SVM model
    y_pred_train = svm_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    print("SVM Training accuracy:", train_accuracy)

    y_pred_test = svm_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print("SVM Testing accuracy:", test_accuracy)

    en_svm_train_time = time.time()    
    print(f"Elapsed time: {en_svm_train_time-st_svm_train_time} seconds")
    return en_svm_train_time-st_svm_train_time


def nnCalculation(input_coordinates,y):
    ## Neural Network Area starts

    # Step 1: Split the dataset into training and testing subsets
    st_nn_train_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(input_coordinates, y, test_size=0.2,    random_state=42)

    # Step 2: Create and train the Neural Network model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, verbose=0)

    # Step 3: Evaluate the Neural Network model
    y_pred_train = np.round(model.predict(X_train)).flatten()
    train_accuracy = accuracy_score(y_train, y_pred_train)
    print("Neural Network Training accuracy:", train_accuracy)

    y_pred_test = np.round(model.predict(X_test)).flatten()
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print("Neural Network Testing accuracy:", test_accuracy)

    en_nn_train_time = time.time()
    
    print(f"Elapsed time: {en_nn_train_time-st_nn_train_time} seconds")
    return en_nn_train_time-st_nn_train_time


def plotGraph(x1, y1, y2):
    X = x1
    X_axis = np.arange(len(X))
  
    plt.bar(X_axis - 0.2, y1, 0.4, label = 'SVM')
    plt.bar(X_axis + 0.2, y2, 0.4, label = 'NN')
    
    plt.xticks(X_axis, X)
    plt.xlabel("No of Inputs")
    plt.ylabel("Time Taken to Train")
    plt.title("Train time comparison of SVM vs NN")
    plt.legend()
    plt.show()

x1 = np.array([])
y1 = np.array([])
y2 = np.array([])
input_list = [10, 100, 1000, 10000, 100000]

for inp in input_list:	
    input_points = inp
    X1 = np.random.randint(0,100,input_points)
    X2 = np.random.randint(0,100,input_points)
    input_coordinates = np.column_stack((X1,X2))
    y = np.random.randint(0,2,input_points)
    print(f"Dataset points: {input_points}")
    svm_time = svmCalculation(input_coordinates, y)
    nn_time = nnCalculation(input_coordinates, y)
    y1 = np.append(y1, svm_time)
    y2 = np.append(y2, nn_time)
    x1 = np.append(x1,input_points)

plotGraph(x1, y1, y2)

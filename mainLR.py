from sklearn import datasets
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt

def costFunction(X, y, theta):
    predictions = np.matmul(X, theta)
    val = np.square(predictions - y.reshape((len(y), 1)), dtype = np.float64)
    m = len(y)
    J = 1/(2*m) * sum(val)
    return J

def GradientDecent(X, y, theta):
    alpha = float(input("Enter Learning rate: "))
    m = len(y)
    J_his = np.zeros((50, 1))
    features = np.shape(X)[1]
    temp = np.zeros((features, 1))

    for iterator in range(50):
        val = np.matmul(X , theta) - y
        for i in range(features-1):
            a = sum(val * X[:, i].reshape((len(X[:, i]), 1)))
            temp[i, 0] = theta[i, 0] - (alpha/m) * sum(a)
        for i in range(features):
            theta[i, 0] = temp[i, 0]
        J_his[iterator] = costFunction(X, y, theta)
        del(val)
    return J_his, theta

def scalefeatures(X):
    feat = np.shape(X)[1]
    mean = np.zeros((1, feat-1))
    sigma = np.zeros((1, feat-1))
    for i in range(feat-1):
        y = X[:, i]
        mean[0][i] = np.mean(y)
        sigma[0][i] = np.std(y)
    for i in range(feat - 1):
        for j in range(len(X[:, 1] - 1)):
            X[j][i] = (X[j][i] - mean[0][i]) / sigma[0][i]   
    return X
    
def predict(X, y, theta):
    prediction = np.matmul(X, theta)
    error = np.square(prediction - y)
    avg = np.average(y)
    denom = np.square(prediction - avg)
    ans = 1 - sum(error)/ sum(denom)
    return ans
    
def predict_one(new_X, theta):
    new_X = np.append(1, new_X)
    return np.matmul(new_X, theta)
    
def normalequation(X, y, theta):
    temp = np.matmul(np.transpose(X), X)
    temp = np.linalg.inv(temp)
    temp = np.matmul(temp, np.transpose(X))
    ans = np.matmul(temp, y)
    for i in range(len(ans)):
        theta[i] = ans[i]
    return theta
    
data = datasets.load_boston()
X = data.data
y = data.target

print("Which Algorithm you want to use \n 1 for Gradient Decent \n 2 for Normal Equation:")
a = int(input("CHoice: "))

if a == 1:
    X = scalefeatures(X)
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype = np.float64)
    y = y.reshape((len(y), 1))
    X = np.append(np.ones((np.shape(X)[0], 1)), X, axis = 1)
    theta = np.zeros((np.shape(X)[1], 1), dtype = np.float64)
    X_train, X_test,Y_train, Y_test = model_selection.train_test_split(X, y)
    cost = costFunction(X_train, Y_train, theta)
    J_val, theta = GradientDecent(X_train, Y_train, theta)
    a = [x for x in range(len(J_val))]
    plt.plot(a, J_val)
    plt.show()
    cost = costFunction(X_train, Y_train, theta)
    main_error = predict(X_test, Y_test, theta)

elif a == 2:
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype = np.float64)
    y = y.reshape((len(y), 1))
    X = np.append(np.ones((np.shape(X)[0], 1)), X, axis = 1)
    theta = np.zeros((np.shape(X)[1], 1), dtype = np.float64)
    X_train, X_test,Y_train, Y_test = model_selection.train_test_split(X, y)
    theta = normalequation(X_train, Y_train, theta)
    main_error1 = predict(X_test, Y_test, theta)

else: 
    print("Invalid Choice!")

##Input Prompt
print("Enter new input:")
features = data.feature_names
print(features)

new_val = [None] * len(features)
for i in range(len(features)):
    new_val[i] = float(input("Enter {} :".format(features[i])))
    
new_val = np.array(new_val, dtype = np.float64)

ans = predict_one(new_val, theta)
print("Expected output is: {}".format(ans))


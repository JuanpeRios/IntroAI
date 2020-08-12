import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt
import matplotlib.pyplot as plt

#Clase 2
def DatasetSintetico(c,n):
    """
        :param c: Numero de clusters
        :param n: Realizaciones del cluster
        :return: Dataset sintetico
    """
    X = np.eye(c)
    X = np.repeat(X,n,axis=0)
    ruido = np.random.normal(0,1,np.shape(X))
    X = X+0.2*ruido
    idp = np.random.permutation(c*n)
    return X[idp]

def k_means(X,n):
    """
    :param X: Dataset a clusterizar
    :param n: Cantidad de clusters
    :return: Centroides y a cual pertenece cada elemento del dataset
    """
    iteraciones = 10
    aux = np.random.randint(0,np.shape(X)[0],n)
    Centroides = X[aux]
    for i in range(iteraciones):
        CentroidesExp = Centroides[:,None]
        resta = X - CentroidesExp
        distancia = np.sqrt(np.sum((resta**2),axis=2))
        arg = np.argmin(distancia, axis=0)
        for i in range(n):
            Centroides[i]=np.mean(X[arg == i, :],axis=0)
    return Centroides, arg

def PCAManual(X,dim):
    x2 = (X - np.mean(X,axis=0))
    cov_1 = np.cov(x2.T)
    w, v = np.linalg.eig(cov_1)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:, idx]
    return np.matmul(x2, v[:, :dim])

#######################################################
#Clase 3
class Data(object):

    def __init__(self, dataset):
        self.dataset = dataset

    def split(self, percentage): # 0.8
        X = self.dataset[:,0][::,None]
        y = self.dataset[:,1][::,None]
        permuted_idxs = np.random.permutation(X.shape[0])
        train_idxs = permuted_idxs[0:int(percentage * X.shape[0])]
        test_idxs = permuted_idxs[int(percentage * X.shape[0]): X.shape[0]]
        X_train = X[train_idxs]
        X_test = X[test_idxs]
        y_train = y[train_idxs]
        y_test = y[test_idxs]
        return X_train, X_test, y_train, y_test

class BaseModel(object):

    def __init__(self):
        self.model = None

    def fit(self, X, Y):
        return NotImplemented

    def predict(self, X):
        return NotImplemented

class LinearRegression(BaseModel):

    def fit(self, X, y):
        if len(X.shape) == 1:
            W = X.T.dot(y) / X.T.dot(X)
        else:
            W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.model = W

    def predict(self, X):
        return self.model * X

class LinearRegressionWithB(BaseModel):

    def fit(self, X, y):
        X_expanded = np.vstack((X.T, np.ones(len(X)))).T
        W = np.linalg.inv(X_expanded.T.dot(X_expanded)).dot(X_expanded.T).dot(y)
        self.model = W

    def predict(self, X):
        X_expanded = np.vstack((X.T, np.ones(len(X)))).T
        return X_expanded.dot(self.model)

class ConstantModel(BaseModel):

    def fit(self, X, Y):
        W = Y.mean()
        self.model = W

    def predict(self, X):
        return np.ones(len(X)) * self.model

class Metric(object):
    def __call__(self, target, prediction):
        return NotImplemented


class MSE(Metric):
    def __call__(self, target, prediction):
        return np.square(np.subtract(target,prediction)).mean()

#######################################################
#Clase 4
def k_folds(X_train, y_train, k=5):
    l_regression = LinearRegression()
    error = MSE()

    chunk_size = int(len(X_train) / k)
    mse_list = []
    for i in range(0, len(X_train), chunk_size):
        end = i + chunk_size if i + chunk_size <= len(X_train) else len(X_train)
        new_X_valid = X_train[i: end]
        new_y_valid = y_train[i: end]
        new_X_train = np.concatenate([X_train[: i], X_train[end:]])
        new_y_train = np.concatenate([y_train[: i], y_train[end:]])

        l_regression.fit(new_X_train, new_y_train)
        prediction = l_regression.predict(new_X_valid)
        mse_list.append(error(new_y_valid, prediction))

    mean_MSE = np.mean(mse_list)

    return mean_MSE

def GradienteDescendiente(Xtrain, Ytrain, lr=0.01, epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    n,m = Xtrain.shape

    # initialize random weights
    W = np.random.randn(m).reshape(m, 1)

    for i in range(epochs):
        prediction = np.matmul(Xtrain, W)  # nx1
        error = Ytrain - prediction  # nx1

        grad_sum = np.sum(error * Xtrain, axis=0)
        grad_mul = -2/n * grad_sum  # 1xm
        gradient = np.transpose(grad_mul).reshape(-1, 1)  # mx1

        W = W - (lr * gradient)

    return W

def GradienteEstocastico(Xtrain, Ytrain, lr=0.01, epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    n,m = Xtrain.shape

    # initialize random weights
    W = np.random.randn(m).reshape(m, 1)

    for i in range(epochs):
        idx = np.random.permutation(Xtrain.shape[0])
        Xtrain = Xtrain[idx]
        Ytrain = Ytrain[idx]

        for j in range(n):
            prediction = np.matmul(Xtrain[j].reshape(1, -1), W)  # 1x1
            error = Ytrain[j] - prediction  # 1x1

            grad_sum = error * Xtrain[j]
            grad_mul = -2/n * grad_sum  # 2x1
            gradient = np.transpose(grad_mul).reshape(-1, 1)  # 2x1

            W = W - (lr * gradient)

    return W

def GradienteMiniBatch(Xtrain, Ytrain, lr=0.01, epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    b = 16
    n,m = Xtrain.shape

    # initialize random weights
    W = np.random.randn(m).reshape(m, 1)

    for i in range(epochs):
        idx = np.random.permutation(Xtrain.shape[0])
        Xtrain = Xtrain[idx]
        Ytrain = Ytrain[idx]

        batch_size = int(len(Xtrain) / b)
        for i in range(0, len(Xtrain), batch_size):
            end = i + batch_size if i + batch_size <= len(Xtrain) else len(Xtrain)
            batch_X = Xtrain[i: end]
            batch_y = Ytrain[i: end]

            prediction = np.matmul(batch_X, W)  # nx1
            error = batch_y - prediction  # nx1

            grad_sum = np.sum(error * batch_X, axis=0)
            grad_mul = -2/n * grad_sum  # 1xm
            gradient = np.transpose(grad_mul).reshape(-1, 1)  # mx1

            W = W - (lr * gradient)

    return W
#######################################################
#Clase 2
centroides, indice = k_means(DatasetSintetico(3,10),3)

x = np.array([ [0.4, 4800, 5.5], [0.7, 12104, 5.2], [1, 12500, 5.5], [1.5, 7002, 4.0] ])
pca = PCA(n_components=3)
x_std = StandardScaler(with_std=False).fit_transform(x)
pca.fit_transform(x_std)
xPCA = PCAManual(x,2)

#######################################################
#Clase 3

my_data = genfromtxt('income.data.csv', delimiter=',')
Mediciones = my_data[:,1:]
Datos = Data(Mediciones)
Xtrain,Xtest,Ytrain,Ytest = Datos.split(0.8)

RegresionLineal = LinearRegression()
RegresionLineal.fit(Xtrain,Ytrain)
Yrl = RegresionLineal.predict(Xtest)

RegresionLinealB = LinearRegressionWithB()
RegresionLinealB.fit(Xtrain,Ytrain)
Yrlb = RegresionLinealB.predict(Xtest)

Constante = ConstantModel()
Constante.fit(Xtrain,Ytrain)
Yconst = Constante.predict(Xtest)

ECM = MSE()
rl_ecm = ECM(Yrl,Ytest)
rlb_ecm = ECM(Yrlb,Ytest)
const_ecm = ECM(Yconst,Ytest)

# plt.scatter(Xtest, Ytest, color='b', label='dataset')
# plt.plot(Xtest, Yrl, color='r', label='LinearRegresion')
# plt.plot(Xtest, Yrlb, color='g', label='LinearRegresionWithB')
# plt.plot(Xtest, Yconst, color='y', label='ConstantModel')
# plt.show()

#######################################################
#Clase 4

errork = k_folds(Xtrain,Ytrain,k=10)
W1 = GradienteEstocastico(Xtrain,Ytrain)
X1 = np.vstack((Xtrain.T, np.ones(len(Xtrain)))).T
X1test = np.vstack((Xtest.T, np.ones(len(Xtest)))).T
W11 = GradienteEstocastico(X1,Ytrain)
plt.scatter(Xtest, Ytest, color='b', label='dataset')
plt.plot(Xtest, np.matmul(Xtest,W1), color='r', label='GradienteDescendiente')
plt.plot(Xtest, np.matmul(X1test,W11), color='y', label='GradienteDescendiente + b')
plt.show()
print(W11)
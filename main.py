import numpy as np
import os
import requests
import zipfile
import pandas as pd

class Data:
    def __init__(self):
        self.data_path = os.path.join(os.getcwd(), "data")

    def fetch_iris(self):
        url = "https://archive.ics.uci.edu/static/public/53/iris.zip"
        req = requests.get(url)
        
        if not os.path.isdir(self.data_path):
            os.mkdir(self.data_path)
        with open(os.path.join(self.data_path, "iris.zip"), "wb") as f:
            f.write(req.content)
            
        return os.path.join(self.data_path, "iris.zip")

    def get_data_from_zip(self, zip_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(path=self.data_path)

        return pd.read_csv(os.path.join(self.data_path, "iris.data"))

    def train_test_set_split(self, X, y, test_set_size):
        train_set_len = len(X) - int(len(X)*test_set_size) 
        train_X, test_X = X[:train_set_len], X[train_set_len:]
        train_y, test_y = y[:train_set_len], y[train_set_len:]
        return train_X, train_y, test_X, test_y


class Transformer:
    def __init__(self):
        self._label_names = []
        self.sparse_vectors = []
        self._represent_dict = {}
    
    def transform_text_labels(self, labels_list: np.array):
        self._label_names = np.unique(labels_list)
        
        for i, name in enumerate(self._label_names):
            self._represent_dict[name] = i
        
        for label in labels_list:
            sparse_vec = np.zeros((len(self._label_names)))
            sparse_vec[self._represent_dict[label]] = 1
            sparse_vec = sparse_vec.reshape((len(self._label_names), 1))
            self.sparse_vectors.append(sparse_vec)

        self.sparse_vectors = np.array(self.sparse_vectors)


class MultilayerPerceptron:
    def __init__(self, loss_func, input_features, hidden_features, numclass, learning_rate):
        self.loss_func = loss_func

        self.input_features = input_features
        self.hidden_features = hidden_features
        self.numclass = numclass

        self.learning_rate = learning_rate

        self.hidden_layers = [
            {
             "in_size": input_features,
             "out_size": hidden_features,
             "activation_func": ReLU,
             "layer_size": 5
            },
            {
             "in_size": hidden_features,
             "out_size": hidden_features,
             "activation_func": ReLU,
             "layer_size": 5
            },
            {
             "in_size": hidden_features,
             "out_size": numclass,
             "activation_func": SoftMax,
             "layer_size": 5
            }
        ]

        self.weights = []
        self.biases = []

        self._random_init_weights()
        self._random_init_biases()

        
    def _random_init_weights(self):
        #weights beetween x and first layer
        layer0_size = self.hidden_layers[0]["layer_size"]
        w0 = np.random.uniform(0, 0.1, (layer0_size, self.input_features))
        self.weights.append(w0)

        #weights beetween layers
        for layer in self.hidden_layers:
            w = np.random.uniform(0, 0.1, (layer["out_size"], layer["layer_size"]))
            self.weights.append(w)
    
    def _random_init_biases(self):
        #biases beetween x and first layer
        layer0_size = self.hidden_layers[0]["layer_size"]
        b0 = np.random.uniform(0, 0.1, (layer0_size, 1))
        self.biases.append(b0)

        #biases beetween layers
        for layer in self.hidden_layers:
            b = np.random.uniform(0, 0.1, (layer["out_size"], 1))
            self.biases.append(b)

    def compute_layer_by_index(self, x, ind=0):
        """
            ind = 0 means compute y_hat
            and ind = -len(hidden_layers) means first hidden layer
        """
        result = x.reshape((-1, 1))

        i = -len(self.weights)
        while i < ind:
            result = np.dot(self.weights[i], result)
            result += self.biases[i]

            if i == -len(self.weights):
                i += 1
                continue

            if self.hidden_layers[i]["activation_func"]:
                result = self.hidden_layers[i]["activation_func"](result)
            
            i += 1

        return result

    def compute_y_hat(self, X):
        y_hats = []

        for x in X:
            y_hat = self.compute_layer_by_index(np.array(x))
            y_hats.append(y_hat)

        return np.array(y_hats)

    def compute_gradients(self, loss, x):
        weight_gradients = []
        bias_gradients = []

        layer_ind = len(self.hidden_layers)-1

        #compute gradients beetween y_hat and last hidden layer
        del_zL_del_wL = self.compute_layer_by_index(x, -1).T
        del_aL_del_zL = SoftMax_gradient(self.compute_layer_by_index(x))
        del_C_del_aL = 2*loss

        grad_for_W = np.matmul((del_zL_del_wL*del_C_del_aL).T, del_aL_del_zL).T
        grad_for_b = np.matmul(del_aL_del_zL, del_C_del_aL)

        weight_gradients.append(grad_for_W)
        bias_gradients.append(grad_for_b)

        #compute gradients for middle hidden layers
        while layer_ind > 0:
            i = -(len(self.hidden_layers)-layer_ind)

            prev_layer = self.hidden_layers[i-1]
            current_layer = self.hidden_layers[i]

            if del_aL_del_zL.shape[1] != 1:
                del_C_del_aL = np.matmul(self.weights[i].T, np.matmul(del_aL_del_zL, del_C_del_aL))
            else:
                del_C_del_aL = np.matmul(self.weights[i].T, del_aL_del_zL*del_C_del_aL)
            del_zL_del_wL = self.compute_layer_by_index(x, i).T
            del_aL_del_zL = activation_function_grads[prev_layer["activation_func"]](self.compute_layer_by_index(x, i))

            grad_for_W = del_zL_del_wL * del_aL_del_zL * del_C_del_aL.T
            grad_for_b = np.multiply(del_aL_del_zL, del_C_del_aL)

            weight_gradients.append(grad_for_W)
            bias_gradients.append(grad_for_b)
            layer_ind -= 1

        #compute gradients beetween x and first hiddent layer
        i = -(len(self.hidden_layers)-layer_ind)
    
        del_C_del_aL = np.sum(self.weights[0], axis=-1).reshape((-1,1))*del_aL_del_zL*del_C_del_aL
        del_zL_del_wL = x
        del_aL_del_zL = np.ones((5,1))

        grad_for_W = del_zL_del_wL * del_aL_del_zL * del_C_del_aL
        grad_for_b = del_aL_del_zL * del_C_del_aL

        weight_gradients.append(grad_for_W)
        bias_gradients.append(grad_for_b)

        return weight_gradients, bias_gradients

    ### IMPLEMENT FUNCTION TO COMPUTE PREVIOUSE LAYER GRADS depending on layers size

    def compute_average_gradients(self, loss, X):
        w_av_grads, b_av_grads = self.compute_gradients(loss[0], np.array(X[0]))

        for i in range(1, len(X)):
            w_grads, b_grads = self.compute_gradients(loss[i], np.array(X[i]))
            for j in range(0, len(w_av_grads)):
                w_av_grads[j] += w_grads[j]
                b_av_grads[j] += b_grads[j]
        
        for j in range(0, len(w_av_grads)):
            w_av_grads[j] /= len(X)
            b_av_grads[j] /= len(X)

        return w_av_grads, b_av_grads

    def pass_gradients(self, w_grads, b_grads):
        for i in range(0, len(self.weights)):
            self.weights[i] -= self.learning_rate * w_grads[::-1][i]
            self.biases[i] -= self.learning_rate * b_grads[::-1][i]

    def fit(self, X_train, y_train, X_test, y_test, epochs):
        self.train_epoch_losses = []
        self.test_epoch_losses = []

        epsilon = 1e-5

        for epoch in range(0, epochs):
            y_hat = self.compute_y_hat(X_train)

            loss = y_hat - y_train

            cost = -(y_train * np.log(y_hat + epsilon))
            cost = cost.sum(axis=1).mean()

            w_grads, b_grads = self.compute_average_gradients(loss, X_train)

            self.pass_gradients(w_grads, b_grads)

            self.train_epoch_losses.append(cost)
            self.test_epoch_losses.append(self.test_loss(X_test, y_test))

        print(self.train_epoch_losses[::100])
        print(self.test_epoch_losses[::100])

    def test_loss(self, X_test, y_test):
        epsilon = 1e-5

        y_hat = self.compute_y_hat(X_test)

        cost = -(y_test * np.log(y_hat + epsilon))
        cost = cost.sum(axis=1).mean()

        return cost


def MSE(y_hat, y):
    return np.power(y_hat - np.array(y).reshape(y_hat.shape), 2)

def SoftMax(x):
    return np.exp(x-np.max(x)) / np.sum(np.exp(x-np.max(x)))

def ReLU(x):
    return np.max(np.array([np.zeros(x.shape), x]), axis=0)

def ReLU_gradient(x):
    result = []
    for x in np.nditer(x):
        if x > 0:
            result.append(1)
        else:
            result.append(0)
    
    return np.array(result).reshape((-1,1))

def SoftMax_gradient(x):
    x = np.ravel(x)

    result = np.zeros((x.shape[0], x.shape[0]))

    for i in range(0, len(x)):
        for j in range(0, len(x)):
            if j==i:
                result[i,j] = x[i]*(1-x[j])
            else:
                result[i,j] += -x[j]*x[i]

    return result

activation_function_grads = {
    ReLU: ReLU_gradient,
    SoftMax: SoftMax_gradient
}

#data thing
data_manager = Data()

#zip_file_path = data_manager.fetch_iris()
zip_file_path = os.path.join(data_manager.data_path, "iris.zip")
raw_data = data_manager.get_data_from_zip(zip_file_path)

first_row = pd.DataFrame({
    "sepal_length": [5.1],
    "sepal_width": [3.5], 
    "petal_length": [1.4], 
    "petal_width": [0.2], 
    "label": ["Iris-setosa"]
})

rest_of_rows = pd.DataFrame({
    "sepal_length": raw_data["5.1"].values,
    "sepal_width": raw_data["3.5"].values, 
    "petal_length": raw_data["1.4"].values, 
    "petal_width": raw_data["0.2"].values, 
    "label": raw_data["Iris-setosa"].values
})

data = pd.concat([first_row, rest_of_rows])

data_transformer = Transformer()
data_transformer.transform_text_labels(np.array(data["label"].values))

X = np.array(data.copy().drop("label", axis=1).to_numpy())
y = data_transformer.sparse_vectors

X_train, y_train, X_test, y_test = data_manager.train_test_set_split(X, y, 0.3)

#model thing
model = MultilayerPerceptron(MSE, 4, 5, 3, 0.1)
model.fit(X_train, y_train, X_test, y_test, 1500)
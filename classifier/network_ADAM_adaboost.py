import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.beta1 = 0.9  # 一阶矩估计的衰减率
        self.beta2 = 0.999  # 二阶矩估计的衰减率
        self.epsilon = 1e-8  # 为了数值稳定性而加的常数
        self.m_b = [np.zeros(b.shape) for b in self.biases]  # 一阶矩估计的初始化
        self.v_b = [np.zeros(b.shape) for b in self.biases]  # 二阶矩估计的初始化
        self.m_w = [np.zeros(w.shape) for w in self.weights]  # 一阶矩估计的初始化
        self.v_w = [np.zeros(w.shape) for w in self.weights]  # 二阶矩估计的初始化
        self.beta1_t = 1.0  # 一阶矩估计的衰减率的时间步长
        self.beta2_t = 1.0  # 二阶矩估计的衰减率的时间步长


    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def ADAM(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}  {:.2%}".format(j,self.evaluate(test_data),n_test, self.evaluate(test_data)/n_test))
            else:
                print("Epoch {} complete".format(j))


    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            self.m_b = [self.beta1 * m + (1 - self.beta1) * nb
                    for m, nb in zip(self.m_b, nabla_b)]
        self.v_b = [self.beta2 * v + (1 - self.beta2) * np.square(nb)
                    for v, nb in zip(self.v_b, nabla_b)]
        self.m_w = [self.beta1 * m + (1 - self.beta1) * nw
                    for m, nw in zip(self.m_w, nabla_w)]
        self.v_w = [self.beta2 * v + (1 - self.beta2) * np.square(nw)
                    for v, nw in zip(self.v_w, nabla_w)]
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2
        m_b_corrected = [m / (1 - self.beta1_t) for m in self.m_b]
        v_b_corrected = [v / (1 - self.beta2_t) for v in self.v_b]
        m_w_corrected = [m / (1 - self.beta1_t) for m in self.m_w]
        v_w_corrected = [v / (1 - self.beta2_t) for v in self.v_w]

        self.weights = [w - eta * m_w_corr / (np.sqrt(v_w_corr) + self.epsilon)
                        for w, m_w_corr, v_w_corr in zip(self.weights, m_w_corrected, v_w_corrected)]
        self.biases = [b - eta * m_b_corr / (np.sqrt(v_b_corr) + self.epsilon)
                        for b, m_b_corr, v_b_corr in zip(self.biases, m_b_corrected, v_b_corrected)]
    

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
    
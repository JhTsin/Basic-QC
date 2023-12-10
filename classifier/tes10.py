import mnist_loader10
import network_ADAM_adaboost
training_data, validation_data, test_data = mnist_loader10.load_data_wrapper()
net = network_ADAM_adaboost.Network([784,5,10])
net.ADAM(training_data, 100, 100 ,0.01 , test_data=test_data)

# import network_SGD
# training_data, validation_data, test_data = mnist_loader10.load_data_wrapper()
# net = network_SGD.Network([784, 10])
# net.SGD(training_data, 5, 100, 3.0, test_data=test_data)

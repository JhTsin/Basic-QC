import mnist_loader
a = 0
b = 2
c = 7
d = 9
import network_ADAM
training_data, validation_data, test_data = mnist_loader.load_data_wrapper(a,b,c,d)
net = network_ADAM.Network([784,10])
net.ADAM(training_data, 30, 10 ,0.001 , test_data=test_data)

# import network_SGD 
# training_data, validation_data, test_data = mnist_loader.load_data_wrapper(a,b,c,d)
# net = network_SGD.Network([784, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

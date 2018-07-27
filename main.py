from model import Network
import mnist.src.mnist_loader

training_data, validation_data, test_data = mnist.src.mnist_loader.load_data_wrapper()
net = Network(layers=[784, 70, 10])
net.sgd(
    training_data,

    # Hyperparameters
    epochs=30,
    mini_batch_size=10,
    eta=3.0,

    test_data=test_data
)

"""
Log:
layers=[784, 30, 10]      Epoch 29 : 9474 / 10000
layers=[784, 100, 10]     Epoch 0 : 3877 / 10000    to      Epoch 16: 96XX / 10000
layers=[784, 100, 10]     Epoch 0 : 5739 / 10000    to      Epoch 29 : 6887 / 10000
layers=[784, 70, 10]      Epoch 0 : 9158 / 10000    to      Epoch 29 : 9616 / 10000
"""
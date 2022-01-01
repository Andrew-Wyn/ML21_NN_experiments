import matplotlib.pyplot as plt

from mlprj.optimizers import StochasticGradientDescent
from mlprj.datasets import read_monk
from mlprj.model_selection import grid_search_cv, split_train_params
from mlprj.feed_forward import Network, Layer
from mlprj.utility import model_accuracy

from mlprj.losses import *
from mlprj.regularizers import *
from mlprj.initializers import *


def monk_build_model(units, learning_rate, alpha, lambdareg = 0):
    nn = Network(17, [
        Layer(units, "relu", UniformInitializer(min_value=-0.1, max_value=0.1)),
        Layer(1, "sigmoid", UniformInitializer(min_value=-0.1, max_value=0.1))
    ])

    nn.compile(loss=MSE(), regularizer=L2Regularizer(lambdareg), optimizer=StochasticGradientDescent(learning_rate, alpha))

    return nn

def mean_fun(units, training, test, learning_rate, alpha, lambdareg,  epochs, batch_size, trials):
    mean = {"loss_tr" : [], "loss_vl" : [], "accuracy_tr": [], "accuracy_ts": []}
    for i in range(trials):
        model = monk_build_model(units, learning_rate, alpha, lambdareg)
        history = model.training(training, test, epochs = epochs, batch_size = batch_size, verbose = False, accuracy_curve=True)
        if len(mean["loss_tr"]) == 0 :
            mean["loss_tr"] +=  history["loss_tr"]
            mean["loss_vl"] +=  history["loss_vl"]
            mean["accuracy_tr"] +=  history["accuracy_tr"]
            mean["accuracy_ts"] +=  history["accuracy_vl"]
            print("Done!")
        else:
            mean["loss_tr"][:] = [x + y for x, y in zip(mean["loss_tr"], history["loss_tr"])]
            mean["loss_vl"][:] = [x + y for x, y in zip(mean["loss_vl"], history["loss_vl"])]
            mean["accuracy_tr"][:] = [x + y for x, y in zip(mean["accuracy_tr"], history["accuracy_tr"])]
            mean["accuracy_ts"][:] = [x + y for x, y in zip(mean["accuracy_ts"], history["accuracy_vl"])]
            print("Done!")

    mean["loss_tr"][:] = [x / trials for x in mean["loss_tr"]]
    mean["loss_vl"][:] = [x / trials for x in mean["loss_vl"]]
    mean["accuracy_tr"][:] = [x / trials for x in mean["accuracy_tr"]]
    mean["accuracy_ts"][:] = [x / trials for x in mean["accuracy_ts"]]
    return mean

if __name__ == "__main__":
    monk_1_X, monk_1_test_x, monk_1_y, monk_1_test_y, _ = read_monk(1)
    monk_2_X, monk_2_test_x, monk_2_y, monk_2_test_y, _ = read_monk(2)
    monk_3_X, monk_3_test_x, monk_3_y, monk_3_test_y, _ = read_monk(3)   

    mean = mean_fun(16, (monk_3_X, monk_3_y), (monk_3_test_x, monk_3_test_y), learning_rate = 0.8, alpha = 0.6, lambdareg = 0.001, epochs = 50, batch_size = "full", trials = 10)

    plt.plot(mean["loss_tr"])
    plt.plot(mean["loss_vl"])
    plt.show()

    plt.plot(mean["accuracy_tr"])
    plt.plot(mean["accuracy_ts"])
    plt.show()

    print(f"training accuracy: {mean['accuracy_tr'][-1]}")
    print(f"test accuracy: {mean['accuracy_ts'][-1]}")
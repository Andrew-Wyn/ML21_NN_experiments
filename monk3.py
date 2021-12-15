import matplotlib.pyplot as plt

from mlprj.optimizers import StochasticGradientDescent
from mlprj.datasets import *
from mlprj.model_selection import *
from mlprj.losses import *
from mlprj.regularizers import *
from mlprj.initializers import *
from mlprj.feed_forward import Network, Layer
from mlprj.utility import model_accuracy
from mlprj.randomized_nn import *


def monk_3_build_model(learning_rate, alpha, lambdareg = 0):
    nn = Network(17, [
        Layer(16, "relu", GaussianInitializer()),
        Layer(8, "relu", GaussianInitializer()),
        Layer(4, "relu", GaussianInitializer()),
        Layer(1, "sigmoid", GaussianInitializer())
    ])

    nn.compile(loss=MSE(), regularizer=L2Regularizer(lambdareg), optimizer=StochasticGradientDescent(learning_rate, alpha))

    return nn



if __name__ == "__main__":
    # Monk-1 datas
    monk_3_X, monk_3_test_x, monk_3_y, monk_3_test_y = read_monk(3)

    monk_3_params = {
        "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "alpha": [0.1, 0.2, 0.3, 0.4, 0.5],
        "lambdareg": [0, 0.0001, 0.001],
        "epochs": 500,
        "batch_size": "full"
    }

    monk_3_best_params = grid_search_cv(monk_3_build_model, (monk_3_X, monk_3_y), monk_3_params, path="monk3.csv")
    monk_3_best_params_other, monk_3_best_params_training = split_train_params(monk_3_best_params)
    print(monk_3_best_params_other, monk_3_best_params_training)

    model = monk_3_build_model(**monk_3_best_params_other)
    history = model.training((monk_3_X, monk_3_y), **monk_3_best_params_training, verbose = True)

    plt.plot(history["loss_tr"])
    plt.plot(history["loss_vl"])
    plt.show()

    print(f"training accuracy:{model_accuracy(model, monk_3_X, monk_3_y)}")
    print(f"test accuracy:{model_accuracy(model, monk_3_test_x, monk_3_test_y)}")


import matplotlib.pyplot as plt

from mlprj.optimizers import StochasticGradientDescent
from mlprj.datasets import read_cup
from mlprj.model_selection import grid_search_cv, split_train_params
from mlprj.feed_forward import Network, Layer

from mlprj.losses import *
from mlprj.regularizers import *
from mlprj.initializers import *
from mlprj.utility import model_loss


def cup_build_model_train(learning_rate, alpha, lambdareg = 0, decay = None):
    nn = Network(10, [ 
        Layer(50, "relu", GaussianInitializer()),
        Layer(20, "sigmoid", GaussianInitializer()),
        Layer(10, "relu", GaussianInitializer()),
        Layer(2, "linear", GaussianInitializer())
      ])

    nn.compile(loss=MSE(), regularizer=L2Regularizer(lambdareg), optimizer=StochasticGradientDescent(learning_rate, alpha, decay))
    return nn

def cup_build_model_test(learning_rate, alpha, lambdareg = 0, decay = None):
    nn = Network(10, [ 
        Layer(50, "relu", GaussianInitializer()),
        Layer(20, "sigmoid", GaussianInitializer()),
        Layer(10, "relu", GaussianInitializer()),
        Layer(2, "linear", GaussianInitializer())
      ])

    nn.compile(loss=MEE(), regularizer=L2Regularizer(lambdareg), optimizer=StochasticGradientDescent(learning_rate, alpha, decay))
    return nn


if __name__ == "__main__":
    # Cup data
    X, test_x, y, test_y= read_cup()
    
    cup_params = {
        "learning_rate" : [0.01],
        "alpha" : [0.15],
        "lambdareg" : [0.00001],
        "epochs" : 5,
        "batch_size" : "full",
        "early_stopping" : 20
    }
    cup_best_params = grid_search_cv(cup_build_model_train, (X, y), cup_params, k_folds = 5, path="cup")
    cup_best_params_other, cup_best_params_training = split_train_params(cup_best_params, direct = False)
    print(cup_best_params_other, cup_best_params_training)

    model = cup_build_model_test(**cup_best_params_other)
    history = model.training((X, y), **cup_best_params_training, verbose = True)

    plt.plot(history["loss_tr"])
    plt.show()

    print(f"train MEE loss: {model_loss(model, MEE(), X, y)}")
    print(f"test MEE loss: {model_loss(model, MEE(), test_x, test_y)}")
import matplotlib.pyplot as plt

from mlprj.optimizers import StochasticGradientDescent
from mlprj.datasets import *
from mlprj.model_selection import *
from mlprj.losses import *
from mlprj.regularizers import *
from mlprj.initializers import *
from mlprj.feed_forward import Network, Layer
from mlprj.randomized_nn import *
from sklearn.model_selection import train_test_split

def cup_build_model(learning_rate, alpha, lambdareg = 0, decay = None):
    nn = Network(10, [ 
        Layer(50, "relu", GaussianInitializer()),
        Layer(20, "sigmoid", GaussianInitializer()),
        Layer(10, "relu", GaussianInitializer()),
        Layer(2, "linear", GaussianInitializer())
      ])

    nn.compile(loss=MSE(), regularizer=L2Regularizer(lambdareg), optimizer=StochasticGradientDescent(learning_rate, alpha, decay))
    return nn



if __name__ == "__main__":
    # Cup data
    cup_X, cup_y= read_cup()
    cup_train_x, cup_test_x, cup_train_y, cup_test_y = train_test_split(cup_X, cup_y, test_size=0.10, random_state=42)
    
    cup_params = {
        "learning_rate" : [0.1, 0.2],
        "alpha" : [0.1, 0.2],
        "lambdareg" : [0.00001, 0.00005],
        "decay": [(10, 0.01), (20, 0.01)],
        "epochs" : 500,
        "batch_size" : 64,
        "early_stopping" : 20
    }
    cup_best_params = grid_search_cv(cup_build_model, (cup_train_x, cup_train_y), cup_params, k_folds = 5)
    cup_best_params_other, cup_best_params_training = split_train_params(cup_best_params, direct = False)
    print(cup_best_params_other, cup_best_params_training)

    model = cup_build_model(**cup_best_params_other)
    history = model.training((cup_train_x, cup_train_y), (cup_test_x, cup_test_y), **cup_best_params_training, verbose = True)

    plt.plot(history["loss_tr"])
    plt.plot(history["loss_vl"])
    plt.show()

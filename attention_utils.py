import keras.backend as K
import numpy as np
from keras.utils import plot_model
import random

def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


def get_data(n, input_dim, attention_column=1):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    x = np.random.standard_normal(size=(n, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))

    # yの先頭の値を attention_column とするテストデータを作る.
    x[:, attention_column] = y[:, 0]
    return x, y

# データを作成する
# attentionの位置を前後に揺らしてみたときに、attentionが正しく登場するかを確認できる
def get_data_recurrent(n, time_steps, input_dim, base_attention_column=8):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param time_steps: the number of time steps of your series.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """

    information_nugget_value = 1.3
    information_nugget_value_max = 2
    information_nugget_value_min = 0

    x = np.random.standard_normal(size=(n, time_steps, input_dim))
    y = np.random.randint(low=information_nugget_value_min, high=(information_nugget_value_max+1), size=(n, 1))
    attention_columns = []
    
    # attentionの位置を前後に揺らしてみたときに、attentionが正しく登場するかを確認したい
    print(len(x[:, base_attention_column, :]))
    for i in range(len(x)):
        #print(i)
        this_attention_column = base_attention_column+random.randrange(2)
        x[i, this_attention_column, :] = np.tile(y[i]*information_nugget_value, (1, input_dim))
        attention_columns.append(this_attention_column)
    #x[:, attention_column, :] = np.tile(y[:]*information_nugget_value, (1, input_dim))
    return x, y, attention_columns

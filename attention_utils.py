import keras.backend as K
import numpy as np
from keras.utils import plot_model
import random

def get_activations(model, inputs, print_shape_only=True, layer_name=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
    print('----- activations -----')
    activations = []
    inp = model.input

    # 任意レイヤーの重みを出力
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    
    print("outputs",outputs) # [<tf.Tensor 'attention_vec/transpose:0' shape=(?, 20, 32) dtype=float32>]
    print("inp",inp) # Tensor("input_1:0", shape=(?, 20, 1), dtype=float32)
    
    # K.function は特定の層（中間レイヤーの）出力をを取り出す
    # K.learning_phase(): はテスト時か訓練時かを表すプレースホルダー。最初に作成したモデルにはドロップアウトが含まれているが
    # ドロップアウトは訓練時のみに適用され、テスト時には適用されません。そのため、テスト時か訓練時かを教えてあげる必要があります。
    # print("K.learning_phase()",K.learning_phase())
    # print([inp] + [K.learning_phase()])
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    
    # funcs[0]([inputs, 1.])で引数が２つなのは、上記のK.functionのinputの定義は２つだから.
    print("funcs[0]([inputs, 1.])[0]:", funcs[0]([inputs, 0])[0] ) # K.learning_phase()へのinputは、テスト時には0で良い？
    print(inputs) # TIME_STEPS次元
    print(funcs) # 配列の要素数は1, [<keras.backend.tensorflow_backend.Function object at 0x11ca43b70>]
    print(funcs[0])
    #quit()

    # Keras function を実行する
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]

    #print(layer_outputs)
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape) # (1, 20, 32)
        else:
            print(layer_activations)
    
    print(activations)
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
def get_data_recurrent(data_size, time_steps, input_dim, base_attention_column=5, random_position=False):
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

    information_nugget_value = 0.5
    information_nugget_value_max = 1
    information_nugget_value_min = 0

    x = np.random.standard_normal(size=(data_size, time_steps, input_dim))
    # data_size分のテストセットを作成(ランダムで、結果の0,1を決める)
    y = np.random.randint(low=information_nugget_value_min, high=(information_nugget_value_max+1), size=(data_size, 1))
    #print(y)

    attention_columns = []
    
    # attentionの位置を前後に揺らしてみたときに、attentionが正しく登場するかを確認したい
    print(len(x[:, base_attention_column, :]))
    for i in range(len(x)): # data_sizeイテレーション
        # print(i)
        if random_position:
            this_attention_column = base_attention_column+random.randrange(5)
        else:
            this_attention_column = base_attention_column
        # this_attention_column の場所に information nugget を配置する
        random_vec = [0.2]
        x[i, this_attention_column, :] = np.tile(y[i]*information_nugget_value if y[i] > 0 else random_vec[0], (1, input_dim))
        attention_columns.append(this_attention_column)
    # print(max(x[0][0]))
    
    #x[:, attention_column, :] = np.tile(y[:]*information_nugget_value, (1, input_dim))
    return x, y, attention_columns

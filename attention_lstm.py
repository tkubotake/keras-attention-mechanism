from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from keras.utils import plot_model

from attention_utils import get_activations, get_data_recurrent

INPUT_DIM = 2
TIME_STEPS = 20
LSTM_UNITS = 32
# if True, the attention vector is shared across the input_dimensions where the attention is applied.
SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = False


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    print("input_dim",input_dim)
    
    print("inputs shape",inputs.shape)
    a = Permute((2, 1))(inputs)
    print("before shape",a.shape)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    print("after shape",a.shape)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    print("a_probs shape",a_probs.shape)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    print("output_attention_mul shape",output_attention_mul.shape)
    return output_attention_mul


def model_attention_applied_after_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    print("inputs shape",inputs.shape)
    lstm_out = LSTM(LSTM_UNITS, return_sequences=True)(inputs)
    print("lstm_out shape",lstm_out.shape)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


def model_attention_applied_before_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    attention_mul = attention_3d_block(inputs)
    attention_mul = LSTM(LSTM_UNITS, return_sequences=False)(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


if __name__ == '__main__':

    N = 300000
    # N = 300 #-> too few = no training
    inputs_1, outputs, attention_columns = get_data_recurrent(N, TIME_STEPS, INPUT_DIM)
    
    # for debug
    for ntest in [0, 1]:
        print( len(inputs_1[ntest]), inputs_1[ntest], inputs_1[ntest][10] )
        print( outputs[ntest] ) 
        print( attention_columns[ntest] )
    
    if APPLY_ATTENTION_BEFORE_LSTM:
        m = model_attention_applied_before_lstm()
    else:
        m = model_attention_applied_after_lstm()

    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(m.summary())
    plot_model(m, to_file='model_lstm.png',show_shapes=True)     

    m.fit([inputs_1], outputs, epochs=1, batch_size=64, validation_split=0.1)

    attention_vectors = []
    for i in range(300):
        testing_inputs_1, testing_outputs, attention_columns = get_data_recurrent(1, TIME_STEPS, INPUT_DIM)

        # attention_vec を指定して値を取り出している
        # test = get_activations(m,
        #     testing_inputs_1,
        #     print_shape_only=True,
        #     layer_name='attention_vec')
        #print(test)
        # print(len(test[0][0]))
        # print(len(test[0][0][0]))
        # print(np.mean(test[0], axis=2))

        # TODO: attentionの位置を前後に揺らしてみたときに、attentionが正しく登場するかを確認したい
        print("testing_inputs_1",testing_inputs_1)
        print("testing_outputs",testing_outputs)
        print("attention_columns",attention_columns)
        attention_vector = np.mean(get_activations(m,
                                                   testing_inputs_1,
                                                   print_shape_only=True,
                                                   layer_name='attention_vec')[0], axis=2).squeeze()
        print('attention =', attention_vector)
        print('attention =', attention_vector[attention_columns[0]])
        assert (np.sum(attention_vector) - 1.0) < 1e-5
        attention_vectors.append(attention_vector)

        if(testing_outputs.squeeze() == 1):
            quit()
    print(len(attention_vectors))
    attention_vector_final = np.mean(np.array(attention_vectors), axis=0)
    # plot part.
    import matplotlib.pyplot as plt
    import pandas as pd

    pd.DataFrame(attention_vector_final, columns=['attention (%)']).plot(kind='bar',
                                                                         title='Attention Mechanism as '
                                                                               'a function of input'
                                                                               ' dimensions.')
    plt.show()

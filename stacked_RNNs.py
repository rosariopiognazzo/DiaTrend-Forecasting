from tensorflow import keras
from keras import layers

def make_stacked_RNNs(input_shape, dropout, type_model, num_layers=3, hidden_units=64, bidirectional=False):

    input = keras.Input(shape=input_shape) # input shape deve essere (features, timesteps)
    
    if type_model == 'LSTM':
        if bidirectional:
            x = input
            for n in range(num_layers):
                x = layers.Bidirectional(layers.LSTM(hidden_units, return_sequences=True, recurrent_dropout=dropout, unroll=True))(x)
        else:
            x = input
            for n in range(num_layers):
                x = layers.LSTM(hidden_units, return_sequences=True, recurrent_dropout=dropout, unroll=True)(x)

    elif type_model == 'GRU':
        if bidirectional:
            x = input
            for n in range(num_layers):
                x = layers.Bidirectional(layers.GRU(hidden_units, return_sequences=True, recurrent_dropout=dropout, unroll=True))(x)
        else:
            x = input
            for n in range(num_layers):
                x = layers.GRU(hidden_units, return_sequences=True, recurrent_dropout=dropout, unroll=True)(x)
    else:
        raise ValueError("Unsupported RNN type. Use 'LSTM' or 'GRU'.")
    
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs=input, outputs=outputs)
    
    return model
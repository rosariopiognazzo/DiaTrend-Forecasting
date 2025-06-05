from tensorflow import keras
from keras import layers

def make_stacked_RNNs(input_shape, forecast_horizon, dropout=0.2, type_model='LSTM', 
                     num_layers=3, hidden_units=64, bidirectional=False):
    """
    Create optimized stacked RNN model for time series forecasting.
    
    Args:
        input_shape: (timesteps, features) - sequence length and number of features
        forecast_horizon: Number of future timesteps to predict
        dropout: Dropout rate (default: 0.2, set to 0.0 for cuDNN optimization)
        type_model: 'LSTM' or 'GRU'
        num_layers: Number of RNN layers
        hidden_units: Number of units in each RNN layer
        bidirectional: Whether to use bidirectional RNNs
    
    Returns:
        Compiled Keras model for time series forecasting
    """
    
    input_layer = keras.Input(shape=input_shape)
    x = input_layer
    
    # Stack RNN layers - all but last have return_sequences=True
    if type_model == 'LSTM':
        for i in range(num_layers):
            return_sequences = (i < num_layers - 1)  # Last layer returns only final output
            
            if bidirectional:
                x = layers.Bidirectional(
                    layers.LSTM(
                        hidden_units, 
                        return_sequences=return_sequences,
                        recurrent_dropout=dropout if dropout > 0 else 0.0,
                        unroll=False  # Set to False for better GPU optimization
                    )
                )(x)
            else:
                x = layers.LSTM(
                    hidden_units, 
                    return_sequences=return_sequences,
                    recurrent_dropout=dropout if dropout > 0 else 0.0,
                    unroll=False
                )(x)
            
            # Add dropout between layers (except after last layer)
            if i < num_layers - 1 and dropout > 0:
                x = layers.Dropout(dropout)(x)
                
    elif type_model == 'GRU':
        for i in range(num_layers):
            return_sequences = (i < num_layers - 1)
            
            if bidirectional:
                x = layers.Bidirectional(
                    layers.GRU(
                        hidden_units, 
                        return_sequences=return_sequences,
                        recurrent_dropout=dropout if dropout > 0 else 0.0,
                        unroll=False
                    )
                )(x)
            else:
                x = layers.GRU(
                    hidden_units, 
                    return_sequences=return_sequences,
                    recurrent_dropout=dropout if dropout > 0 else 0.0,
                    unroll=False
                )(x)
            
            # Add dropout between layers (except after last layer)
            if i < num_layers - 1 and dropout > 0:
                x = layers.Dropout(dropout)(x)
    else:
        raise ValueError("Unsupported RNN type. Use 'LSTM' or 'GRU'.")
    
    # Final dropout before output layer
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    
    # Output layer for multi-step forecasting
    outputs = layers.Dense(forecast_horizon, activation='linear')(x)
    
    model = keras.Model(inputs=input_layer, outputs=outputs)
    
    return model
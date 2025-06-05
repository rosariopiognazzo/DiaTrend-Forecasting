import numpy as np
import tensorflow as tf
from typing import Optional, List, Dict, Tuple, Union

## POSITIONAL ENCODING OTTIMIZZATO
def positional_encoding(length, depth):
    """
    Positional encoding ottimizzato per time series.
    Supporta lunghezze dinamiche e cache per efficienza.
    """
    depth = depth / 2
    
    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth # (1, depth)
    
    angle_rates = 1 / (10000 ** depths)              # (1, depth)
    angle_rads = positions * angle_rates             # (pos, depth)
    
    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1
    )
    
    return tf.cast(pos_encoding, dtype=tf.float32)

class TimeSeriesEmbedding(tf.keras.layers.Layer):
    """
    Embedding layer ottimizzato per time series usando CNN 1D.
    Supporta skip connections e normalization per miglior training.
    """
    def __init__(self, d_model, num_conv_layers=2, kernel_size=3, 
                 activation='relu', dropout_rate=0.1, use_skip_connections=True):
        super().__init__()
        self.d_model = d_model
        self.num_conv_layers = num_conv_layers
        self.use_skip_connections = use_skip_connections
        
        # Conv layers con skip connections
        self.conv_layers = []
        self.layer_norms = []
        self.dropouts = []
        
        for i in range(num_conv_layers):
            self.conv_layers.append(tf.keras.layers.Conv1D(
                filters=d_model,
                kernel_size=kernel_size,
                padding='same',
                activation=activation,
                name=f'conv1d_{i}'
            ))
            self.layer_norms.append(tf.keras.layers.LayerNormalization(name=f'norm_{i}'))
            self.dropouts.append(tf.keras.layers.Dropout(dropout_rate, name=f'dropout_{i}'))
        
        # Projection layer se necessario
        self.projection = tf.keras.layers.Dense(d_model, name='embedding_projection')
        self.final_norm = tf.keras.layers.LayerNormalization(name='final_embedding_norm')
    
    def call(self, x, training=None):
        # x shape: (batch, seq_len, features)
        
        # Project to d_model se necessario
        if x.shape[-1] != self.d_model:
            x = self.projection(x)
        
        # Apply conv layers con skip connections
        for i, (conv, norm, dropout) in enumerate(zip(self.conv_layers, self.layer_norms, self.dropouts)):
            residual = x
            x = conv(x)
            x = norm(x)
            x = dropout(x, training=training)
            
            # Skip connection se possibile
            if self.use_skip_connections and x.shape == residual.shape:
                x = x + residual
        
        x = self.final_norm(x)
        return x

class MultiHeadAttentionWithWeights(tf.keras.layers.Layer):
    """
    Multi-Head Attention che salva i pesi per interpretabilità.
    """
    def __init__(self, num_heads, key_dim, dropout=0.0, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout,
            **kwargs
        )
        self.last_attention_weights = None
    
    def call(self, query, value=None, key=None, return_attention_scores=True, **kwargs):
        if value is None:
            value = query
        if key is None:
            key = query
            
        output, attention_weights = self.mha(
            query=query,
            value=value,
            key=key,
            return_attention_scores=True,
            **kwargs
        )
        
        # Salva per interpretabilità
        self.last_attention_weights = attention_weights
        
        if return_attention_scores:
            return output, attention_weights
        return output

class GlobalSelfAttention(tf.keras.layers.Layer):
    """
    Global Self Attention ottimizzato per forecasting con interpretabilità.
    """
    def __init__(self, num_heads, key_dim, dropout=0.0, name=None):
        super().__init__(name=name)
        self.mha = MultiHeadAttentionWithWeights(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout
        )
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()
    
    def call(self, x, training=None):
        attn_output, attn_weights = self.mha(x, training=training)
        
        # Residual connection + LayerNorm
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        
        return x
    
    def get_attention_weights(self):
        """Estrae i pesi di attenzione per interpretabilità"""
        return self.mha.last_attention_weights

class FeedForward(tf.keras.layers.Layer):
    """
    Feed Forward layer ottimizzato con GELU activation e skip connections.
    """
    def __init__(self, d_model, dff, dropout_rate=0.1, activation='gelu'):
        super().__init__()
        self.d_model = d_model
        self.dff = dff
        
        self.dense1 = tf.keras.layers.Dense(dff, activation=activation)
        self.dense2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()
    
    def call(self, x, training=None):
        residual = x
        
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        
        # Residual connection + LayerNorm
        x = self.add([residual, x])
        x = self.layer_norm(x)
        
        return x

class EncoderLayer(tf.keras.layers.Layer):
    """
    Encoder layer ottimizzato per forecasting.
    """
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, name=None):
        super().__init__(name=name)
        
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            name=f'{name}_attention' if name else None
        )
        
        self.ffn = FeedForward(
            d_model=d_model,
            dff=dff,
            dropout_rate=dropout_rate
        )
    
    def call(self, x, training=None):
        x = self.self_attention(x, training=training)
        x = self.ffn(x, training=training)
        return x
    
    def get_attention_weights(self):
        """Estrae i pesi di attenzione per interpretabilità"""
        return self.self_attention.get_attention_weights()

class ForecastEncoder(tf.keras.layers.Layer):
    """
    Encoder ottimizzato per forecasting di time series con interpretabilità.
    """
    def __init__(self, num_layers, d_model, num_heads, dff, input_features,
                 dropout_rate=0.1, num_conv_layers=2, kernel_size=3,
                 max_seq_length=512, name='forecast_encoder'):
        super().__init__(name=name)
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        
        # Time series embedding
        self.embedding = TimeSeriesEmbedding(
            d_model=d_model,
            num_conv_layers=num_conv_layers,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate
        )
        
        # Positional encoding (cache per efficienza)
        self.pos_encoding = positional_encoding(max_seq_length, d_model)
        
        # Encoder layers
        self.enc_layers = []
        for i in range(num_layers):
            self.enc_layers.append(
                EncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dff=dff,
                    dropout_rate=dropout_rate,
                    name=f'encoder_layer_{i}'
                )
            )
        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.final_norm = tf.keras.layers.LayerNormalization(name='final_encoder_norm')
    
    def call(self, x, training=None, return_attention_weights=False):
        # x shape: (batch, seq_len, features)
        seq_len = tf.shape(x)[1]
        
        # Embedding
        x = self.embedding(x, training=training)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        pos_encoding = tf.convert_to_tensor(self.pos_encoding)[:seq_len, :]
        x = x + pos_encoding[tf.newaxis, :, :]
        
        # Dropout
        x = self.dropout(x, training=training)
        
        # Store attention weights se richiesto
        attention_weights = []
        
        # Passa attraverso encoder layers
        for layer in self.enc_layers:
            x = layer(x, training=training)
            
            if return_attention_weights:
                attention_weights.append(layer.get_attention_weights())
        
        # Final normalization
        x = self.final_norm(x)
        
        if return_attention_weights:
            return x, attention_weights
        return x
    
    def get_attention_weights(self, x, training=False):
        """
        Estrae i pesi di attenzione per interpretabilità.
        
        Returns:
            List di tensori con shape (batch, num_heads, seq_len, seq_len)
        """
        _, attention_weights = self.call(x, training=training, return_attention_weights=True)
        return attention_weights

class TransformerForecaster(tf.keras.Model):
    """
    Modello completo per forecasting basato su Transformer Encoder
    con supporto per interpretabilità.
    """
    def __init__(self, num_layers, d_model, num_heads, dff, input_features,
                 forecast_horizon, dropout_rate=0.1, num_conv_layers=2,
                 kernel_size=3, aggregation='attention', max_seq_length=512):
        super().__init__()
        
        self.forecast_horizon = forecast_horizon
        self.aggregation = aggregation
        self.d_model = d_model
        
        # Encoder
        self.encoder = ForecastEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            input_features=input_features,
            dropout_rate=dropout_rate,
            num_conv_layers=num_conv_layers,
            kernel_size=kernel_size,
            max_seq_length=max_seq_length
        )
        
        # Aggregation layers
        if aggregation == 'global_avg':
            self.aggregation_layer = tf.keras.layers.GlobalAveragePooling1D()
        elif aggregation == 'global_max':
            self.aggregation_layer = tf.keras.layers.GlobalMaxPooling1D()
        elif aggregation == 'last':
            self.aggregation_layer = lambda x: x[:, -1, :]
        elif aggregation == 'attention':
            # Learnable attention weights per aggregation
            self.attention_dense = tf.keras.layers.Dense(1, use_bias=False)
            self.aggregation_layer = self._attention_aggregation
        
        # Forecasting head ottimizzato
        self.forecast_head = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model, activation='gelu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(dropout_rate),
            
            tf.keras.layers.Dense(d_model // 2, activation='gelu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(dropout_rate),
            
            tf.keras.layers.Dense(forecast_horizon, name='forecast_output')
        ], name='forecast_head')
        
        # Per salvare attention weights
        self.last_attention_weights = None
        self.last_aggregation_weights = None
    
    def _attention_aggregation(self, x):
        """
        Aggregation usando attention learnable.
        
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            aggregated: (batch, d_model)
        """
        # Calcola attention scores
        attention_scores = self.attention_dense(x)  # (batch, seq_len, 1)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        
        # Salva per interpretabilità
        self.last_aggregation_weights = attention_weights
        
        # Weighted sum
        aggregated = tf.reduce_sum(x * attention_weights, axis=1)  # (batch, d_model)
        
        return aggregated
    
    def call(self, x, training=None, return_attention_weights=False):
        # x shape: (batch, seq_len, features)
        
        # Encode
        if return_attention_weights:
            encoded, attention_weights = self.encoder(
                x, training=training, return_attention_weights=True
            )
            self.last_attention_weights = attention_weights
        else:
            encoded = self.encoder(x, training=training)
        
        # Aggregate
        if self.aggregation == 'last':
            aggregated = encoded[:, -1, :]
        else:
            aggregated = self.aggregation_layer(encoded)
        
        # Forecast
        forecast = self.forecast_head(aggregated, training=training)
        
        if return_attention_weights:
            return forecast, attention_weights
        return forecast
    
    def get_attention_maps(self, x, training=False):
        """
        Estrae tutte le attention maps per interpretabilità.
        
        Args:
            x: Input tensor (batch, seq_len, features)
            
        Returns:
            Dict con:
            - 'encoder_attention': List di (batch, num_heads, seq_len, seq_len)
            - 'aggregation_attention': (batch, seq_len, 1) se aggregation='attention'
        """
        # Reset weights precedenti
        self.last_attention_weights = None
        self.last_aggregation_weights = None
        
        # Forward pass con attention weights
        _ = self.call(x, training=training, return_attention_weights=True)
        
        attention_maps = {
            'encoder_attention': self.last_attention_weights
        }
        
        if self.aggregation == 'attention' and self.last_aggregation_weights is not None:
            attention_maps['aggregation_attention'] = self.last_aggregation_weights
        
        return attention_maps
    
    def interpret_prediction(self, x, feature_names=None) -> Dict:
        """
        Genera interpretazione completa della predizione.
        
        Args:
            x: Input tensor (batch, seq_len, features)
            feature_names: Lista nomi delle features
            
        Returns:
            Dict con interpretazione dettagliata
        """
        # Get prediction
        prediction = self.call(x, training=False)
        
        # Get attention maps
        attention_maps = self.get_attention_maps(x, training=False)
        
        # Analizza attention patterns
        encoder_attention = attention_maps['encoder_attention']
        
        # Media su heads e layers per interpretazione semplificata
        avg_attention = []
        for layer_attention in encoder_attention: # type: ignore
            if layer_attention is not None:
                # Media su heads: (batch, seq_len, seq_len)
                layer_avg = tf.reduce_mean(layer_attention, axis=1)
                avg_attention.append(layer_avg)
            
        # Stack e media su layers: (batch, seq_len, seq_len)
        if avg_attention:
            final_attention = tf.reduce_mean(tf.stack(avg_attention), axis=0)
        else:
            final_attention = None
        
        interpretation = {
            'prediction': prediction,
            'attention_maps': attention_maps,
            'average_attention': final_attention,
            'temporal_importance': None,
            'feature_importance': None
        }
        
        # Calcola importanza temporale (attenzione su timesteps)
        if final_attention is not None:
            # Somma attention ricevuta da ogni timestep
            temporal_importance = tf.reduce_sum(final_attention, axis=-1)  # (batch, seq_len)
            interpretation['temporal_importance'] = temporal_importance
        
        # Aggregation attention se disponibile
        if 'aggregation_attention' in attention_maps:
            interpretation['aggregation_importance'] = attention_maps['aggregation_attention']
        
        return interpretation

# Funzioni utility per visualizzazione
def plot_attention_weights(attention_weights, timesteps, layer_idx=0, head_idx=0):
    """
    Plotta i pesi di attenzione per interpretabilità.
    
    Args:
        attention_weights: Output di get_attention_maps()
        timesteps: Lista di timestamps o indici
        layer_idx: Quale layer visualizzare
        head_idx: Quale head visualizzare
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    encoder_attention = attention_weights['encoder_attention']
    
    if encoder_attention and len(encoder_attention) > layer_idx:
        # Estrai attention del layer/head specifico
        layer_attention = encoder_attention[layer_idx]  # (batch, num_heads, seq_len, seq_len)
        
        if layer_attention.shape[0] > 0:  # Se c'è almeno un batch
            attention_matrix = layer_attention[0, head_idx].numpy()  # (seq_len, seq_len)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(attention_matrix, 
                       xticklabels=timesteps, 
                       yticklabels=timesteps,
                       cmap='Blues',
                       cbar_kws={'label': 'Attention Weight'})
            plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
            plt.xlabel('Key Position')
            plt.ylabel('Query Position')
            plt.tight_layout()
            plt.show()

def plot_temporal_importance(interpretation, timesteps=None):
    """
    Plotta l'importanza temporale per interpretabilità.
    """
    import matplotlib.pyplot as plt
    
    temporal_importance = interpretation.get('temporal_importance')
    
    if temporal_importance is not None:
        # Prendi il primo sample del batch
        importance = temporal_importance[0].numpy()
        
        plt.figure(figsize=(12, 4))
        x_labels = timesteps if timesteps is not None else range(len(importance))
        
        plt.bar(x_labels, importance, alpha=0.7, color='skyblue')
        plt.title('Temporal Importance (Attention-based)')
        plt.xlabel('Timestep')
        plt.ylabel('Importance Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

import numpy as np
import tensorflow as tf

## POSITIONAL ENCODING
# Classical positional encoding from: https://www.tensorflow.org/text/tutorials/transformer
def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
    self.pos_encoding = positional_encoding(length=2048, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    pos_encoding_dense = tf.convert_to_tensor(self.pos_encoding)
    x = x + pos_encoding_dense[tf.newaxis, :length, :]
    return x

## ATTENTIONS
# (1) Tipologia di <Attention>: Cross Attention
# Questa tipolgia si trova al centro del Transformer e connette l'encoder e il decoder.
class CrossAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

  def call(self, x, context):
    attn_output, attn_scores = self.mha(
        query=x,
        key=context,
        value=context,
        return_attention_scores=True)

    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x

# (2) Tipologia di <Attention>: Global Self Attention
# Questo layer è il responsabile per il processamento iniziale del contesto della sequenza e della propagazione dell'informazione estratta.
class GlobalSelfAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x
  
# (3) Tipologia di <Attention>: Causal Self Attention
# è simile al lavoro che compie la global ma lo fa per l'output sentence del decoder.
class CausalSelfAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        use_causal_mask = True) 
    # Questo parametro permette di garantire al modello di non guardare i token futuri durante il calcolo dell'attenzione
    # ossia ogni location dell'attenzione può guardare solo le posizioni precedenti e se stessa.
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

## FEED FORWARD LAYER
# Il trasformer include layer di feed forward sia nell'encoder che nel decoder
# 2 linear networks con ReLU come attivazione e dropout
class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
    return x
  
# Nuova classe per estrarre embeddings da sequenze di time series tramite CNN 1D
class TimeSeriesEmbedding(tf.keras.layers.Layer):
  def __init__(self, d_model, num_conv_layers=1, kernel_size=3, activation='relu'):
    super().__init__()
    conv_layers = []
    for _ in range(num_conv_layers):
      conv_layers.append(tf.keras.layers.Conv1D(
        filters=d_model,
        kernel_size=kernel_size,
        padding='same',
        activation=activation
      ))
    self.conv_layers = tf.keras.Sequential(conv_layers)
    self.layernorm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    # x shape: (batch, seq_len, features)
    x = self.conv_layers(x)
    x = self.layernorm(x)
    return x
  
## ENCODER
# L'encoder che compone il Trasformer è composto da uno stack di N encoder layer
# Ogni encoder layer è composto da:
# 1. Global Self Attention
# 2. Feed Forward Layer
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x

# Encoder:
class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads,
               dff, vocab_size=None, dropout_rate=0.1,
               use_timeseries_embedding=False, input_features=None,
               num_conv_layers=1, kernel_size=3):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    self.use_timeseries_embedding = use_timeseries_embedding
    
    if use_timeseries_embedding:
      assert input_features is not None, "input_features must be specified for time series embedding"
      self.embedding = TimeSeriesEmbedding(
        d_model=d_model,
        num_conv_layers=num_conv_layers,
        kernel_size=kernel_size
      )
    else:
      assert vocab_size is not None, "vocab_size must be specified for text embedding"
      self.embedding = PositionalEmbedding(
        vocab_size=vocab_size, d_model=d_model)

    self.pos_encoding = positional_encoding(length=2048, depth=d_model)
    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    # x shape: (batch, seq_len, features) se time series, altrimenti (batch, seq_len) per token IDs
    x = self.embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

    # Add positional encoding (sempre applicato, anche dopo CNN per time series)
    if self.use_timeseries_embedding:
      # Per time series, aggiungiamo positional encoding dopo la CNN
      length = tf.shape(x)[1]
      pos_encoding_dense = tf.convert_to_tensor(self.pos_encoding)
      x = x + pos_encoding_dense[tf.newaxis, :length, :]
    # Per PositionalEmbedding (testi), il positional encoding è già incluso

    # Add dropout.
    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    return x  # Shape `(batch_size, seq_len, d_model)`.

# Decoder:
# Il decoder è composto da uno stack di N decoder layer
# Ogni decoder layer è composto da:
# 1. Causal Self Attention
# 2. Cross Attention
# 3. Feed Forward Layer
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1):
    super(DecoderLayer, self).__init__()

    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x, context):
    x = self.causal_self_attention(x=x)
    x = self.cross_attention(x=x, context=context)

    # Cache the last attention scores for plotting later
    self.last_attn_scores = self.cross_attention.last_attn_scores

    x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
    return x

# Encoder:
class Decoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
               dropout_rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                             d_model=d_model)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_layers = [
        DecoderLayer(d_model=d_model, num_heads=num_heads,
                     dff=dff, dropout_rate=dropout_rate)
        for _ in range(num_layers)]

    self.last_attn_scores = None

  def call(self, x, context):
    # `x` is token-IDs shape (batch, target_seq_len)
    x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

    x = self.dropout(x)

    for i in range(self.num_layers):
      x  = self.dec_layers[i](x, context)

    self.last_attn_scores = self.dec_layers[-1].last_attn_scores

    # The shape of x is (batch_size, target_seq_len, d_model).
    return x

## TRANSFORMER
# Mette insieme l'encoder e il decoder ed aggiunge un layer finale denso + softmax che permette di convertire l'output in token probabilities.
class Transformer(tf.keras.Model):
  def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size=None, target_vocab_size=None, dropout_rate=0.1,
               use_timeseries_embedding=False, input_features=None,
               num_conv_layers=1, kernel_size=3):
    super().__init__()
    self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=input_vocab_size,
                           dropout_rate=dropout_rate,
                           use_timeseries_embedding=use_timeseries_embedding,
                           input_features=input_features,
                           num_conv_layers=num_conv_layers,
                           kernel_size=kernel_size)

    self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=target_vocab_size,
                           dropout_rate=dropout_rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inputs):
    # To use a Keras model with `.fit` you must pass all your inputs in the
    # first argument.
    context, x  = inputs

    context = self.encoder(context)  # (batch_size, context_len, d_model)

    x = self.decoder(x, context)  # (batch_size, target_len, d_model)

    # Final linear layer output.
    logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
      del logits._keras_mask
    except AttributeError:
      pass

    # Return the final output and the attention weights.
    return logits
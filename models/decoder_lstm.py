import tensorflow as tf
from tensorflow.keras import layers

# --- 1. ATENCIÓN ---
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = self.V(tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# --- 2. DECODER (BLINDADO CONTRA ERRORES DE KERAS 3) ---
class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        
        # Usamos GRU que es más fácil de manejar (retorna 2 cosas)
        self.gru = layers.GRU(self.units,
                              return_sequences=True,
                              return_state=True,
                              recurrent_initializer='glorot_uniform')
        
        self.fc1 = layers.Dense(self.units)
        self.fc2 = layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # 1. Atención
        context_vector, attention_weights = self.attention(features, hidden)

        # 2. Embedding
        x = self.embedding(x)

        # 3. Concatenar
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # 4. RNN (GRU)
        # TRUCO: Usamos 'output, *state'
        # Esto funciona si la RNN devuelve 2 cosas (GRU) o 3 cosas (LSTM)
        # El * agrupa lo sobrante en una lista, y cogemos el primer elemento.
        result = self.gru(x, initial_state=hidden)
        output = result[0]
        state = result[1] # Nuevo estado oculto

        # 5. Clasificación
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

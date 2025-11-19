import tensorflow as tf

class Trainer:
    def __init__(self, encoder, decoder, optimizer):
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none'
        )

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    @tf.function  # Ahora SÍ funciona bien aquí
    def train_step(self, img_tensor, target, start_token_id):
        loss = 0
        
        # Inicializar estado oculto
        hidden = self.decoder.reset_state(batch_size=target.shape[0])

        # Input inicial para el decoder: <start>
        dec_input = tf.expand_dims([start_token_id] * target.shape[0], 1)

        with tf.GradientTape() as tape:
            # Usamos self.encoder
            features = self.encoder(img_tensor, training=True)

            for i in range(1, target.shape[1]):
                predictions, hidden, _ = self.decoder(dec_input, features, hidden)
                loss += self.loss_function(target[:, i], predictions)
                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = loss / int(target.shape[1])
        
        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return total_loss

def get_optimizer(learning_rate=1e-4):
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)

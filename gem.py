import tensorflow as tf

from tensorflow import keras


@keras.saving.register_keras_serializable(package="keras-gem", name="GeM")
class GeM(keras.layers.Layer):
    def __init__(self, init_norm=3.0, normalize=True, **kwargs):
        self.init_norm = init_norm
        self.normalize = normalize

        super(GeM, self).__init__(**kwargs)

    def build(self, input_shape):
        feature_size = input_shape[-1]
        self.p = self.add_weight(name="norms", shape=(feature_size,),
                                 initializer=keras.initializers.constant(self.init_norm),
                                 trainable=True)
        super(GeM, self).build(input_shape)

    def call(self, inputs):
        x = inputs
        x = tf.math.maximum(x, 1e-6)
        x = tf.pow(x, self.p)

        x = keras.layers.GlobalAveragePooling2D()(x)
        x = tf.pow(x, 1.0 / self.p)

        if self.normalize:
            x = tf.nn.l2_normalize(x, 1)

        return x

    def compute_output_shape(self, input_shape):
        return tuple([None, input_shape[-1]])

    def get_config(self):
        config = super().get_config()
        config.update(
            {"init_norm": self.init_norm,
             "normalize": self.normalize}
        )
        return config

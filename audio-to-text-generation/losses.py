import tensorflow as tf

class CTCLoss(tf.keras.losses.Loss):
    def __init__(self, blank_token_int, name='ctc_loss'):
        super().__init__(name=name)
        self.blank_token_int = blank_token_int

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.int32)
        logits = tf.transpose(y_pred, perm=[1, 0, 2])
        logit_length = tf.fill(tf.shape(y_pred)[0:1], tf.shape(y_pred)[1])
        non_blank_mask = tf.not_equal(y_true, self.blank_token_int)
        label_length = tf.reduce_sum(tf.cast(non_blank_mask, tf.int32), axis=-1)
        logits = tf.cast(logits, tf.float32)

        loss = tf.nn.ctc_loss(
            labels=y_true,
            logits=logits,
            label_length=label_length,
            logit_length=logit_length,
            blank_index=self.blank_token_int
        )
        return loss

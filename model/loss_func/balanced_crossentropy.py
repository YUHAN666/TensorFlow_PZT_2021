import tensorflow as tf


def balanced_crossentropy_with_logits(labels, logits, negative_ratio=3, scale=5, least_negative_count=10000.0):
    positive_mask = labels
    negative_mask = 1 - labels

    positive_count = tf.reduce_sum(positive_mask)

    negative_count = tf.cond(positive_count > tf.constant(0.0),
                             lambda: tf.reduce_min([tf.reduce_sum(negative_mask), positive_count * negative_ratio]),
                             lambda: least_negative_count)  # 防止因positive count等于零导致tf.reshape报错

    # loss = tf.keras.backend.binary_crossentropy(labels, logits, from_logits=True)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    positive_loss = loss * positive_mask
    negative_loss = loss * negative_mask

    negative_loss, _ = tf.nn.top_k(tf.reshape(negative_loss, (-1,)), tf.cast(negative_count, tf.int32))
    balanced_loss = (tf.reduce_sum(positive_loss) + tf.reduce_sum(negative_loss)) / (
                positive_count + negative_count + 1e-6)
    balanced_loss = balanced_loss * scale

    return balanced_loss

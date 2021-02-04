import tensorflow as tf


def balanced_crossentropy_with_logits(labels, logits, negative_ratio=3, scale=5):
    positive_mask = labels
    negative_mask = 1 - labels

    positive_count = tf.reduce_sum(positive_mask)
    if positive_count == 0:
        negative_count = tf.reduce_sum(negative_mask) // 3
    else:
        negative_count = tf.reduce_min([tf.reduce_sum(negative_mask), positive_count * negative_ratio])

    # loss = tf.keras.backend.binary_crossentropy(labels, logits, from_logits=True)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    positive_loss = loss * positive_mask
    negative_loss = loss * negative_mask

    negative_loss, _ = tf.nn.top_k(tf.reshape(negative_loss, (-1,)), tf.cast(negative_count, tf.int32))
    balanced_loss = (tf.reduce_sum(positive_loss) + tf.reduce_sum(negative_loss)) / (
                positive_count + negative_count + 1e-6)
    balanced_loss = balanced_loss * scale

    return balanced_loss

import tensorflow as tf

def batch_to_time_major(inputs):
    inputs = tf.split(inputs,  num_or_size_splits=inputs.get_shape().as_list()[1], axis=1)
    inputs = [tf.squeeze(e, axis=1) for e in inputs]
    return inputs
    
def sample2D(probability):
    probability = tf.clip_by_value(probability,1e-7,1.0)
    return tf.squeeze(tf.multinomial(tf.log(probability),1))
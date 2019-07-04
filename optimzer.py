import tensorflow as tf

def START(loss_op, W, W_avg, lr=0.1, gamma=1000):
        new_grads= tf.gradients(loss_op, W)
        update_ops_bn = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops = []
        with tf.control_dependencies(update_ops_bn):
            for p, p_avg, g in zip(W, W_avg, new_grads):
                new_p = (gamma*p + lr*p_avg - lr*gamma*g)/(lr + gamma)
                update_ops.append(p.assign(new_p))
        return update_ops
    
def SGD(loss_op, lr=0.1):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    update_ops_bn = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops_bn):
        train_op = optimizer.minimize(loss_op)
    return train_op
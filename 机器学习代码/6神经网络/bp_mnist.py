import tensorflow as tf
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist_data/", one_hot=True)


#定义参数
learning_rate = 0.005
training_epochs = 20
batch_size = 100
display_step = 1
batch_count = int(mnist.train.num_examples/batch_size)

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'weight1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'weight2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'bias1': tf.Variable(tf.random_normal([n_hidden_1])),
    'bias2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron_model(x):
    layer_1 = tf.add(tf.matmul(x, weights['weight1']), biases['bias1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['weight2']), biases['bias2'])
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

logits = multilayer_perceptron_model(X)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
#optimizer = tf.train.MomentumOptimizer(learning_rate,0.2)
#optimizer = tf.train.AdagradOptimizer(learning_rate)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss_op)
init = tf.global_variables_initializer()#参数初始化

with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        # Loop over all batches
        for i in range(batch_count):
            train_x, train_y =  mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: train_x, Y: train_y})
            # Compute average loss
            avg_cost += c / batch_count
        if epoch % display_step == 0:
            print("Epoch:", '%02d' % (epoch+1), "avg cost={:.6f}".format(avg_cost))

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

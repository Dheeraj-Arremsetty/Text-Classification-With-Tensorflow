import tensorflow as tf
import numpy as np
from text_classification import create_feature_sets_and_labels
directory_paths = ["./data/mini_newsgroups/alt.atheism",
                   "./data/mini_newsgroups/rec.autos",
                   "./data/mini_newsgroups/sci.med",]
                   # "./data/mini_newsgroups/misc.forsale",
                   # "./data/mini_newsgroups/talk.religion.misc",]
                   # "./data/mini_newsgroups/talk.politics.misc",
                   # "./data/mini_newsgroups/talk.politics.guns",
                   # "./data/mini_newsgroups/sci.space",
                   # "./data/mini_newsgroups/sci.crypt",]
                   # "./data/mini_newsgroups/comp.windows.x"]

directory_paths = ["./data/20_newsgroups/alt.atheism",
                   "./data/20_newsgroups/rec.autos",]
print "Stateddd"
train_x, train_y, test_x, test_y,classification = create_feature_sets_and_labels(directory_paths)
print "Received Data"
print "classifications are:",classification

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

#TotalNumber of classes
n_classes = len(classification)#len(directory_paths)
#10 classes => 0,1,2,3,4,5,6,7,8,9

#batches of features passed to NN one batch at each time
batch_size = 100
# print train_x[0]
dataLength = len(train_x[0])
x = tf.placeholder('float',[None, dataLength])
y = tf.placeholder('float')

def neural_network_model(data):

    #(weights*inputData) + biases

    hidden_1_layer = {"weights":tf.Variable(tf.random_normal([dataLength, n_nodes_hl1])),
                      "biases":tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {"weights": tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      "biases": tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {"weights": tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      "biases": tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {"weights": tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      "biases": tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer["weights"]), hidden_1_layer["biases"])
    #Activation function - (sigmoid func) rectified linear
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer["weights"]), hidden_2_layer["biases"])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer["weights"]), hidden_3_layer["biases"])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['weights']) , output_layer['biases'])

    return output



def train_neural_network(x):
    prediction = neural_network_model(x)
    print "Neural network is ready"
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))


    optimizer = tf.train.AdamOptimizer().minimize(cost) #default learning_rate = 0.001

    # cycles = FeedForwars+Backpropo
    #if cpu is low give it less
    hm_epochs = 15

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            i=0
            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                # print "train_x[start:end]",len(train_x)
                # print "len(batch_x)=",len(batch_x)
                # print "len(batch_y)=", len(batch_y)
                # print "dataLength",dataLength
                _, c = sess.run([optimizer, cost], feed_dict = {x:batch_x, y:batch_y})
                i += batch_size
                # print "number of iterations:",i
                epoch_loss += c

            print "Epoch",epoch,"completed out of ",hm_epochs,"loss:",epoch_loss

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        print "Accuaracy",accuracy.eval({x:test_x, y:test_y})

print "Started Neural Network"
train_neural_network(x)
print "Completed neural network"
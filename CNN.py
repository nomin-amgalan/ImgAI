'''
CNN model for classifrying the overall image type

Datasets used:
 - https://www.kaggle.com/ikarus777/best-artworks-of-all-time/version/1

'''

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import skimage
from skimage import transform
from skimage.color import rgb2gray

from PIL import Image
from scipy import signal
from scipy import misc
import os

# Showing less logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



classes = {4: "Photos", 3: "Screenshots", 1: "Paintings", 2: "Other"}
numOfClasses = len(classes)
training_path = "CNN_training_data"
testing_path = "CNN_testing_data"
main_path = "/Users/nominamgalan/Desktop/ImgAI"



def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:

        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]                    # Fix the extension
        for f in file_names:
            
            info = skimage.data.imread(f)
            #print("skimage.data.imread: " ,info)
            images.append(info)
            labels.append(int(d))
    return images, labels

# PREPARING DATA SET

training_data_dir = os.path.join(main_path, training_path)
images, labels = load_data(training_data_dir)
images1k = [transform.resize(image, (1000, 1000)) for image in images]
print(type(images1k))
#images1k = images1k.reshape(1000, 1000)

images1k = np.asarray(images1k)
images1k = rgb2gray(images1k)



# PREPARING NEURAL NETWORK

# Initialize placeholders 
x = tf.compat.v1.placeholder(dtype = tf.float32, shape = [None, 1000, 1000])
y = tf.compat.v1.placeholder(dtype = tf.int32, shape = [None])

# Flatten the input data
images_flat = tf.contrib.layers.flatten(x)

# Fully connected layer 
logits = tf.contrib.layers.fully_connected(images_flat, 2, tf.nn.relu)

# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, 
                                                                    logits = logits))
# Define an optimizer 
train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



# RUNNING TRAINING

tf.compat.v1.set_random_seed(1234)

sess = tf.compat.v1.Session()
saver = tf.compat.v1.train.Saver()

sess.run(tf.compat.v1.global_variables_initializer())

for i in range(20):
        print('EPOCH', i)
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images1k, y: labels})
        if i % 10 == 0:
            print("Loss: ", loss)
        print('DONE WITH EPOCH')

# PREPARING TESTING SET

print("Starting preparation of testing dataset")

test_data_directory = os.path.join(main_path, testing_path)

# Load the test data
test_images, test_labels = load_data(test_data_directory)

# Transform the images to 28 by 28 pixels
test_images1k = [transform.resize(image, (1000, 1000)) for image in test_images]

# Convert to grayscale

test_images1k = rgb2gray(np.array(test_images1k))

# RUNNING TESTING


print("TEST IMAGES READY")

# Run predictions against the full test set.
predicted = sess.run([correct_pred], feed_dict={x: test_images1k})[0]

# Calculate correct matches 
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

# Calculate the accuracy
accuracy = match_count / len(test_labels)

# Print the accuracy
print("Accuracy: {:.3f}".format(accuracy))


saver.save(sess, 'my-test-model')
sess.close()

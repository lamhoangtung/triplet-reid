import json
import os
from importlib import import_module

import cv2
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

sess = tf.Session()

# Read config
config = json.loads(open(os.path.join(
    '/Users/lamhoangtung/Downloads/deep_fashion_2/default_param/', 'args.json'), 'r').read())

# Input img
net_input_size = (
    config['net_input_height'], config['net_input_width'])

path = tf.placeholder(tf.string)
image_encoded = tf.read_file(path)
image_decoded = tf.image.decode_jpeg(image_encoded, channels=3)
image_resized = tf.image.resize_images(image_decoded, net_input_size)
img = tf.expand_dims(image_resized, axis=0)

# Create the model and an embedding head.
model = import_module('nets.' + config['model_name'])
head = import_module('heads.' + config['head_name'])

endpoints, _ = model.endpoints(img, is_training=False)
with tf.name_scope('head'):
    endpoints = head.head(
        endpoints, config['embedding_dim'], is_training=False)

# Initialize the network/load the checkpoint.
checkpoint = tf.train.latest_checkpoint(config['experiment_root'])
print('Restoring from checkpoint: {}'.format(checkpoint))
tf.train.Saver().restore(sess, checkpoint)

emb = sess.run(endpoints['emb'],  feed_dict={
               path: '/Users/lamhoangtung/Downloads/deep_fashion_2/022870_item1.jpg'})[0]

print(emb)

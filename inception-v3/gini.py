#!/usr/bin/env python
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import matplotlib.pylab as plt

from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import pairwise_distances

import seaborn as sns
import cPickle as pkl

FLAGS = tf.app.flags.FLAGS
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 0})
#Layers names based on TensorBoard graph
layer_names = [#'DecodeJpeg',
#'Cast', 'ExpandDims', 'ResizeBilinear',
'Sub',
'Mul',
          'conv', 'conv_1', 'conv_2',
          'pool',
          'conv_3', 'conv_4',
          'pool_1',
          'mixed/join', 'mixed_1/join', 'mixed_2/join', 'mixed_3/join', 'mixed_4/join', 'mixed_5/join',
          'mixed_6/join', 'mixed_7/join', 'mixed_8/join', 'mixed_9/join', 'mixed_10/join',
          'pool_3',
          'softmax']
postfix = ':0'

# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
    'model_dir', '/tmp/imagenet',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_file', '/home/olivia/neural-code/img/dog.jpg',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long


class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  # with tf.gfile.FastGFile(os.path.join(
    #   FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
  model_filename = './graph/classify_image_graph_def.pb'
  with tf.gfile.FastGFile(model_filename, 'rb') as f:

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image):
  """Runs inference on an image.

  Args:
    image: Image file name.

  Returns:
    Nothing
  """
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()


  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    layer_tensors = []
    for layer_name in layer_names:
        layer_tensors.append(sess.graph.get_tensor_by_name(layer_name+postfix))


    states = sess.run(layer_tensors,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(states[-1])

    #summary = sess.run(merged,  {'DecodeJpeg/contents:0': image_data})
    #summary_writer.add_summary(summary)

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()

    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))
  return states#, labels#states[0], states[1], states[2], states[3], states[4], states[5], states[6]


def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def create_pattern_at_a_layer_matrix(state, save_as):#(state, labels, save_as):
  print(state)
  print(state.shape)

  #corrmat = 1-pairwise_distances(states, metric="correlation")
  #birds, mammals, vehicles, household objects, tools, fruit

  # Set up the matplotlib figure
  f, ax = plt.subplots(figsize=(12,9))

  # Draw the heatmap using seaborn
  sns.heatmap(state)#, vmax=1, vmin=-1, square=True, xticklabels = labels, yticklabels = False, cmap = "RdBu_r", center = 0)
  f.tight_layout()
  # This sets the yticks "upright" with 0, as opposed to sideways with 90.
  plt.yticks(rotation=90)
  plt.xticks(rotation=90)

  ax.set_xticklabels(labels, ha = 'center', va='center', size= 10)
  plt.savefig('./fig/'+save_as+'.png', bbox_inches='tight')
  #plt.savefig('./fig/'+save_as+'_heatmap.pdf', bbox_inches='tight')
  plt.close()
  del f, ax

def create_all_patterns_at_a_layer_matrix(states, labels, save_as):
  print('Saving: '+save_as+'_heatmap.png')
  #corrmat = 1-pairwise_distances(states, metric="correlation")
  #birds, mammals, vehicles, household objects, tools, fruit

  # Set up the matplotlib figure
  f, ax = plt.subplots(figsize=(100,10))

  # Draw the heatmap using seaborn
  sns.heatmap(states, vmax=1, vmin=0, square=True, xticklabels = False, yticklabels = labels[::-1], center=0.5)#, cmap = "RdBu_r", center = 0)
  f.tight_layout()
  # This sets the yticks "upright" with 0, as opposed to sideways with 90.
  plt.yticks(rotation=0)
  # plt.xticks(rotation=0)

  ax.set_yticklabels(labels[::-1], ha = 'left', va = 'center', size= 26)
  #ax.xaxis.set(ticks=np.arange(0, len(labels)))

  plt.savefig('./fig/'+save_as+'_heatmap.png', bbox_inches='tight')
  plt.savefig('./fig/'+save_as+'_heatmap.pdf', bbox_inches='tight')
  plt.close()
  del f, ax

def create_similarity_matrix(states, labels, save_as):
  print('Saving: '+save_as+'_heatmap.png')
  corrmat = 1-pairwise_distances(states, metric="correlation")
  #birds, mammals, vehicles, household objects, tools, fruit

  # Set up the matplotlib figure
  f, ax = plt.subplots(figsize=(12,9))

  # Draw the heatmap using seaborn
  sns.heatmap(corrmat, vmax=1, vmin=-1, square=True, xticklabels = False, yticklabels = labels[::-1], cmap = "RdBu_r", center = 0)
  f.tight_layout()
  # This sets the yticks "upright" with 0, as opposed to sideways with 90.
  plt.yticks(rotation=0)
  # plt.xticks(rotation=0)

  ax.set_yticklabels(labels[::-1], ha = 'left', va = 'center', size= 26)
  #ax.xaxis.set(ticks=np.arange(0, len(labels)))

  plt.savefig('./fig/'+save_as+'_heatmap.png', bbox_inches='tight')
  plt.savefig('./fig/'+save_as+'_heatmap.pdf', bbox_inches='tight')
  plt.close()
  del f, ax

def make_2d(array):
  dims = np.asarray(array).shape
  #print ( dims[1:])
  dim2 = 1
  for dim in dims[1:]:
    dim2 *= dim
  array = np.asarray(array)
  return array.reshape((dims[0], dim2))

def run_on_images(images, labels, fig_labels):
  # pickled_states = './load/figure_states_dump.pkl'
  # if not tf.gfile.Exists(pickled_states):
  # Creates graph from saved GraphDef.
  create_graph()

  states = [[] for image in images]
  #labels = [os.path.split(image)[1][0:8] for image in images]

  for i, image in enumerate(images):
    print(labels[i])
    states[i] = run_inference_on_image(image)

  #     pkl.dump(states, open(pickled_states, 'w'))
  # else:
  #     states = pkl.load(open(pickled_states, 'r'))


  layers = len(states[0])
  gini_coef = np.zeros((len(images)))
  for sts in states:
  #for each image
    for state in sts:
    #for each layer
      state = make_2d(state)
  for l in range(layers):
      layer = np.empty((len(images), states[0][l].flatten().shape[0]))
      for i, image in enumerate(images):
        layer[i] = states[i][l].flatten()
        states[i][l] = 0

        # if l == 12:
        #     print(layer[i].shape)
        #     create_pattern_at_a_layer_matrix(layer[i].reshape((640, 490)), 'state_at_layer_'+str(l))#str(i))
      layer_2d = make_2d(layer)

      for i, image in enumerate(images):
            gini_coef[i] = gini(layer_2d[i])
            # if i == 0:
                #print(np.amin(layer_2d[i]), np.amax(layer_2d[i]), np.mean(layer_2d[i]))
      print(l ,'layer dim: ', layer_2d.shape, 'layer gini: ', np.mean(gini_coef))
    #    create_all_patterns_at_a_layer_matrix(layer_2d, fig_labels, 'patterns_at_layer_'+str(l))
      #create_similarity_matrix(layer_2d, fig_labels, 'layer_'+str(l))

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    array_sum = np.sum(array) #sum over whole array
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * array_sum)) #Gini coefficient


def main(_):
  maybe_download_and_extract()
  images_dir  = './img/'
  categories = ['lion', 'tiger', #mammals
                  'robin', 'partridge', #birds
                 'sportscar', 'moped', #vehicles
                 'guitar', 'banjo' #musical instruments
                ]
  paths = [images_dir+category for category in categories]
  images = []
  labels = []
  for p, path in enumerate(paths):
    for i, image in enumerate(os.listdir(path)):
     if os.path.isfile(os.path.join(path, image)):
        images.append(os.path.join(path, image))
        labels.append(categories[p]+' '+str(i))

  fig_labels = ['' for l in labels]
  cat_num = len(categories)
  print(len(fig_labels), len(labels))

  for i in range(cat_num):
    print(labels[5*i+2][0:-2])
    fig_labels[5*i+2] = labels[5*i+2][0:-2]

  run_on_images(images, labels, fig_labels)

if __name__ == '__main__':
   tf.app.run()

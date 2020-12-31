import sys
import getopt
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

import mrcnn
from mrcnn import model as modellib
from mrcnn.model import MaskRCNN

from matplotlib import pyplot
from matplotlib.patches import Rectangle

import keras
from keras.models import load_model

from MaskRCNNConfig import MaskRCNNConfig
from RoadBarrierDataset import RoadBarrierDataset

from utils.visualize import visualize_dataset

matplotlib.use('TkAgg')

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

logs_path = './logs'
models_path = './retrained_models'

def main(argv):
  """ This is the main entrypoint of the retraining process. 
  
  Proper usage: python main.py [-f ./path/to/data] [-w ./path/to/checkpoint] [-e num_epochs] [-v] [-d] [-h] [-i] [-e]
    * -f: The path to the training and validation data (see images/README for proper format)
    * -w: The path to the checkpoint you would like to begin at. The default is mask_rcnn_coco.h5.
    * -e: The number of epochs to retrain on; default is 10
    * -v: "verbose"; prints out more things...
    * -d: "display"; displays the validation images and ground truths when they are loaded
    * -i: "inference"; only run inference using the model specified with -w (no retraining)

  
  It performs the following steps:
    1. Creates a config class for retraining (see MaskRCNNConfig.py)
    2. Loads and prepares the training and validation datasets based on the 
       input path provided
    3. Loads a MaskRCNN model using the weights specified
    4. Retrains the MaskRCNN model on the new training/validation datasets
    5. Displays the results of the best model (best validation loss) performing 
       inference on the validation set
  """
  verbose = False
  should_visualize = False
  should_retrain = True
  base_path = './road-block-images'
  weights_path = './mask_rcnn_coco.h5'
  num_epochs = 10

  # handle passed arguments
  try:
      opts, args = getopt.getopt(argv, 'hdvif:w:e:', ['files=', 'weights=', 'epochs='])
  except getopt.GetoptError:
      print('Incorrect usage.')
      print_help_message()
      sys.exit(2)

  for opt, arg in opts:
    if opt in ('-h', '--help'):
        print_help_message()
        sys.exit()
    elif opt in ('-d', '--display'):
        should_visualize = True
    elif opt in ('-f', '--files'):
        base_path = arg
    elif opt in ('-v', '--verbose'):
        verbose = True
    elif opt in ('-w', '--weights'):
      weights_path = arg
    elif opt in ('-i', '--inference'):
      should_retrain = False
    elif opt in ('-e', '--epochs'):
      try:
        num_epochs = int(arg)
      except ValueError as e:
        print('Error: number of epochs must be an integer value.')
        print(e)
        sys.exit(2)
  
  # instantiate maskRCNN Config class (hyperparameters for retraining)
  config = MaskRCNNConfig()
  if verbose:
    config.display()
  
  # load and prepare the training and test sets
  # train set
  train_dir = base_path + '/train'

  train_set = RoadBarrierDataset()
  train_set.load_dataset(train_dir, is_train=True)
  train_set.prepare()

  # test set
  test_dir = base_path + '/val'

  test_set = RoadBarrierDataset()
  test_set.load_dataset(test_dir, is_train=False)
  test_set.prepare()

  if should_visualize: # visualize the ground truth for the test set
    visualize_dataset(test_set, config)
    # Place a limit as shown below if you don't want to display too many images
    # visualize_dataset(test_set, config, limit=1)

  if should_retrain: # retrain existing model on dataset
    print("Loading Mask R-CNN model...")
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=models_path)

    #load the weights for COCO
    model.load_weights(weights_path, 
                      by_name=True, 
                      exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

    # **** Setup callback functions for training process ****
    training_run = str(time.time())

    # setup Tensorboard
    log_dir = logs_path + '/' + training_run

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,  
            write_graph=True, write_images=True)
  
    # save best validation loss checkpoint
    best_val_model_path = models_path + '/best_val.' + training_run + '.h5'
  
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
      filepath=best_val_model_path,
      save_weights_only=True,
      monitor='val_loss',
      mode='max',
      save_best_only=True)

    # Retrain the model on the new dataset
    model.train(train_set, 
                test_set, 
                learning_rate=config.LEARNING_RATE,
                custom_callbacks = [
                  tensorboard_callback,
                  model_checkpoint_callback],
                epochs=num_epochs, 
                layers='heads')
  
    model_path = models_path  + '/final.' + training_run + '.h5'
    model.keras_model.save_weights(model_path)

  # load best model (best validation loss)
  model = modellib.MaskRCNN(mode="inference", config=config, model_dir=models_path)
  if should_retrain: # use best model found in retraining process
    model.load_weights(best_val_model_path, by_name=True)
  else: # use given model
    model.load_weights(weights_path, by_name=True)

  # visualize detections on the test set with best scoring checkpoint
  visualize_dataset(test_set, config, model=model)

def print_help_message():
  """ Prints a message that details the correct usage of the project """
  print('Correct usage: python main.py -f <inputDataPath> -w <checkpointPath> -e <numEpochs>\n')
  print('Options:')
  print('\t-h: help')
  print('\t-v: verbose')
  print('\t-d: display')
  print('\t-i: inference')

if __name__ == '__main__':
  with tf.device('/device:GPU:3'):
    main(sys.argv[1:])

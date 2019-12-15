"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
import csv

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02

# Testing constants
#DNN_HIDDEN_UNITS_DEFAULT = '100'
#LEARNING_RATE_DEFAULT = 2e-3
#MAX_STEPS_DEFAULT = 3
#BATCH_SIZE_DEFAULT = 200
#EVAL_FREQ_DEFAULT = 100
#NEG_SLOPE_DEFAULT = 0.02

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################

  # Turn into array of 1's for each row if argmax is the same - then take mean
  mask = np.where(predictions.argmax(axis=1) == targets.argmax(axis=1), 1, 0)
  accuracy = np.mean(mask)

  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  # Get negative slope parameter for LeakyReLU
  neg_slope = FLAGS.neg_slope

  ########################
  # PUT YOUR CODE HERE  #
  #######################

  # Rebranding some of the flags
  eta = FLAGS.learning_rate
  batch_size = FLAGS.batch_size
  eval_freq = FLAGS.eval_freq

  # Get full cifar10 dataset and test data (reshape x for NN)
  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
  x_test, y_test = cifar10['test'].images, cifar10['test'].labels

  # Dimensions
  depth  = x_test[0].shape[0]
  width  = x_test[0].shape[1]
  height = x_test[0].shape[2]
  x_test = x_test.reshape((x_test.shape[0], depth * width * height))

  # Initialize NN and loss module
  NN = MLP(n_inputs = depth * width * height, n_hidden = dnn_hidden_units, n_classes = y_test.shape[1], neg_slope = FLAGS.neg_slope)
  crossent = CrossEntropyModule()

  # Keep track for plots
  trainloss = []
  trainacc  = []
  testloss  = []
  testacc   = []
  steps     = []

  # Do mini-batch gradient descent
  for step in range(FLAGS.max_steps):

    # New mini-batch and reshape x
    x_mini, y_mini = cifar10['train'].next_batch(batch_size)
    x_mini = x_mini.reshape((x_mini.shape[0], x_mini.shape[1] * x_mini.shape[2] * x_mini.shape[3]))

    # Feed-forward of x, get loss and gradient of loss
    x_mini = NN.forward(x_mini)
    loss = crossent.forward(x_mini, y_mini)
    lossgrad = crossent.backward(x_mini, y_mini)

    # Backprop has no return type
    NN.backward(lossgrad)

    # Do weight and gradient updates
    for layer in NN.nn:
      layer.params['weight'] = layer.params['weight'] - eta * layer.grads['weight']
      layer.params['bias']   = layer.params['bias']   - eta * layer.grads['bias']

    # Evaluate if step is multitude of 500
    if (step % FLAGS.eval_freq) == 0:
      
      # Initialize loss and accuracy 
      temp_loss = 0
      temp_acc  = 0

      # Prevent memory overload so divvy up in smaller batches
      prevent_overload = 50
      for i in range(prevent_overload):

        # Break it up to run over all test data
        size = cifar10['test'].num_examples // prevent_overload
        imgs, labels = cifar10['test'].next_batch(size)

        # Reshape and forward
        new_input = imgs.reshape(imgs.shape[0], imgs.shape[1] * imgs.shape[2] * imgs.shape[3])
        output = NN.forward(new_input)

        # Do cross entropy
        entropy = crossent.forward(output, labels)

        # Accumulate loss and accuracy and divide both by prevent_overload
        temp_loss = temp_loss + entropy / prevent_overload
        temp_acc  = temp_acc  + accuracy(output, labels) / prevent_overload
      
      # Print acc
      print("Loss at step ", step, " = ", temp_loss)
      print("Acc  at step ", step, " = ", temp_acc)

      # Append all losses and accuracies to list
      trainloss.append(loss.item())
      trainacc.append(accuracy(x_mini, y_mini).item())
      testloss.append(temp_loss)
      testacc.append(temp_acc)
      steps.append(step)

  # Write data to csv file    
  # First build string to specify a file
  # Commented out to prevent overwriting of files
  #dropout = False
  #filename = "nump-(" + str(FLAGS.dnn_hidden_units) + ")" + "-" + str(FLAGS.max_steps) + "-" + str(FLAGS.learning_rate) + "-" + str(dropout) + ".csv"

  # Actual writing
  #with open("csv/" + filename, 'w', newline='') as myfile:
     #wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     #wr.writerow(trainloss)
     #wr.writerow(trainacc)
     #wr.writerow(testloss)
     #wr.writerow(testacc)
     #wr.writerow(steps)

  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--neg_slope', type=float, default=NEG_SLOPE_DEFAULT,
                      help='Negative slope parameter for LeakyReLU')
  FLAGS, unparsed = parser.parse_known_args()

  main()
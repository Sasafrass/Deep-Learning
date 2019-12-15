"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import torch
import torch.nn as nn
import torch.optim as optim
import csv 

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02

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

  # Detach grad and convert to numpy
  predictions = predictions.cpu().detach().numpy()
  targets = targets.cpu().detach().numpy()

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
  torch.manual_seed(42)

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

  # Get full cifar10 dataset and make torch tensors
  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)  
  x_test = torch.tensor(cifar10['test'].images, requires_grad = False)
  y_test = torch.tensor(cifar10['test'].labels, requires_grad = False)

  # Dimensions
  depth  = x_test[0].shape[0]
  width  = x_test[0].shape[1]
  height = x_test[0].shape[2]
  x_test = x_test.reshape((x_test.shape[0], depth * width * height))

  # Initialize device GPU or CPU
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # Initialize the Neural Network
  NN = MLP(n_inputs = depth * width * height, n_hidden = dnn_hidden_units, n_classes = y_test.shape[1], neg_slope = FLAGS.neg_slope).to(device)
  
  # Set loss to Cross Entropy and initialize optimizer
  crossent = nn.CrossEntropyLoss()
  optimizer = optim.Adam(NN.parameters(), lr = FLAGS.learning_rate) # Perhaps add momentum 

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

    # Make 'em torch-y
    x_mini = torch.from_numpy(x_mini).to(device)
    y_mini = torch.from_numpy(y_mini).to(device)
    y_mini = y_mini.long()

    # Clear the gradient buffer and do forward pass
    optimizer.zero_grad()
    x_mini = NN.forward(x_mini)

    # Do some loss
    loss = crossent(x_mini, torch.argmax(y_mini, dim=1))

    # Do a backward pass and update
    loss.backward()
    optimizer.step()

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

        # Make 'em torch-y
        imgs = torch.from_numpy(imgs).to(device)
        labels = torch.from_numpy(labels).long().to(device)

        # Reshape and forward
        new_input = imgs.reshape(imgs.shape[0], imgs.shape[1] * imgs.shape[2] * imgs.shape[3])
        output = NN.forward(new_input)

        # Do cross entropy
        entropy = crossent.forward(output, torch.argmax(labels, dim = 1))

        # Accumulate loss and accuracy and divide both by prevent_overload
        temp_loss = temp_loss + entropy / prevent_overload
        temp_acc  = temp_acc  + accuracy(output, labels) / prevent_overload

      # Test set acc and train set acc for plots only
      trainloss.append(loss.item())
      trainacc.append(accuracy(x_mini, y_mini).item())
      testloss.append(temp_loss.item())
      testacc.append(temp_acc.item())
      steps.append(step)
      
      # Print acc
      print("Loss at step ", step, " = ", temp_loss)
      print("Acc  at step ", step, " = ", temp_acc, "\n")

      # Print losses and accuracies
      # Mostly for writing purposes so commented out 
      # print("Trainloss: ", trainloss)
      # print("Trainacc:  ", trainacc)
      # print("Testloss:  ", testloss)
      # print("Test acc:  ", testacc)
      # print("Step    :  ", steps)

  # Commented out to prevent creation of files
  # Write data to csv file    
  # First build string to specify a file
  #dropout = True
  #filename = "(" + str(FLAGS.dnn_hidden_units) + ")" + "-" + str(FLAGS.max_steps) + "-" + str(FLAGS.learning_rate) + "-" + str(dropout) + ".csv"

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
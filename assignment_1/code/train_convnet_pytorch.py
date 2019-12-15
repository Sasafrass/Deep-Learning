"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim
import csv

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

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
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)
  torch.manual_seed(42)

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
  
  # Initialize ConvNet
  NN = ConvNet(depth, y_test.shape[0])

  # Initialize device GPU or CPU and port model to gpu
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # Set loss to Cross Entropy and initialize optimizer
  crossent = nn.CrossEntropyLoss()
  optimizer = optim.Adam(NN.parameters(), lr = FLAGS.learning_rate)

  #NN.to(device)
  NN.cuda()

  # Keep track for plots
  trainloss = []
  trainacc  = []
  testloss  = []
  testacc   = []
  steps     = []

# Do mini-batch gradient descent
  for step in range(FLAGS.max_steps):

    # New mini-batch and reshape x
    x_mini, y_mini = cifar10['train'].next_batch(FLAGS.batch_size)

    # Make 'em torch-y
    x_mini = torch.from_numpy(x_mini).to(device)
    y_mini = torch.from_numpy(y_mini).to(device)
    y_mini = y_mini.long()

    # Clear the gradient buffer and do forward pass
    optimizer.zero_grad()
    x_mini = NN.forward(x_mini)

    # Do some loss
    loss = crossent(x_mini, torch.argmax(y_mini, dim = 1))

    # Do a backward pass and update
    loss.backward()
    optimizer.step()

    if (step % FLAGS.eval_freq) == 0:
      
      # Initialize loss and acc
      temp_loss = 0
      temp_acc  = 0

      # Prevent memory overload so divvy up in smaller batches
      prevent_overload = 50
      for i in range(prevent_overload):
        
        # Break it up to run over all test data
        size = cifar10['test'].num_examples // prevent_overload # Int division is necessary
        imgs, labels = cifar10['test'].next_batch(size)

        # Make 'em torch-y
        imgs = torch.from_numpy(imgs).to(device)
        labels = torch.from_numpy(labels).to(device)
        labels = labels.long()

        # Indices for evaluating accuracy
        acc_indices = torch.argmax(labels, dim = 1)

        # Feed forward and compute loss
        output = NN.forward(imgs)
        entropy = crossent(output, acc_indices).item() 

        # Update loss and accuracy including division for actual value
        temp_loss = temp_loss + entropy / prevent_overload        
        temp_acc  = temp_acc + accuracy(output, labels).item() / prevent_overload

      # Print loss and accuracy
      print("Loss at step ", step, " = ", temp_loss)
      print("Acc  at step ", step, " = ", temp_acc)

      # Add losses, accuracies and steps to file
      #print(loss.item())
      #print(accuracy(x_mini, y_mini).item())
      #print(temp_loss)
      #print(temp_acc)
      #print(step)

      trainloss.append(loss.item())
      trainacc.append(accuracy(x_mini, y_mini).item())
      testloss.append(temp_loss)
      testacc.append(temp_acc)
      steps.append(step)

  # Commented out to prevent creation and overwriting of files
  # Create file str
  #dropout = False
  #filename = "CNN-(" + ")" + "-" + str(FLAGS.max_steps) + "-" + str(FLAGS.learning_rate) + "-" + str(dropout) + ".csv"

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
  FLAGS, unparsed = parser.parse_known_args()

  main()
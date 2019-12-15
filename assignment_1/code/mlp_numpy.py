"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes, neg_slope):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
      neg_slope: negative slope parameter for LeakyReLU
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    # Helper to create specific modules
    def Module(kind, left = 0, right = 0, neg_slope = neg_slope):

      # For linear
      if kind == "Linear":
        return LinearModule(left, right)

      # For LeakyReLU
      if kind == "LeakyReLU":
        return LeakyReLUModule(neg_slope)

      # Softmax
      else:
        return SoftMaxModule()

    # Initialize lists for linear layers and activations
    self.nn = []
    self.activations = []

    # Create first linear layer and Leaky ReLU
    linear = Module("Linear", n_inputs, n_hidden[0])
    leaky  = Module("LeakyReLU", neg_slope = neg_slope)

    # Initialize the layers
    self.nn.append(linear)
    self.activations.append(leaky)

    # Initialize rest of layers
    for i in range(1, len(n_hidden)):

      # Create linear and leaky
      linear = Module("Linear", n_hidden[i-1], n_hidden[i])
      leaky  = Module("LeakyReLU", neg_slope = neg_slope)

      # Add them to respective lists
      self.nn.append(linear)
      self.activations.append(leaky)

    # Create last linear layer and softmax
    linear  = Module("Linear",  n_hidden[-1], n_classes)
    softmax = Module("SoftMax") 

    # Add final layers
    self.nn.append(linear)
    self.activations.append(softmax)

    # Store number of hidden layers for later use
    self.numhidden = len(n_hidden)

    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    # Loop over all layers
    for i in range(self.numhidden + 1):

      # Pass through linear forward, then through activation
      x = self.nn[i].forward(x)
      x = self.activations[i].forward(x)

    out = x
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    TODO:
    Implement backward pass of the network.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################

    # Traverse the list backwards
    for i in reversed(range(self.numhidden + 1)):

      dout = self.activations[i].backward(dout)
      dout = self.nn[i].backward(dout)

    ########################
    # END OF YOUR CODE    #
    #######################

    return

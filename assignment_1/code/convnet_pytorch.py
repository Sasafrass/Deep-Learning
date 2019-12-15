"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import torch-y stuff
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    
    # Initialization is inspired by 60 minute blitz
    super(ConvNet, self).__init__()

    # In, out, kernel, stride, padding (for Conv2d)
    # Kernel, stride, padding (for pool)

    # Do conv1 with n_channels (for variability instead of 3)
    self.conv1   = nn.Conv2d(n_channels, 64, 3, stride = 1, padding = 1)

    # Instantiate a pooling filter because it's always the same 
    # Also instantiate a batch norm
    self.pool    = nn.MaxPool2d(3, stride = 2, padding = 1)
    self.bnorm1  = nn.BatchNorm2d(64)

    # Do next set of convolutions
    self.conv2   = nn.Conv2d(64,128, 3, stride = 1, padding = 1)
    self.bnorm2  = nn.BatchNorm2d(128)

    # Conv3 and batch norm
    self.conv3_a = nn.Conv2d(128, 256, 3, stride = 1, padding = 1)
    self.conv3_b = nn.Conv2d(256, 256, 3, stride = 1, padding = 1)
    self.bnorm3  = nn.BatchNorm2d(256)

    # 4th set of convolutions
    self.conv4_a = nn.Conv2d(256, 512, 3, stride = 1, padding = 1)
    self.conv4_b = nn.Conv2d(512, 512, 3, stride = 1, padding = 1)
    self.bnorm4  = nn.BatchNorm2d(512)

    # 5th set of convolutions
    self.conv5_a = nn.Conv2d(512, 512, 3, stride = 1, padding = 1)
    self.conv5_b = nn.Conv2d(512, 512, 3, stride = 1, padding = 1)   
    self.bnorm5  = nn.BatchNorm2d(512)

    # Final linear layer (for variability output n_classes instead of 10)
    self.fc1     = nn.Linear(512, 10)

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

    # Do the forward pass
    x = self.pool(F.relu(self.bnorm1(self.conv1(x))))
    x = self.pool(F.relu(self.bnorm2(self.conv2(x))))

    # First we do conv3_a..
    x = self.conv3_a(x)

    # .. then we're doing the full layer thing and conv4_a
    x = self.pool(F.relu(self.bnorm3(self.conv3_b(x))))
    x = self.conv4_a(x)

    # convpool conv
    x = self.pool(F.relu(self.bnorm4(self.conv4_b(x))))
    x = self.conv5_a(x)

    # Last pooling
    x = self.pool(F.relu(self.bnorm5(self.conv5_b(x))))

    # Reshape to proper size with view (32 input rows and 512 columns for training)
    x = x.view(-1, 512)
    out = self.fc1(x)


    ########################
    # END OF YOUR CODE    #
    #######################

    return out

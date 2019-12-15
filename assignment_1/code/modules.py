"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.params = {'weight': np.random.normal(0, 0.0001, (out_features, in_features)), 'bias': np.zeros((out_features, 1))}
    self.grads = {'weight': np.zeros((out_features, in_features)), 'bias': np.zeros((out_features, 1))}
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    ####################### 
    out = self.params['weight'] @ x.T + self.params['bias']
    self.x = x

    ########################
    # END OF YOUR CODE    #
    #######################

    return out.T

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    # With respect to x
    dx = dout @ self.params['weight']

    # With respect to w
    self.grads['weight'] = dout.T @ self.x

    # With respect to bias
    self.grads["bias"] = dout.T.sum(axis=1).reshape(self.grads["bias"].shape)

    ########################
    # END OF YOUR CODE    #
    #######################
    
    return dx

class LeakyReLUModule(object):
  """
  Leaky ReLU activation module.
  """
  def __init__(self, neg_slope):
    """
    Initializes the parameters of the module.

    Args:
      neg_slope: negative slope parameter.

    TODO:
    Initialize the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.params = {'a' : neg_slope}
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    # Save x for later use in backprop
    self.x = x

    # We can use np.maximum() for element-wise maximum
    zeroes = np.zeros((x.shape))
    out = np.maximum(x, zeroes) + (self.params['a'] * np.minimum(x, zeroes))

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    # Construct "diagonal" matrix (same as element wise product with n-1 dim matrix)
    diagonal = np.where(self.x >= 0, 1, self.params['a'])
    dx = dout * diagonal
    
    ########################
    # END OF YOUR CODE    #
    #######################    

    return dx


class SoftMaxModule(object):
  """
  Softmax activation module.
  """

  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    
    # Save x in module for later use
    self.x = x

    # Max trick
    b = np.max(x)
    xminb = x - b

    # Divide exponents of input by sum of exponents
    #out = np.exp(xminb) / (np.tile(np.sum(np.exp(xminb), axis = 1), (10,1)).T)
    out = np.exp(xminb)/np.exp(xminb).sum(axis=1, keepdims=True)
    
    self.out = out

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.
    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    # Retrieve our saved output
    output = self.out

    # Use dout to retrieve dimensions
    num_inputs  = dout.shape[0]
    num_classes = dout.shape[1]

    # We can create a 200x10x10 tensor with the 10x10 diagonals filled with xi from self.out
    diagonal = np.zeros((num_inputs, num_classes, num_classes))
    diagonal[:,np.arange(num_classes), np.arange(num_classes)] = output

    # Get tensor of outer products to subtract from diagonal using einsum
    outer = np.einsum('ij,ik->ijk',output, output)
    result = diagonal - outer

    # Use np.einsum again for proper dot products for all batch elements to calculate total gradient
    dx = np.einsum('ij,ijk->ik', dout, result)

    #######################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """

  def forward(self, x, y):
    """
    Forward pass.
    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss

    TODO:
    Implement forward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    # Small number to prevent division by zero
    e = 1e-9
    out = - np.sum (y * np.log(x+e), axis = 1).mean()

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.
    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    # To prevent division by 0
    e = 1e-9

    # DIVIDE BY NUMBER OF UNITS IN MINIBATCH TOOK WAY TOO LONG TO DEBUG
    stable_x = x + e
    dx = (-y / stable_x)/x.shape[0]

    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

# To test functions
if __name__ == '__main__':
  test = LinearModule(40, 10)
  test.forward(np.zeros((40)))

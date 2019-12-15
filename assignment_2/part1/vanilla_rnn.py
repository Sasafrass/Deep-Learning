################################################################################
# MIT License
#
# Copyright (c) 2019
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(VanillaRNN, self).__init__()
        
        # Save sequence length for for loop
        self.seq_length = seq_length
        self.device = device

        # Initialize weights
        self.whx = nn.Parameter(torch.empty(num_hidden,input_dim).normal_(mean = 0,std = 1/num_hidden)) # num hidden by input dim tensor
        self.whh = nn.Parameter(torch.empty(num_hidden, num_hidden).normal_(mean = 0, std = 1/num_hidden))
        self.wph = nn.Parameter(torch.empty(num_classes, num_hidden).normal_(mean = 0, std = 1/num_hidden))

        # Initialize biases
        self.bh = nn.Parameter(torch.zeros(num_hidden, 1))
        self.bp = nn.Parameter(torch.zeros(num_classes, 1))

        # Initialize hidden 
        self.h_init = nn.Parameter(torch.zeros(num_hidden, 1))

    def forward(self, x):
        # Implementation here ...

        # Give ht its first value
        ht = self.h_init

        # Whoops
        tanh = nn.Tanh()

        # Loop over sequence length and do vanilla RNN update rule
        for i in range(self.seq_length):
            
            # xt is the right input for time-step t, transpose because of batch implementation
            xt = torch.unsqueeze(x[:, i].t(), 0).to(self.device)
            ht = tanh(self.whx @ xt + self.whh @ ht + self.bh)

        # Predict last digit
        pt = self.wph @ ht + self.bp

        # Make dimensions match up 
        pt = pt.t()

        # Return pt
        return pt

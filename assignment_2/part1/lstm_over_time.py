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

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(LSTM, self).__init__()
        # Initialization here ...

        # Save sequence length
        self.seq_length = seq_length
        self.device = device
        self.input_dim = input_dim

        # Initialize weights wrt inputs
        self.wgx = nn.Parameter(torch.empty(num_hidden,input_dim).normal_(mean = 0,std = 1/num_hidden))
        self.wix = nn.Parameter(torch.empty(num_hidden,input_dim).normal_(mean = 0,std = 1/num_hidden))
        self.wfx = nn.Parameter(torch.empty(num_hidden,input_dim).normal_(mean = 0,std = 1/num_hidden))
        self.wox = nn.Parameter(torch.empty(num_hidden,input_dim).normal_(mean = 0,std = 1/num_hidden))

        # Initialize weights wrt hidden
        self.wgh = nn.Parameter(torch.empty(num_hidden,num_hidden).normal_(mean = 0,std = 1/num_hidden))
        self.wih = nn.Parameter(torch.empty(num_hidden,num_hidden).normal_(mean = 0,std = 1/num_hidden))
        self.wfh = nn.Parameter(torch.empty(num_hidden,num_hidden).normal_(mean = 0,std = 1/num_hidden))
        self.woh = nn.Parameter(torch.empty(num_hidden,num_hidden).normal_(mean = 0,std = 1/num_hidden))

        # Initialize biases
        self.bg = nn.Parameter(torch.zeros(num_hidden, 1))
        self.bi = nn.Parameter(torch.zeros(num_hidden, 1))
        self.bf = nn.Parameter(torch.ones(num_hidden, 1))
        self.bo = nn.Parameter(torch.zeros(num_hidden, 1))

        # Initialize the output parameters
        self.wph = nn.Parameter(torch.empty(num_classes,num_hidden).normal_(mean = 0,std = 0.0001))
        self.bp = nn.Parameter(torch.zeros(num_classes, 1))

        # Initialize h and Ct 
        self.h_init = nn.Parameter(torch.zeros(num_hidden, 1)).requires_grad_(True)
        self.c_init = nn.Parameter(torch.zeros(num_hidden, 1))

        # Initialize an htlist
        #self.htlist = torch.empty(num_hidden, self.seq_length)
        self.htlist = []

    def forward(self, x):
        # Implementation here ...

        # Initialize ht and Ct
        # ht constantly refers to h(t-1)!
        ht = self.h_init
        ht.requires_grad_(True)
        ct = self.c_init  
        ct.requires_grad_(True)

        # tanh and sigmoid
        tanh = nn.Tanh().requires_grad_(True)
        sigmoid = nn.Sigmoid().requires_grad_(True)

        # be conSEQUENTIAL
        for i in range(self.seq_length):
            # xt is the right input for time-step t, transpose because of batch implementation
            xt = torch.unsqueeze(x[:, i].t(), 0).to(self.device)

            # Implement the gates
            gt = tanh(   self.wgx @ xt + self.wgh @ ht + self.bg)
            it = sigmoid(self.wix @ xt + self.wih @ ht + self.bi)
            ft = sigmoid(self.wfx @ xt + self.wfh @ ht + self.bf)
            ot = sigmoid(self.wox @ xt + self.woh @ ht + self.bo).requires_grad_(True)

            # Update Ct and ht values
            ct = gt * it + ct * ft
            ht = tanh(ct) * ot
            
            # Retain grad and add to htlist
            ht.retain_grad()
            self.htlist.append(ht)
            
            #print("ht grad at step ", str(i) , ": " , ht.grad)

        # Update probabilities - transpose for right interpretation
        pt = (self.wph @ ht + self.bp).t()

        # Return probabilities
        return pt 
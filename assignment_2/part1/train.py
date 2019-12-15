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

import sys
sys.path.append("..")

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from part1.dataset import PalindromeDataset
from part1.vanilla_rnn import VanillaRNN
from part1.lstm import LSTM

import csv 

# You may want to look into tensorboard for logging
# from torch.utils.tensorboard import SummaryWriter

################################################################################

# Create own accuracy function
def calc_accuracy(predictions, targets):

    # Take argmaxes from preds for predictions
    preds = torch.argmax(predictions, dim = 1).to(config.device)
    
    # Return sum of where preds == targets divided by length of targets for accuracy
    return torch.sum(preds == targets).item() / len(targets)

def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device(config.device)
    
    if config.sequences == '10':
        sequences = [config.input_length]
        print(sequences)
    else: 
        sequences = config.sequences.split(",")
        sequences = [int(sequence) for sequence in sequences]
        print(sequences)

    # Accuracies list for plotting
    accuracies = []

    # Main loop over all sequence sizes - default = input_length
    for seq_size in sequences:

        # Initialize the model that we are going to use
        if config.model_type == "RNN":
            model = VanillaRNN(seq_size, config.input_dim, config.num_hidden, config.num_classes, device = device).to(device)  
        else:
            model = LSTM(seq_size, config.input_dim, config.num_hidden, config.num_classes, device = device).to(device)

        # Setup the loss and optimizer
        criterion = nn.CrossEntropyLoss() 
        optimizer = optim.RMSprop(model.parameters(), lr = config.learning_rate)

        # Initialize the dataset and data loader (note the +1)
        # dataset = PalindromeDataset(config.input_length+1)
        dataset = PalindromeDataset(seq_size + 1)
        data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

        # Test set for accuracy plots
        test_size   = 10000
        test_loader = DataLoader(dataset, test_size, num_workers = 1)

        # Loop once to retrieve test inputs and targets 
        for step, (inputs, targets) in enumerate(data_loader):
            test_inputs  = inputs.to(device)
            test_targets = targets.to(device)

            # Break after one iteration
            if step == 0:
                break

        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            # Only for time measurement of step through network
            t1 = time.time()

            # Add more code here ...
            # Clear the gradient buffers 
            optimizer.zero_grad()

            # Get predictions - put both targets and preds to device
            predictions = model(batch_inputs).to(device)
            batch_targets = batch_targets.to(device)    

            ############################################################################
            # QUESTION: what happens here and why?
            ############################################################################
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
            ############################################################################

            # Add more code here ...
            loss = criterion(predictions, batch_targets)   # fixme
            accuracy = calc_accuracy(predictions, batch_targets)  # fixme

            # Backward the loss and do step 
            loss.backward()
            optimizer.step()

            # Just for time measurement
            t2 = time.time() + 0.01
            examples_per_second = config.batch_size/float(t2-t1)

            if step % 10 == 0:
                
                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                    "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                        config.train_steps, config.batch_size, examples_per_second,
                        accuracy, loss
                ))

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break

        print('Done training with sequence size:', seq_size)
        
        test_accuracy = calc_accuracy(model(test_inputs), test_targets)
        accuracies.append(test_accuracy)
        print("Accuracies: ", accuracies)

    # Write stuff to csv
    filename = config.filename
    with open("csv/" + filename, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(sequences)
        wr.writerow(accuracies)
    



 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    # Own parameters
    # for list of sequences for easier plotting
    parser.add_argument('--sequences', type = str, default = '10',
                      help='Comma separated list of number of units in each hidden layer')
    # For filename 
    parser.add_argument('--filename', type = str, default = 'dummy.csv',
                      help='Filename for csv output - standard is dummy to prevent overwriting of important data')

    config = parser.parse_args()

    # Train the model
    train(config)
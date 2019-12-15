# MIT License
#
# Copyright (c) 2019 Tom Runia
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

import os
import time
from datetime import datetime
import argparse
import csv

import numpy as np
import random

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional

from part2.dataset import TextDataset
from part2.model import TextGenerationModel

################################################################################

def calc_accuracy(predictions, targets):
    
    # Calculates accuracy with new preds, creating boolean mask and dividing by total num of elements in targets
    new_preds = torch.argmax(predictions, dim = 2)
    return torch.sum(new_preds == targets).item() / (targets.shape[0] * targets.shape[1])

def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length) 
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size, config.lstm_num_hidden,
                                config.lstm_num_layers, device = device).to(device)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = config.learning_rate)#, weight_decay = config.weight_decay) # fixme

    # Save vocab size
    vocab_size = dataset.vocab_size

    # Lists to output
    sentences = []
    steps = []
    accuracies = []
    losses = []

    # To keep track of proper step
    maxstep = 0

    for epoch in range(config.num_epochs):

        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            # Only for time measurement of step through network
            t1 = time.time()

            # Zero the gradient buffers
            optimizer.zero_grad()

            #######################################################
            # Add more code here ... 
            
            # Inputs and outputs are one-hot encoded - push all to device
            batch_inputs = torch.nn.functional.one_hot(batch_inputs, vocab_size).long().to(device)
            batch_targets = batch_targets.to(device)
            outputs = model(batch_inputs).to(device)
        
            #######################################################

            # Initialize loss 
            loss = 0.0

            # Calculate loss over all time steps and calculate accuracy
            loss = criterion(outputs.permute(0, 2, 1), batch_targets)
            accuracy = calc_accuracy(outputs, batch_targets) 

            # Backward the loss and do optimizer step
            loss.backward()
            optimizer.step()

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            # Keep track of true step throughout epochs
            truestep = maxstep * epoch + step

            if truestep % config.print_every == 0:

                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                    "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), truestep,
                        int(config.train_steps) * config.num_epochs, config.batch_size, examples_per_second,
                        accuracy, loss
                ))

            if step % config.sample_every == 0:
                # Generate some sentences by sampling from the model
                pass

                # BELANGRIJK: ZIE HET PIAZZA BERICHT HIEROVER!! IPV 'STEP ==' IS DIT WELLICHT STEP %

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break

            if truestep % config.generate_every == 0:

                # Empty list to build sequence
                sequence = np.zeros(config.seq_length)

                # Generating, so we don't want to train the model 
                with torch.no_grad():

                    # Generate a first letter randomly (one-hot encoded)
                    x = torch.zeros(vocab_size).to(device)
                    rand = random.randint(0,vocab_size-1)
                    sequence[0] = rand
                    x[rand] = 1 

                    # Turn into one-hot
                    x = x.unsqueeze(0).unsqueeze(1)
                    output = x

                    # get seq_length - 1 more predictions   
                    #for i in range(1,config.seq_length):
                    for i in range(1, config.seq_length):
                        x = model(output) 
                        x = x[:,i-1,:]

                        # Check whether we're using greedy sampling or not
                        if config.temperature:

                            # Epsilon for numerical stability
                            epsilon = 1e-8
                            x = x * config.heat
                            m = torch.distributions.Categorical(logits = (x + epsilon))
                            x = m.sample()
                        else:
                            x = torch.argmax(x).item()
                        sequence[i] = x

                        # Turn x into one hot again for next step 
                        output = one_hotter(sequence, vocab_size, length = i + 1)

                #print("len: ", len(sequence))
                #print(sequence)
                sentence = dataset.convert_to_string(np.array(sequence).astype(int)) 
                print(sentence)

                # Add info to lists for csv file
                steps.append(truestep)
                sentences.append(sentence)
                accuracies.append(accuracy)
                losses.append(loss.item())

        # Keep track of maxstep for display of proper step
        maxstep = step

    print('Done training.')

    # Save model
    torch.save(model.state_dict(), "models/trained_" + config.output_file + "_" + str(config.lstm_num_hidden) + "_" + str(config.learning_rate) + ".pth")
    
    # Write stuff to csv
    filename = config.output_file
    with open("csv/" + filename + ".csv", 'w', newline='', encoding = 'utf-8') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(steps)
        wr.writerow(sentences)
        wr.writerow(accuracies)
        wr.writerow(losses)

def one_hotter(sequence, vocab_size, length):

    out = torch.zeros(size = (1,length, vocab_size)).to(config.device)
    #print("out shape: ", out.shape)

    for i in range(length):
        #print("seq: ", sequence)
        out[:,i,int(sequence[i])] = 1

    return out

 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on", default = "assets/book_EN_democracy_in_the_US.txt")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=1000, help='How often to sample from the model')

    # Device
    parser.add_argument('--device', type=str, default='cuda:0', help='How often to sample from the model')

    # Extra params
    parser.add_argument('--generate_every', type=int, default=1000, help='How often to generate new sequence')
    parser.add_argument('--num_epochs', type=int, default=60, help='How many epochs to run for')
    parser.add_argument('--output_file', type=str, default="dummy", help='File which to write output to')
    parser.add_argument('--temperature', type=bool, default=False, help='Sampling method')
    parser.add_argument('--heat', type=float, default=1.0, help='Sampling method')

    config = parser.parse_args()

    # Train the model
    train(config)


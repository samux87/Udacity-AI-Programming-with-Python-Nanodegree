#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

import pandas as pd
from PIL import Image
import time

import matplotlib.pyplot as plt
import os, random

import json
import argparse

def main():
    
    print('Hi! Welcome to the Training script')
    
    # Collect args and parse them
    parser = argparse.ArgumentParser(description='Arguments to load the training script. USAGE: train.py data_dir --arch --hidden_units --learning_rate --epochs')

    parser.add_argument('data_dir', action="store", type=str, help='Directory where all the data is (required)')
    parser.add_argument('--arch', action="store", type=str, default="vgg16", help='Model architecture (default = vgg16)')
    parser.add_argument('--hidden_units', action="store", type=int, default=1000, help='The number of hidden units (default = 1000)')
    parser.add_argument('--learning_rate', action="store", type=float, default=0.001, help='The learning rate (default = 0.001)')
    parser.add_argument('--epochs', action="store", type=int, default=5, help='The number of epochs (default = 5)')

    inputargs = parser.parse_args()
    
   
    # Load the Data
    data_dir = inputargs.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # DONE: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])
                                          ])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])
                                          ])

    # DONE: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    image_datasets = {'train': train_data, 'valid': valid_data, 'test': test_data}


    # DONE: Using the image datasets and the trainforms, define the dataloaders

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    # Label mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    # DONE: Build and train your network
    model = getattr(models, inputargs.arch)(pretrained=True)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Create a new classifier
    from collections import OrderedDict
    intermediary_nodes = inputargs.hidden_units
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, intermediary_nodes)),
                              ('relu', nn.ReLU()),
                              ('dropout', nn.Dropout(p=0.5)),

                              ('fc2', nn.Linear(intermediary_nodes, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier

    model.classifier

    # Now lets train the classifier
    criterion = nn.NLLLoss()
    learn_rate = inputargs.learning_rate
    optimizer    = optim.Adam(model.classifier.parameters(), lr=learn_rate)

    cuda  = torch.cuda.is_available()
    if cuda:
        model.to("cuda:0")
        device="cuda"
        print('Using Cuda')
    else:
        model.to("cpu")
        device="cpu"
        print('Using CPU')



    print ("Beginning training")

    epochs = inputargs.epochs
    steps = 0
    running_loss = 0
    print_every = 20

    for e in range(epochs):
        print ('Just started epoch #' + str(e+1))

        model.train()

        for i, (inputs, labels) in enumerate(trainloader):
            steps += 1

            optimizer.zero_grad()

            inputs, labels = Variable(inputs), Variable(labels)

            # Move to the GPU
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()


            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


            if steps % print_every == 0:
                # Put the network in evaluation mode
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():

                    valid_loss = 0
                    accuracy = 0
                    for j, (inputs_valid, labels_valid) in enumerate(validloader):

                        inputs_valid, labels_valid = Variable(inputs_valid), Variable(labels_valid)

                        # Move to the GPU
                        if cuda:
                            inputs_valid, labels_valid = inputs_valid.cuda(), labels_valid.cuda()

                        outputs_valid = model.forward(inputs_valid)
                        loss_valid = criterion(outputs_valid, labels_valid)
                        valid_loss += loss_valid.item()

                        ps = torch.exp(outputs_valid)

                        equality = (labels_valid.data == ps.max(dim=1)[1])

                        accuracy += equality.type(torch.FloatTensor).mean()

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss),
                      "Valid Loss: {:.3f}.. ".format(valid_loss),
                      "Valid Accuracy %: {:.3f}".format(accuracy/len(validloader) * 100))

                running_loss = 0

                # Make sure training is back on
                model.train()


    print("Done training")

    
    # DONE: Test the network

    test_loss = 0
    accuracy = 0

    model.eval()  # Put the model in evaluation mode

    for k, (inputs, labels) in enumerate(testloader):

        with torch.no_grad():
            inputs, labels = Variable(inputs), Variable(labels)

            # Move to the GPU
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            ps = torch.exp(outputs)
            equality = (labels.data == ps.max(dim=1)[1])

            accuracy += equality.type(torch.FloatTensor).mean()

    print("Test Loss: {:.3f}.. ".format(test_loss))
    print("Test Accuracy %: {:.3f}".format(accuracy/len(testloader) * 100))

    
    # DONE: Save the checkpoint 
    checkpoint = {'class_to_idx': image_datasets['train'].class_to_idx,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'arch': inputargs.arch,
                  'intermediary_nodes': intermediary_nodes,
                  'learn_rate': learn_rate}

    torch.save(checkpoint, 'checkpoint.pth')
    print('Saved model with the following keys:')
    print(model.state_dict().keys())


# Call to main function to run the program
if __name__ == "__main__":
    main()


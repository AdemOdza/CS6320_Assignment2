import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser


unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU() # The rectified linear unit; one valid choice of activation function
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)
        self.softmax = nn.LogSoftmax() # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
        self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        ### START OF MODIFIED CODE ###
        # Hidden Layer Representation
        hidden_output = self.activation(self.W1(input_vector))
        # Output Layer Representation
        final_output = self.W2(hidden_output)
        # softmax to get prob distribution
        predicted_vector = self.softmax(final_output)
        ### END OF MODIFIED CODE ###

        return predicted_vector


# Returns:
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index
        index2word[index] = word
    vocab.add(unk)
    return vocab, word2index, index2word


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index))
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data


### START OF MODIFIED CODE ###
# We modified the load_data function to also load the test data
def load_data(train_data, val_data, test_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)
    with open(test_data) as test_f:
        test = json.load(test_f)
    tra = []
    val = []
    tes = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in test:
        tes.append((elt["text"].split(),int(elt["stars"]-1)))

    return tra, val, tes
### END OF MODIFIED CODE ###

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    ### START OF MODIFIED CODE ###
    parser.add_argument("--test_data", default = "test.json", help = "path to test data")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate") # Added learning rate as a script argument for ease
    ### END OF MODIFIED CODE ###
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    # fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # load data
    print("========== Loading data ==========")
    ### START OF MODIFIED CODE ###
    train_data, valid_data, test_data = load_data(args.train_data, args.val_data, args.test_data)
    ### END OF MODIFIED CODE ###

    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    ### START OF MODIFIED CODE ###
    test_data = convert_to_vector_representation(test_data, word2index)
    ### END OF MODIFIED CODE ###


    model = FFNN(input_dim = len(vocab), h = args.hidden_dim)
    ### START OF MODIFIED CODE ###
    optimizer = optim.SGD(model.parameters(),lr=args.learning_rate, momentum=0.9)
    ### END OF MODIFIED CODE ###

    ### START OF MODIFIED CODE ###
    # We use these two lists to compute and store the training losses and validation accuracies for each epoch
    train_losses = []
    val_accuracies = []
    ### END OF MODIFIED CODE ###

    print("========== Training for {} epochs ==========".format(args.epochs))
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        ### START OF MODIFIED CODE ###
        curr_epoch_loss = 0.0
        ### END OF MODIFIED CODE ###
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data) # Good practice to shuffle order of training data
        minibatch_size = 16
        N = len(train_data)
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size # type: ignore
            ### START OF MODIFIED CODE ###
            curr_epoch_loss += loss.item() #sum the loss
            ### END OF MODIFIED CODE ###
            loss.backward()
            optimizer.step()

        ### START OF MODIFIED CODE ###
        # Store the average train loss for the current epoch
        avg_epoch_loss = curr_epoch_loss / (N // minibatch_size)
        train_losses.append(avg_epoch_loss)
        ### END OF MODIFIED CODE ###

        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        ### START OF MODIFIED CODE ###
        print("Training loss for epoch {}: {}".format(epoch + 1, avg_epoch_loss))
        ### END OF MODIFIED CODE ###
        print("Training time for this epoch: {}".format(time.time() - start_time))

        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        minibatch_size = 16
        N = len(valid_data)
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size # type: ignore

        ### START OF MODIFIED CODE ###
        # Store the validation accuracy for the current epoch
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        ### END OF MODIFIED CODE ###

        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Validation time for this epoch: {}".format(time.time() - start_time))

    ### START OF MODIFIED CODE ###
    print("Training losses: ", train_losses)
    print("Validation accuracies: ", val_accuracies)
    ### END OF MODIFIED CODE ###

    ### START OF MODIFIED CODE ###
    # Evaluate on test data
    print("========== Testing ==========")
    model.eval()
    correct = 0
    total = 0
    predictions = []

    for input_vector, gold_label in tqdm(test_data):
        predicted_vector = model(input_vector)
        predicted_label = torch.argmax(predicted_vector).item()
        predictions.append(predicted_label)
        correct += int(predicted_label == gold_label)
        total += 1

    test_accuracy = correct / total
    print("Test accuracy: {}".format(test_accuracy))
    ### END OF MODIFIED CODE ###

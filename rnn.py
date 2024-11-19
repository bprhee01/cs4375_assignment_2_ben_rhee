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
import string
from argparse import ArgumentParser
import pickle

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h):
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 2 # Increased number of layers
        
        # Reduced dropout rates
        self.dropout = 0.2
        self.embedding_dropout = nn.Dropout(0.1)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(h * 2, num_heads=4, dropout=0.1)
        
        # Made it to Bidirectional LSTM
        self.rnn = nn.LSTM(input_dim, h, self.numOfLayer,
                          dropout=self.dropout,
                          bidirectional=True,
                          batch_first=False)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(h * 2)
        self.layer_norm2 = nn.LayerNorm(h * 2)
        
        # Output layers
        self.output_dropout = nn.Dropout(0.1)
        self.intermediate = nn.Linear(h * 2, h)
        self.activation = nn.ReLU()
        self.W = nn.Linear(h, 5)

        # Initialize weights
        self._initialize_weights()

        self.softmax = nn.LogSoftmax(dim=1)
        # Added label smoothing
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.1) ## Label smoothing

    ## added proper weight initialization
    def _initialize_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                init.orthogonal_(param, gain=0.8)
            elif 'bias' in name:
                init.zeros_(param)
        
        init.xavier_uniform_(self.intermediate.weight, gain=0.8)
        init.zeros_(self.intermediate.bias)
        init.xavier_uniform_(self.W.weight, gain=0.8)
        init.zeros_(self.W.bias)

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        # Apply embedding dropout
        inputs = self.embedding_dropout(inputs)
        
        # LSTM forward pass
        output, (h_n, c_n) = self.rnn(inputs)
        
        # Apply attention mechanism
        attn_output, _ = self.attention(output, output, output)
        
        # Get last hidden states
        last_hidden_forward = h_n[-2,:,:]
        last_hidden_backward = h_n[-1,:,:]
        last_hidden = torch.cat((last_hidden_forward, last_hidden_backward), dim=1)
        
        # First layer norm
        last_hidden = self.layer_norm1(last_hidden)
        
        # Residual connection with attention
        last_hidden = last_hidden + attn_output[-1]
        
        # Second layer norm
        last_hidden = self.layer_norm2(last_hidden)
        
        # Output pathway
        last_hidden = self.output_dropout(last_hidden)
        intermediate = self.activation(self.intermediate(last_hidden))
        z = self.W(intermediate)
        
        return self.softmax(z)

def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))
    return tra, val


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
    # Option 3 will be the most time consuming, so we do not recommend starting with this

    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim)
    #Decreased learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

    # Incluuded learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2,
        verbose=True
    )

    # Early stopping parameters
    best_validation_accuracy = 0
    patience = 5
    patience_counter = 0
    
    epoch = 0
    while epoch < args.epochs:
        random.shuffle(train_data)
        model.train()
        print("Training started for epoch {}".format(epoch + 1))
        
        correct = 0
        total = 0
        # Increased batch size
        minibatch_size = 64
        N = len(train_data)

        loss_total = 0
        loss_count = 0
        
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            batch_loss = None
            
            for example_index in range(minibatch_size):
                idx = minibatch_index * minibatch_size + example_index
                if idx >= len(train_data):
                    break
                    
                input_words, gold_label = train_data[idx]
                input_words = " ".join(input_words)
                input_words = input_words.translate(str.maketrans("", "", string.punctuation)).split()
                
                # Text augmentation
                if random.random() < 0.1:  # 10% chance to drop words
                    input_words = [w for w in input_words if random.random() > 0.1]
                
                if len(input_words) == 0:  # Skip empty sequences
                    continue
                
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() 
                          else word_embedding['unk'] for i in input_words]
                
                vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                output = model(vectors)
                
                example_loss = model.compute_Loss(output.view(1, -1), 
                                                torch.tensor([gold_label]))
                
                predicted_label = torch.argmax(output)
                correct += int(predicted_label == gold_label)
                total += 1
                
                if batch_loss is None:
                    batch_loss = example_loss
                else:
                    batch_loss += example_loss

            if total > 0:  # Only backprop if we processed some examples
                batch_loss = batch_loss / minibatch_size
                loss_total += batch_loss.item()
                loss_count += 1
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                batch_loss.backward()
                optimizer.step()

        avg_loss = loss_total / loss_count if loss_count > 0 else 0
        print(f"Average loss: {avg_loss}")
        print("Training completed for epoch {}".format(epoch + 1))
        training_accuracy = correct / total if total > 0 else 0
        print("Training accuracy for epoch {}: {}".format(epoch + 1, training_accuracy))

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        print("Validation started for epoch {}".format(epoch + 1))
        
        with torch.no_grad():
            for input_words, gold_label in tqdm(valid_data):
                input_words = " ".join(input_words)
                input_words = input_words.translate(str.maketrans("", "", string.punctuation)).split()
                
                if len(input_words) == 0:  # Skip empty sequences
                    continue
                    
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() 
                          else word_embedding['unk'] for i in input_words]
                
                vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                output = model(vectors)
                predicted_label = torch.argmax(output)
                correct += int(predicted_label == gold_label)
                total += 1

        print("Validation completed for epoch {}".format(epoch + 1))
        validation_accuracy = correct / total if total > 0 else 0
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, validation_accuracy))

        # Learning rate scheduling
        scheduler.step(validation_accuracy)
        
        # Early stopping check
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping! Best validation accuracy:", best_validation_accuracy)
                # Load best model
                model.load_state_dict(torch.load('best_model.pt'))
                break

        epoch += 1
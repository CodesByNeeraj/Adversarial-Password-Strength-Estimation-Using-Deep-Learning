#GAN 
#convert text data into tensors
#imports

import torch.nn as nn

#residual block (concept from skip connection)
class ResidualBlock(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.block = nn.Sequential(
            #pre-activation
            nn.ReLU(),
            #kernel size = 5 which means this model looks ta 5 characters at a time
            #dates are 4 digits long, humans love to put dates in passwords
            #common passwords / sequences are 4-6 characters long
            #padding = (kernel_size-1)/2
            nn.Conv1d(dim,dim,kernel_size=5,padding=2),
            nn.ReLU(),
            #mixes the data one more time to find deeper patterns
            nn.Conv1d(dim,dim,kernel_size=5,padding=2),
        )
        
    def forward(self,x):
        #we have a batch of password data labeled x
        #shape is (batch,128,10)
        #skip connection (raw input data x) + (augments) the output from the data (x) flowing through a block
        #0.3 is taken from research paper
        return x + (0.3*self.block(x))
        
#generates text using noise
class Generator(nn.Module):
    #vocab size refers to total number of unique characters our model is allowed to use
    #eg: 69 --> call letters + numbers + special characters
    def __init__(self,seq_len,vocab_size,hidden_dim=128):
        super().__init__()
        #output password length
        self.seq_len = seq_len
        #128 features of that character position, eg: is this a capital letter, is this a number, is this usually part of the data etc
        self.hidden_dim = hidden_dim
        #width 10 positions , height = 128
        #convulational layers require a 2d matrix so we multiply here
        #takes noise (128) and expands to 2d matrix
        self.linear = nn.Linear(hidden_dim,seq_len*hidden_dim)
        
        self.blocks = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),    
        )
        #translating 128 hidden features into defined vocab size
        #kernel size = 1: if lets say vocab size is 69, then this essentially means
        #given that we know 128 things about position 1, give me 69 scores, ie: one for 'A', one for 'B' etc
        self.out_layer = nn.Conv1d(hidden_dim,vocab_size,kernel_size=1)
        
        #applies softmax function to all columns in the 2d matrix
        #converting raw scores to probabilities
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,noise):
        #eg:(batch,1280)
        x = self.linear(noise)
        
        #eg:(batch,128,10) --> new shape ({hidden_dim} features for each of the {seq_len} positions)
        x = x.view(-1,self.hidden_dim,self.seq_len)
        
        #eg:(batch,128,10)
        x = self.blocks(x)
        
        #eg:(batch,vocab_size,10)
        x = self.out_layer(x)
        
        return self.softmax(x)
    
#input: password
#uses residual blocks to find patterns
#outputs a realness score
#WGAN --> no Softmax at the end because we want the raw score without any squashing
#this makes the math more stable and prevent the training from crashing
class Discriminator(nn.Module):
    def __init__(self,seq_len,vocab_size,hidden_dim=128):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        #we start with vocab size and compress it to hidden_dim
        self.entry_layer = nn.Conv1d(vocab_size,hidden_dim,kernel_size=1)
        
        self.blocks = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )
        #we flatten the grid and output 1 single number
        self.linear = nn.Linear(seq_len*hidden_dim,1)
        
    def forward(self,x):
        x = self.entry_layer(x)
        #we analyse the password structure
        x = self.blocks(x)
        #flatten grid into long line of 1280 numbers
        x = x.view(-1,self.hidden_dim*self.seq_len)
        #final score
        return self.linear(x)
        
        
        
        
            
            
            
            
    
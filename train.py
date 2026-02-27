import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import Generator, Discriminator
from utils import initialize_weights, TextDataset

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 10
vocab_size = 69
hidden_dim = 128
batch_size = 64
l_r = 1e-4
num_epochs = 1000
critic_iters = 10
lambda_gp = 10

#use gpu if computer have or is connected to cloud gpu or else use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

def gradient_penalty(disc, real_samples, fake_samples):

    # Random weight term for interpolation between real and fake samples
    #alpha determines where on the link we pick
    #if alpha==0, we look at fake image, if ==1 we look at real image, if ==0.7, we look at 70% real and 30% fake
    alpha = torch.rand((real_samples.size(0), 1, 1)).to(device)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    #critic
    d_interpolates = disc(interpolates)
    
    #dummy tensor
    fake = torch.ones(real_samples.size(0), 1).to(device)
    
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # Reshape gradients to calculate norm
    gradients = gradients.view(gradients.size(0), -1)
    
    # Calculate gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty
    
    
def get_infinite_batches(dataloader):
    # This acts just like the paper's inf_train_gen()
    while True:
        for batch in dataloader:
            yield batch

def train():
    
    print(f"Training on: {device}")
    
    dataset = TextDataset("./data/rockyou.txt")
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

    #initialising models
    gen = Generator(seq_len,vocab_size,hidden_dim).to(device)
    disc = Discriminator(seq_len,vocab_size,hidden_dim).to(device)

    #kaiming initialization
    #initialize_weights(gen)
    #initialize_weights(disc)

    #optimizers using RMSprop (based on WGAN research)
    optimizer_G = optim.Adam(gen.parameters(),lr=l_r,betas=(0.5, 0.9))
    optimizer_D = optim.Adam(disc.parameters(),lr=l_r,betas=(0.5, 0.9))
    
    #set models to training mode
    gen.train()
    disc.train()
    
    data_iterator = iter(get_infinite_batches(dataloader))
    print("Starting WGAN training now...")
    
    # Notice we loop over iterations now, not epochs (just like the paper)
    total_iterations = 200000
    
    #training loop
    for iteration in range(total_iterations):
        for _ in range(critic_iters):
            #real_indxes --> tensor of shape (64,10): 64 passwords with each of size 10
            real_indxes = next(data_iterator).to(device)
            curr_batch_size = real_indxes.size(0)
            
            real_data_one_hot = F.one_hot(real_indxes.long(),num_classes=vocab_size)
            real_data = real_data_one_hot.permute(0,2,1).float()
            
            #training the discriminator
            optimizer_D.zero_grad()
            
            noise = torch.randn(curr_batch_size,128).to(device)
            fake_data = gen(noise)
            
            critic_real = disc(real_data)
            #we detach to freeze the generator
            critic_fake = disc(fake_data.detach())
            
            gp = gradient_penalty(disc, real_data, fake_data.detach())
            
            # Critic Loss = -(Real - Fake) + (Lambda * GP)
            d_loss = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) 
                + (lambda_gp * gp)
            )
            d_loss.backward()
            optimizer_D.step()
                
        #train generator
        optimizer_G.zero_grad()
        noise = torch.randn(curr_batch_size, 128).to(device)
        fake_data = gen(noise)
        
        # Generator wants Critic to output High Real Score
        gen_fake_score = disc(fake_data)
        g_loss = -torch.mean(gen_fake_score)
        
        g_loss.backward()
        optimizer_G.step()
            
        # --- LOGGING ---
        if iteration % 100 == 0:
            print(
                f"Iteration [{iteration}/{total_iterations}] "
                f"Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f} "
                f"GP: {gp.item():.4f}"
            )

if __name__ == "__main__":
    train()
                


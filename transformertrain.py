import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import TextDataset
from transformermodels import PasswordTransformer

seq_len    = 10
batch_size = 64
num_epochs = 10
save_path  = "transformer_weights.pth"
vocab_path = "transformer_vocab.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

dataset    = TextDataset("./data/train.txt")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
vocab_size = dataset.vocab_size

model     = PasswordTransformer(vocab_size, seq_len).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ignore_index=0 so padding tokens don't pollute the loss signal
loss_fn   = nn.CrossEntropyLoss(ignore_index=0)

total_params = sum(p.numel() for p in model.parameters())
print(f"Vocab size: {vocab_size} | Model parameters: {total_params:,}")
print("Starting transformer training...")

model.train()

for epoch in range(num_epochs):
    total_loss  = 0.0
    num_batches = 0

    for batch in dataloader:
        batch = batch.to(device)

        # next-character prediction: feed chars 0..8, predict chars 1..9
        # (batch, 9)
        x_input  = batch[:, :-1] 
        # (batch, 9)  
        y_target = batch[:, 1:] 
        
        # (batch, 9, vocab_size)
        logits = model(x_input)  
        
        # (batch*9, vocab_size)
        logits   = logits.reshape(-1, vocab_size)
        # (batch*9,)  
        y_target = y_target.reshape(-1)          

        loss = loss_fn(logits, y_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss  += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"Epoch [{epoch+1}/{num_epochs}]  Avg Loss: {avg_loss:.4f}")

print("Training complete! Saving model...")
torch.save(model.state_dict(), save_path)
dataset.save_vocab(vocab_path)
print(f"Weights saved to {save_path}")
print(f"Vocab saved to  {vocab_path}")
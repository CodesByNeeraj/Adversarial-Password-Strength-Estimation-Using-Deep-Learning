from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.init as init

class TextDataset(Dataset):
    def __init__(self,text_data):
        self.text_data = text_data
    def __len__(self):
        return len(self.text_data)
    
    def __getitem__(self,idx):
        return self.text_data[idx]
    
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m,(nn.Conv1d,nn.Linear)):
            init.kaiming_uniform_(m.weight,nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias,0)
            
    

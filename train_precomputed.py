from models.best import Video, Audio
from dataloader import AVE
from utils import record_variables
from torch.utils.data.dataloader import DataLoader
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = AVE('AVE_Dataset/', 'train', 'settings.json', precomputed=True)
test_data = AVE('AVE_Dataset/', 'test', 'settings.json', precomputed=True)
val_data = AVE('AVE_Dataset/', 'val', 'settings.json', precomputed=True)
train_loader = DataLoader(train_data, 21, shuffle=True, num_workers=3, pin_memory=True)
test_loader = DataLoader(test_data, 1, shuffle=True, num_workers=1, pin_memory=True)
val_loader = DataLoader(val_data, 1, shuffle=True, num_workers=1, pin_memory=True)
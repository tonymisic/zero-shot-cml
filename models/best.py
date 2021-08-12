from torch import nn
from torch.functional import F
import torch

class Video(nn.Module):
    def __init__(self, video_size=25088, out=64, normalize=True):
        super(Video, self).__init__()
        self.normalize = normalize
        self.layer1 = nn.Linear(video_size, 4096)
        self.layer2 = nn.Linear(4096, 1024)
        self.layer3 = nn.Linear(1024, 512)
        self.layer4 = nn.Linear(512, out)

    def forward(self, video):
        video = F.relu(self.layer1(video))
        video = F.relu(self.layer2(video))
        video = F.relu(self.layer3(video))
        video = self.layer4(video)
        if self.normalize:
            return torch.sigmoid(video)
        else:
            return video

class Audio(nn.Module):
    def __init__(self, audio_size=128, out=64, normalize=True):
        super(Audio, self).__init__()
        self.normalize = normalize
        self.layer1 = nn.Linear(audio_size, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, out)
    
    def forward(self, audio):
        audio = F.relu(self.layer1(audio))
        audio = F.relu(self.layer2(audio))
        audio = F.relu(self.layer3(audio))
        audio = self.layer4(audio)
        if self.normalize:
            return torch.sigmoid(audio)
        else:
            return audio
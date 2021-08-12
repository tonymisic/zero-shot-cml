import torch, torch.nn as nn
import torch.nn.functional as F

class SelfAttention(torch.nn.Module):
    """Self-Attention Module
    """
    def __init__(self, embed_dim, heads=1):
        super(SelfAttention, self).__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim, heads)
    def forward(self, x):
        return self.attention(x, x, x)

class AudioGuidedAttention(torch.nn.Module):
    """Visual Attention guided by audio input, from AVE/DMRN Paper
    """
    def __init__(self, input=128, output=128, linear_in=25088):
        super(AudioGuidedAttention, self).__init__()
        self.video_linear = torch.nn.Linear(linear_in, output)
        self.audio_linear = torch.nn.Linear(input, output)
        self.fc3 = torch.nn.Linear(output, output)
    def forward(self, video, audio):
        video = F.relu(self.video_linear(video))
        audio = F.relu(self.audio_linear(audio))
        return F.softmax(self.fc3(torch.tanh(video + audio)), dim=0)

class MLP(nn.Module):
    def __init__(self, input, output):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, output)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

def dam(global_audio, global_video, audio, video, segments=10):
    """Dual-Attention Matching Module A2V and V2A
    """
    # Global Video to Audio
    pred_audio = torch.zeros([segments])
    for i in range(segments):
        pred_audio[i] = torch.dot(global_video, audio[i])
    # Global Audio to Video
    pred_video = torch.zeros([segments])
    for i in range(segments):
        pred_video[i] = torch.dot(global_audio, video[i])
    # combine and return
    return torch.div(torch.add(torch.sigmoid(pred_audio), torch.sigmoid(pred_video)), 2)

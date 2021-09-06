import torch, torch.nn as nn
import torch.nn.functional as F

class AdjustVideo(torch.nn.Module):
    def __init__(self, embed_dim=256):
        super(AdjustVideo, self).__init__()
        self.linear = torch.nn.Linear(512, embed_dim)
    def forward(self, x):
        return self.linear(x)

class AdjustAudio(torch.nn.Module):
    def __init__(self, embed_dim=256):
        super(AdjustAudio, self).__init__()
        self.linear = torch.nn.Linear(128, embed_dim)
    def forward(self, x):
        return self.linear(x)

class SelfAttention(torch.nn.Module):
    """Self-Attention Module
    """
    def __init__(self, embed_dim, heads=1, video=False):
        super(SelfAttention, self).__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim, heads)
        self.video = video
        #self.video_layer = torch.nn.Linear(512, 256)
        #self.audio_layer = torch.nn.Linear(128, 256)
    def forward(self, x):
        if self.video:
            #x = self.video_layer(x)
            return self.attention(x, x, x)
        else:
            #x = self.audio_layer(x)
            return self.attention(x, x, x)

class AudioGuidedAttention(torch.nn.Module):
    """Visual Attention guided by audio input, from AVE/DMRN Paper
    """
    def __init__(self, linear_in=512):
        super(AudioGuidedAttention, self).__init__()
        self.video_linear = torch.nn.Linear(linear_in, 512)
        self.audio_linear = torch.nn.Linear(128, 512)

    def forward(self, video, audio):
        original_video = video
        video = F.relu(self.video_linear(video))
        audio = F.relu(self.audio_linear(audio))
        return torch.multiply(original_video, F.softmax(torch.tanh(video + audio), dim=-1))

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
    """Dual-Attention Matching Module
    """
    # Global Video to Audio
    pred_audio = torch.zeros([segments])
    for i in range(segments):
        pred_audio[i] = torch.sigmoid(torch.dot(global_video, audio[i]))
    # Global Audio to Video
    pred_video = torch.zeros([segments])
    for i in range(segments):
        pred_video[i] = torch.sigmoid(torch.dot(global_audio, video[i]))
    # combine and return
    return torch.div(torch.add(pred_audio, pred_video), 2)

def remove_background(tensor, start, end):
    return tensor[start:end, :]

def zero_pad_background(tensor, start, end, temporal_labels):
    for i in range(tensor.size(0)):
        if torch.sum(temporal_labels[i]) < 10:
            tensor[i, 0:start[i], :] = torch.zeros(tensor[i, 0:start[i], :].size())
            tensor[i, end[i]::, :] = torch.zeros(tensor[i, end[i]::, :].size())
    return tensor
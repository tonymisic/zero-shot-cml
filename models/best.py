import torch, torch.nn as nn
import torch.nn.functional as F

class AudioSelfAttention(nn.Module):
    """Audio Self-Attention Model
    """
    def __init__(self, embed_dim, heads=1):
        super(AudioSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, heads)
        self.linear = nn.Linear(embed_dim, 256)
    def forward(self, x):
        x, _ = self.attention(x, x, x)
        x = x.permute([1,0,2]).flatten(0,1)
        return self.linear(x)
    
class VideoSelfAttention(nn.Module):
    """Video Self-Attention Model
    """
    def __init__(self, embed_dim, heads=1):
        super(VideoSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, heads)
        self.linear = nn.Linear(embed_dim, 256)
    def forward(self, x):
        x, _ = self.attention(x, x, x)
        x = x.permute([1,0,2]).flatten(0,1)
        return self.linear(x)
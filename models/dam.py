import torch, torch.nn as nn
import torch.nn.functional as F

class SelfAttention(torch.nn.Module):
    """Self-Attention Module
    """
    def __init__(self, embed_dim, heads=1, video=False):
        super(SelfAttention, self).__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim, heads)
        self.video = video
        self.video_layer = torch.nn.Linear(embed_dim, 128)
    def forward(self, x):
        if self.video:
            return self.video_layer(self.attention(x, x, x))
        else:
            return self.attention(x, x, x)

class AudioGuidedAttention(torch.nn.Module):
    """Visual Attention guided by audio input, from AVE/DMRN Paper
    """
    def __init__(self, linear_in=512):
        super(AudioGuidedAttention, self).__init__()
        self.video_linear = torch.nn.Linear(linear_in, 512)
        self.audio_linear = torch.nn.Linear(128, 512)
        self.fc3 = torch.nn.Linear(512, 128)

    def forward(self, video, audio):
        original_video = video
        video = F.relu(self.video_linear(video))
        audio = F.relu(self.audio_linear(audio))
        return torch.multiply(original_video, F.softmax(torch.tanh(video + audio), dim=-1))

class AudioGuidedAttentionFromPaper(torch.nn.Module):
    """Visual Attention guided by audio input, from AVE/DMRN Paper
    """
    def __init__(self, linear_in=512):
        super(AudioGuidedAttentionFromPaper, self).__init__()
        self.relu = nn.ReLU()
        self.affine_audio = nn.Linear(128, linear_in)
        self.affine_video = nn.Linear(512, linear_in)
        self.affine_v = nn.Linear(linear_in, 49, bias=False)
        self.affine_g = nn.Linear(linear_in, 49, bias=False)
        self.affine_h = nn.Linear(49, 1, bias=False)

    def forward(self, video, audio):
        v_t = video.view(video.size(0) * video.size(1), -1, 512)
        V = v_t

        # Audio-guided visual attention
        v_t = self.relu(self.affine_video(v_t))
        a_t = audio.view(-1, audio.size(-1))
        a_t = self.relu(self.affine_audio(a_t))
        content_v = self.affine_v(v_t) \
                    + self.affine_g(a_t).unsqueeze(2)
        z_t = self.affine_h((torch.tanh(content_v))).squeeze(2)
        alpha_t = F.softmax(z_t, dim=-1).view(z_t.size(0), -1, z_t.size(1)) # attention map
        c_t = torch.bmm(alpha_t, V).view(-1, 512)
        video_t = c_t.view(video.size(0), -1, 512) # attended visual features
        return video_t

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
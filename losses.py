import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    """My contrastive loss for event start prediction
    """
    def __init__(self, temp=0.07, weight=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temp = temp
        self.weight = weight

    def forward(self, video, audio, mask):
        assert video.size() == audio.size() and video.size(0) == mask.size(0)
        batch_size = video.size(0)
        batch_loss_a, batch_loss_v = torch.Tensor([0.0]), torch.Tensor([0.0])
        for i in range(batch_size): # i'th sample in batch
            idx = torch.argmax(mask[i]) # index of event start
            pos_sim = torch.exp(F.cosine_similarity(video[i, idx], audio[i, idx], dim=0) / self.temp)
            v2a_neg_sim, a2v_neg_sim = torch.Tensor([0.0]), torch.Tensor([0.0])
            # Audio Start to Video
            for j in range(video.size(1)): 
                if j != idx:
                    a2v_neg_sim = torch.add(a2v_neg_sim, torch.exp(F.cosine_similarity(video[i, j], audio[i, idx], dim=0) / self.temp))
            batch_loss_a = torch.add(batch_loss_a, -torch.log(pos_sim / a2v_neg_sim))
            # Video Start to Audio
            for j in range(video.size(1)): 
                if j != idx:
                    v2a_neg_sim = torch.add(v2a_neg_sim, torch.exp(F.cosine_similarity(audio[i, j], video[i, idx], dim=0) / self.temp))
            batch_loss_v = torch.add(batch_loss_v, -torch.log(pos_sim / v2a_neg_sim))
        #evenly weighted sum
        return torch.div(torch.add(torch.mul(batch_loss_a, self.weight), torch.mul(batch_loss_v, self.weight)), batch_size) 
import torch

def localize(target_audio, target_video, query_audio, query_video, labels, device):
    if torch.sum(labels) == 10:
        return 1, 1
    # Global Video to Audio
    sims_to_audio = torch.zeros([10]).to(device)
    for j in range(10):
        sims_to_audio[j] = torch.dot(query_video, target_audio[j])
    v2a = max_contiguos_sum(sims_to_audio, labels, device)
    # Global Audio to Video
    sims_to_video = torch.zeros([10]).to(device)
    for j in range(10):
        sims_to_video[j] = torch.dot(query_audio, target_video[j])
    a2v = max_contiguos_sum(sims_to_video, labels, device)
    return v2a, a2v

def max_contiguos_sum(similarities, labels, device):
    length, sums = int(torch.sum(labels)), []
    window = torch.arange(0, length)
    for i in range(10 - length + 1):
        sums.append(torch.sum(similarities[i + window.type(torch.LongTensor)]))
    start = sums.index(max(sums))
    prediction = torch.zeros([10]).to(device)
    prediction[start:start + length] = 1
    if torch.equal(prediction, labels):
        return 1
    else:
        return 0

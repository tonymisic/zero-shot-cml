import torch

def localize(target_audio, target_video, query_audio, query_video, labels, device):
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

def reconstruction_localize(labels, vae_video, vae_audio, device, video, audio):
    length = int(torch.sum(labels))
    sims_to_video, sims_to_audio = torch.zeros([10 - length + 1]).to(device), torch.zeros([10 - length + 1]).to(device)
    for i in range(10 - length + 1):
        with torch.no_grad():
            reconstructed_audio, mu1, _ = vae_video(video[labels == 1, :], vae_audio) # query
            reconstructed_video, mu2, _ = vae_audio(audio[i + length - 1, :], vae_video) # target
            loss1, loss2 = torch.dist(mu1, mu2, 2), torch.dist(reconstructed_audio, reconstructed_video, 2)
        sims_to_audio[i] = loss1.item() * 0.5 + loss2.item() * 0.5
        with torch.no_grad():
            reconstructed_audio, mu1, _ = vae_video(video[i + length - 1, :], vae_audio) # query
            reconstructed_video, mu2, _ = vae_audio(audio[labels == 1, :], vae_video) # target
            loss1, loss2 = torch.dist(mu1, mu2, 2), torch.dist(reconstructed_audio, reconstructed_video, 2)
        sims_to_video[i] = loss1.item() * 0.5 + loss2.item() * 0.5
    v2a, a2v = min_contiguos_sum(sims_to_audio, labels, device), min_contiguos_sum(sims_to_video, labels, device)
    return v2a, a2v

def min_contiguos_sum(similarities, labels, device):
    length = int(torch.sum(labels))
    start = torch.argmin(similarities)
    prediction = torch.zeros([10]).to(device)
    prediction[start:start + length] = 1
    if torch.equal(prediction, labels):
        return 1
    else:
        return 0
import torch
import torch.nn.functional as F

def max_similarity(query, target, event_start, sim='cosine'):
    sims = []
    for i in range(target.size(0)):
        if sim == 'cosine':
           sims.append(F.cosine_similarity(query, target[i], dim=0))
    if int(event_start) == sims.index(max(sims)):
        return 1
    else:
        return 0

def localize(target_audio, target_video, query_audio, query_video, labels, device):
    # Global Video to Audio
    sims_to_audio = torch.zeros([10]).to(device)
    for j in range(10):
        sims_to_audio[j] = torch.dot(query_video, target_audio[j])
    v2a, start_v2a, end_v2a = max_contiguos_sum(sims_to_audio, labels, device)
    # Global Audio to Video
    sims_to_video = torch.zeros([10]).to(device)
    for j in range(10):
        sims_to_video[j] = torch.dot(query_audio, target_video[j])
    a2v, start_a2v, end_a2v = max_contiguos_sum(sims_to_video, labels, device)
    return v2a, a2v, start_v2a, end_v2a, start_a2v, end_a2v

def max_contiguos_sum(similarities, labels, device):
    length, sums = int(torch.sum(labels)), []
    window = torch.arange(0, length)
    for i in range(10 - length + 1):
        sums.append(torch.sum(similarities[i + window.type(torch.LongTensor)]))
    start = sums.index(max(sums))
    prediction = torch.zeros([10]).to(device)
    prediction[start:start + length] = 1
    if torch.equal(prediction, labels):
        return 1, start, start + length 
    else:
        return 0, 9, 1

def test_cmm_a2v(video_feature, audio_feature, vae_video, vae_audio):
	with torch.no_grad():
		reconstructed_audio, mu1, _ = vae_video(video_feature, vae_audio)
		reconstructed_video, mu2, _ = vae_audio(audio_feature, vae_video)
		loss1 = torch.dist(mu1,mu2, 2)
		loss2 = torch.dist(reconstructed_audio, reconstructed_video)
	return loss1.item() * 0.5 + loss2.item() * 0.5

def test_cmm_v2a(video_feature, audio_feature, vae_video, vae_audio):
	with torch.no_grad():
		reconstructed_audio, mu1, _ = vae_video(video_feature, vae_audio)
		reconstructed_video, mu2, _ = vae_audio(audio_feature, vae_video)
		loss1 = torch.dist(mu1,mu2, 2)
		loss2 = torch.dist(reconstructed_audio, reconstructed_video, 2)
	return  loss1.item() * 0.5 + loss2.item() * 0.5 

def VAE_CML(video, audio, temporal_label, vae_audio, vae_video, device):
    v2a, a2v = 0, 0
    l = int(torch.sum(temporal_label))
    start_idx = torch.argmax(temporal_label)
    seg = torch.zeros([l]).type(torch.LongTensor)
    for i in range(l):
        seg[i] = int(start_idx + i)
    # given audio clip
    score = torch.zeros([10 - l + 1])
    for nn_count in range(10 - l + 1):
        s = 0
        for i in range(l):
            s += test_cmm_a2v(video[nn_count + i:nn_count + i + 1, :], audio[seg[i:i + 1], :], vae_video, vae_audio)
        score[nn_count] = s
    min_id = int(torch.argmin(score))
    pred_aud = torch.zeros([10])
    pred_aud[min_id : min_id + int(l)] = 1
    if torch.equal(pred_aud.to(device), temporal_label):
        v2a = 1
    # given video clip
    score = torch.zeros([10 - l + 1])
    for nn_count in range(10 - l + 1):
        s = 0
        for i in range(l):
            s += test_cmm_v2a(video[seg[i:i + 1], :], audio[nn_count + i:nn_count + i + 1, :], vae_video, vae_audio)
        score[nn_count] = s
    min_id = int(torch.argmin(score))
    pred_vid = torch.zeros([10])
    pred_vid[min_id : min_id + int(l)] = 1
    if torch.equal(pred_vid.to(device), temporal_label):
        a2v = 1
    return v2a, a2v
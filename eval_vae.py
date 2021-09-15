from torch.autograd import Variable
import numpy as np
import torch.utils.data
import h5py
import wandb
import torch.nn as nn
import torch
import json
from models.vae import *
from utils import GeneralizedZeroShot
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
'''
Main training script
'''
wandb.init(project="VAE Baseline",
    config={
        "task": "Standard CML",
        "learning_rate": 0.00001,
        "dataset": "AVE",
        "device": "GTX1080",
        "epochs": 999,
        "starting_epoch": 0,
        "batch_size": 21,
        "eval_classes": [0, 1, 2, 3, 4],
        "testSplit": 0.8
    }
)
# splittings
settings = json.load(open('settings.json'))
rootdir = 'AVE_Dataset/'
ZSL = True
gzs = GeneralizedZeroShot('AVE_Dataset/', precomputed=True)
gzs.split_precomputed()
#data loader
with h5py.File(rootdir + settings['precomputed']['folder'] + settings['precomputed']['temporal'], 'r') as hf:
    closs_labels = hf['avadataset'][:]
with h5py.File(rootdir + settings['precomputed']['folder'] + settings['precomputed']['video'], 'r') as hf:
    video_features = hf['avadataset'][:]
with h5py.File(rootdir + settings['precomputed']['folder'] + settings['precomputed']['audio'], 'r') as hf:
    audio_features = hf['avadataset'][:]
if not ZSL:
    with h5py.File(rootdir + settings['precomputed']['folder'] + 'train_order.h5', 'r') as hf:
        train_l = hf['order'][:]
    with h5py.File(rootdir + settings['precomputed']['folder'] + 'val_order.h5', 'r') as hf:
        val_l = hf['order'][:]
    with h5py.File(rootdir + settings['precomputed']['folder'] + 'test_order.h5', 'r') as hf:
        test_l = hf['order'][:]
else:
    with h5py.File(rootdir + settings['precomputed']['folder'] + settings['precomputed']['zsl_train'], 'r') as hf:
        train_l = hf['dataset'][:]
    with h5py.File(rootdir + settings['precomputed']['folder'] + settings['precomputed']['zsl_val'], 'r') as hf:
        val_l = hf['dataset'][:]
    with h5py.File(rootdir + settings['precomputed']['folder'] + settings['precomputed']['zsl_test'], 'r') as hf:
        test_l = hf['dataset'][:]
with h5py.File(rootdir + settings['precomputed']['folder'] + settings['precomputed']['spatial'], 'r') as hf:
	labels = hf['avadataset'][:]

closs_labels = np.array(closs_labels)  # 4143 * 10
audio_features = np.array(audio_features)  # 4143 * 10 * 128
video_features = np.mean(np.array(video_features), axis=(2, 3))  # 4143 * 10 * 512
closs_labels = closs_labels.astype("float32")
audio_features = audio_features.astype("float32")
video_features = video_features.astype("float32")

x_audio_train = np.zeros((len(train_l)*10, 128))
print(len(train_l), len(val_l), len(test_l))
x_video_train = np.zeros((len(train_l)*10, 512))
x_audio_val = np.zeros((len(val_l)*10, 128))
x_video_val = np.zeros((len(val_l)*10, 512))
x_audio_test = np.zeros((len(test_l)*10, 128))
x_video_test = np.zeros((len(test_l)*10, 512))
y_train = np.zeros((len(train_l)*10))
y_val = np.zeros((len(val_l)*10))
y_test = np.zeros((len(test_l)*10))

for i in range(len(train_l)):
    id = train_l[i]
    for j in range(10):
        x_audio_train[10*i + j, :] = audio_features[id, j, :]
        x_video_train[10*i + j, :] = video_features[id, j, :]
        y_train[10*i + j] = closs_labels[id, j]

for i in range(len(val_l)):
    id = val_l[i]
    for j in range(10):
        x_audio_val[10 * i + j, :] = audio_features[id, j, :]
        x_video_val[10 * i + j, :] = video_features[id, j, :]
        y_val[10 * i + j] = closs_labels[id, j]

for i in range(len(test_l)):
    id = test_l[i]
    for j in range(10):
        x_audio_test[10 * i + j, :] = audio_features[id, j, :]
        x_video_test[10 * i + j, :] = video_features[id, j, :]
        y_test[10 * i + j] = closs_labels[id, j]


def norm(x):
	m, n = x.shape
	# print(m,n)
	for i in range(m):
		x[i, :] /= np.max(np.abs(x[i, :]))
	return x


def avgpooling(x):
	m = nn.AvgPool2d(7)
	return m(x)


print("data loading finished!")
################ VAE model #################


def calculate_loss(x, reconstructed_x, mu, logvar):

	loss_MSE = nn.MSELoss()
	mse_loss = loss_MSE(x, reconstructed_x)
	# kl_loss = -0.5 * torch.sum((1 + logvar - mu.pow(2) - logvar.exp()), dim = 1)
	kl_loss = -0.5 * torch.sum((1 + logvar - mu.pow(2) - logvar.exp()))
	return mse_loss + kl_loss


def euclidean_dis(x, reconstructed_x):
	dis = torch.dist(x, reconstructed_x, 2)
	return dis


def compute_accuracy(predictions, labels):
    c = 0
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            c += 1
    return c / len(predictions)


def compute_precision(predictions, labels):
    c = 0
    for i in range(len(predictions)):
        if predictions[i] == 1 and predictions[i] == labels[i]:
            c += 1
    return c


def test_cmm_a2v(video_feature, audio_feature):
	vae_audio.eval()
	vae_video.eval()
	video_feature = video_feature.float()
	audio_feature = audio_feature.float()
	with torch.no_grad():
		reconstructed_audio, mu1, logvar1 = vae_video(video_feature, vae_audio)
		reconstructed_video, mu2, logvar2 = vae_audio(audio_feature, vae_video)
		z1 = vae_video.reparameterize(mu1, logvar1)
		z2 = vae_audio.reparameterize(mu2, logvar2)
		loss1 = euclidean_dis(mu1, mu2)
		loss2 = euclidean_dis(reconstructed_audio, reconstructed_video)
	return loss1.item()*0.5 + loss2.item()*0.5


def test_cmm_v2a(video_feature, audio_feature):
	vae_audio.eval()
	vae_video.eval()
	video_feature = video_feature.float()
	audio_feature = audio_feature.float()
	with torch.no_grad():
		reconstructed_audio, mu1, logvar1 = vae_video(video_feature, vae_audio)
		reconstructed_video, mu2, logvar2 = vae_audio(audio_feature, vae_video)
		z1 = vae_video.reparameterize(mu1, logvar1)
		z2 = vae_audio.reparameterize(mu2, logvar2)
		loss1 = euclidean_dis(mu1, mu2)
		loss2 = euclidean_dis(reconstructed_audio, reconstructed_video)
	return loss1.item()*0.5 + loss2.item()*0.5


latent_dim = 100
training_size = len(train_l)
testing_size = len(test_l)
val_size = len(val_l)
audio_encode = audio_encoder(latent_dim)
audio_decode = general_decoder(latent_dim)
video_encode = visual_encoder(latent_dim)
video_decode = general_decoder(latent_dim)
vae_audio = VAE(latent_dim, audio_encode, audio_decode)
vae_video = VAE(latent_dim, video_encode, video_decode)
vae_audio.cuda()
vae_video.cuda()
vae_audio.load_state_dict(torch.load('savefiles/vae/msvae_a.pkl'))
vae_video.load_state_dict(torch.load('savefiles/vae/msvae_v.pkl'))
print("Model load completed!")
x_video_train = norm(x_video_train)
x_audio_train = norm(x_audio_train)
x_video_val = norm(x_video_val)
x_audio_val = norm(x_audio_val)
x_video_test = norm(x_video_test)
x_audio_test = norm(x_audio_test)
print("Testing begins!")
N = y_test.shape[0]
s = 0
e = s + 10
count_num = 0
audio_count = 0
video_count = 0
video_acc = 0
audio_acc = 0
pos_len = 0
xx = 0
for video_id in range(int(N / 10)):
    s = video_id * 10
    e = s + 10
    x_test = y_test[s: e]
    if np.sum(x_test) == 10:
        continue
    count_num += 1
    xx += np.sum(x_test)
    nb = np.argwhere(x_test == 1)
    seg = np.zeros(len(nb)).astype('int8')
    for i in range(len(nb)):
        seg[i] = nb[i][0]
    l = len(seg)
    x_test_video_feature = x_video_test[s:e, :]
    x_test_video_feature = torch.from_numpy(x_test_video_feature)
    x_test_video_feature = x_test_video_feature.float()
    x_test_video_feature = x_test_video_feature.cuda()
    x_test_video_feature = Variable(x_test_video_feature)
    x_test_audio_feautre = x_audio_test[s:e, :]
    x_test_audio_feautre = torch.from_numpy(x_test_audio_feautre)
    x_test_audio_feautre = x_test_audio_feautre.float()
    x_test_audio_feautre = x_test_audio_feautre.cuda()
    x_test_audio_feautre = Variable(x_test_audio_feautre)
    score = []
    for nn_count in range(10 - l + 1):
        s = 0
        for i in range(l):
            s += test_cmm_a2v(x_test_video_feature[nn_count + i:nn_count +
                                i + 1, :], x_test_audio_feautre[seg[i:i + 1], :])
        score.append(s)
    score = np.array(score).astype('float32')
    score_1 = score
    min_id = int(np.argmin(score))
    pred_vid = np.zeros(10)
    for i in range(min_id, min_id + int(l)):
        pred_vid[i] = 1
    if np.argmin(score) == seg[0]:
        audio_count += 1
    video_acc += compute_precision(x_test, pred_vid)
    ind = np.where(x_test - pred_vid == 0)[0]
    acc_v = len(ind)
    score = []
    for nn_count in range(10 - l + 1):
        s = 0
        for i in range(l):
            s += test_cmm_v2a(x_test_video_feature[seg[i:i + 1], :],
                                x_test_audio_feautre[nn_count + i:nn_count + i + 1, :])
        score.append(s)
    score = np.array(score).astype('float32')
    score_2 = score
    if np.argmin(score) == seg[0]:
        video_count += 1
    pred_aid = np.zeros(10)
    min_id = int(np.argmin(score))
    for i in range(min_id, min_id + int(l)):
        pred_aid[i] = 1
    audio_acc += compute_precision(x_test, pred_aid)
    pos_len += len(seg)
    ind = np.where(x_test - pred_aid == 0)[0]
    acc_a = len(ind)
wandb.log({"Testing V2A": video_count / count_num, "Testing A2V": audio_count / count_num})
print("Validation begins!")
N = y_val.shape[0]
s = 0
e = s + 10
count_num = 0
audio_count = 0
video_count = 0
video_acc = 0
audio_acc = 0
pos_len = 0
xx = 0
for video_id in range(int(N / 10)):
    s = video_id * 10
    e = s + 10
    x_val = y_val[s: e]
    if np.sum(x_val) == 10:
        continue
    count_num += 1
    xx += np.sum(x_val)
    nb = np.argwhere(x_val == 1)
    seg = np.zeros(len(nb)).astype('int8')
    for i in range(len(nb)):
        seg[i] = nb[i][0]
    l = len(seg)
    x_val_video_feature = x_video_val[s:e, :]
    x_val_video_feature = torch.from_numpy(x_val_video_feature)
    x_val_video_feature = x_val_video_feature.float()
    x_val_video_feature = x_val_video_feature.cuda()
    x_val_video_feature = Variable(x_val_video_feature)
    x_val_audio_feautre = x_audio_val[s:e, :]
    x_val_audio_feautre = torch.from_numpy(x_val_audio_feautre)
    x_val_audio_feautre = x_val_audio_feautre.float()
    x_val_audio_feautre = x_val_audio_feautre.cuda()
    x_val_audio_feautre = Variable(x_val_audio_feautre)
    score = []
    for nn_count in range(10 - l + 1):
        s = 0
        for i in range(l):
            s += test_cmm_a2v(x_val_video_feature[nn_count + i:nn_count +
                                i + 1, :], x_val_audio_feautre[seg[i:i + 1], :])
        score.append(s)
    score = np.array(score).astype('float32')
    score_1 = score
    min_id = int(np.argmin(score))
    pred_vid = np.zeros(10)
    for i in range(min_id, min_id + int(l)):
        pred_vid[i] = 1
    if np.argmin(score) == seg[0]:
        audio_count += 1
    video_acc += compute_precision(x_val, pred_vid)
    ind = np.where(x_val - pred_vid == 0)[0]
    acc_v = len(ind)
    score = []
    for nn_count in range(10 - l + 1):
        s = 0
        for i in range(l):
            s += test_cmm_v2a(x_val_video_feature[seg[i:i + 1], :],
                                x_val_audio_feautre[nn_count + i:nn_count + i + 1, :])
        score.append(s)
    score = np.array(score).astype('float32')
    score_2 = score
    if np.argmin(score) == seg[0]:
        video_count += 1
    pred_aid = np.zeros(10)
    min_id = int(np.argmin(score))
    for i in range(min_id, min_id + int(l)):
        pred_aid[i] = 1
    audio_acc += compute_precision(x_val, pred_aid)
    pos_len += len(seg)
    ind = np.where(x_val - pred_aid == 0)[0]
    acc_a = len(ind)
wandb.log({"Zero-Shot V2A": video_count / count_num, "Zero-Shot A2V": audio_count / count_num})
from models.best import AudioSelfAttention, VideoSelfAttention
from models.dam import remove_background
from dataloader import AVE
from utils import GeneralizedZeroShot, SupConLoss, create_positives, create_mask
from metrics import max_similarity
from torch.utils.data.dataloader import DataLoader
import torch, wandb, torch.optim as optim 
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="2"
'''
Main training script
'''
wandb.init(project="Best Baseline",
    config={
        "task": "Standard CML",
        "learning_rate": 0.001,
        "dataset": "AVE",
        "device": "GTX1080",
        "epochs": 300,
        "starting_epoch" : 0,
        "batch_size": 42,
        "threshold": 0.5,
        "lambda": 0.5,
        "eval_classes": "Manifold 2",
        "testSplit": 0.8
    }
)
# splitting
gzs = GeneralizedZeroShot('AVE_Dataset/', precomputed=True)
gzs.split_precomputed()
# devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# loaders
train_data = AVE('AVE_Dataset/', 'train', 'settings.json', precomputed=True, ZSL=True)
train_loader = DataLoader(train_data, wandb.config['batch_size'], shuffle=True, num_workers=3, pin_memory=True)
test_data = AVE('AVE_Dataset/', 'test', 'settings.json', precomputed=True, ZSL=True)
test_loader = DataLoader(test_data, 1, shuffle=True, num_workers=1, pin_memory=True)
val_data = AVE('AVE_Dataset/', 'val', 'settings.json', precomputed=True, ZSL=True)
val_loader = DataLoader(val_data, 1, shuffle=True, num_workers=1, pin_memory=True)
# models
audio_attention_model = AudioSelfAttention(128)
video_attention_model = VideoSelfAttention(512)
audio_attention_model.to(device), video_attention_model.to(device)
# lossess
criterion = torch.nn.CosineEmbeddingLoss()
criterion.to(device)
optimizer_video = optim.SGD(video_attention_model.parameters(), lr=wandb.config['learning_rate'], momentum=0.9)
optimizer_audio = optim.SGD(audio_attention_model.parameters(), lr=wandb.config['learning_rate'], momentum=0.9)
# epochs
epoch = wandb.config['starting_epoch']
running_loss, run_temp, run_spat, iteration = 0.0, 0.0, 0.0, 0
while epoch <= wandb.config['epochs']:
    ### --------------- TRAIN --------------- ###
    running_spatial, running_temporalV2A, running_temporalA2V, running_temporal, running_classification, batch = 0.0, 0.0, 0.0, 0.0, 0.0, 1
    for video, audio, temporal_labels, spatial_labels, class_names, back_start, back_end in train_loader:
        batch_size = video.size(0)
        optimizer_audio.zero_grad(), optimizer_video.zero_grad()
        # initialize features
        video, local_audio = torch.mean(video, dim=(2,3)).to(device), audio.to(device)
        spatial_labels, temporal_labels = spatial_labels.type(torch.LongTensor).to(device), temporal_labels.to(device)
        # self-attention on audio and video features
        audio_attention = audio_attention_model(local_audio.permute([1,0,2]))
        video_attention = video_attention_model(video.permute([1,0,2]))
        V2A_accuracies, A2V_accuracies = torch.zeros([batch_size]), torch.zeros([batch_size])
        for i in range(batch_size):
            # background removal for both audio and video
            query_video = remove_background(video[i], int(back_start[i]), int(back_end[i]))
            query_audio = remove_background(local_audio[i], int(back_start[i]), int(back_end[i]))
            # use self-attention to sample audio and video
            query_video_attention = video_attention_model(query_video.unsqueeze(0).permute([1,0,2]))
            query_audio_attention = audio_attention_model(query_audio.unsqueeze(0).permute([1,0,2]))
            V2A_accuracies[i] = max_similarity(query_video_attention[0], audio_attention[i * 10:i * 10 + 10], back_start[i])
            A2V_accuracies[i] = max_similarity(query_audio_attention[0], video_attention[i * 10:i * 10 + 10], back_start[i])
        A2V_ACCURACY, V2A_ACCURACY = float(torch.sum(A2V_accuracies)) / batch_size, float(torch.sum(V2A_accuracies)) / batch_size
        ######### ------------------------------ LOSS AND RECORD ---------------------------------------- ###########
        labels = create_positives(back_start).flatten().to(device)
        LOSS = criterion(audio_attention, video_attention, labels)
        LOSS.backward()
        optimizer_audio.step(), optimizer_video.step()
        # apply and record iteration
        batch += 1
        iteration += 1
        running_loss += float(LOSS)
        running_temporalV2A += V2A_ACCURACY
        running_temporalA2V += A2V_ACCURACY
        wandb.log({"Loss": running_loss / iteration})
    print("Samples: " + str(batch * wandb.config['batch_size']))
    wandb.log({"Training A2V": running_temporalA2V / batch})
    wandb.log({"Training V2A": running_temporalV2A / batch})
    torch.save(audio_attention_model.state_dict(), 'savefiles/audio/epoch' + str(epoch) + '.pth')
    torch.save(video_attention_model.state_dict(), 'savefiles/video/epoch' + str(epoch) + '.pth')
    print("Saved Models for Epoch:" + str(epoch))
    ### --------------- TEST --------------- ###
    running_spatial, running_temporalV2A, running_temporalA2V, running_temporal, running_classification, batch = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    for video, audio, temporal_labels, spatial_labels, class_names, back_start, back_end in test_loader:
        if torch.sum(temporal_labels[0]) < 10:
            batch_size = video.size(0)
            optimizer_audio.zero_grad(), optimizer_video.zero_grad()
            # initialize features
            video, local_audio = torch.mean(video, dim=(2,3)).to(device), audio.to(device)
            spatial_labels, temporal_labels = spatial_labels.type(torch.LongTensor).to(device), temporal_labels.to(device)
            # self-attention on audio and video features
            audio_attention = audio_attention_model(local_audio.permute([1,0,2]))
            video_attention = video_attention_model(video.permute([1,0,2]))
            V2A_accuracies, A2V_accuracies = torch.zeros([batch_size]), torch.zeros([batch_size])
            for i in range(batch_size):
                # background removal for both audio and video
                query_video = remove_background(video[i], int(back_start[i]), int(back_end[i]))
                query_audio = remove_background(local_audio[i], int(back_start[i]), int(back_end[i]))
                # use self-attention to sample audio and video
                query_video_attention = video_attention_model(query_video.unsqueeze(0).permute([1,0,2]))
                query_audio_attention = audio_attention_model(query_audio.unsqueeze(0).permute([1,0,2]))
                V2A_accuracies[i] = max_similarity(query_video_attention[0], audio_attention[i * 10:i * 10 + 10], back_start[i])
                A2V_accuracies[i] = max_similarity(query_audio_attention[0], video_attention[i * 10:i * 10 + 10], back_start[i])
            A2V_ACCURACY, V2A_ACCURACY = float(torch.sum(A2V_accuracies)) / batch_size, float(torch.sum(V2A_accuracies)) / batch_size
            batch += 1
            running_temporalV2A += V2A_ACCURACY
            running_temporalA2V += A2V_ACCURACY
    print("Samples: " + str(batch))
    wandb.log({"Testing A2V": running_temporalA2V / batch})
    wandb.log({"Testing V2A": running_temporalV2A / batch})
    ### --------------- TEST --------------- ###
    running_spatial, running_temporalV2A, running_temporalA2V, running_temporal, running_classification, batch = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    for video, audio, temporal_labels, spatial_labels, class_names, back_start, back_end in val_loader:
        if torch.sum(temporal_labels[0]) < 10:
            batch_size = video.size(0)
            optimizer_audio.zero_grad(), optimizer_video.zero_grad()
            # initialize features
            video, local_audio = torch.mean(video, dim=(2,3)).to(device), audio.to(device)
            spatial_labels, temporal_labels = spatial_labels.type(torch.LongTensor).to(device), temporal_labels.to(device)
            # self-attention on audio and video features
            audio_attention = audio_attention_model(local_audio.permute([1,0,2]))
            video_attention = video_attention_model(video.permute([1,0,2]))
            V2A_accuracies, A2V_accuracies = torch.zeros([batch_size]), torch.zeros([batch_size])
            for i in range(batch_size):
                # background removal for both audio and video
                query_video = remove_background(video[i], int(back_start[i]), int(back_end[i]))
                query_audio = remove_background(local_audio[i], int(back_start[i]), int(back_end[i]))
                # use self-attention to sample audio and video
                query_video_attention = video_attention_model(query_video.unsqueeze(0).permute([1,0,2]))
                query_audio_attention = audio_attention_model(query_audio.unsqueeze(0).permute([1,0,2]))
                V2A_accuracies[i] = max_similarity(query_video_attention[0], audio_attention[i * 10:i * 10 + 10], back_start[i])
                A2V_accuracies[i] = max_similarity(query_audio_attention[0], video_attention[i * 10:i * 10 + 10], back_start[i])
            A2V_ACCURACY, V2A_ACCURACY = float(torch.sum(A2V_accuracies)) / batch_size, float(torch.sum(V2A_accuracies)) / batch_size
            batch += 1
            running_temporalV2A += V2A_ACCURACY
            running_temporalA2V += A2V_ACCURACY
    print("Samples: " + str(batch))
    wandb.log({"Zero-Shot A2V": running_temporalA2V / batch})
    wandb.log({"Zero-Shot V2A": running_temporalV2A / batch})
    epoch += 1
    print("Epoch: " + str(epoch - 1) + " finished!")
print("Finished, finished.")
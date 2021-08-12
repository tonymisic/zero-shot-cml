from models.dam import SelfAttention, AudioGuidedAttention, MLP, dam
from dataloader import AVE
from utils import GeneralizedZeroShot
from metrics import temporal_accuracy
from torch.utils.data.dataloader import DataLoader
import torch, wandb, torch.optim as optim 
'''
Main training script
'''
wandb.init(project="DAM Baseline",
    config={
        "task": "Zero Shot CML",
        "learning_rate": 0.001,
        "dataset": "AVE",
        "device": "GTX1080",
        "epochs": 999,
        "starting_epoch" : 0,
        "batch_size": 21,
        "threshold": 0.5,
        "eval_classes": [0,1,2,3,4],
        "testSplit": 0.8
    }
)
# splitting
gzs = GeneralizedZeroShot('AVE_Dataset/', precomputed=True)
gzs.split_precomputed()
# devices
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# loaders
train_data = AVE('AVE_Dataset/', 'train', 'settings.json', precomputed=True)
test_data = AVE('AVE_Dataset/', 'test', 'settings.json', precomputed=True)
val_data = AVE('AVE_Dataset/', 'val', 'settings.json', precomputed=True)
train_loader = DataLoader(train_data, wandb.config['batch_size'], shuffle=True, num_workers=3, pin_memory=True)
test_loader = DataLoader(test_data, 1, shuffle=True, num_workers=1, pin_memory=True)
val_loader = DataLoader(val_data, 1, shuffle=True, num_workers=1, pin_memory=True)
# models
audio_attention_model = SelfAttention(128)
video_attention_model = SelfAttention(128)
guided_model = AudioGuidedAttention()
classifier = MLP(256, 29)
# losses
criterion = torch.nn.CrossEntropyLoss()
event_criterion = torch.nn.BCELoss()
criterion.to(device), event_criterion.to(device)
optimizer_classifier = optim.SGD(classifier.parameters(), lr=wandb.config['learning_rate'], momentum=0.9)
optimizer_guided = optim.SGD(guided_model.parameters(), lr=wandb.config['learning_rate'], momentum=0.9)
optimizer_video = optim.SGD(video_attention_model.parameters(), lr=wandb.config['learning_rate'], momentum=0.9)
optimizer_audio = optim.SGD(audio_attention_model.parameters(), lr=wandb.config['learning_rate'], momentum=0.9)
classifier.to(device), video_attention_model.to(device), audio_attention_model.to(device), guided_model.to(device)
# training, seen testing, unseen validation
epoch = wandb.config['starting_epoch']
while epoch <= wandb.config['epochs']:
    ### --------------- TRAIN --------------- ###
    running_loss, run_temp, run_spat, batch = 0.0, 0.0, 0.0, 0
    running_spatial, running_temporal = 0.0, 0.0
    for video, audio, temporal_labels, spatial_labels, class_names, back_start, back_end in train_loader:
        optimizer_classifier.step(), optimizer_audio.step(), optimizer_video.step(), optimizer_guided.step()
        video, audio = torch.flatten(video, start_dim=2).to(device), audio.to(device)
        spatial_labels, temporal_labels = spatial_labels.to(device), temporal_labels.to(device)
        final_seg_videos = torch.zeros([wandb.config['batch_size'], 10, 128]).to(device)
        for i in range(wandb.config['batch_size']):
            for j in range(10):
                final_seg_videos[i,j,:] = torch.dot(guided_model(video[i,j,:].clone(), audio[i,j,:].clone()), audio[i,j,:].clone())
        for i in range(final_seg_videos.clone().size(0)):
            if torch.sum(temporal_labels[i]) < 10:
                audio[i, 0:back_start[i], :] = torch.zeros(audio[i, 0:back_start[i], :].size())
                final_seg_videos[i, 0:back_start[i], :] = torch.zeros(final_seg_videos[i, 0:back_start[i], :].clone().size())
                audio[i, back_end[i]::, :] = torch.zeros(audio[i, back_end[i]::, :].size())
                final_seg_videos[i, back_end[i]::, :] = torch.zeros(final_seg_videos[i, back_end[i]::, :].clone().size())
        a_atten, a_weight = audio_attention_model(audio.permute([1,0,2]))
        v_atten, v_weight = video_attention_model(final_seg_videos.permute([1,0,2]))
        video_global = torch.mean(v_atten, dim=0)
        audio_global = torch.mean(a_atten, dim=0)
        event_relevance = classifier(torch.cat([video_global, audio_global], dim=1))
        predictions, ground_truth = torch.max(event_relevance, 1)[1], torch.max(spatial_labels, 2)[1]
        # use temporal labels as predictions of spatial labels.
        SEL_ACCURACY = float(torch.sum(predictions == ground_truth)) / wandb.config['batch_size']
        temporal_preds = torch.zeros([temporal_labels.size(0), temporal_labels.size(1)]).to(device)
        for i in range(temporal_preds.size(0)):
            temporal_preds[i] = dam(audio_global[i], video_global[i], a_atten[:, i, :], v_atten[:, i, :]).round()
        LOSS_SPATIAL = 0.5 * criterion(predictions, ground_truth) 
        LOSS_TEMPORAL = 0.5 * (event_criterion(temporal_preds, temporal_labels) / 10)
        LOSS = LOSS_SPATIAL + LOSS_TEMPORAL
        LOSS.backward()
        CML_ACCURACY = temporal_accuracy(temporal_preds, temporal_labels, wandb.config['threshold'], device)
        batch += 1
        run_temp += LOSS_TEMPORAL / batch
        run_spat += LOSS_SPATIAL / batch
        running_loss += LOSS / batch
        running_spatial += SEL_ACCURACY / batch
        running_temporal += CML_ACCURACY / batch
        wandb.log({"batch": batch})
        wandb.log({"Combined Loss": running_loss})
        wandb.log({"Temporal Loss": run_temp})
        wandb.log({"Spatial Loss": run_spat})
    wandb.log({"Training CML": running_temporal})
    wandb.log({"Training SEL": running_spatial})
    torch.save(audio_attention_model.state_dict(), 'savefiles/audio/epoch' + str(epoch) + '.pth')
    torch.save(video_attention_model.state_dict(), 'savefiles/video/epoch' + str(epoch) + '.pth')
    torch.save(classifier.state_dict(), 'savefiles/classifier/epoch' + str(epoch) + '.pth')
    torch.save(guided_model.state_dict(), 'savefiles/guided/epoch' + str(epoch) + '.pth')
    print("Saved Models for Epoch:" + str(epoch))
    ### --------------- TEST --------------- ###
    running_loss, batch = 0.0, 0
    running_spatial, running_temporal = 0.0, 0.0
    for video, audio, temporal_labels, spatial_labels, class_names, back_start, back_end in test_loader:
        optimizer_audio.zero_grad(), optimizer_classifier.zero_grad(), optimizer_video.zero_grad(), optimizer_guided.zero_grad()
        video, audio = torch.flatten(video, start_dim=2).to(device), audio.to(device)
        spatial_labels, temporal_labels = spatial_labels.to(device), temporal_labels.to(device)
        final_seg_videos = torch.zeros([1, 10, 128]).to(device)
        for i in range(1):
            for j in range(10):
                final_seg_videos[i,j,:] = torch.dot(guided_model(video[i,j,:].clone(), audio[i,j,:].clone()), audio[i,j,:].clone())
        a_atten, a_weight = audio_attention_model(audio.permute([1,0,2]))
        v_atten, v_weight = video_attention_model(final_seg_videos.permute([1,0,2]))
        video_global = torch.mean(v_atten, dim=0)
        audio_global = torch.mean(a_atten, dim=0)
        event_relevance = classifier(torch.cat([video_global, audio_global], dim=1))
        predictions, ground_truth = torch.max(event_relevance, 1)[1], torch.max(spatial_labels, 2)[1]
        SEL_ACCURACY = torch.sum(predictions == ground_truth)
        temporal_preds = torch.zeros([temporal_labels.size(0), temporal_labels.size(1)]).to(device)
        for i in range(temporal_preds.size(0)):
            temporal_preds[i] = dam(audio_global[i], video_global[i], a_atten[:, i, :], v_atten[:, i, :]).round()
        CML_ACCURACY = temporal_accuracy(temporal_preds, temporal_labels, wandb.config['threshold'], device)
        batch += 1
        running_spatial += SEL_ACCURACY / batch
        running_temporal += CML_ACCURACY / batch
    wandb.log({"Testing CML": running_temporal})
    wandb.log({"Testing SEL": running_spatial})
    ### --------------- VAL --------------- ###
    running_loss, batch = 0.0, 0
    running_spatial, running_temporal = 0.0, 0.0
    for video, audio, temporal_labels, spatial_labels, class_names, back_start, back_end in val_loader:
        optimizer_audio.zero_grad(), optimizer_classifier.zero_grad(), optimizer_video.zero_grad(), optimizer_guided.zero_grad()
        video, audio = torch.flatten(video, start_dim=2).to(device), audio.to(device)
        spatial_labels, temporal_labels = spatial_labels.to(device), temporal_labels.to(device)
        final_seg_videos = torch.zeros([1, 10, 128]).to(device)
        for i in range(1):
            for j in range(10):
                final_seg_videos[i,j,:] = torch.dot(guided_model(video[i,j,:].clone(), audio[i,j,:].clone()), audio[i,j,:].clone())
        a_atten, a_weight = audio_attention_model(audio.permute([1,0,2]))
        v_atten, v_weight = video_attention_model(final_seg_videos.permute([1,0,2]))
        video_global = torch.mean(v_atten, dim=0)
        audio_global = torch.mean(a_atten, dim=0)
        event_relevance = classifier(torch.cat([video_global, audio_global], dim=1))
        predictions, ground_truth = torch.max(event_relevance, 1)[1], torch.max(spatial_labels, 2)[1]
        SEL_ACCURACY = torch.sum(predictions == ground_truth)
        temporal_preds = torch.zeros([temporal_labels.size(0), temporal_labels.size(1)]).to(device)
        for i in range(temporal_preds.size(0)):
            temporal_preds[i] = dam(audio_global[i], video_global[i], a_atten[:, i, :], v_atten[:, i, :]).round()
        CML_ACCURACY = temporal_accuracy(temporal_preds, temporal_labels, wandb.config['threshold'], device)
        batch += 1
        running_spatial += SEL_ACCURACY / batch
        running_temporal += CML_ACCURACY / batch
    wandb.log({"ZeroShot CML": running_temporal})
    wandb.log({"ZeroShot SEL": running_spatial})
    wandb.log({"epoch": epoch + 1})
    epoch += 1
    print("Epoch: " + str(epoch - 1) + " finished!")
print("Finished, finished.")
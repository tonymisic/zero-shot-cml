from models.dam import SelfAttention, AudioGuidedAttention, MLP, dam, remove_background, zero_pad_background
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
        "lamdba": 0.5,
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
train_data = AVE('AVE_Dataset/', 'train', 'settings.json', precomputed=True, ZSL=True)
test_data = AVE('AVE_Dataset/', 'test', 'settings.json', precomputed=True, ZSL=True)
train_loader = DataLoader(train_data, wandb.config['batch_size'], shuffle=True, num_workers=3, pin_memory=True)
test_loader = DataLoader(test_data, 1, shuffle=True, num_workers=1, pin_memory=True)
# models
audio_attention_model = SelfAttention(128)
video_attention_model = SelfAttention(128)
guided_model = AudioGuidedAttention()
classifier = MLP(256, 29)
# losses
criterion = torch.nn.CrossEntropyLoss()
event_criterion = torch.nn.BCEWithLogitsLoss()
criterion.to(device), event_criterion.to(device)
optimizer_classifier = optim.SGD(classifier.parameters(), lr=wandb.config['learning_rate'], momentum=0.9)
optimizer_guided = optim.SGD(guided_model.parameters(), lr=wandb.config['learning_rate'], momentum=0.9)
optimizer_video = optim.SGD(video_attention_model.parameters(), lr=wandb.config['learning_rate'], momentum=0.9)
optimizer_audio = optim.SGD(audio_attention_model.parameters(), lr=wandb.config['learning_rate'], momentum=0.9)
classifier.to(device), video_attention_model.to(device), audio_attention_model.to(device), guided_model.to(device)
# training, seen testing, unseen validation
epoch = wandb.config['starting_epoch']
running_loss, run_temp, run_spat, iteration = 0.0, 0.0, 0.0, 0
while epoch <= wandb.config['epochs']:
    ### --------------- TRAIN --------------- ###
    running_spatial, running_temporal, batch = 0.0, 0.0, 0
    for video, audio, temporal_labels, spatial_labels, class_names, back_start, back_end in train_loader:
        optimizer_classifier.zero_grad(), optimizer_audio.zero_grad(), optimizer_video.zero_grad(), optimizer_guided.zero_grad()
        video, audio = torch.flatten(video, start_dim=2).to(device), audio.to(device)
        spatial_labels, temporal_labels = spatial_labels.type(torch.LongTensor).to(device), temporal_labels.to(device)
        # audio-guided attention
        guided_video = torch.zeros([video.size(0), 10, 128]).to(device)
        for i in range(video.size(0)):
            for j in range(10):
                guided_video[i,j,:] = torch.dot(guided_model(video[i,j,:].clone(), audio[i,j,:].clone()), audio[i,j,:].clone())
        # background removal for both audio and video
        trimmed_video = zero_pad_background(guided_video, back_start, back_end, temporal_labels)
        trimmed_audio = zero_pad_background(audio, back_start, back_end, temporal_labels)
        # self-attetion on local audio and video features
        a_atten, a_weight = audio_attention_model(trimmed_audio.permute([1,0,2]))
        v_atten, v_weight = video_attention_model(trimmed_video.permute([1,0,2]))
        # global pooling of local features
        video_global = torch.mean(v_atten, dim=0)
        audio_global = torch.mean(a_atten, dim=0)
        # classifier prediction
        classifier_output = classifier(torch.cat([video_global, audio_global], dim=1))
        # temporal predictions
        temporal_preds = torch.zeros(temporal_labels.size()).to(device)
        for i in range(temporal_preds.size(0)):
            temporal_preds[i] = dam(audio_global[i], video_global[i], a_atten[:, i, :], v_atten[:, i, :]).round() # 0.5 threshold by rounding to nearest int
        ground_truth = torch.argmax(spatial_labels, 2).flatten()
        # expand spatial prediction based on temporal prediction
        expanded_spatial = torch.zeros([classifier_output.size(0), 10, classifier_output.size(2)]).to(device)
        for i in range(expanded_spatial.size(0)):
            for j in range(10):
                if temporal_preds[i][j] == 1:
                    expanded_spatial[i][j] = classifier_output[i][j]
                elif temporal_preds[i][j] == 0:
                   expanded_spatial[i][j] = 28
        # Cross Entropy Loss per segment on spatial predictions
        LOSS_SPATIAL = criterion(expanded_spatial.flatten(start_dim=0, end_dim=1), )
        # Binary Cross Entropy per segment on temporal predictions
        LOSS_TEMPORAL = event_criterion(temporal_preds, temporal_labels)
        # combine and apply losses as per equation 8
        LOSS = wandb.config['lambda'] * LOSS_SPATIAL + wandb.config['lambda']* LOSS_TEMPORAL
        LOSS.backward()
        optimizer_classifier.step(), optimizer_audio.step(), optimizer_video.step(), optimizer_guided.step()
        # calculate accuracies
        SEL_ACCURACY = float(torch.sum(final_spatial == ground_truth)) / (10 * final_seg_videos.size(0))
        CML_ACCURACY = temporal_accuracy(temporal_preds, temporal_labels, wandb.config['threshold'], device)
        # apply and record iteration
        batch += 1
        iteration += 1
        run_temp += LOSS_TEMPORAL
        run_spat += LOSS_SPATIAL 
        running_loss += LOSS
        running_spatial += SEL_ACCURACY 
        running_temporal += CML_ACCURACY 
        wandb.log({"batch": batch})
        wandb.log({"Combined Loss": running_loss / iteration})
        wandb.log({"Temporal Loss": run_temp / iteration})
        wandb.log({"Spatial Loss": run_spat / iteration})
    wandb.log({"Training CML": running_temporal / batch})
    wandb.log({"Training SEL": running_spatial / batch})
    # save current epoch
    torch.save(audio_attention_model.state_dict(), 'savefiles/audio/epoch' + str(epoch) + '.pth')
    torch.save(video_attention_model.state_dict(), 'savefiles/video/epoch' + str(epoch) + '.pth')
    torch.save(classifier.state_dict(), 'savefiles/classifier/epoch' + str(epoch) + '.pth')
    torch.save(guided_model.state_dict(), 'savefiles/guided/epoch' + str(epoch) + '.pth')
    print("Saved Models for Epoch:" + str(epoch))
    ### --------------- TEST --------------- ###
    running_spatial, running_temporal, batch = 0.0, 0.0, 0
    for video, audio, temporal_labels, spatial_labels, class_names, back_start, back_end in test_loader:
        optimizer_classifier.step(), optimizer_audio.step(), optimizer_video.step(), optimizer_guided.step()
        video, audio = torch.flatten(video, start_dim=2).to(device), audio.to(device)
        spatial_labels, temporal_labels = spatial_labels.type(torch.LongTensor).to(device), temporal_labels.to(device)
        final_seg_videos = torch.zeros([audio.size(0), 10, 128]).to(device)
        for i in range(audio.size(0)):
            for j in range(10):
                final_seg_videos[i,j,:] = torch.dot(guided_model(video[i,j,:].clone(), audio[i,j,:].clone()), audio[i,j,:].clone())
        a_atten, a_weight = audio_attention_model(audio.permute([1,0,2]))
        v_atten, v_weight = video_attention_model(final_seg_videos.permute([1,0,2]))
        video_global = torch.mean(v_atten, dim=0)
        audio_global = torch.mean(a_atten, dim=0)
        event_relevance = classifier(torch.cat([video_global, audio_global], dim=1))
        predictions, ground_truth = torch.max(event_relevance, 1)[1], torch.max(spatial_labels, 2)[1]
        final_spatial = torch.zeros([final_seg_videos.size(0), 10]).to(device)
        for i in range(final_seg_videos.size(0)):
            final_spatial[i][temporal_labels[i] == 1] = predictions[i].float()
            final_spatial[i][temporal_labels[i] == 0] = 28
        SEL_ACCURACY = float(torch.sum(final_spatial == ground_truth)) / (10 * final_seg_videos.size(0))
        temporal_preds = torch.zeros([temporal_labels.size(0), temporal_labels.size(1)]).to(device)
        for i in range(temporal_preds.size(0)): # needs a V2A and A2V 
            temporal_preds[i] = dam(audio_global[i], video_global[i], a_atten[:, i, :], v_atten[:, i, :]).round()
        CML_ACCURACY = temporal_accuracy(temporal_preds, temporal_labels, wandb.config['threshold'], device)
        batch += 1
        running_spatial += SEL_ACCURACY
        running_temporal += CML_ACCURACY
    wandb.log({"Testing CML": running_temporal / batch})
    wandb.log({"Testing SEL": running_spatial / batch})
    epoch += 1
    print("Epoch: " + str(epoch - 1) + " finished!")
print("Finished, finished.")
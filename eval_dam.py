from models.dam import SelfAttention, AudioGuidedAttention, MLP, dam, remove_background
from dataloader import AVE
from utils import GeneralizedZeroShot
from metrics import localize
from torch.utils.data.dataloader import DataLoader
import torch, wandb
'''
Main training script
'''
wandb.init(project="DAM Baseline",
    config={
        "task": "Standard CML",
        "learning_rate": 0.001,
        "dataset": "AVE",
        "device": "GTX1080",
        "epochs": 999,
        "starting_epoch" : 0,
        "batch_size": 21,
        "threshold": 0.5,
        "lambda": 0.5,
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
val_data = AVE('AVE_Dataset/', 'test', 'settings.json', precomputed=True, ZSL=False)
val_loader = DataLoader(val_data, 1, shuffle=True, num_workers=1, pin_memory=True)
# models
testing_epoch = 100
audio_attention_model = SelfAttention(128)
audio_attention_model.load_state_dict(torch.load('savefiles/audio/epoch' + str(testing_epoch) + '.pth'))
video_attention_model = SelfAttention(128)
video_attention_model.load_state_dict(torch.load('savefiles/video/epoch' + str(testing_epoch) + '.pth'))
guided_model = AudioGuidedAttention(linear_in=512)
guided_model.load_state_dict(torch.load('savefiles/guided/epoch' + str(testing_epoch) + '.pth'))
classifier = MLP(256, 29)
classifier.load_state_dict(torch.load('savefiles/classifier/epoch' + str(testing_epoch) + '.pth'))
classifier.to(device), video_attention_model.to(device), audio_attention_model.to(device), guided_model.to(device)
classifier.eval(), video_attention_model.eval(), audio_attention_model.eval(), guided_model.eval()
### --------------- TEST --------------- ###
running_spatial, running_temporalV2A, running_temporalA2V, running_temporal, running_classification, batch = 0.0, 0.0, 0.0, 0.0, 0.0, 0
correct_ones, count_ones = 0, 0
correct_zeros, count_zeros = 0, 0
for video, audio, temporal_labels, spatial_labels, class_names, back_start, back_end in val_loader:
    if torch.sum(temporal_labels[0]) < 10:
        batch_size = video.size(0)
        video, local_audio = torch.mean(video, dim=(2,3)).to(device), audio.to(device)
        spatial_labels, temporal_labels = spatial_labels.type(torch.LongTensor).to(device), temporal_labels.to(device)
        # audio-guided attention
        guided_video = torch.zeros([batch_size, 10, 128]).to(device)
        for i in range(batch_size):
            for j in range(10):
                guided_video[i,j,:] = guided_model(video[i,j,:].clone(), local_audio[i,j,:].clone())
        ######### ------------------------------ CROSS MODAL LOCALIZATION ---------------------------------------- ###########
        video_global_cml = torch.zeros([batch_size, 128]).to(device)
        audio_global_cml = torch.zeros([batch_size, 128]).to(device)
        for i in range(batch_size):
            # background removal for both audio and video
            query_video = remove_background(guided_video[i], int(back_start[i]), int(back_end[i]))
            query_audio = remove_background(local_audio[i], int(back_start[i]), int(back_end[i]))
            # use self-attention to sample audio and video
            v_attention_cml, _ = video_attention_model(query_video.unsqueeze(0).permute([1,0,2]))
            a_attention_cml, _ = audio_attention_model(query_audio.unsqueeze(0).permute([1,0,2]))
            # mean to get global feature
            video_global_cml[i] = torch.mean(v_attention_cml, dim=0)
            audio_global_cml[i] = torch.mean(a_attention_cml, dim=0)
        # temporal predictions
        temporal_preds = torch.zeros(temporal_labels.size()).to(device)
        V2A_accuracies, A2V_accuracies = torch.zeros([batch_size]), torch.zeros([batch_size])
        for i in range(batch_size):
            V2A_accuracies[i], A2V_accuracies[i] = localize(local_audio[i], guided_video[i], audio_global_cml[i], video_global_cml[i], temporal_labels[i], device)
            temporal_preds[i] = dam(audio_global_cml[i], video_global_cml[i], local_audio[i], guided_video[i])
        A2V_ACCURACY, V2A_ACCURACY = float(torch.sum(A2V_accuracies)) / batch_size, float(torch.sum(V2A_accuracies)) / batch_size
        TEMPORAL_ACC = float(torch.sum(temporal_preds.round() == temporal_labels)) / (batch_size * 10)
        for i in range(temporal_preds.size(1)):
            if temporal_labels[0][i] == 1:
                if temporal_preds[0][i].round() == 1:
                    correct_ones += 1
                count_ones += 1
            else:
                if temporal_preds[0][i].round() == 0:
                    correct_zeros += 1
                count_zeros += 1
        ######### ------------------------------ SUPERVISED EVENT LOCALIZATION ---------------------------------------- ###########
        # self-attetion on local audio and video features
        a_attention, _ = audio_attention_model(local_audio.permute([1,0,2]))
        v_attention, _ = video_attention_model(guided_video.permute([1,0,2]))
        # global pooling of local features
        video_global_sel = torch.mean(v_attention, dim=0)
        audio_global_sel = torch.mean(a_attention, dim=0)
        # classifier prediction
        classifier_output = classifier(torch.cat([video_global_sel, audio_global_sel], dim=1))
        # calculate category accuracy and SEL
        gt_classes = torch.min(torch.argmax(spatial_labels, 2), 1).values.to(device)
        ground_truth = torch.argmax(spatial_labels, 2).to(device)
        spatial_pred = torch.argmax(classifier_output, 1).to(device)
        expanded_spatial = torch.zeros(temporal_preds.size()).to(device)
        for i in range(batch_size):
            for j in range(expanded_spatial.size(1)):
                if temporal_preds[i][j] >= wandb.config['threshold']:
                    expanded_spatial[i][j] = spatial_pred[i]
                else:
                    expanded_spatial[i][j] = 28
        SEL_ACCURACY = float(torch.sum(expanded_spatial == ground_truth)) / (batch_size * 10)
        CLASSIFICATION_ACCURACY = float(torch.sum(spatial_pred == gt_classes)) / batch_size
        # apply and record iteration
        batch += 1
        running_spatial += SEL_ACCURACY
        running_classification += CLASSIFICATION_ACCURACY 
        running_temporalV2A += V2A_ACCURACY
        running_temporalA2V += A2V_ACCURACY
        running_temporal += TEMPORAL_ACC
        wandb.log({"batch": batch})

wandb.log({"1's Accuracy": float(correct_ones) / count_ones,
           "0's Accuracy": float(correct_zeros) / count_zeros })
wandb.log({"Testing A2V": running_temporalA2V / batch})
wandb.log({"Testing V2A": running_temporalV2A / batch})
wandb.log({"Testing SEL": running_spatial / batch})
wandb.log({"Testing Classification": running_classification / batch})
wandb.log({"Testing Temporal": running_temporal / batch})
print("Finished, finished.")
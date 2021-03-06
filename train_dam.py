from models.dam import SelfAttention, AudioGuidedAttention, MLP, AdjustAudio, AdjustVideo, dam, remove_background
from dataloader import AVE
from utils import GeneralizedZeroShot
from metrics import localize
from visualization import histogram
from torch.utils.data.dataloader import DataLoader
import torch, wandb, torch.optim as optim
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="2"
'''
Main training script
'''
wandb.init(project="DAM Baseline",
    config={
        "task": "ZSL",
        "learning_rate": 0.01,
        "dataset": "AVE",
        "device": "GTX1080",
        "epochs": 200,
        "starting_epoch" : 0,
        "batch_size": 21,
        "threshold": 0.5,
        "lambda": 0.5,
        "eval_classes": "Manifold Test",
        "testSplit": 0.8
    }
)
# splitting
gzs = GeneralizedZeroShot('AVE_Dataset/', precomputed=True)
gzs.split_precomputed()
# devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# loaders
ZSL = True
train_data = AVE('AVE_Dataset/', 'train', 'settings.json', precomputed=True, ZSL=ZSL)
train_loader = DataLoader(train_data, wandb.config['batch_size'], shuffle=True, num_workers=1, pin_memory=True)
test_data = AVE('AVE_Dataset/', 'test', 'settings.json', precomputed=True, ZSL=ZSL)
test_loader = DataLoader(test_data, 1, shuffle=True, num_workers=1, pin_memory=True)
val_data = AVE('AVE_Dataset/', 'val', 'settings.json', precomputed=True, ZSL=ZSL)
val_loader = DataLoader(val_data, 1, shuffle=True, num_workers=1, pin_memory=True)
# models
audio_attention_model = SelfAttention(256)
video_attention_model = SelfAttention(256, video=True)
guided_model = AudioGuidedAttention(linear_in=512)
classifier = MLP(512, 29)
video_linear, audio_linear = AdjustVideo(), AdjustAudio()
classifier.to(device), audio_attention_model.to(device), video_attention_model.to(device), guided_model.to(device)
video_linear.to(device), audio_linear.to(device)
# losses
criterion = torch.nn.CrossEntropyLoss()
event_criterion = torch.nn.BCELoss()
criterion.to(device), event_criterion.to(device)
optimizer_classifier = optim.SGD(classifier.parameters(), lr=wandb.config['learning_rate'], momentum=0.9)
optimizer_guided = optim.SGD(guided_model.parameters(), lr=wandb.config['learning_rate'], momentum=0.9)
optimizer_video = optim.SGD(video_attention_model.parameters(), lr=wandb.config['learning_rate'], momentum=0.9)
optimizer_audio = optim.SGD(audio_attention_model.parameters(), lr=wandb.config['learning_rate'], momentum=0.9)
opti_lin_v = optim.SGD(video_linear.parameters(), lr=wandb.config['learning_rate'], momentum=0.9)
opti_lin_a = optim.SGD(audio_linear.parameters(), lr=wandb.config['learning_rate'], momentum=0.9)
# epochs
epoch = wandb.config['starting_epoch']
running_loss, run_temp, run_spat, iteration = 0.0, 0.0, 0.0, 0
while epoch <= wandb.config['epochs']:
    ### --------------- TRAIN --------------- ###
    starts_gt, ends_gt = None, None
    start_v2a, end_v2a, start_a2v, end_a2v = None, None, None, None
    running_spatial, running_temporalV2A, running_temporalA2V, running_temporal, running_classification, batch = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    for video, audio, temporal_labels, spatial_labels, class_names, back_start, back_end in train_loader:
        # create list of event starts and ends
        if batch == 0:
            starts_gt = back_start.detach()
            ends_gt = back_end.detach()
        else:
            starts_gt = torch.cat((starts_gt, back_start.detach()), dim=0)
            ends_gt = torch.cat((ends_gt, back_end.detach()), dim=0)
        # initialization
        batch_size = video.size(0)
        optimizer_classifier.zero_grad(), optimizer_audio.zero_grad(), optimizer_video.zero_grad(), optimizer_guided.zero_grad()
        opti_lin_v.zero_grad(), opti_lin_a.zero_grad()
        video, local_audio = torch.mean(video, dim=(2,3)).to(device), audio.to(device)
        spatial_labels, temporal_labels = spatial_labels.type(torch.LongTensor).to(device), temporal_labels.to(device)
        # audio-guided attention
        guided_video = torch.zeros([batch_size, 10, 512]).to(device)
        for i in range(batch_size):
            for j in range(10):
                guided_video[i,j,:] = guided_model(video[i,j,:].clone(), local_audio[i,j,:].clone())
        local_audio = audio_linear(local_audio).to(device)
        guided_video = video_linear(guided_video).to(device)
        # self-attention on local audio and video features
        a_attention, _ = audio_attention_model(local_audio.permute([1,0,2]))
        v_attention, _ = video_attention_model(guided_video.permute([1,0,2]))
        # global pooling of local features
        video_global_sel = torch.mean(v_attention, dim=0)
        audio_global_sel = torch.mean(a_attention, dim=0)
        ######### ------------------------------ CROSS MODAL LOCALIZATION ---------------------------------------- ###########
        video_global_cml = torch.zeros([batch_size, 256]).to(device)
        audio_global_cml = torch.zeros([batch_size, 256]).to(device)
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
        V2A_starts, V2A_ends, A2V_starts, A2V_ends = torch.zeros([batch_size]), torch.zeros([batch_size]), torch.zeros([batch_size]), torch.zeros([batch_size])
        for i in range(batch_size):
            V2A_accuracies[i], A2V_accuracies[i], V2A_starts[i], V2A_ends[i], A2V_starts[i], A2V_ends[i] = localize(local_audio[i], guided_video[i], audio_global_cml[i], video_global_cml[i], temporal_labels[i], device)
            temporal_preds[i] = dam(audio_global_cml[i], video_global_cml[i], local_audio[i], guided_video[i])
        TEMPORAL_ACC = float(torch.sum(temporal_preds.round() == temporal_labels)) / (batch_size * 10)
        A2V_ACCURACY, V2A_ACCURACY = float(torch.sum(A2V_accuracies)) / batch_size, float(torch.sum(V2A_accuracies)) / batch_size
        # Binary Cross Entropy per segment on temporal predictions
        LOSS_TEMPORAL = event_criterion(temporal_preds.flatten(), temporal_labels.flatten())
        # create list of event starts and ends
        if batch == 0:
            start_v2a, end_v2a, start_a2v, end_a2v = V2A_starts.detach(), V2A_ends.detach(), A2V_starts.detach(), A2V_ends.detach()
        else:
            start_v2a = torch.cat((start_v2a, V2A_starts.detach()), dim=0)
            end_v2a = torch.cat((end_v2a, V2A_ends.detach()), dim=0)
            start_a2v = torch.cat((start_a2v, A2V_starts.detach()), dim=0)
            end_a2v = torch.cat((end_a2v, A2V_ends.detach()), dim=0)
        ######### ------------------------------ SUPERVISED EVENT LOCALIZATION ---------------------------------------- ###########
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
        # Cross Entropy Loss per segment on spatial predictions
        LOSS_SPATIAL = criterion(classifier_output, gt_classes)
        ######### ------------------------------ LOSS AND RECORD ---------------------------------------- ###########
        # combine and apply losses as per equation 8
        LOSS = wandb.config['lambda'] * LOSS_SPATIAL + wandb.config['lambda'] * LOSS_TEMPORAL
        LOSS.backward()
        optimizer_classifier.step(), optimizer_audio.step(), optimizer_video.step(), optimizer_guided.step()
        opti_lin_a.step(), opti_lin_v.step()
        # apply and record iteration
        batch += 1
        iteration += 1
        run_temp += float(LOSS_TEMPORAL)
        run_spat += float(LOSS_SPATIAL)
        running_loss += float(LOSS)
        running_spatial += SEL_ACCURACY 
        running_classification += CLASSIFICATION_ACCURACY 
        running_temporalV2A += V2A_ACCURACY
        running_temporalA2V += A2V_ACCURACY
        running_temporal += TEMPORAL_ACC
        wandb.log({"batch": batch})
        wandb.log({"Combined Loss": running_loss / iteration})
        wandb.log({"Temporal Loss": run_temp / iteration})
        wandb.log({"Spatial Loss": run_spat / iteration})
    # visualize epoch predictions
    print("Samples: " + str(batch))
    wandb.log({"Training A2V": running_temporalA2V / batch})
    wandb.log({"Training V2A": running_temporalV2A / batch})
    wandb.log({"Training SEL": running_spatial / batch})
    wandb.log({"Training Temporal": running_temporal / batch})
    wandb.log({"Training Classification": running_classification / batch})
    torch.save(audio_attention_model.state_dict(), 'savefiles/audio/epoch' + str(epoch) + '.pth')
    torch.save(video_attention_model.state_dict(), 'savefiles/video/epoch' + str(epoch) + '.pth')
    torch.save(classifier.state_dict(), 'savefiles/classifier/epoch' + str(epoch) + '.pth')
    torch.save(guided_model.state_dict(), 'savefiles/guided/epoch' + str(epoch) + '.pth')
    histogram(starts_gt.type(torch.IntTensor), start_v2a.type(torch.IntTensor), start_a2v.type(torch.IntTensor), file="histograms/dam/starts/histogram_starts" + str(epoch) + "train.jpg", time='start')
    histogram(ends_gt.type(torch.IntTensor), end_v2a.type(torch.IntTensor), end_a2v.type(torch.IntTensor), file="histograms/dam/ends/histogram_ends" + str(epoch) + "train.jpg", time='end')
    print("Saved Models for Epoch:" + str(epoch))
    ### --------------- TEST --------------- ###
    starts_gt, ends_gt = None, None
    start_v2a, end_v2a, start_a2v, end_a2v = None, None, None, None
    running_spatial, running_temporalV2A, running_temporalA2V, running_temporal, running_classification, batch = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    for video, audio, temporal_labels, spatial_labels, class_names, back_start, back_end in test_loader:
        if torch.sum(temporal_labels[0]) < 10:
            # create list of event starts and ends
            if batch == 0:
                starts_gt = back_start.detach()
                ends_gt = back_end.detach()
            else:
                starts_gt = torch.cat((starts_gt, back_start.detach()), dim=0)
                ends_gt = torch.cat((ends_gt, back_end.detach()), dim=0)
            # initialization
            batch_size = video.size(0)
            optimizer_classifier.zero_grad(), optimizer_audio.zero_grad(), optimizer_video.zero_grad(), optimizer_guided.zero_grad()
            video, local_audio = torch.mean(video, dim=(2,3)).to(device), audio.to(device)
            spatial_labels, temporal_labels = spatial_labels.type(torch.LongTensor).to(device), temporal_labels.to(device)
            # audio-guided attention
            guided_video = torch.zeros([batch_size, 10, 512]).to(device)
            for i in range(batch_size):
                for j in range(10):
                    guided_video[i,j,:] = guided_model(video[i,j,:].clone(), local_audio[i,j,:].clone())
            local_audio = audio_linear(local_audio).to(device)
            guided_video = video_linear(guided_video).to(device)
            # self-attention on local audio and video features
            a_attention, _ = audio_attention_model(local_audio.permute([1,0,2]))
            v_attention, _ = video_attention_model(guided_video.permute([1,0,2]))
            # global pooling of local features
            video_global_sel = torch.mean(v_attention, dim=0)
            audio_global_sel = torch.mean(a_attention, dim=0)
            ######### ------------------------------ CROSS MODAL LOCALIZATION ---------------------------------------- ###########
            video_global_cml = torch.zeros([batch_size, 256]).to(device)
            audio_global_cml = torch.zeros([batch_size, 256]).to(device)
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
            V2A_starts, V2A_ends, A2V_starts, A2V_ends = torch.zeros([batch_size]), torch.zeros([batch_size]), torch.zeros([batch_size]), torch.zeros([batch_size])
            for i in range(batch_size):
                V2A_accuracies[i], A2V_accuracies[i], V2A_starts[i], V2A_ends[i], A2V_starts[i], A2V_ends[i] = localize(local_audio[i], guided_video[i], audio_global_cml[i], video_global_cml[i], temporal_labels[i], device)
                temporal_preds[i] = dam(audio_global_cml[i], video_global_cml[i], local_audio[i], guided_video[i])
            TEMPORAL_ACC = float(torch.sum(temporal_preds.round() == temporal_labels)) / (batch_size * 10)
            A2V_ACCURACY, V2A_ACCURACY = float(torch.sum(A2V_accuracies)) / batch_size, float(torch.sum(V2A_accuracies)) / batch_size
            # create list of event starts and ends
            if batch == 0:
                start_v2a, end_v2a, start_a2v, end_a2v = V2A_starts.detach(), V2A_ends.detach(), A2V_starts.detach(), A2V_ends.detach()
            else:
                start_v2a = torch.cat((start_v2a, V2A_starts.detach()), dim=0)
                end_v2a = torch.cat((end_v2a, V2A_ends.detach()), dim=0)
                start_a2v = torch.cat((start_a2v, A2V_starts.detach()), dim=0)
                end_a2v = torch.cat((end_a2v, A2V_ends.detach()), dim=0)
            ######### ------------------------------ SUPERVISED EVENT LOCALIZATION ---------------------------------------- ###########
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
    print("Samples: " + str(batch))
    histogram(starts_gt.type(torch.IntTensor), start_v2a.type(torch.IntTensor), start_a2v.type(torch.IntTensor), file="histograms/dam/starts/histogram_starts" + str(epoch) + "test.jpg", time='start')
    histogram(ends_gt.type(torch.IntTensor), end_v2a.type(torch.IntTensor), end_a2v.type(torch.IntTensor), file="histograms/dam/ends/histogram_ends" + str(epoch) + "test.jpg", time='end')
    wandb.log({"Testing A2V": running_temporalA2V / batch})
    wandb.log({"Testing V2A": running_temporalV2A / batch})
    wandb.log({"Testing SEL": running_spatial / batch})
    wandb.log({"Testing Classification": running_classification / batch})
    wandb.log({"Testing Temporal": running_temporal / batch})
    ### --------------- VAL --------------- ###
    starts_gt, ends_gt = None, None
    start_v2a, end_v2a, start_a2v, end_a2v = None, None, None, None
    running_spatial, running_temporalV2A, running_temporalA2V, running_temporal, running_classification, batch = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    for video, audio, temporal_labels, spatial_labels, class_names, back_start, back_end in val_loader:
        if torch.sum(temporal_labels[0]) < 10:    
            # init
            batch_size = video.size(0)
            optimizer_classifier.zero_grad(), optimizer_audio.zero_grad(), optimizer_video.zero_grad(), optimizer_guided.zero_grad()
            video, local_audio = torch.mean(video, dim=(2,3)).to(device), audio.to(device)
            spatial_labels, temporal_labels = spatial_labels.type(torch.LongTensor).to(device), temporal_labels.to(device)
            # audio-guided attention
            guided_video = torch.zeros([batch_size, 10, 512]).to(device)
            for i in range(batch_size):
                for j in range(10):
                    guided_video[i,j,:] = guided_model(video[i,j,:].clone(), local_audio[i,j,:].clone())
            local_audio = audio_linear(local_audio).to(device)
            guided_video = video_linear(guided_video).to(device)
            # self-attention on local audio and video features
            a_attention, _ = audio_attention_model(local_audio.permute([1,0,2]))
            v_attention, _ = video_attention_model(guided_video.permute([1,0,2]))
            # global pooling of local features
            video_global_sel = torch.mean(v_attention, dim=0)
            audio_global_sel = torch.mean(a_attention, dim=0)
            ######### ------------------------------ CROSS MODAL LOCALIZATION ---------------------------------------- ###########
            video_global_cml = torch.zeros([batch_size, 256]).to(device)
            audio_global_cml = torch.zeros([batch_size, 256]).to(device)
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
            V2A_starts, V2A_ends, A2V_starts, A2V_ends = torch.zeros([batch_size]), torch.zeros([batch_size]), torch.zeros([batch_size]), torch.zeros([batch_size])
            for i in range(batch_size):
                V2A_accuracies[i], A2V_accuracies[i], V2A_starts[i], V2A_ends[i], A2V_starts[i], A2V_ends[i] = localize(local_audio[i], guided_video[i], audio_global_cml[i], video_global_cml[i], temporal_labels[i], device)
                temporal_preds[i] = dam(audio_global_cml[i], video_global_cml[i], local_audio[i], guided_video[i])
            TEMPORAL_ACC = float(torch.sum(temporal_preds.round() == temporal_labels)) / (batch_size * 10)
            A2V_ACCURACY, V2A_ACCURACY = float(torch.sum(A2V_accuracies)) / batch_size, float(torch.sum(V2A_accuracies)) / batch_size
            ######### ------------------------------ SUPERVISED EVENT LOCALIZATION ---------------------------------------- ###########
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
    print("Samples: " + str(batch))
    wandb.log({"Zero-Shot A2V": running_temporalA2V / batch})
    wandb.log({"Zero-Shot V2A": running_temporalV2A / batch})
    wandb.log({"Zero-Shot SEL": running_spatial / batch})
    wandb.log({"Zero-Shot Classification": running_classification / batch})
    wandb.log({"Zero-Shot Temporal": running_temporal / batch})
    epoch += 1
    print("Epoch: " + str(epoch - 1) + " finished!")
print("Finished, finished.")
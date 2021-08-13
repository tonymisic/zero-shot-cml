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
        final_spatial = torch.zeros([wandb.config['batch_size'], 10]).to(device)
        for i in range(wandb.config['batch_size']):
            final_spatial[i][temporal_labels[i] == 1] = predictions[i].float()
            final_spatial[i][temporal_labels[i] == 0] = 28
        SEL_ACCURACY = torch.sum(final_spatial == spatial_labels)
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
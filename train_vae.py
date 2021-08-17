from models.vae import audio_encoder, visual_encoder, VAE, general_decoder
from dataloader import AVE
from utils import GeneralizedZeroShot
from metrics import reconstruction_localize
from torch.utils.data.dataloader import DataLoader
import torch, wandb, torch.optim as optim 
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
        "starting_epoch" : 0,
        "batch_size": 21,
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
train_loader = DataLoader(train_data, wandb.config['batch_size'], shuffle=True, num_workers=3, pin_memory=True)
test_data = AVE('AVE_Dataset/', 'test', 'settings.json', precomputed=True, ZSL=True)
test_loader = DataLoader(test_data, 1, shuffle=True, num_workers=1, pin_memory=True)
# models
audio_encode = audio_encoder(100)
video_encode = visual_encoder(100)
general_decode = general_decoder(100)
vae_audio = VAE(100, audio_encode, general_decode)
vae_video = VAE(100, video_encode, general_decode)
audio_encode.to(device), video_encode.to(device), general_decode.to(device), vae_audio.to(device), vae_video.to(device)
# losses
loss_MSE = torch.nn.MSELoss()
loss_MSE.to(device)
optimizer_audio = optim.Adam(vae_audio.parameters(), lr = wandb.config['learning_rate'])
optimizer_video = optim.Adam(vae_video.parameters(), lr = wandb.config['learning_rate'])
# losses directly from code
def caluculate_losses(x_visual, x_audio, x_reconstruct, mu, logvar, epoch):
	x_input = torch.cat((x_visual, x_audio), 1)
	mse_loss = loss_MSE(x_input, x_reconstruct)
	kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	_ , mu1, logvar1 = vae_audio(x_audio)
	z1 = vae_audio.reparameterize(mu1, logvar1)
	_, mu2, logvar2 = vae_video(x_visual)
	z2 = vae_video.reparameterize(mu2, logvar2)
	latent_loss = torch.dist(z1, z2, 2)
	if epoch < 10:
		final_loss = mse_loss + kl_loss * 0.1 + latent_loss
	else:
		final_loss = mse_loss + kl_loss * 0.01 + latent_loss
	return final_loss, kl_loss, mse_loss, latent_loss
# training
epoch = wandb.config['starting_epoch']
running_loss, running_kl, running_mse, runnning_latent, iteration = 0.0, 0.0, 0.0, 0.0, 0
BEST_SCORE = 0
while epoch <= wandb.config['epochs']:
    ### --------------- TRAIN --------------- ###
    running_spatial, running_temporalV2A, running_temporalA2V, batch = 0.0, 0.0, 0.0, 0
    for video, audio, temporal_labels, spatial_labels, class_names, back_start, back_end in train_loader:
        batch_size = video.size(0)
        optimizer_audio.zero_grad(), optimizer_video.zero_grad()
        # other work just ignores the 3rd and 4th dims, no clue why
        video = torch.mean(video,dim=(2,3), keepdim=True).squeeze().to(device)
        audio = audio.to(device)
        flattened_video = torch.flatten(video, start_dim=0, end_dim=1).to(device)
        flattened_audio = torch.flatten(audio, start_dim=0, end_dim=1).to(device)
        spatial_labels, temporal_labels = spatial_labels.type(torch.LongTensor).to(device), temporal_labels.to(device)
        # reconstruction
        if epoch == 0:
            x_reconstruct, mu, logvar = vae_video(flattened_video)
        else:
            x_reconstruct, mu, logvar = vae_video(flattened_video, vae_audio)
        x_reconstruct_from_a, mu2, logvar2 = vae_audio(flattened_audio, vae_video)
        V2A_accuracies, A2V_accuracies = torch.zeros([batch_size]), torch.zeros([batch_size])
        for i in range(batch_size):
            V2A_accuracies[i], A2V_accuracies[i] = reconstruction_localize(temporal_labels[i], vae_video, vae_audio, device, video[i], audio[i])
        A2V_ACCURACY, V2A_ACCURACY = float(torch.sum(A2V_accuracies)) / batch_size, float(torch.sum(V2A_accuracies)) / batch_size
        running_temporalV2A += V2A_ACCURACY
        running_temporalA2V += A2V_ACCURACY
        # calculate losses
        loss1, kl_loss1, mse_loss1, latent_loss1 = caluculate_losses(flattened_video, flattened_audio, x_reconstruct, mu, logvar, epoch)
        loss2, kl_loss2, mse_loss2, latent_loss2 = caluculate_losses(flattened_video, flattened_audio, x_reconstruct_from_a, mu2, logvar2, epoch)
        FINAL_LOSS = loss1 + loss2
        kl_loss = kl_loss1 + kl_loss2
        mse_loss = mse_loss1 + mse_loss2
        latent_loss = latent_loss1 + latent_loss2
        FINAL_LOSS.backward()
        optimizer_audio.step(), optimizer_video.step()
        # apply and record iteration
        batch += 1
        iteration += 1
        running_loss += float(FINAL_LOSS / 2)
        running_kl += float(kl_loss / 2)
        running_mse += float(mse_loss / 2)
        runnning_latent += float(latent_loss / 2)
        wandb.log({"batch": batch})
        wandb.log({"Combined Loss": running_loss / iteration})
        wandb.log({"KL Loss": running_kl / iteration})
        wandb.log({"MSE Loss": running_mse / iteration})
        wandb.log({"Latent Loss": runnning_latent / iteration})
    wandb.log({"Training A2V": running_temporalA2V / batch})
    wandb.log({"Training V2A": running_temporalV2A / batch})
    ### --------------- TEST --------------- ###
    running_temporalV2A, running_temporalA2V, batch = 0.0, 0.0, 0
    for video, audio, temporal_labels, spatial_labels, class_names, back_start, back_end in test_loader:
        if torch.sum(temporal_labels[0]) < 10:
            batch_size = video.size(0)
            optimizer_audio.zero_grad(), optimizer_video.zero_grad()
            # other work just ignores the 3rd and 4th dims, no clue why
            video = torch.mean(video,dim=(2,3), keepdim=True).squeeze(2).squeeze(2).to(device)
            audio = audio.to(device)
            spatial_labels, temporal_labels = spatial_labels.type(torch.LongTensor).to(device), temporal_labels.to(device)
            # calculate scores
            V2A_ACCURACY, A2V_ACCURACY = reconstruction_localize(temporal_labels[0], vae_video, vae_audio, device, video[0], audio[0])
            # apply and record iteration
            batch += 1
            running_temporalV2A += V2A_ACCURACY
            running_temporalA2V += A2V_ACCURACY
    AVERAGE = (0.5*(running_temporalA2V / batch) + 0.5*(running_temporalV2A / batch))
    if AVERAGE > BEST_SCORE:
        BEST_SCORE = AVERAGE
        # save current epoch
        torch.save(vae_video.state_dict(), 'savefiles/vae/best_video_epoch.pth')
        torch.save(vae_audio.state_dict(), 'savefiles/vae/best_audio_epoch.pth')
        print("Saved Models for Epoch:" + str(epoch))
    wandb.log({"Testing A2V": running_temporalA2V / batch})
    wandb.log({"Testing V2A": running_temporalV2A / batch})
    epoch += 1
    print("Epoch: " + str(epoch - 1) + " finished!")
print("Finished, finished.")
from models.vae import audio_encoder, visual_encoder, VAE, general_decoder
from dataloader import AVE
from utils import GeneralizedZeroShot
from metrics import VAE_CML
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
val_data = AVE('AVE_Dataset/', 'val', 'settings.json', precomputed=True, ZSL=False)
val_loader = DataLoader(val_data, 1, shuffle=True, num_workers=1, pin_memory=True)
# models
audio_encode = audio_encoder(100)
video_encode = visual_encoder(100)
general_decode = general_decoder(100)
vae_audio = VAE(100, audio_encode, general_decode)
vae_video = VAE(100, video_encode, general_decode)
vae_audio.load_state_dict(torch.load('savefiles/vae/msvae_a.pkl'))
vae_video.load_state_dict(torch.load('savefiles/vae/msvae_v.pkl'))
audio_encode.eval(), video_encode.eval(), general_decode.eval(), vae_audio.eval(), vae_video.eval()
audio_encode.to(device), video_encode.to(device), general_decode.to(device), vae_audio.to(device), vae_video.to(device)
# training
epoch = wandb.config['starting_epoch']
running_loss, running_kl, running_mse, runnning_latent, iteration = 0.0, 0.0, 0.0, 0.0, 0
### --------------- TEST --------------- ###
running_temporalV2A, running_temporalA2V, batch = 0.0, 0.0, 0
for video, audio, temporal_labels, spatial_labels, class_names, back_start, back_end in val_loader:
    if torch.sum(temporal_labels[0]) < 10:
        batch_size = video.size(0)
        # other work just ignores the 3rd and 4th dims, no clue why
        video = torch.mean(video,dim=(2,3)).to(device)
        video[0] /= torch.max(torch.abs(video[0])).to(device)
        audio = audio.to(device)
        audio[0] /= torch.max(torch.abs(audio[0])).to(device)
        spatial_labels, temporal_labels = spatial_labels.type(torch.LongTensor).to(device), temporal_labels.to(device)
        # calculate scores
        V2A_ACCURACY, A2V_ACCURACY = VAE_CML(video[0], audio[0], temporal_labels[0], vae_audio, vae_video, device)
        # apply and record iteration
        batch += 1
        running_temporalV2A += V2A_ACCURACY
        running_temporalA2V += A2V_ACCURACY
wandb.log({"Testing A2V": running_temporalA2V / batch})
wandb.log({"Testing V2A": running_temporalV2A / batch})
print("Finished, finished.")
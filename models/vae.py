''' Code from "Learning Audio Visual Correlations with VAE's" in ICASSP 2021
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import matplotlib.pyplot as plt


class visual_encoder(nn.Module):
	def __init__(self, latent_dim):
		super(visual_encoder,self).__init__()

		self.lin_lays = nn.Sequential(
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
        )
		self.mu = nn.Linear(128, latent_dim)
		self.var = nn.Linear(128, latent_dim)


	def forward(self, x):
		# x shape: [batch_size, input_dim]
		hidden = self.lin_lays(x)
		# hidden shape: [batch_size, hidden_dim]
		# latent parameters
		mean = self.mu(hidden)

		log_var = self.var(hidden)

		return mean, log_var

class audio_encoder(nn.Module):
	def __init__(self, latent_dim):
		super(audio_encoder,self).__init__()

		self.lin_lays = nn.Sequential(
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
        )

		self.mu = nn.Linear(128, latent_dim)
		self.var = nn.Linear(128, latent_dim)


	def forward(self, x):
		# x shape: [batch_size, input_dim]
		hidden = self.lin_lays(x)
		mean = self.mu(hidden)
		log_var = self.var(hidden)

		return mean, log_var

class general_decoder(nn.Module):
	def __init__(self, latent_dim):
		super(general_decoder, self).__init__()

		self.lin_lays = nn.Sequential(
			nn.Linear(latent_dim, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 640),
			nn.ReLU(),
			nn.Linear(640, 640),
			nn.ReLU(),
			nn.Linear(640, 640),
			)
		
	def forward(self, x):
		generated_x = self.lin_lays(x)
		return generated_x

class VAE(nn.Module):
	"""Variational Autoencoder module that allows cross-embedding

	Arguments:
	in_dim(list): Input dimension.
	z_dim(int): Noise dimension
	encoder(nn.Module): The encoder module. Its output dim must be 2*z_dim
	decoder(nn.Module): The decoder module. Its output dim must be in_dim.prod()
	"""
	def __init__(self, z_dim, encoder, decoder):
		super(VAE, self).__init__()
		self.z_dim = z_dim
		self.encoder = encoder
		self.decoder = decoder


	def reparameterize(self, mu, logvar):
		if self.training:
		#### used only for training process ####
			std = torch.exp(logvar / 2)
			eps = Variable(std.data.new(std.size()).normal_())
			z = eps.mul(std).add_(mu)
			return z
		else:
			return mu
		#### otherwise, reparameterize returns to mu ####


	def forward(self, x, vae_decoder=None):
		mu, logvar = self.encoder(x)
		z = self.reparameterize(mu, logvar)

		if not vae_decoder:
			dec = self.decoder
		else:
			dec = vae_decoder.decoder

		return dec(z), mu, logvar

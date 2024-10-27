#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import GradScaler, autocast  # Mixed precision training
import torchvision
import matplotlib.pyplot as plt


# In[10]:


# Hyperparameters
CHANNELS_IMG = 1
FEATURES_DISC = 32  # Reduced features for faster training
FEATURES_GEN = 32
BATCH_SIZE = 64  # Reduced batch size for faster processing per epoch
NOISE_DIM = 100
LEARNING_RATE = 2e-4
EPOCHS = 5


# In[11]:


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[25]:


# Data loading with increased number of workers for faster loading
celeba_root = os.path.expanduser("~/Downloads/celeba/celeb_dataset/images")
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),  # Ensure the image is grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalization for single-channel images
])

dataset = datasets.MNIST(root="celeba_root", train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)



# In[13]:


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_disc):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img, features_disc, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_disc, features_disc * 2, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(features_disc * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_disc * 2, features_disc * 4, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(features_disc * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_disc * 4, 1, kernel_size=4, stride=2, padding=0)  # Output: single value
        )

    def forward(self, x):
        return torch.sigmoid(self.disc(x))  # Apply Sigmoid here


# In[14]:


class Generator(nn.Module):
    def __init__(self, noise_dim, channels_img, features_gen):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, features_gen * 16, kernel_size=4, stride=1, padding=0),  # 4x4
            nn.BatchNorm2d(features_gen * 16),
            nn.ReLU(),
            nn.ConvTranspose2d(features_gen * 16, features_gen * 8, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(features_gen * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(features_gen * 8, features_gen * 4, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(features_gen * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(features_gen * 4, channels_img, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.Tanh()  # Output: single-channel image in range [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)


# In[15]:


# Model, Optimizers, and Loss Function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
disc = Discriminator(1, FEATURES_DISC).to(device)  # Change channels_img to 1
gen = Generator(NOISE_DIM, 1, FEATURES_GEN).to(device)  # Change channels_img to 1
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# Define the criterion for binary cross-entropy
criterion = nn.BCELoss()  # Ensure this is used


# In[16]:


# Initialize GradScaler only if CUDA is available
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None


# In[17]:


for epoch in range(EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
        fake = gen(noise)

        # Train Discriminator
        opt_disc.zero_grad()
        
        # Mixed precision training if scaler is defined
        if scaler:
            with torch.cuda.amp.autocast():
                disc_real = disc(real).reshape(-1)
                loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
                disc_fake = disc(fake.detach()).reshape(-1)
                loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
                loss_disc = (loss_disc_real + loss_disc_fake) / 2

            scaler.scale(loss_disc).backward()
            scaler.step(opt_disc)
            scaler.update()
        else:
            # Regular precision training for CPU or non-CUDA setup
            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2

            loss_disc.backward()
            opt_disc.step()

        # Train Generator
        opt_gen.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                output = disc(fake).reshape(-1)
                loss_gen = criterion(output, torch.ones_like(output))

            scaler.scale(loss_gen).backward()
            scaler.step(opt_gen)
            scaler.update()
        else:
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))

            loss_gen.backward()
            opt_gen.step()

    print(f"Epoch [{epoch + 1}/{EPOCHS}] - Loss D: {loss_disc.item()}, Loss G: {loss_gen.item()}")

print("Training complete.")


# In[19]:


import os
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# Define directories
real_dir = 'generated_images/real_images'
fake_dir = 'generated_images/fake_images'
os.makedirs(real_dir, exist_ok=True)
os.makedirs(fake_dir, exist_ok=True)

# Function to save a grid of images
def save_image_grid(images, epoch, directory, prefix="img"):
    img_grid = vutils.make_grid(images, normalize=True)
    plt.imshow(img_grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.title(f"{prefix.capitalize()} Images - Epoch {epoch+1}")
    plt.axis("off")
    
    # Save the image grid
    plt.savefig(f"{directory}/{prefix}_epoch_{epoch+1}.png")
    plt.close()

# After each epoch, save real and fake image grids
for epoch in range(EPOCHS):
    # Generate and save a batch of fake images
    with torch.no_grad():
        noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
        fake_images = gen(noise)
        save_image_grid(fake_images, epoch, fake_dir, prefix="fake")

    # Save a batch of real images
    real_batch = next(iter(loader))[0].to(device)  # Fetch a batch of real images
    save_image_grid(real_batch, epoch, real_dir, prefix="real")

    print(f"Epoch {epoch+1}: Saved real and fake images.")


# In[ ]:





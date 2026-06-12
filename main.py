"""
DCGAN on CIFAR-10 with FID Evaluation and Latent Space Interpolation
---------------------------------------------------------------------
Author: (Your Name)
Purpose: Showcase GAN implementation for SciML and software engineering roles.
         Understandable, production-style code with extensive comments.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# -------------------------------
# 1. Configuration (Hyperparameters)
# -------------------------------
class Config:
    # Data
    dataset = 'CIFAR10'          # CIFAR10 gives 32x32 color images
    image_size = 32
    channels = 3                 # RGB
    # Model
    latent_dim = 100             # Size of random noise vector z
    feature_g = 64               # Base feature maps in generator
    feature_d = 64               # Base feature maps in discriminator
    # Training
    batch_size = 128
    epochs = 50
    lr = 0.0002                  # Standard DCGAN learning rate
    beta1 = 0.5                  # Adam beta1 (0.5 works well)
    beta2 = 0.999
    num_workers = 2              # For DataLoader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Evaluation
    fid_sample_interval = 5      # Compute FID every 5 epochs
    save_interval = 10           # Save model every 10 epochs
    # Output
    out_dir = 'gan_output'
    os.makedirs(out_dir, exist_ok=True)

cfg = Config()
print(f"Using device: {cfg.device}")

# -------------------------------
# 2. Data Loading (CIFAR-10)
# -------------------------------
def get_dataloader():
    transform = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Scale to [-1, 1]
    ])
    dataset = dset.CIFAR10(root='./data', download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size,
                            shuffle=True, num_workers=cfg.num_workers)
    return dataloader

dataloader = get_dataloader()

# -------------------------------
# 3. Models: Generator and Discriminator
# -------------------------------
def weights_init(m):
    """Custom initialization for DCGAN: all weights from Normal(mean=0, std=0.02)"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    """Maps latent vector z (100-dim) to a fake image (3x32x32)."""
    def __init__(self, latent_dim, feature_g, channels):
        super().__init__()
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, feature_g*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g*8),
            nn.ReLU(True),
            # state: (feature_g*8) x 4 x 4
            nn.ConvTranspose2d(feature_g*8, feature_g*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g*4),
            nn.ReLU(True),
            # state: (feature_g*4) x 8 x 8
            nn.ConvTranspose2d(feature_g*4, feature_g*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g*2),
            nn.ReLU(True),
            # state: (feature_g*2) x 16 x 16
            nn.ConvTranspose2d(feature_g*2, feature_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),
            # state: (feature_g) x 32 x 32
            nn.ConvTranspose2d(feature_g, channels, 4, 2, 1, bias=False),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, z):
        # z shape: (batch, latent_dim) -> reshape to (batch, latent_dim, 1, 1)
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.main(z)

class Discriminator(nn.Module):
    """Classifies real vs fake images (outputs a single logit)."""
    def __init__(self, channels, feature_d):
        super().__init__()
        self.main = nn.Sequential(
            # Input: 3 x 32 x 32
            nn.Conv2d(channels, feature_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state: feature_d x 16 x 16
            nn.Conv2d(feature_d, feature_d*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d*2),
            nn.LeakyReLU(0.2, inplace=True),
            # state: feature_d*2 x 8 x 8
            nn.Conv2d(feature_d*2, feature_d*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d*4),
            nn.LeakyReLU(0.2, inplace=True),
            # state: feature_d*4 x 4 x 4
            nn.Conv2d(feature_d*4, feature_d*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d*8),
            nn.LeakyReLU(0.2, inplace=True),
            # state: feature_d*8 x 2 x 2
            nn.Conv2d(feature_d*8, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()  # Probability of being real
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

# Instantiate models
netG = Generator(cfg.latent_dim, cfg.feature_g, cfg.channels).to(cfg.device)
netD = Discriminator(cfg.channels, cfg.feature_d).to(cfg.device)

# Apply custom initialization
netG.apply(weights_init)
netD.apply(weights_init)

print(netG)
print(netD)

# -------------------------------
# 4. Loss function and Optimizers
# -------------------------------
criterion = nn.BCELoss()  # Binary Cross Entropy for real/fake

optimizerG = optim.Adam(netG.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
optimizerD = optim.Adam(netD.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))

# -------------------------------
# 5. Helper functions
# -------------------------------
def generate_fixed_noise():
    """Fixed noise for visualizing progress across epochs."""
    return torch.randn(64, cfg.latent_dim, device=cfg.device)

fixed_noise = generate_fixed_noise()

def save_samples(epoch):
    """Save a grid of generated images."""
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
        grid = vutils.make_grid(fake, padding=2, normalize=True)
        vutils.save_image(grid, os.path.join(cfg.out_dir, f'fake_samples_epoch_{epoch}.png'))

def compute_fid(real_loader, netG, num_samples=10000):
    """
    Compute FID between real dataset and generated images.
    Lower is better (realistic and diverse).
    """
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(cfg.device)
    # Real images: sample from dataloader
    real_count = 0
    for batch, _ in real_loader:
        real_imgs = batch.to(cfg.device)
        # Inception expects images in [0,1] but our real_imgs are [-1,1]
        real_imgs = (real_imgs + 1) / 2  # -> [0,1]
        fid.update(real_imgs, real=True)
        real_count += real_imgs.size(0)
        if real_count >= num_samples:
            break
    # Fake images: generate from noise
    fake_count = 0
    while fake_count < num_samples:
        noise = torch.randn(cfg.batch_size, cfg.latent_dim, device=cfg.device)
        with torch.no_grad():
            fake_imgs = netG(noise)
        fake_imgs = (fake_imgs + 1) / 2
        fid.update(fake_imgs, real=False)
        fake_count += fake_imgs.size(0)
    return fid.compute().item()

def interpolate_latent_space(z1, z2, steps=10):
    """Linearly interpolate between two latent vectors and generate images."""
    alphas = np.linspace(0, 1, steps)
    interpolated = []
    for alpha in alphas:
        z = (1 - alpha) * z1 + alpha * z2
        with torch.no_grad():
            img = netG(z.unsqueeze(0)).cpu()
        interpolated.append(img)
    return torch.cat(interpolated, dim=0)

# -------------------------------
# 6. Training Loop
# -------------------------------
def train():
    print("Starting training...")
    G_losses = []
    D_losses = []
    best_fid = float('inf')

    for epoch in range(cfg.epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        num_batches = 0

        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(cfg.device)
            batch_size = real_imgs.size(0)

            # Labels for BCE loss (real=1, fake=0)
            real_labels = torch.ones(batch_size, device=cfg.device)
            fake_labels = torch.zeros(batch_size, device=cfg.device)

            # ---------------------
            # 1. Train Discriminator
            # ---------------------
            netD.zero_grad()

            # Real images forward
            outputs_real = netD(real_imgs)
            loss_real = criterion(outputs_real, real_labels)

            # Fake images forward
            noise = torch.randn(batch_size, cfg.latent_dim, device=cfg.device)
            fake_imgs = netG(noise)
            outputs_fake = netD(fake_imgs.detach())  # detach to avoid G gradients
            loss_fake = criterion(outputs_fake, fake_labels)

            lossD = loss_real + loss_fake
            lossD.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
            optimizerD.step()

            # ---------------------
            # 2. Train Generator
            # ---------------------
            netG.zero_grad()

            noise = torch.randn(batch_size, cfg.latent_dim, device=cfg.device)
            fake_imgs = netG(noise)
            outputs = netD(fake_imgs)
            lossG = criterion(outputs, real_labels)  # Try to fool D
            lossG.backward()
            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
            optimizerG.step()

            epoch_d_loss += lossD.item()
            epoch_g_loss += lossG.item()
            num_batches += 1

            # Print progress every 100 batches
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{cfg.epochs}] [Batch {i}/{len(dataloader)}] "
                      f"Loss_D: {lossD.item():.4f} Loss_G: {lossG.item():.4f}")

        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        G_losses.append(avg_g_loss)
        D_losses.append(avg_d_loss)

        print(f"Epoch {epoch} finished. Avg D loss: {avg_d_loss:.4f}, Avg G loss: {avg_g_loss:.4f}")

        # Save generated samples
        save_samples(epoch)

        # Evaluate FID periodically
        if epoch % cfg.fid_sample_interval == 0 or epoch == cfg.epochs - 1:
            fid_score = compute_fid(dataloader, netG, num_samples=5000)  # Use 5000 for speed
            print(f"FID at epoch {epoch}: {fid_score:.2f}")
            if fid_score < best_fid:
                best_fid = fid_score
                torch.save(netG.state_dict(), os.path.join(cfg.out_dir, 'best_generator.pth'))
                print(f"New best generator saved (FID={best_fid:.2f})")

        # Save checkpoint
        if epoch % cfg.save_interval == 0 or epoch == cfg.epochs - 1:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': netG.state_dict(),
                'discriminator_state_dict': netD.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
                'G_losses': G_losses,
                'D_losses': D_losses,
            }, os.path.join(cfg.out_dir, f'checkpoint_epoch_{epoch}.pth'))

    # Plot loss curves
    plt.figure(figsize=(10,5))
    plt.plot(G_losses, label='Generator Loss')
    plt.plot(D_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Losses')
    plt.savefig(os.path.join(cfg.out_dir, 'loss_plot.png'))
    plt.show()

    print(f"Training complete. Best FID: {best_fid:.2f}")

# -------------------------------
# 7. Demo: Latent Space Interpolation
# -------------------------------
def demo_interpolation():
    """Load the best generator and show interpolation between two random points."""
    netG.load_state_dict(torch.load(os.path.join(cfg.out_dir, 'best_generator.pth')))
    netG.eval()
    z1 = torch.randn(cfg.latent_dim, device=cfg.device)
    z2 = torch.randn(cfg.latent_dim, device=cfg.device)
    interp_imgs = interpolate_latent_space(z1, z2, steps=8)
    # Convert to grid and save
    grid = vutils.make_grid(interp_imgs, nrow=8, normalize=True)
    vutils.save_image(grid, os.path.join(cfg.out_dir, 'interpolation.png'))
    print("Interpolation saved to interpolation.png")

# -------------------------------
# 8. Run Everything
# -------------------------------
if __name__ == "__main__":
    train()
    demo_interpolation()
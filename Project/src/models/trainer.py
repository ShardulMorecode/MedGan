import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from generator import Generator
from discriminator import Discriminator
import gc
import time

# ✅ Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  

# ✅ Dataset Class (Optimized for Faster Loading)
class MedicalImageDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Assuming grayscale images
        ])
        self.images = [os.path.join(data_path, img) for img in os.listdir(data_path)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")  
        image.load()  # ✅ Load immediately to avoid PIL overhead
        image = self.transform(image)
        return image, image

# ✅ Perceptual Loss (VGG19)
class PerceptualLoss(nn.Module):
    def __init__(self, layer_index=16):  # Use early layers for feature extraction
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features  
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:layer_index]).eval()
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # ✅ Freeze VGG layers

        self.criterion = nn.L1Loss()  

    def forward(self, generated, target):
        gen_features = self.feature_extractor(generated)
        target_features = self.feature_extractor(target)
        return self.criterion(gen_features, target_features)

# ✅ Initialize Models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# ✅ Loss Functions
adversarial_loss = nn.MSELoss()
perceptual_loss = PerceptualLoss().to(device) 
l1_loss = nn.L1Loss()  # ✅ Explicitly adding L1 loss

# ✅ Optimizers
lr = 0.0001
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))  
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999)) 

# ✅ Learning Rate Schedulers
scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.5)
scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=10, gamma=0.5)

# ✅ Mixed Precision Training Setup
scaler = torch.amp.GradScaler()  # 🚀 Enable mixed precision

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()  # ✅ Fix Windows multiprocessing issue

    # ✅ Data Loader (Optimized)
    dataset = MedicalImageDataset("D:/Final Year/Mega project/Project/data/processed/stare_aug") 
    torch.backends.cudnn.benchmark = False  # Prevents unnecessary memory usage
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1, pin_memory=True, persistent_workers=True )  

    # ✅ Speed Test for 10 Batches
    start = time.time()
    for i, _ in enumerate(dataloader):
        if i == 10: 
            break
    print(f"✅ DataLoader Time (10 batches): {time.time() - start:.2f} sec")

    dummy_tensor = torch.randn(1, 3, 256, 256, device=device)
    start = time.time()
    _ = generator(dummy_tensor)
    torch.cuda.synchronize()
    print(f"🚀 Model Forward Pass Time: {time.time() - start:.4f} sec")

    # ✅ Training Loop
    EPOCHS = 30

    for epoch in range(EPOCHS):
        if (epoch + 1) % 5 == 0:  # Clean every 5 epochs
            gc.collect()
            torch.cuda.empty_cache()
        print(f"🚀 Starting Epoch {epoch+1}/{EPOCHS}...")
        for batch_idx, (real_images, _) in enumerate(dataloader):
            print(f"🟢 Processing Batch {batch_idx+1}/{len(dataloader)}...")
            real_images = real_images.to(device)
            #gc.collect()
            #torch.cuda.empty_cache()  # ✅ Free memory before processing batch

            # ✅ Add Noise to Discriminator Inputs
            real_images_noisy = real_images + 0.1 * torch.randn_like(real_images).to(device)
            
            # ✅ Generate Fake Images
            with torch.autocast(device_type="cuda", dtype=torch.float16):  # 🚀 Enable Mixed Precision
                fake_images = generator(real_images).detach()  # ✅ Detach to avoid memory leak
                fake_images_noisy = fake_images + 0.1 * torch.randn_like(fake_images).to(device)

                # ✅ Train Discriminator
                optimizer_D.zero_grad()
                outputs_real = discriminator(real_images_noisy)

                # ✅ One-Sided Label Smoothing
                real_labels = (torch.ones_like(outputs_real) * (0.8 + 0.2 * torch.rand_like(outputs_real))).to(device)
                fake_labels = (torch.zeros_like(outputs_real) + 0.1 * torch.rand_like(outputs_real)).to(device)

                d_loss_real = adversarial_loss(outputs_real, real_labels)
                outputs_fake = discriminator(fake_images_noisy)
                d_loss_fake = adversarial_loss(outputs_fake, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                scaler.scale(d_loss).backward()
                scaler.step(optimizer_D)
                scaler.update()

                # ✅ Train Generator
                optimizer_G.zero_grad()
                fake_images = generator(real_images)
                outputs = discriminator(fake_images)
                g_loss_adv = adversarial_loss(outputs, real_labels)  
                g_loss_perceptual = perceptual_loss(fake_images, real_images)  
                g_loss_l1 = l1_loss(fake_images, real_images)  
                g_loss = g_loss_adv + 10 * g_loss_perceptual + 5 * g_loss_l1  # Adjusted weighting

                scaler.scale(g_loss).backward()
                scaler.step(optimizer_G)
                scaler.update()

                # ✅ Balance Training: If D Loss is too low, update G again
                if d_loss.item() < 0.1:
                    optimizer_G.zero_grad()
                    fake_images = generator(real_images)
                    outputs = discriminator(fake_images)
                    g_loss_adv = adversarial_loss(outputs, real_labels)
                    g_loss_perceptual = perceptual_loss(fake_images, real_images)
                    g_loss_l1 = l1_loss(fake_images, real_images)
                    g_loss = g_loss_adv + 10 * g_loss_perceptual + 5 * g_loss_l1
                    scaler.scale(g_loss).backward()
                    scaler.step(optimizer_G)
                    scaler.update()

                # ✅ Free up memory after batch
                del real_images, real_images_noisy, fake_images_noisy, outputs_real, outputs_fake, outputs
                gc.collect()
                torch.cuda.empty_cache()

        # ✅ Step LR Schedulers
        scheduler_G.step()
        scheduler_D.step()

        print(f"Epoch [{epoch+1}/{EPOCHS}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    print("✅ Training Complete!")

    # ✅ Save the Models
    torch.save(generator.state_dict(), "D:/Final Year/Mega project/Project/saved_models/generator_v11.pth")
    torch.save(discriminator.state_dict(), "D:/Final Year/Mega project/Project/saved_models/discriminator_v11.pth")

from tqdm.auto import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch

def train(epochs,data_loader,netG,netD,clip_model,criterion,optimizerG,optimizerD,nz,fixed_noise,fixed_text_embedding,device):
    # Lists to keep track of progress
    img_list = []
    losses_G = []
    losses_D = []
    D_x_list = []
    D_G_z_list = []

    # Ensure models are on the correct device and in training mode
    netG.to(device)
    netD.to(device)
    clip_model.to(device) # Ensure clip_model is also on the device
    netG.train()
    netD.train()
    clip_model.eval() # CLIP model remains in evaluation mode

    for epoch in tqdm(range(epochs), desc="Epochs-"):
        for i, data in enumerate(data_loader, 0):
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            netD.zero_grad()

            # Get real images and their tokenized captions
            real_cpu = data['transformed_image'].to(device)
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)

            # Generate text embeddings for the current batch
            with torch.no_grad(): # Text encoder should not be trained during GAN training
                text_embeddings = clip_model.get_text_features(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            b_size = real_cpu.size(0)
            
            # Train with all-real batch
            label_real_tensor = torch.full((b_size,), 0.9, dtype=torch.float, device=device) # Smoothed real label
            output_real = netD(real_cpu, text_embeddings).view(-1)
            errD_real = criterion(output_real, label_real_tensor)
            errD_real.backward()
            D_x = output_real.mean().item()

            # Train with all-fake batch
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise, text_embeddings)
            label_fake_tensor = torch.full((b_size,), 0.1, dtype=torch.float, device=device) # Smoothed fake label
            output_fake = netD(fake.detach(), text_embeddings).view(-1) # Detach fake for D training
            errD_fake = criterion(output_fake, label_fake_tensor)
            errD_fake.backward()
            D_G_z1 = output_fake.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            # (2) Update G network: maximize log(D(G(z)))
            netG.zero_grad()
            label_generator_target = torch.full((b_size,), 0.9, dtype=torch.float, device=device) # G wants D to see fakes as real
            output_G = netD(fake, text_embeddings).view(-1) # Do NOT detach fake for G training
            errG = criterion(output_G, label_generator_target)
            errG.backward()
            optimizerG.step()
            D_G_z2 = output_G.mean().item()

            # Output training stats
            if i % 50 == 0:
                print(f'[{epoch}/{epochs}][{i}/{len(data_loader)}]\tLoss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}\tD(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

            # Save Losses and D(G(z)) for plotting later
            losses_G.append(errG.item())
            losses_D.append(errD.item())
            D_x_list.append(D_x)
            D_G_z_list.append(D_G_z2) # Use D_G_z2 (after G update) for Generator's performance

            # Check how the generator is doing by saving G's output on fixed_noise and fixed_text_embedding
            if (i % 100 == 0) or ((epoch == epochs-1) and (i == len(data_loader)-1)):
                with torch.no_grad():
                    netG.eval() # Put generator in eval mode for inference
                    generated_images = netG(fixed_noise, fixed_text_embedding).detach().cpu()
                    img_list.append(vutils.make_grid(generated_images, padding=2, normalize=True))
                    netG.train() # Put generator back in train mode
    print("Training finished.")
    return losses_G,losses_D,D_x_list,D_G_z_list,img_list

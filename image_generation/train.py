from tqdm.auto import tqdm
import torch
import torchvision.utils as vutils

def train(epochs,nz,train_dataloader,netD,netG,optimizerD,optimizerG,loss_fn,device):
    # Lists to keep track of progress
    img_list = []
    losses_G = []
    losses_D = []
    D_x_list = []
    D_G_z_list = []
    fixed_noise = torch.randn(train_dataloader.batch_size, nz, 1, 1, device=device)
    for epoch in tqdm(range(epochs),desc="Epochs-"):
        for i,data in enumerate(train_dataloader,0):
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            netD.zero_grad()
            # Train with all-real batch
            real_cpu=data[0].to(device)
            b_size=real_cpu.size(0)
            label_real_tensor=torch.full((b_size,),0.9,dtype=torch.float,device=device)
            # Forward pass real batch through D
            output_real=netD(real_cpu).mean(dim=[2,3]).view(-1)
            # Calculate loss on all-real batch
            errD_real=loss_fn(output_real,label_real_tensor)
            # Calculate gradients for D in real batch
            errD_real.backward()
            D_x=output_real.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise=torch.randn(b_size,nz,1,1,device=device)
            # Generate fake image batch with G
            fake=netG(noise)
            label_fake_tensor=torch.full((b_size,),0.1,dtype=torch.float,device=device)
            # Classify all fake batch with D
            output_fake=netD(fake.detach()).mean(dim=[2,3]).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake=loss_fn(output_fake,label_fake_tensor)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1=output_fake.mean().item()
            # Compute error of D as sum over the real and fake batches
            errD=errD_real+errD_fake
            # Update D
            optimizerD.step()

            # (2) Update G network: maximize log(D(G(z)))
            netG.zero_grad()
            label_real_tensor=torch.full((b_size,),0.9,dtype=torch.float,device=device)
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output_G=netD(fake).mean(dim=[2,3]).view(-1)
            # Calculate G's loss based on this output
            errG=loss_fn(output_G,label_real_tensor)
            # Calculate gradients for G
            errG.backward()
            # Update G
            optimizerG.step()
            D_G_z2=output_G.mean().item()

            # Output training stats
            if i % 50 == 0:
                    print(f'[{epoch}/{epochs}][{i}/{len(train_dataloader)}]	Loss_D: {errD.item():.4f}	Loss_G: {errG.item():.4f}	D(x): {D_x:.4f}	D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

            # Save Losses for plotting later
            losses_G.append(errG.item())
            losses_D.append(errD.item())
            D_x_list.append(D_x)
            D_G_z_list.append(D_G_z2)

            # Check how the generator is doing by saving G's output on fixed_noise
            if (i % 100 == 0) or ((epoch == epochs-1) and (i == len(train_dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
    
    return losses_G,losses_D,img_list
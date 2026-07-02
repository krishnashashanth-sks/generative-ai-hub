import torch

def train_model(num_epochs,train_dataloader,text_encoder,video_encoder,video_generator,video_discriminator,optimizer_G,optimizer_D,adversarial_loss_fn,reconstruction_loss_fn,text_video_alignment_loss_fn,device,reconstruction_loss_weight,alignment_loss_weight,latent_dim):
  print("Start Training")
  # Label smoothing factors
  smooth_real_label = 0.9
  smooth_fake_label = 0.1

  for epoch in range(num_epochs):
    for i,batch in enumerate(train_dataloader):
      input_ids=batch['input_ids'].to(device)
      attention_mask=batch['attention_mask'].to(device)
      real_video_frames_dataloader_shape=batch['video_frames'].to(device)

      # --- Discriminator Training ---
      optimizer_D.zero_grad()

      # Labels for adversarial loss: real (1s) and fake (0s) with smoothing
      real_labels_smoothed = torch.full((real_video_frames_dataloader_shape.size(0), 1), smooth_real_label, device=device)
      fake_labels_smoothed = torch.full((real_video_frames_dataloader_shape.size(0), 1), smooth_fake_label, device=device)
      # Original labels for Generator's adversarial loss (Generator still wants to fool D)
      real_labels_for_G = torch.ones(real_video_frames_dataloader_shape.size(0), 1, device=device)


      real_video_frames_discriminator_shape=real_video_frames_dataloader_shape.permute(0,2,1,3,4)
      output_real=video_discriminator(real_video_frames_discriminator_shape)
      loss_D_real=adversarial_loss_fn(output_real,real_labels_smoothed) # Use smoothed real labels

      with torch.no_grad():
        text_embeddings_for_D=text_encoder(input_ids,attention_mask)
        noise_vector_for_D=torch.randn(input_ids.size(0),latent_dim).to(device)
        generated_video_frames_for_D=video_generator(text_embeddings_for_D,noise_vector_for_D)

      output_fake=video_discriminator(generated_video_frames_for_D.detach())
      loss_D_fake=adversarial_loss_fn(output_fake,fake_labels_smoothed) # Use smoothed fake labels

      loss_D=loss_D_real+loss_D_fake
      loss_D.backward()
      optimizer_D.step()

      # --- Generator Training ---
      optimizer_G.zero_grad()
      text_embeddings_for_G=text_encoder(input_ids,attention_mask)
      noise_vector_for_G=torch.randn(input_ids.size(0),latent_dim).to(device)
      generated_video_frames_for_G=video_generator(text_embeddings_for_G,noise_vector_for_G)

      output_fake_G=video_discriminator(generated_video_frames_for_G)
      adversarial_loss_G=adversarial_loss_fn(output_fake_G,real_labels_for_G) # Generator still wants to make fakes look real

      generated_video_frames_for_encoder=generated_video_frames_for_G.permute(0,2,1,3,4)
      generated_video_embeddings=video_encoder(generated_video_frames_for_encoder.contiguous())

      reconstruction_loss=reconstruction_loss_fn(generated_video_frames_for_G,real_video_frames_discriminator_shape)
      alignment_loss=text_video_alignment_loss_fn(text_embeddings_for_G,generated_video_embeddings)

      loss_G=adversarial_loss_G+reconstruction_loss_weight*reconstruction_loss+alignment_loss_weight*alignment_loss
      loss_G.backward()
      optimizer_G.step()

      if (i + 1) % 1 == 0: # Print every batch for demonstration with small dataset
        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_dataloader)}] | "
                      f"D_loss: {loss_D.item():.4f} | G_loss: {loss_G.item():.4f} "
                      f"(G_Adv: {adversarial_loss_G.item():.4f}, G_Rec: {reconstruction_loss.item():.4f}, G_Align: {alignment_loss.item():.4f})")
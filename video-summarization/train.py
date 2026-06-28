import torch

def train(num_epochs,train_dataloader,optimizer,loss_function,model,val_dataloader):
    print("\n--- Starting Training and Evaluation Loop ---")

    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        total_train_loss = 0

        for i, batch in enumerate(train_dataloader):
            features = batch['features'].squeeze(0) # Remove batch_size=1 dimension if present from DataLoader
            ground_truth = batch['ground_truth'].squeeze(0) # Remove batch_size=1 dimension if present

            # Ensure ground_truth is 1D or (sequence_length, 1) and features is (sequence_length, input_size)
            # Our model outputs (batch_size, sequence_length, 1), so ground_truth needs to match this for BCELoss
            if ground_truth.dim() == 1: # Convert (sequence_length) to (sequence_length, 1)
                ground_truth = ground_truth.unsqueeze(-1)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(features.unsqueeze(0)) # Add batch dimension back for model input

            # Calculate loss
            loss = loss_function(outputs, ground_truth.unsqueeze(0)) # Add batch dimension back for loss calculation

            # Backward and optimize
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}')

        # Evaluation loop
        model.eval() # Set model to evaluation mode
        total_val_loss = 0
        with torch.no_grad(): # Disable gradient calculation for evaluation
            for i, batch in enumerate(val_dataloader):
                features = batch['features'].squeeze(0)
                ground_truth = batch['ground_truth'].squeeze(0)

                if ground_truth.dim() == 1:
                    ground_truth = ground_truth.unsqueeze(-1)

                outputs = model(features.unsqueeze(0))
                val_loss = loss_function(outputs, ground_truth.unsqueeze(0))
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

    print("--- Training and Evaluation Complete ---")
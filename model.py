import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionRecognitionModel(nn.Module):
    
    def __init__(self, num_emotions, input_height, kernel_size=12, padding=5):
        """
        Initialize the EmotionRecognitionModel.

        Args:
            num_emotions (int): Number of emotion classes.
            input_height (int): Height of the input images.
        """
        super(EmotionRecognitionModel, self).__init__()

        self.kernel_size = kernel_size
        self.padding = padding
        print(f"Kernel size: {self.kernel_size}, Padding: {self.padding}")    
       # Convolutional layers for feature extraction
        self.conv_block = nn.Sequential(
            # First conv block
            
            nn.Conv2d(1, 32, kernel_size=kernel_size, stride=1, padding=padding),  # Output: (32, H, W)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output: (32, H/2, W/2)
            nn.Dropout(p=0.3),

            # Second conv block
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=padding),  # Output: (64, H/2, W/2)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output: (64, H/4, W/4)
            nn.Dropout(p=0.3),

            # Third conv block
            nn.Conv2d(64, 128, kernel_size=kernel_size, stride=1, padding=padding),  # Output: (128, H/4, W/4)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output: (128, H/8, W/8)
            nn.Dropout(p=0.3),

        )

        # Calculate the output height after convolution and pooling layers
        conv_output_height = input_height
        for _ in range(3):
            conv_output_height = (conv_output_height + 2 * self.padding - self.kernel_size) // 1 + 1  # Adjusted kernel size and padding
            conv_output_height //= 2

          # Fully connected layer to reduce dimensionality before LSTM
        self.fc1 = nn.Linear(128 * conv_output_height, 128)
        self.dropout_fc1 = nn.Dropout(p=0.3)

        # Bidirectional LSTM
        self.bilstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2,
                              batch_first=True, bidirectional=True, dropout=0.3)

        # Attention layer
        self.attention_layer = nn.Linear(64 * 2, 1)

        # Output layer
        self.fc2 = nn.Linear(64 * 2, num_emotions)
        # No need for softmax layer if using CrossEntropyLoss

    def forward(self, x):
        batch_size = x.size(0)

        # CNN feature extraction
        x = self.conv_block(x)

        # Prepare for LSTM
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(batch_size, x.size(1), -1)

        # Fully connected layer
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_fc1(x)

        # BiLSTM
        lstm_out, _ = self.bilstm(x)

        # Attention mechanism
        attn_weights = F.softmax(self.attention_layer(lstm_out), dim=1)
        attn_weights = attn_weights.permute(0, 2, 1)

        # Weighted sum of LSTM outputs
        attn_output = torch.bmm(attn_weights, lstm_out)
        attn_output = attn_output.squeeze(1)

        # Output layer
        logits = self.fc2(attn_output)

        return logits



# Initialize model
#num_emotions = np.unique(augmented_labels).shape[0]


# Define the loss function once
loss_fn = nn.CrossEntropyLoss()

def make_train_step(model, optimizer):
    def train_step(X, Y):
        # Set model to train mode
        model.train()

        # Move data to the same device as model
        X = X.to(next(model.parameters()).device)
        Y = Y.to(next(model.parameters()).device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output_logits = model(X)

        # Compute loss
        loss = loss_fn(output_logits, Y)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Compute predictions and accuracy
        with torch.no_grad():
            predictions = torch.argmax(output_logits, dim=1)
            correct = (predictions == Y).sum().item()
            accuracy = correct / Y.size(0) * 100.0

        return loss.item(), accuracy
    return train_step

def make_validate_fnc(model):
    def validate(X, Y):
        model.eval()
        with torch.no_grad():
            # Move data to the same device as model
            X = X.to(next(model.parameters()).device)
            Y = Y.to(next(model.parameters()).device)

            output_logits = model(X)
            loss = loss_fn(output_logits, Y)
            predictions = torch.argmax(output_logits, dim=1)
            correct = (predictions == Y).sum().item()
            accuracy = correct / Y.size(0) * 100.0
        return loss.item(), accuracy, predictions
    return validate


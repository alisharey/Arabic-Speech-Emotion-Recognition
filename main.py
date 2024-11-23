# Numerical and Data Handling
import numpy as np
import pandas as pd
import os

# Audio Processing and Visualization
import librosa
import librosa.display

# Plotting
import matplotlib.pyplot as plt

import seaborn as sns


    


from utils import *
from model import  make_train_step, make_validate_fnc, EmotionRecognitionModel
from transformer import EmotionRecognitionModelTransfomer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import argparse



# Training and validation functions
def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0
    correct_preds = 0
    total_samples = 0

    for X_batch, Y_batch in loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = loss_fn(outputs, Y_batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        epoch_loss += loss.item() * X_batch.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct_preds += (predicted == Y_batch).sum().item()
        total_samples += Y_batch.size(0)

    epoch_loss /= total_samples
    accuracy = (correct_preds / total_samples) * 100
    return epoch_loss, accuracy

def validate_epoch(model, loader, loss_fn, device):
    model.eval()
    epoch_loss = 0
    correct_preds = 0
    total_samples = 0

    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            outputs = model(X_batch)
            loss = loss_fn(outputs, Y_batch)

            # Accumulate metrics
            epoch_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_preds += (predicted == Y_batch).sum().item()
            total_samples += Y_batch.size(0)

    epoch_loss /= total_samples
    accuracy = (correct_preds / total_samples) * 100
    return epoch_loss, accuracy




def train(fold, X_train, Y_train, X_val, Y_val, device, # Hyperparameters
    EPOCHS = 100,
    BATCH_SIZE = 32,
    LEARNING_RATE = 1e-4,
    WEIGHT_DECAY = 1e-5,
    EARLY_STOPPING_PATIENCE = 20
    ):
   
    print(f'Selected device is {device}')


    # Initialize model
    num_emotions = np.unique(Y_train).shape[0]
    model = EmotionRecognitionModel(num_emotions,input_height=128, kernel_size=kernal_size, padding=padding).to(device)
    #model = EmotionRecognitionModelTransfomer(num_emotions,input_height=128).to(device)
    print('Number of trainable parameters:', sum(p.numel() for p in model.parameters()))

    # Optimizer and loss function
    #optimizer = ADOPT(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
   

    loss_fn = nn.CrossEntropyLoss()


    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Prepare datasets and data loaders
    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(Y_train).long())
    val_dataset = TensorDataset(torch.tensor(X_val).float(), torch.tensor(Y_val).long())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    

    # Training loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Early stopping variables
    best_val_loss = float('inf')
    best_val_acc = 0
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = validate_epoch(model, val_loader, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Learning rate scheduling
        scheduler.step(val_loss)

        print(f'Epoch {epoch}/{EPOCHS}: '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%,'
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"new best val_loss {val_loss}")
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), f'best_model{fold}.pth')
       
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered!")
                break
        
        

# Test function
def test_model(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct_preds = 0
    total_samples = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            outputs = model(X_batch)
            loss = loss_fn(outputs, Y_batch)

            # Accumulate loss and accuracy
            total_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_preds += (predicted == Y_batch).sum().item()
            total_samples += Y_batch.size(0)

            # Store predictions and labels for further analysis if needed
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(Y_batch.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = (correct_preds / total_samples) * 100

    return avg_loss, accuracy, all_predictions, all_labels



def train_test_model(include_augmentation=True):
    """
    Train and test the emotion recognition model using k-fold cross-validation.
    This function performs the following steps:
    1. Loads the dataset and creates k-folds.
    2. For each fold:
        a. Loads and augments the spectrograms.
        b. Scales the training and testing data.
        c. Saves the datasets for the current fold.
        d. Trains the model on the training data.
        e. Tests the model on the testing data and records the accuracy.
    3. Prints the test accuracies for each fold and the average accuracy.
    Parameters:
    None
    Returns:
    None
    """
    # Use the function to load the dataset
    # Replace with your actual base path
    base_path = 'ksu_emotions/data/SPEECH'
    SAMPLE_RATE = 16_000  # Sample rate of the audio files
    duration = 10
    num_augmentations = 2 if include_augmentation else 0
    
    folds = create_folds(Data=load_ksu_dataset(base_path))
    #folds = create_folds(Data=EYASE_dataset('/home/ali/AR-Emotion-Recognition/data/EYASE'), SAMPLE_RATE=SAMPLE_RATE)

    test_accuracies = np.empty(shape=len(folds))
    
    for f, (train_df, test_df) in enumerate(folds):
        fold_num = f + 1
        print(f"Fold {fold_num}, {train_df.shape, test_df.shape}:")
        mel_spectrograms, signals, mel_spectrograms_test = load_spectograms(train_df, test_df, num_augmentations, duration, SAMPLE_RATE)
        k = train_df.shape[0]
        print(f"signals'shape {signals.shape}")

        if include_augmentation:
        # applying augmentations
            for i,signal in enumerate(signals):
                augmented_signals = add_augmentation(signal, augmented_num=num_augmentations)
                #augmented_signals = addAWGN(signal)
                for j in range(augmented_signals.shape[0]):
                    mel_spectrogram = get_log_mel_spectrogram(augmented_signals[j,:], sample_rate=SAMPLE_RATE)       
                    mel_spectrograms[k] = mel_spectrogram
                    k += 1
                    train_df = pd.concat([train_df, train_df.iloc[i:i+1]], ignore_index=True)
                print("\r Processed {}/{} files".format(i,len(signals)),end='')

        del signals

            

        X_train = np.expand_dims(mel_spectrograms,1)        
        Y_train = np.array(train_df.Emotion)
        X_test = np.expand_dims(mel_spectrograms_test, 1)
        Y_test = np.array(test_df.Emotion)
        print(train_df.Emotion.shape, X_train.shape)


        X_train, X_test = scale(X_train, X_test)
        save_datasets(X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, filename=f"dataset_fold{fold_num}")   

        del mel_spectrograms, mel_spectrograms_test, test_df, train_df
         # Set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        BATCH_SIZE = 32
        train(fold=fold_num, X_train=X_train, Y_train=Y_train, X_val=X_test, Y_val=Y_test, device=device, BATCH_SIZE=BATCH_SIZE)


        num_emotions = np.unique(Y_test).shape[0]
        input_height = 128  # Height of your spectrograms
        model = EmotionRecognitionModel(num_emotions, input_height, kernel_size=kernal_size, padding=padding).to(device)
        model.load_state_dict(torch.load(f'best_model{fold_num}.pth'))
        
        # Prepare the test dataset and DataLoader
        test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(Y_test).long())
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        loss_fn = nn.CrossEntropyLoss()
        test_loss, test_acc, predictions, labels = test_model(model, test_loader, loss_fn, device)
        test_accuracies[f] = test_acc
        print(f"fold {fold_num}, test accuracy {test_acc}")

    print(test_accuracies)
    print(f"average accuracy {np.mean(test_accuracies)}")
    
        


def train_test_using_files(include_augmentation=True):
    """
    Train and evaluate an emotion recognition model using k-fold cross-validation.
    This function performs the following steps:
    
    1. For each fold:
        a. Loads the training and testing datasets.
        b. Sets the device to 'cuda' if available, otherwise 'cpu'.
        c. Trains the model on the training dataset.
        d. Loads the best model for the current fold.
        e. Evaluates the model on the testing dataset.
        f. Stores the test accuracy for the current fold.
    2. Prints the test accuracies for all folds.
    3. Prints the average test accuracy across all folds.    
    Parameters:
    None
    Returns:
    None
    """
    folds = 5

    test_accuracies = np.empty(folds)
    for f in range (folds):
        fold_num = f + 1
        
        
        X_train, X_test, Y_train, Y_test=load_datasets(filename=f"dataset_fold{fold_num}")
        
        # use only the first third of X_train, Y_train if include_augmentation is false
        if not include_augmentation:
            X_train = X_train[:int(X_train.shape[0]/3)]
            Y_train = Y_train[:int(Y_train.shape[0]/3)]

        
            

        print(f"Fold {fold_num}, {X_train.shape, X_test.shape}:")   

        
         # Set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        BATCH_SIZE = 32
        
        train(fold=fold_num, X_train=X_train, Y_train=Y_train, X_val=X_test, Y_val=Y_test, device=device, BATCH_SIZE=BATCH_SIZE)


        # getting test accuracy
        num_emotions = np.unique(Y_test).shape[0]
        input_height = 128  # Height of your spectrograms
        model = EmotionRecognitionModel(num_emotions, input_height, kernel_size=kernal_size, padding=padding).to(device)
        model.load_state_dict(torch.load(f'best_model{fold_num}.pth'))
        
        # Prepare the test dataset and DataLoader
        test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(Y_test).long())
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        loss_fn = nn.CrossEntropyLoss()
        test_loss, test_acc, predictions, labels = test_model(model, test_loader, loss_fn, device)
        test_accuracies[f] = test_acc
        print(f"fold {fold_num}, test accuracy {test_acc}")

    print(test_accuracies)
    print(f"average accuracy {np.mean(test_accuracies)}")
   
    

def test(kernel_size=10, padding=4):
    test_accs = []
    test_losses = []
    all_predictions = []
    all_labels = []
    
    for i in range(5):
       fold_num = i + 1
       _, X_test, _, Y_test = load_datasets(f"dataset_fold{fold_num}")  # Assuming load_datasets is defined elsewhere
       device = 'cuda' if torch.cuda.is_available() else 'cpu'
       BATCH_SIZE = 32
       num_emotions = np.unique(Y_test).shape[0]
       input_height = 128  # Height of your spectrograms
       model = EmotionRecognitionModel(num_emotions, input_height, kernel_size=kernel_size, padding=padding).to(device)  
       model.load_state_dict(torch.load(f'best_model{fold_num}.pth'))
       # Prepare the test dataset and DataLoader
       test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(Y_test).long())
       test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
       loss_fn = nn.CrossEntropyLoss()
       test_loss, test_acc, predictions, labels = test_model(model, test_loader, loss_fn, device)  # Assuming test_model is defined elsewhere
       print(f"Fold {fold_num}, Test Accuracy {test_acc}, Test Loss {test_loss}")
       test_accs.append(test_acc)
       test_losses.append(test_loss)
       all_predictions.extend(predictions)
       all_labels.extend(labels)  

    # Calculate the average confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    cm_normalized = cm // len(test_accs)
    print(cm_normalized)

    # Define emotion labels
    emotion_labels = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Angry', 'Fear']

    # Plot the normalized confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
    plt.title('Averaged Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the confusion matrix as an image file (e.g., PNG)
    plt.savefig('averaged_confusion_matrix.png')  # Saves the figure as 'normalized_confusion_matrix.png'
    plt.close()  # Close the figure to free up memory

    # Calculate averages
    avg_test_acc = np.mean(test_accs)
    avg_test_loss = np.mean(test_losses)

    print("\nAverage Results:")
    print(f"Average Test Accuracy: {avg_test_acc:.2f}%")
    print(f"Average Test Loss: {avg_test_loss:.3f}")

    # Compute overall confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Overall Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the confusion matrix as an image file (e.g., PNG)
    plt.savefig('confusion_matrix.png')  # Saves the figure as 'confusion_matrix.png'
    plt.close()  # Close the figure to free up memory

    # Print overall classification report
    emotion_labels = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Angry', 'Fear']
    print("\nOverall Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=emotion_labels))

    # Calculate overall F1 scores
    micro_f1 = f1_score(all_labels, all_predictions, average='micro')
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    weighted_f1 = f1_score(all_labels, all_predictions, average='weighted')

    print("\nOverall F1 Scores:")
    print(f"Micro F1: {micro_f1:.3f}")
    print(f"Macro F1: {macro_f1:.3f}")
    print(f"Weighted F1: {weighted_f1:.3f}")


# Hyperparameters for the model (optional)
kernal_size = 10
padding = 4

def main(args):
    print(f"saved files used {args.use_saved_files}, augementation included {args.include_augmentation}, test only {args.test_only}")
    if args.test_only:
        test()
    else:
        if args.use_saved_files:
            train_test_using_files(include_augmentation=args.include_augmentation)  
        else:
            train_test_model(include_augmentation=args.include_augmentation)
        test(kernal_size, padding)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and test emotion recognition model.')
    parser.add_argument('--use_saved_files', action='store_true', help='Use saved files for training and testing')
    parser.add_argument('--test_only', action='store_true', help='Run only the test phase')
    parser.add_argument('--include_augmentation', action='store_true', help='Include data augmentation during training')

    args = parser.parse_args()
    
    main(args)
    



    

    
# Emotion Recognition Model


This repository contains the implementation of an emotion recognition model using deep learning techniques. The model is designed to recognize emotions from speech data using spectrograms and various neural network architectures.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/AR-Emotion-Recognition.git
    cd AR-Emotion-Recognition
    ```

2. Install the required dependencies:

Create and activate a conda environment with Python 3.12.1 and install pip:
    ```bash
    conda create -n emotion_recognition python=3.12.1 pip
    conda activate emotion_recognition
    ```

    ```bash
    pip install -r requirements.txt
    ```
    to get the required libraries

## Usage
First exract the KSUEmotion dataset in the root folder.

To train and test the emotion recognition model, run the following command:
```bash
python main.py 
```
After the first run the data will be saved as arrays and can be reused usign the following commands for training.

```bash
python main.py --use_saved_files
```
to train with augementataion you use the previous commands and add `--include_augmentation` flag.

```bash
python main.py --use_saved_files --include_augmentation
```


To only test the model using pre-trained weights, you need to do at least run one training to save the arrays first then run:
```bash
python main.py --test_only
```

## Training and Evaluation

The training process involves the following steps:
1. Load and preprocess the dataset.
2. Perform data augmentation on the training data.
3. Train the model using k-fold cross-validation.
4. Evaluate the model on the test data.

### Training

To train the model, use the `train_test_model` function which performs k-fold cross-validation and saves the best model for each fold. Note that arrays of the folds are saved for testing and each fold is about 2.7 gb in size.

After that `train_test_using_files` be used for trainning for faster processing. 

### Evaluation

The `test` function evaluates the model on the test data and computes various metrics such as accuracy, confusion matrix, and F1 scores.

## Results

The results of the model are evaluated using metrics such as accuracy, confusion matrix, and F1 scores. The confusion matrix and classification report are saved as images for further analysis.



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
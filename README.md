# Emotion Recognition Model
## Model Architecture

The emotion recognition model is built using a CNN-BiLSTM-Attention architecture. The architecture is designed to capture both spatial and temporal features from the input data, making it well-suited for emotion recognition tasks.

![Model Architecture](Architecture.png)

This repository contains the implementation of an emotion recognition model using a CNN-BiLSTM-Attention architecture on KSUEmotion dataset.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/AR-Emotion-Recognition.git
    cd AR-Emotion-Recognition
    ```


2. Create and activate a conda environment with Python 3.12.1 and install pip:
    ```bash
    conda create -n emotion_recognition python=3.12.1 pip
    conda activate emotion_recognition
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    

## Usage
First exract the KSUEmotion dataset in the root folder.

To train and test the emotion recognition model, run the following command:
```bash
python main.py --include_augmentation
```
After the first run the data will be processed data will be saved as arrays and can be re-used using the following command for training.

```bash
python main.py --use_saved_files
```
to train with augementataion you can use the previous commands and add `--include_augmentation` flag.

```bash
python main.py --use_saved_files --include_augmentation
```


To only test the model using pre-trained weights, you need to do at least one training run to save the arrays then run:
```bash
python main.py --test_only
```



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

# CodeAlpha Handwritten Character Recognition

## Project Overview
This project builds a CNN to recognize handwritten digits/characters using the MNIST dataset.

## Features
- Image preprocessing and normalization
- CNN model design and training
- Accuracy evaluation and saved model export

## Setup Instructions
1. Clone this repository:
    ```bash
    git clone https://github.com/Saurabh3312/CodeAlpha_Handwritten_Recognition.git
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the training script:
    ```bash
    python src/train_model.py
    ```
4. Test the model:
    ```bash
    python src/test_model.py
    ```

## Usage
- The `src/train_model.py` script downloads MNIST, trains a CNN, and saves the model under `models/`.
- Use `src/test_model.py` to load the saved model and evaluate on test images.


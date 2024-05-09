# NLP Word2Vec and FastText Sentiment Analysis with RNN

This repository contains the implementation of a sentiment analysis project that leverages Word2Vec and FastText embeddings with a Recurrent Neural Network (RNN). The project aims to classify movie reviews from the IMDB dataset as positive or negative.

## Project Structure

- `NLP_WORD2VEC_FASTTEXT_SENTIMENT-RNN.ipynb`: Main Jupyter notebook with all preprocessing, training, and evaluation code.
- `aclImdb_v1.tar.gz`: Dataset archive, containing positive and negative movie reviews.
- `word2vec_embeddings.pth`: Saved Word2Vec model weights.
- `ft_skipgram.bin`: FastText model trained with the skip-gram approach.
- `sentiment_rnn.pth`: Trained RNN model for sentiment analysis.

## Setup and Installation

To run this project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/NLP-Sentiment-Analysis.git
   cd NLP-Sentiment-Analysis
2. **Install dependencies:**
    ```bash
    pip install torch torchvision torchaudio fasttext
3. **Extract and prepare the dataset:**
    ```bash
    tar -xzvf "/content/drive/MyDrive/aclImdb_v1.tar.gz" -C /path/to/destination
    
4. **Mount Google Drive (if using Google Colab):**
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    
Usage
Word2Vec Training:
Train Word2Vec embeddings using the provided scripts within the Jupyter notebook. Adjust the hyperparameters as necessary.
FastText Training:
Compile and install FastText, then train the FastText model using the skip-gram method.
RNN Sentiment Analysis:
Train the RNN using the pretrained embeddings from Word2Vec or FastText. Evaluate the model on the test dataset and view the performance metrics.
Performance Visualization
Below is the performance visualization of the model:


Contributing
Contributions are welcome! Here are some ways you can contribute:

Improve the model architecture or training routines.
Add new datasets or experiment with different preprocessing techniques.
Improve the documentation or add example use cases.
Please submit a pull request with your proposed changes.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
Dataset from the ACL IMDB dataset.
PyTorch for providing the deep learning framework.
FastText for efficient text classification and representation learning.

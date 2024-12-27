# IMDb Sentiment Analysis

## 1. Introduction

This project involves developing a sentiment analysis system for IMDb movie reviews. Using a custom-trained deep learning model, the application predicts whether a review is positive or negative. A Flask-based web interface provides real-time predictions based on user input.

## 2. Dataset and Preprocessing

The dataset consists of IMDb movie reviews, evenly balanced between positive and negative sentiments. The data preprocessing pipeline includes:

- Removing duplicates and cleaning text.
- Lowercasing, tokenizing, and lemmatizing words.
- Removing stopwords and correcting spelling errors using SymSpell.
- Encoding sentiments as binary classes: `1` for positive and `0` for negative.

The dataset was split into training, validation, and test sets for model development and evaluation.

## 3. Model Development

The model was developed using a Bidirectional LSTM network. Key features include:

- **GloVe Embeddings**: Pre-trained word vectors for semantic representation.
- **Bidirectional LSTM**: Captures contextual dependencies in text.
- **Dropout Layers**: Reduces overfitting.
- **Binary Classification**: Outputs a sentiment score via a sigmoid activation.

### Training Process

- The model was trained using padded input sequences, with early stopping applied to optimize performance.
- The final model was saved as `model.pkl`.

### Performance Metrics

The model's performance was evaluated using accuracy and loss metrics on the training and validation datasets. Below are the training and validation curves:

![Training and Validation Metrics](train_val.png)

- **Accuracy**: The model steadily improved during training, achieving high accuracy on both the training and validation sets.
- **Loss**: Training and validation loss decreased over epochs, indicating effective learning.

The confusion matrix illustrates the model's performance on the test set:

![Confusion Matrix](matrix.png)

## 4. Web Application

The trained model was deployed using Flask, providing a simple and intuitive interface:

- **Input**: Users enter a review in the text box.
- **Output**: The predicted sentiment score is displayed on the page.

![Web Interface Screenshot](ui.png)

## 5. Challenges and Future Work

### Challenges

- **Spelling and Grammar Variations**: Handling diverse text patterns remains a challenge.
- **Complex Sentiments**: Understanding nuanced sentiments, such as sarcasm, is a limitation.

### Future Work

- Incorporate transformer models like BERT for improved contextual understanding.
- Extend the application to support multiple languages.
- Deploy the application on cloud platforms for scalability.

## 6. Conclusion

This project combines data preprocessing, a custom-trained deep learning model, and a web application to deliver accurate sentiment predictions. The results demonstrate the effectiveness of the approach, with room for future enhancements.

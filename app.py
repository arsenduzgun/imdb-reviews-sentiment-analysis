from flask import Flask, request, jsonify, render_template
import re
import contractions
import spacy
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import words
from spacy.lang.en.stop_words import STOP_WORDS
from symspellpy.symspellpy import SymSpell, Verbosity
from tensorflow.keras.preprocessing.sequence import pad_sequences
import math
import pickle
import joblib

app = Flask(__name__)

# Global initialization
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")

# Initialize SymSpell globally
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = "frequency_dictionary_en_82_765.txt"
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Load tokenizer once
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load model once
model = joblib.load('model.pkl')

# Custom stopwords
stopwords_to_keep = {
    "not", "no", "nor", "never", "neither",
    "but", "though", "although", "however", "yet",
    "very", "too", "much", "more", "most", "almost",
    "should", "could", "might", "may", "would", "can", "must",
    "only", "against", "because", "than"
}
custom_stopwords = STOP_WORDS - stopwords_to_keep

# Set of valid words
valid_words = set(words.words())

# Get WordNet POS tag
def get_wordnet_pos(spacy_tag):
    if spacy_tag == "ADJ":
        return wordnet.ADJ
    elif spacy_tag == "VERB":
        return wordnet.VERB
    elif spacy_tag == "NOUN":
        return wordnet.NOUN
    elif spacy_tag == "ADV":
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Correct spelling using SymSpell
def symspell_correct(word):
    suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
    return suggestions[0].term if suggestions else word

# Preprocess text
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Expand contractions
    text = contractions.fix(text)

    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize text
    tokens = word_tokenize(text)

    # SymSpell correction
    tokens = [symspell_correct(word) for word in tokens]

    # Remove stopwords
    tokens = [word for word in tokens if word not in custom_stopwords]

    # Use spaCy for POS tagging
    doc = nlp(" ".join(tokens))

    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(token.text, get_wordnet_pos(token.pos_)) for token in doc]

    # Filter valid words
    tokens = [word for word in tokens if word in valid_words]

    return tokens

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['text']

    # Preprocess the input text
    input_preprocessed = [preprocess_text(input_text)]

    # Convert text to sequences
    input_seq = tokenizer.texts_to_sequences(input_preprocessed)

    # Pad the sequence
    max_length = 100
    input_pad = pad_sequences(input_seq, maxlen=max_length, padding='post')

    # Predict using the loaded model
    prediction = model.predict(input_pad)

    positivity_score = prediction[0][0]*100
    
    positivity_score_rounded =  math.ceil(positivity_score) if positivity_score < .5 else math.floor(positivity_score)

    return jsonify({'prediction': str(positivity_score_rounded)})

if __name__ == '__main__':
    app.run(debug=True)

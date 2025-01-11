from flask import Flask, request, jsonify, render_template
import re
import contractions
import spacy
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, words
from spacy.lang.en.stop_words import STOP_WORDS
from symspellpy.symspellpy import SymSpell, Verbosity
from tensorflow.keras.preprocessing.sequence import pad_sequences
import math
import pickle
import joblib

app = Flask(__name__)

# Initialize global resources
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")

# Initialize SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = "frequency_dictionary_en_82_765.txt"
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load prediction model
model = joblib.load('model.pkl')

# Define custom stopwords (excluding critical words)
stopwords_to_keep = {
    "not", "no", "nor", "never", "neither", "but", "though", "although", "however", "yet",
    "very", "too", "much", "more", "most", "almost", "should", "could", "might", "may",
    "would", "can", "must", "only", "against", "because", "than"
}
custom_stopwords = STOP_WORDS - stopwords_to_keep

# Set of valid words for filtering
valid_words = set(words.words())

# Helper function: Map spaCy POS tags to WordNet POS tags
def get_wordnet_pos(spacy_tag):
    pos_map = {
        "ADJ": wordnet.ADJ,
        "VERB": wordnet.VERB,
        "NOUN": wordnet.NOUN,
        "ADV": wordnet.ADV
    }
    return pos_map.get(spacy_tag, wordnet.NOUN)

# Helper function: Correct spelling using SymSpell
def symspell_correct(word):
    suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
    return suggestions[0].term if suggestions else word

# Preprocess text input
def preprocess_text(text):
    # Lowercase and expand contractions
    text = contractions.fix(text.lower())

    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize and correct spelling
    tokens = [symspell_correct(word) for word in word_tokenize(text)]

    # Remove stopwords
    tokens = [word for word in tokens if word not in custom_stopwords]

    # Lemmatize using spaCy POS tagging
    doc = nlp(" ".join(tokens))
    tokens = [lemmatizer.lemmatize(token.text, get_wordnet_pos(token.pos_)) for token in doc]

    # Filter valid words
    return [word for word in tokens if word in valid_words]

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['text']

    # Preprocess the input text
    preprocessed_text = [preprocess_text(input_text)]

    # Convert text to sequences and pad
    input_seq = tokenizer.texts_to_sequences(preprocessed_text)
    max_length = 100
    input_pad = pad_sequences(input_seq, maxlen=max_length, padding='post')

    # Predict and calculate positivity score
    prediction = model.predict(input_pad)
    positivity_score = prediction[0][0] * 100

    # Round score based on value
    positivity_score_rounded = math.ceil(positivity_score) if positivity_score < 50 else math.floor(positivity_score)

    return jsonify({'prediction': str(positivity_score_rounded)})

# Main entry point
if __name__ == '__main__':
    app.run(debug=True)

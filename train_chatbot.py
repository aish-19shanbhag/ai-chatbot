"""
Train an intent-classification chatbot model.

Reads intent patterns from intents.json, builds a bag-of-words
representation, and trains a small feedforward network to classify
user input into an intent tag.
"""

import argparse
import json
import pickle
import random
from pathlib import Path

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

IGNORE_TOKENS = {"?", "!", ".", ","}


def ensure_nltk_data() -> None:
    """Download required NLTK corpora if they aren't already present."""
    for resource in ("punkt", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def load_intents(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_vocabulary(intents: dict, lemmatizer: WordNetLemmatizer) -> tuple[list, list, list]:
    """Extract the vocabulary, class labels, and (tokens, tag) documents from the intents file."""
    words, classes, documents = [], [], []

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            documents.append((tokens, intent["tag"]))
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

    words = sorted({lemmatizer.lemmatize(w.lower()) for w in words if w not in IGNORE_TOKENS})
    classes = sorted(set(classes))
    return words, classes, documents


def build_training_data(documents: list, words: list, classes: list, lemmatizer: WordNetLemmatizer):
    """Convert documents into bag-of-words feature vectors and one-hot label vectors."""
    training = []
    output_template = [0] * len(classes)

    for tokens, tag in documents:
        lemmatized = {lemmatizer.lemmatize(t.lower()) for t in tokens}
        bag = [1 if w in lemmatized else 0 for w in words]

        output_row = list(output_template)
        output_row[classes.index(tag)] = 1
        training.append((bag, output_row))

    random.shuffle(training)
    train_x = np.array([t[0] for t in training])
    train_y = np.array([t[1] for t in training])
    return train_x, train_y


def build_model(input_size: int, output_size: int) -> Sequential:
    """A small 3-layer feedforward classifier over the bag-of-words input."""
    model = Sequential([
        Dense(128, input_shape=(input_size,), activation="relu"),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(output_size, activation="softmax"),
    ])
    optimizer = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the intent-classification chatbot model.")
    parser.add_argument("--intents", type=Path, default=Path("intents.json"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("chatbot_model.keras"))
    args = parser.parse_args()

    ensure_nltk_data()
    lemmatizer = WordNetLemmatizer()

    intents = load_intents(args.intents)
    words, classes, documents = build_vocabulary(intents, lemmatizer)
    print(f"{len(documents)} documents, {len(classes)} classes, {len(words)} unique words")

    pickle.dump(words, open("words.pkl", "wb"))
    pickle.dump(classes, open("classes.pkl", "wb"))

    train_x, train_y = build_training_data(documents, words, classes, lemmatizer)

    model = build_model(input_size=len(words), output_size=len(classes))
    model.fit(train_x, train_y, epochs=args.epochs, batch_size=args.batch_size, verbose=1)
    model.save(args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()

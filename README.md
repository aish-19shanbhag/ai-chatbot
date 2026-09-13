# AI Chatbot

A rule-based intent-classification chatbot built with a bag-of-words feature representation and a small feedforward neural network, with a Tkinter desktop GUI.

## Overview

The bot classifies user input into a predefined set of intents (`intents.json`), then responds with a matching pre-written response. It's a lightweight, from-scratch alternative to embedding-based chatbots — the whole pipeline is a few hundred lines and trains in seconds on CPU.

## How It Works

1. **Vocabulary building**: patterns from `intents.json` are tokenized and lemmatized into a fixed vocabulary.
2. **Bag-of-words encoding**: each input sentence becomes a binary vector marking which vocabulary words are present.
3. **Classification**: a 3-layer feedforward network (128 → 64 → num_intents) predicts the most likely intent.
4. **Response selection**: a random pre-written response is chosen from the matched intent's response list.

## Tech Stack

- Python, TensorFlow/Keras, NLTK
- Tkinter for the desktop GUI

## Files

- `train_chatbot.py` — builds the vocabulary and trains the classifier
- `chatgui.py` — loads the trained model and runs the chat GUI
- `intents.json` — the intents, patterns, and responses the bot is trained on
- `words.pkl`, `classes.pkl` — vocabulary and label artifacts saved during training

## Running the Project

```bash
pip install -r requirements.txt
python train_chatbot.py
python chatgui.py
```

## Author

Aishwarya Ramanath Shanbhag

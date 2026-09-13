"""
Tkinter GUI for the intent-classification chatbot.

Loads the trained model plus its vocabulary/class artifacts, and
provides a simple chat window for interacting with the bot.
"""

import json
import pickle
import random
import tkinter as tk
from pathlib import Path

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

ERROR_THRESHOLD = 0.25


class ChatbotEngine:
    """Wraps the trained model and vocabulary to turn raw text into a bot response."""

    def __init__(self, model_path: Path, intents_path: Path, words_path: Path, classes_path: Path):
        self.model = load_model(model_path)
        self.intents = json.loads(open(intents_path, encoding="utf-8").read())
        self.words = pickle.load(open(words_path, "rb"))
        self.classes = pickle.load(open(classes_path, "rb"))
        self.lemmatizer = WordNetLemmatizer()

    def _bag_of_words(self, sentence: str) -> np.ndarray:
        tokens = nltk.word_tokenize(sentence)
        tokens = [self.lemmatizer.lemmatize(t.lower()) for t in tokens]
        return np.array([1 if w in tokens else 0 for w in self.words])

    def _predict_intent(self, sentence: str) -> list[dict]:
        bag = self._bag_of_words(sentence)
        predictions = self.model.predict(np.array([bag]), verbose=0)[0]

        results = [(i, p) for i, p in enumerate(predictions) if p > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return [{"intent": self.classes[i], "probability": str(p)} for i, p in results]

    def respond(self, message: str) -> str:
        predictions = self._predict_intent(message)
        if not predictions:
            return "I'm not sure I understand — could you rephrase that?"

        tag = predictions[0]["intent"]
        for intent in self.intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
        return "I'm not sure I understand — could you rephrase that?"


class ChatWindow(tk.Frame):
    """A minimal chat window: message log, input box, and send button."""

    def __init__(self, master: tk.Tk, engine: ChatbotEngine):
        super().__init__(master)
        self.engine = engine
        self._build_ui()

    def _build_ui(self):
        self.chat_log = tk.Text(self, bd=0, bg="white", height=8, width=50, font="Arial")
        self.chat_log.config(state=tk.DISABLED)

        scrollbar = tk.Scrollbar(self, command=self.chat_log.yview)
        self.chat_log["yscrollcommand"] = scrollbar.set

        self.entry_box = tk.Text(self, bd=0, bg="white", width=29, height=5, font="Arial")
        self.entry_box.bind("<Return>", self._on_send)

        send_button = tk.Button(
            self, font=("Verdana", 12, "bold"), text="Send", width=12, height=5,
            bd=0, bg="#32de97", activebackground="#3c9d9b", fg="#ffffff",
            command=self._on_send,
        )

        scrollbar.place(x=376, y=6, height=386)
        self.chat_log.place(x=6, y=6, height=386, width=370)
        self.entry_box.place(x=128, y=401, height=90, width=265)
        send_button.place(x=6, y=401, height=90)

    def _on_send(self, event=None):
        message = self.entry_box.get("1.0", "end-1c").strip()
        self.entry_box.delete("0.0", tk.END)
        if not message:
            return

        self.chat_log.config(state=tk.NORMAL)
        self.chat_log.insert(tk.END, f"You: {message}\n\n")

        response = self.engine.respond(message)
        self.chat_log.insert(tk.END, f"Bot: {response}\n\n")

        self.chat_log.config(state=tk.DISABLED)
        self.chat_log.yview(tk.END)


def main():
    base_dir = Path(__file__).parent
    model_path = base_dir / "chatbot_model.keras"
    if not model_path.exists():
        model_path = base_dir / "chatbot_model.h5"  # fall back to a legacy-trained model

    engine = ChatbotEngine(
        model_path=model_path,
        intents_path=base_dir / "intents.json",
        words_path=base_dir / "words.pkl",
        classes_path=base_dir / "classes.pkl",
    )

    root = tk.Tk()
    root.title("AI Chatbot")
    root.geometry("400x500")
    root.resizable(width=False, height=False)

    ChatWindow(root, engine).pack(fill=tk.BOTH, expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()

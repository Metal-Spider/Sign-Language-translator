"""
Building sentences from recognised signs.

- Word signs (e.g. HELLO) are added as whole words.
- Single-character signs (A, B, 7...) are fingerspelled: they join up into
  the word in progress until a word sign or #SPACE finishes it.
- Control signs (#SPACE, #DEL, #CLEAR, #SPEAK, #OK) edit the sentence.
"""

import os

from . import config as cfg

_WORDS_FILE = os.path.join(os.path.dirname(__file__), "data", "common_words.txt")


def load_common_words():
    try:
        with open(_WORDS_FILE) as f:
            return [w.strip().upper() for w in f if w.strip() and not w.startswith("#")]
    except OSError:
        return []


def is_letter_sign(label):
    return len(label) == 1 and label.isalnum()


class Suggester:
    """Suggests completions for a word being fingerspelled."""

    def __init__(self, common_words=None):
        words = common_words if common_words is not None else load_common_words()
        n = max(len(words), 1)
        # earlier in the list = more common = higher score
        self.common = {w: 1.0 - i / n for i, w in enumerate(words)}

    def suggest(self, prefix, sign_labels=(), history=None, k=cfg.NUM_SUGGESTIONS):
        if not prefix:
            return []
        prefix = prefix.upper()
        history = history or {}
        scores = {}
        for w, s in self.common.items():
            if w.startswith(prefix):
                scores[w] = s
        for label in sign_labels:
            if label.startswith(prefix) and not label.startswith("#") and len(label) > 1:
                scores[label] = scores.get(label, 0) + 2.0
        for w, count in history.items():
            if w.startswith(prefix):
                scores[w] = scores.get(w, 0) + 3.0 + count
        ranked = sorted(scores, key=lambda w: (-scores[w], len(w), w))
        return [w for w in ranked if w != prefix][:k]


class Sentence:
    def __init__(self):
        self.words = []
        self.spelling = ""          # fingerspelled word in progress

    def text(self):
        return " ".join(self.words + ([self.spelling] if self.spelling else []))

    def commit_spelling(self):
        """Finish the word being spelled. Returns it (or None)."""
        if not self.spelling:
            return None
        word, self.spelling = self.spelling, ""
        self.words.append(word)
        return word

    def delete_last(self):
        if self.spelling:
            self.spelling = self.spelling[:-1]
        elif self.words:
            self.words.pop()

    def clear(self):
        self.words, self.spelling = [], ""

    def accept(self, word):
        """Replace the word being spelled with a suggestion."""
        self.spelling = ""
        self.words.append(word)
        return word

    def add(self, label, suggestions=()):
        """
        Apply a recognised sign. Returns (action, text):
          ("word", w)    a whole word was added (or a spelled word finished)
          ("letter", c)  a letter was added to the word being spelled
          ("speak", s)   the user asked to speak the sentence
          ("edit", "")   the sentence was edited (delete / clear)
          ("none", "")   nothing happened
        """
        if label == "#SPACE":
            word = self.commit_spelling()
            return ("word", word) if word else ("none", "")
        if label == "#DEL":
            self.delete_last()
            return "edit", ""
        if label == "#CLEAR":
            self.clear()
            return "edit", ""
        if label == "#SPEAK":
            self.commit_spelling()
            return "speak", self.text()
        if label == "#OK":
            if self.spelling and suggestions:
                return "word", self.accept(suggestions[0])
            word = self.commit_spelling()
            return ("word", word) if word else ("none", "")
        if label.startswith("#"):
            return "none", ""
        if is_letter_sign(label):
            self.spelling += label
            return "letter", label
        self.commit_spelling()
        self.words.append(label)
        return "word", label

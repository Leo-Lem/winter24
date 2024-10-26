from collections import Counter
from spacy import load


class Vocabulary:
    """ Vocabulary object to store the mapping between words and indices. """

    def __init__(self, threshold: int, sentences: list[str] = [], spacy_model: str = "en_core_web_sm"):
        self.threshold = threshold
        self.id_to_token = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.spacy_model = load(spacy_model)

        self._learn(sentences)

    def __len__(self) -> int: return len(self.id_to_token)

    @property
    def token_to_id(self) -> dict[str, int]:
        return {v: k for k, v in self.id_to_token.items()}

    def numericalize(self, text: str) -> list[int]:
        """ For each token in the text, return the index in the vocabulary. """
        token_to_id = self.token_to_id
        return [token_to_id[token] if token in token_to_id else token_to_id["<UNK>"]
                for token in self._tokenize(text)]

    def _learn(self, sentences: list[str]):
        """ Build the vocabulary from the sentences. """
        freq = Counter()
        for sentence in sentences:
            for token in self._tokenize(sentence):
                freq[token] += 1

                if token not in self.id_to_token.values():
                    self.id_to_token[len(self.id_to_token)] = token

    def _tokenize(self, text: str) -> list[str]:
        """ Tokenize the text using spacy model. """
        return [token.text.lower()
                for token in self.spacy_model.tokenizer(text)]

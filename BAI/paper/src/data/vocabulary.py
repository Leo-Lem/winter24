from collections import Counter
from importlib import import_module
from spacy import Language
from spacy.cli import download


class Vocabulary:
    """ Vocabulary object to store the mapping between words and indices. """

    def __init__(self, sentences: list[str], threshold: int, spacy_model_name: str = "en_core_web_sm"):
        """ Initialize the vocabulary and learn the words from the sentences.

        Args:
            threshold: The minimum frequency for a word to be included in the vocabulary.
            sentences: A list of sentences to learn the vocabulary from.
            spacy_model_name: The name of the spaCy model to use for tokenization.
        """

        self.id_to_token = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.threshold = threshold
        self.spacy_model = self._download_and_init(spacy_model_name)

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

    def _download_and_init(self, model_name: str) -> Language:
        """ Load a spaCy model, download it if it has not been installed yet.

        Args:
            model_name: The name of the spaCy model to load.
        """
        try:
            model_module = import_module(model_name)
        except ModuleNotFoundError:
            download(model_name)
            model_module = import_module(model_name)

        return model_module.load()

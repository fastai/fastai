import pickle
from pathlib import Path
from typing import Iterable, Sequence, Union, Tuple

import torch
from fastai.text.learner import RNNLearner
from fastai.text.models import SequentialRNN
from fastai.text.transform import Tokenizer, Vocab, PAD


class LanguageModelPredictor:
    """Wrapper on a fastai language model and additional artifacts, useful for sentence probability inference."""
    def __init__(self, tokenizer: Tokenizer, vocabulary: Vocab, model: SequentialRNN):
        self._tokenizer = tokenizer
        self._vocabulary = vocabulary
        self._model = model
        self._model.eval()
        self._pad = self._vocabulary.stoi[PAD]

    @staticmethod
    def from_learner(learn: RNNLearner) -> 'LanguageModelPredictor':
        """Create new inferer and initialize it from a learner instance."""
        return LanguageModelPredictor(
            tokenizer=learn.data.train_ds.tokenizer,
            vocabulary=learn.data.train_ds.vocab,
            model=learn.model
        )

    @staticmethod
    def from_pickle(path: Union[Path, str]) -> 'LanguageModelPredictor':
        """Load the model with it's preprocessing pipeline from file."""
        with Path(path).open('rb') as f:
            return pickle.load(f)

    def loglikelihood(self, sentences: Iterable[str], length_norm=False) -> torch.FloatTensor:
        """Return a tensor of -log-probability for each sentence."""
        tokenized = self._tokenizer._process_all_1(sentences)
        num_sentences = len(tokenized)

        numericalized, mask = self._numericalize(tokenized)

        predictions, _, __ = self._model(numericalized)

        word_indices = numericalized.reshape(-1)
        word_logprobs = torch.nn.functional.log_softmax(
            predictions[range(len(word_indices)), word_indices].reshape(num_sentences, -1),
            dim=1
        )

        sentence_logprobs = (word_logprobs * mask).sum(dim=1)

        if length_norm:
            sentence_logprobs = sentence_logprobs / _maybe_cuda(torch.FloatTensor(list(map(len, tokenized))))

        return sentence_logprobs

    def perplexity(self, sentences: Iterable[str]) -> torch.FloatTensor:
        """Return a tensor of perplexity for each sentence."""
        return -self.loglikelihood(sentences, length_norm=True)

    def select_most_likely(self, sentences: Iterable[str], length_norm=False) -> str:
        """Select the most likely sentence."""
        sentences = list(sentences)
        logprobs = self.loglikelihood(sentences, length_norm=length_norm)
        return sentences[int(logprobs.argmax())]

    def save(self, path: Union[Path, str]):
        """Serialize the model and preprocessing pipeline to file."""
        with path.open('wb') as f:
            pickle.dump(self, file=f)

    def _numericalize(self, tokens: Iterable[Sequence[str]]) -> Tuple[torch.LongTensor, torch.LongTensor]:
        numericalized = [self._vocabulary.numericalize(tok) for tok in tokens]

        max_len = max(map(len, numericalized))
        pad_per_sequence = [max_len - len(seq) for seq in numericalized]

        padded_numericalized = _maybe_cuda(torch.LongTensor([
            seq + [self._pad] * num_pads
            for seq, num_pads in zip(numericalized, pad_per_sequence)
        ]))
        padding_mask = _maybe_cuda(torch.tensor([
            [1.] * len(seq) + [0.] * num_pads
            for seq, num_pads in zip(numericalized, pad_per_sequence)
        ]))

        return padded_numericalized, padding_mask


def _maybe_cuda(tensor: torch.Tensor) -> torch.Tensor:
    """Helper to allocate the tensor on the GPU if available, else on the CPU."""
    return tensor.cuda() if torch.cuda.device_count() >= 1 else tensor
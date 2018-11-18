import pickle
from pathlib import Path
from typing import Iterable, Sequence, Union

import torch

from fastai.torch_core import  maybe_cuda_alloc
from fastai.data_block import PreProcessor
from fastai.text import TokenizeProcessor, NumericalizeProcessor
from fastai.text.learner import RNNLearner
from fastai.text.models import SequentialRNN


__all__ = ['LanguageModelPredictor']


class LanguageModelPredictor:
    """Wrapper on a fastai language model and additional artifacts, useful for sentence probability inference."""
    def __init__(self, preprocessors: Sequence[PreProcessor], model: SequentialRNN):
        self._tokenizer = [p for p in preprocessors if isinstance(p, TokenizeProcessor)][0]
        self._numericalizer = [p for p in preprocessors if isinstance(p, NumericalizeProcessor)][0]
        self._model = model
        self._model.eval()

    @staticmethod
    def from_learner(learn: RNNLearner) -> 'LanguageModelPredictor':
        """Create new inferer and initialize it from a learner instance."""
        return LanguageModelPredictor(
            learn.data.processor,
            model=learn.model
        )

    @staticmethod
    def from_pickle(path: Union[Path, str]) -> 'LanguageModelPredictor':
        """Load the model with it's preprocessing pipeline from file."""
        with Path(path).open('rb') as f:
            return pickle.load(f)

    def loglikelihood(self, sentences: Sequence[str], length_norm=False) -> torch.FloatTensor:
        """Return a tensor of -log-probability for each sentence."""
        tokenized = self._tokenizer.process_batch(sentences)
        numericalized, mask = self._numericalizer.process_batch(tokenized)

        predictions, _, __ = self._model(numericalized)

        num_sentences = len(tokenized)
        word_indices = numericalized.reshape(-1)
        word_logprobs = torch.nn.functional.log_softmax(
            predictions[range(len(word_indices)), word_indices].reshape(num_sentences, -1),
            dim=1
        )

        sentence_logprobs = (word_logprobs * mask).sum(dim=1)

        if length_norm:
            sentence_logprobs = sentence_logprobs / maybe_cuda_alloc(torch.FloatTensor(list(map(len, tokenized))))

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

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from onnxruntime import InferenceSession
from tokenizers import Tokenizer

from src.core.exception import InvalidInputError
from src.core.sentiment_analyzer import SentimentAnalyzer


@pytest.fixture
def mock_embedder():
    yield MagicMock()


@pytest.fixture
def mock_classifier():
    yield MagicMock()


@pytest.fixture
def mock_tokenizer():
    yield MagicMock(spec=Tokenizer)


@pytest.fixture
def sentiment_analyzer(
    mock_embedder: InferenceSession,
    mock_classifier: InferenceSession,
    mock_tokenizer: Tokenizer,
):
    yield SentimentAnalyzer(mock_embedder, mock_classifier, mock_tokenizer)


def test_predict_with_empty_input(sentiment_analyzer):
    with pytest.raises(InvalidInputError, match="Text cannot be empty"):
        sentiment_analyzer.predict("")
    with pytest.raises(InvalidInputError, match="Text cannot be empty"):
        sentiment_analyzer.predict("   ")
    with pytest.raises(InvalidInputError, match="Text cannot be empty"):
        sentiment_analyzer.predict(None)


def test_predict_negative_label(
    sentiment_analyzer, mock_embedder, mock_classifier, mock_tokenizer
):
    text = "Im very sad"

    # tokenizer zwraca obiekt z ids i attention_mask
    mock_encoded = SimpleNamespace(ids=[1, 2, 3], attention_mask=[1, 1, 1])
    mock_tokenizer.encode.return_value = mock_encoded

    # embedder.run zwraca macierz embeddings
    mock_embedder.run.return_value = [np.array([[-1.0] * 10])]

    # classifier.get_inputs zwraca input name
    mock_input = SimpleNamespace(name="input")
    mock_classifier.get_inputs.return_value = [mock_input]

    # classifier.run zwraca etykietę 0 → "negative"
    mock_classifier.run.return_value = [[0]]

    # when
    result = sentiment_analyzer.predict(text)

    # then
    assert result == "negative"
    mock_tokenizer.encode.assert_called_once_with(text)
    mock_embedder.run.assert_called_once()
    mock_classifier.run.assert_called_once()


def test_predict_neutral_label(
    sentiment_analyzer, mock_embedder, mock_classifier, mock_tokenizer
):
    text = "Im neutral"

    mock_encoded = SimpleNamespace(ids=[1, 2, 3], attention_mask=[1, 1, 1])
    mock_tokenizer.encode.return_value = mock_encoded
    mock_embedder.run.return_value = [np.array([[0.0] * 10])]
    mock_input = SimpleNamespace(name="input")
    mock_classifier.get_inputs.return_value = [mock_input]
    mock_classifier.run.return_value = [[1]]  # label 1 → "neutral"

    result = sentiment_analyzer.predict(text)
    assert result == "neutral"
    mock_tokenizer.encode.assert_called_once_with(text)
    mock_embedder.run.assert_called_once()
    mock_classifier.run.assert_called_once()


def test_predict_positive_label(
    sentiment_analyzer, mock_embedder, mock_classifier, mock_tokenizer
):
    text = "Im happy"

    mock_encoded = SimpleNamespace(ids=[1, 2, 3], attention_mask=[1, 1, 1])
    mock_tokenizer.encode.return_value = mock_encoded
    mock_embedder.run.return_value = [np.array([[1.0] * 10])]
    mock_input = SimpleNamespace(name="input")
    mock_classifier.get_inputs.return_value = [mock_input]
    mock_classifier.run.return_value = [[2]]  # label 2 → "positive"

    result = sentiment_analyzer.predict(text)
    assert result == "positive"
    mock_tokenizer.encode.assert_called_once_with(text)
    mock_embedder.run.assert_called_once()
    mock_classifier.run.assert_called_once()

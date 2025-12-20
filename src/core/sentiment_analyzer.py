import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from src.core.exception import InvalidInputError


class SentimentAnalyzer:
    def __init__(
        self,
        embedding_session: ort.InferenceSession,
        classifier_session: ort.InferenceSession,
        tokenizer: Tokenizer,
    ):
        self.embedding_session = embedding_session
        self.classifier_session = classifier_session
        self.tokenizer = tokenizer

    def predict(self, text: str) -> str:
        if text is None or text.strip() == "":
            raise InvalidInputError("Text cannot be empty")

        encoded = self.tokenizer.encode(text)

        input_ids = np.array([encoded.ids])
        attention_mask = np.array([encoded.attention_mask])

        embedding_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        embeddings = self.embedding_session.run(None, embedding_inputs)[0]
        embeddings = np.asarray(embeddings)

        classifier_input_name = self.classifier_session.get_inputs()[0].name
        classifier_inputs = {classifier_input_name: embeddings.astype(np.float32)}
        prediction = self.classifier_session.run(None, classifier_inputs)[0]

        return self._map_label(prediction[0])

    def _map_label(self, label: int) -> str:
        if label == 0:
            return "negative"
        elif label == 1:
            return "neutral"
        elif label == 2:
            return "positive"
        else:
            return "unknown"

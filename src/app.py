from fastapi import FastAPI, HTTPException
from mangum import Mangum
from onnxruntime import InferenceSession
from tokenizers import Tokenizer

from src.api.models.sentiment_analysis import SentimentCommand, SentimentResponse
from src.core.exception import InvalidInputError
from src.core.sentiment_analyzer import SentimentAnalyzer
from src.utils.config import Settings
from src.utils.logger import log

settings = Settings()

app = FastAPI()
handler = Mangum(app)

embedding_session = InferenceSession(settings.ONNX_EMBEDDING_MODEL_PATH)
classifier_session = InferenceSession(settings.ONNX_CLASSIFIER_PATH)
tokenizer = Tokenizer.from_file(settings.TOKENIZER_PATH)

analyzer = SentimentAnalyzer(embedding_session, classifier_session, tokenizer)


@app.get("/health")
async def health_check():
    return {"status": "Running!"}


@app.post("/predict")
async def predict_sentiment(
    command: SentimentCommand,
) -> SentimentResponse:
    log.debug(f"Analyzing sentiment for text: {command.text}")
    try:
        prediction = analyzer.predict(command.text)
        return SentimentResponse(prediction=prediction)
    except InvalidInputError as e:
        log.error(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail=str(e))

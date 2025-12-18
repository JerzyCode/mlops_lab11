from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ENVIRONMENT: str = "prod"
    CLASSIFIER_PATH: str = "model/classifier.joblib"
    ENBEDDER_PATH: str = "model/sentence_transformer.model"

    S3_BUCKET_NAME: str = "jerzyb-s3-bucket-mlops-lab"
    S3_CLASSIFIER_KEY: str = "mlops_model/classifier.joblib"
    S3_EMBEDDER_KEY: str = "mlops_model/sentence_transformer.model"

    CLASSIFIER_JOBLIB_PATH: str = "model/classifier.joblib"
    SENTENCE_TRANSFORMER_DIR: str = "model/sentence_transformer"
    TOKENIZER_PATH: str = "model/tokenizer"

    ONNX_EMBEDDING_MODEL_PATH: str = "model/onnx/sentence_transformer.onnx"
    ONNX_CLASSIFIER_PATH: str = "model/onnx/classifier.onnx"

    EMBEDDING_DIM: int = 384

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        if value not in ("test", "prod"):
            raise ValueError("ENVIRONMENT must be one of: test, prod")
        return value

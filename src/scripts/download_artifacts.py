import os

import boto3

from src.utils.config import Settings


def download_artifacts(settings: Settings):
    s3 = boto3.client("s3")

    # ensure local directories exist
    os.makedirs(os.path.dirname(settings.CLASSIFIER_JOBLIB_PATH), exist_ok=True)
    os.makedirs(settings.SENTENCE_TRANSFORMER_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(settings.ONNX_CLASSIFIER_PATH), exist_ok=True)

    print("Downloading classifier from S3...")
    try:
        s3.download_file(
            Bucket=settings.S3_BUCKET_NAME,
            Key=settings.S3_CLASSIFIER_KEY,
            Filename=settings.CLASSIFIER_JOBLIB_PATH,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download classifier: {e}")

    print("Downloading sentence transformer from S3...")
    try:
        s3.download_file(
            Bucket=settings.S3_BUCKET_NAME,
            Key=settings.S3_EMBEDDER_KEY,
            Filename=settings.ENBEDDER_PATH,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download embedder: {e}")

    print("Artifacts downloaded successfully.")

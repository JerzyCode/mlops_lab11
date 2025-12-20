import os

import boto3

from src.utils.config import Settings


def download_artifacts(settings: Settings):
    s3 = boto3.client("s3")

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
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=settings.S3_BUCKET_NAME, Prefix=settings.S3_EMBEDDER_KEY
        ):
            if "Contents" in page:
                for obj in page["Contents"]:
                    s3_key = obj["Key"]
                    if s3_key.endswith("/"):
                        continue

                    relative_path = s3_key.replace(settings.S3_EMBEDDER_KEY, "").lstrip(
                        "/"
                    )
                    local_file = os.path.join(
                        settings.SENTENCE_TRANSFORMER_DIR, relative_path
                    )

                    os.makedirs(os.path.dirname(local_file), exist_ok=True)
                    print(f"Downloading {s3_key} to {local_file}")
                    s3.download_file(settings.S3_BUCKET_NAME, s3_key, local_file)

        print(f"Sentence transformer downloaded to {settings.SENTENCE_TRANSFORMER_DIR}")
    except Exception as e:
        raise RuntimeError(f"Failed to download embedder: {e}")

    print("Artifacts downloaded successfully.")


if __name__ == "__main__":
    settings = Settings()
    download_artifacts(settings)

# export_classifier_to_onnx.py

import joblib
from settings import Settings
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def export_classifier_to_onnx(settings: Settings):
    print(f"Loading classifier from {settings.CLASSIFIER_JOBLIB_PATH}...")
    classifier = joblib.load(settings.CLASSIFIER_JOBLIB_PATH)

    # define input shape: (batch_size, embedding_dim)
    initial_type = [("float_input", FloatTensorType([None, settings.EMBEDDING_DIM]))]

    print("Converting to ONNX...")
    onnx_model = convert_sklearn(
        classifier,
        initial_types=initial_type,
        target_opset=13,  # safe default, adjust if needed
    )
    print(f"Saving ONNX model to {settings.ONNX_CLASSIFIER_PATH}...")

    # TODO: save the onnx_model to settings.ONNX_CLASSIFIER_PATH
    with open(settings.ONNX_CLASSIFIER_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())

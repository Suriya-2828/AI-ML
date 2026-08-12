import os
import sys
import logging
import numpy as np
import torch
import onnx
import tensorflow as tf

from model_definition import SimpleCNN


# -----------------------------
# Logging Configuration
# -----------------------------

def log_message(message):
    print(message)
    logging.info(message)


# -----------------------------
# File Paths
# -----------------------------
MODEL_PATH = "model.pth"
CALIB_DIR = "calib"
ONNX_PATH = "model.onnx"
SAVED_MODEL_DIR = "saved_model"
TFLITE_PATH = "model_int8.tflite"


# -----------------------------
# Load PyTorch Model
# -----------------------------
def load_model():

    log_message("Loading PyTorch model...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("model.pth not found.")

    model = SimpleCNN()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    log_message("Model loaded successfully.")

    return model


# -----------------------------
# Validate Calibration Files
# -----------------------------
def validate_calibration():

    log_message("Checking calibration data...")

    if not os.path.exists(CALIB_DIR):
        raise FileNotFoundError("Calibration folder not found.")

    files = sorted(
        [
            f for f in os.listdir(CALIB_DIR)
            if f.endswith(".npy")
        ]
    )

    if len(files) == 0:
        raise ValueError("Calibration folder is empty.")

    samples = []

    for file in files:

        path = os.path.join(CALIB_DIR, file)

        data = np.load(path)

        if data.shape != (1, 28, 28):
            raise ValueError(
                f"{file} has invalid shape {data.shape}"
            )

        if np.isnan(data).any():
            raise ValueError(
                f"{file} contains NaN values."
            )

        if np.isinf(data).any():
            raise ValueError(
                f"{file} contains Inf values."
            )

        samples.append(data.astype(np.float32))

    log_message(
        f"{len(samples)} calibration files validated."
    )

    return samples


# -----------------------------
# Export ONNX
# -----------------------------
def export_onnx(model):

    log_message("Exporting ONNX model...")

    dummy = torch.randn(
        1,
        1,
        28,
        28
    )

    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"]
    )

    log_message("ONNX export completed.")


# -----------------------------
# Verify ONNX
# -----------------------------
def verify_onnx():

    log_message("Checking ONNX model...")

    model = onnx.load(ONNX_PATH)

    onnx.checker.check_model(model)

    log_message("ONNX model is valid.")

# -----------------------------
# Convert ONNX to TensorFlow
# -----------------------------
def convert_to_tensorflow():

    log_message("Converting ONNX model to TensorFlow SavedModel...")

    try:
        from onnx_tf.backend import prepare
    except ImportError:
        raise ImportError(
            "onnx-tf is not installed.\n"
            "Install using: pip install onnx-tf"
        )

    onnx_model = onnx.load(ONNX_PATH)

    tf_rep = prepare(onnx_model)

    tf_rep.export_graph(SAVED_MODEL_DIR)

    log_message("TensorFlow SavedModel created successfully.")


# -----------------------------
# Representative Dataset
# -----------------------------
def representative_dataset(calibration_samples):

    for sample in calibration_samples:

        sample = np.expand_dims(sample, axis=0)

        sample = sample.astype(np.float32)

        yield [sample]


# -----------------------------
# Convert to INT8 TFLite
# -----------------------------
def convert_to_tflite(calibration_samples):

    log_message("Converting TensorFlow model to INT8 TFLite...")

    converter = tf.lite.TFLiteConverter.from_saved_model(
        SAVED_MODEL_DIR
    )

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    converter.representative_dataset = lambda: representative_dataset(
        calibration_samples
    )

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = tf.int8

    converter.inference_output_type = tf.int8

    try:

        tflite_model = converter.convert()

    except Exception as e:

        raise RuntimeError(
            "TFLite conversion failed:\n" + str(e)
        )

    with open(TFLITE_PATH, "wb") as file:

        file.write(tflite_model)

    log_message("INT8 TFLite model saved successfully.")


# -----------------------------
# Check for Floating Point Fallback
# -----------------------------
def check_quantization():

    interpreter = tf.lite.Interpreter(
        model_path=TFLITE_PATH
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()

    output_details = interpreter.get_output_details()

    input_dtype = input_details[0]["dtype"]

    output_dtype = output_details[0]["dtype"]

    if input_dtype != np.int8:

        raise RuntimeError(
            "Input tensor is not INT8."
        )

    if output_dtype != np.int8:

        raise RuntimeError(
            "Output tensor is not INT8."
        )

    log_message("Verified Fully INT8 Model.")

# -----------------------------
# Verify TFLite Model
# -----------------------------
def verify_tflite():

    log_message("Verifying TFLite model...")

    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\n========== TFLite Model Information ==========")

    print("Input Tensor Dtype :", input_details[0]["dtype"])
    print("Output Tensor Dtype:", output_details[0]["dtype"])

    print("\nInput Scale and Zero Point :")
    print(input_details[0]["quantization"])

    print("\nOutput Scale and Zero Point :")
    print(output_details[0]["quantization"])

    model_size = os.path.getsize(TFLITE_PATH) / 1024

    print("\nModel Size : {:.2f} KiB".format(model_size))

    input_shape = input_details[0]["shape"]

    sample = np.random.randint(
        -128,
        127,
        size=input_shape,
        dtype=np.int8
    )

    interpreter.set_tensor(
        input_details[0]["index"],
        sample
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )

    print("\nInference Successful")
    print("Output Shape :", output.shape)
    print("Output Values :")
    print(output)

    log_message("TFLite verification completed.")


# -----------------------------
# Main Function
# -----------------------------
def main():

    try:

        model = load_model()

        calibration_samples = validate_calibration()

        export_onnx(model)

        verify_onnx()

        convert_to_tensorflow()

        convert_to_tflite(calibration_samples)

        check_quantization()

        verify_tflite()

        log_message("Conversion Pipeline Completed Successfully.")

        print("\n======================================")
        print("Task Completed Successfully")
        print("Generated Files:")
        print("1. model.onnx")
        print("2. saved_model/")
        print("3. model_int8.tflite")
        print("4. conversion_log.txt")
        print("======================================")

    except Exception as error:

        log_message(str(error))

        print("\nERROR")
        print(error)

        sys.exit(1)


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":

    main()

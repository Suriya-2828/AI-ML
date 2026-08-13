# Manufacturing Defect Detection

## Objective

To build a simple machine learning pipeline for classifying manufacturing defects from images.

## Dataset

NEU Surface Defect Dataset

Classes:
- Crazing
- Inclusion
- Patches
- Pitted
- Rolled
- Scratches

## Pipeline

1. Load images
2. Resize images to 64 × 64
3. Normalize pixel values
4. Split dataset into training and testing sets
5. Train a K-Nearest Neighbors classifier
6. Evaluate the model
7. Save the trained model

## Deployment Considerations

- Save the model using Joblib.
- Load the saved model in a web application.
- Accept uploaded images.
- Resize the image and predict the defect type.

## Conclusion

The pipeline successfully demonstrates preprocessing, training, evaluation, and model saving for manufacturing defect detection.

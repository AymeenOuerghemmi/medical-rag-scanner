Place your fine-tuned weights at: models/model.pt

The architecture expected is a ResNet18 with the final layer adapted to 4 classes:
['normal', 'pneumonia', 'covid19', 'pulmonary_embolism']

You can fine-tune using the provided train_stub.py as a starting point.

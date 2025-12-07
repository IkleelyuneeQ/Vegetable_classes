#🥦 Vegetable Classification Using ResNet-50

This project focuses on building a vegetable image classification model using PyTorch and ResNet-50. The dataset contains images of 15 different vegetables, divided into train, validation, and test folders. The model learns to classify images into one of these categories after preprocessing, normalization, and training.

## Requirents
pip install torch torchvision matplotlib pandas tqdm torchinfo


Vegetable_Images/
└── Vegetable Images/
    ├── train/
    ├── validation/
    └── test/


train/
  ├── Tomato/
  ├── Potato/
  ├── Cabbage/
  └── ...


classes = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 
 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 
 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']


## Workflow Summary
1️⃣ Load Dataset

The dataset is loaded using torchvision.datasets.ImageFolder.
Images are converted to RGB format and resized to 224 × 224.

2️⃣ Transformation Steps

Convert grayscale images to RGB

Resize to 224×224

Convert to PyTorch tensors

Compute dataset mean and standard deviation

Normalize images

Normalization helps model convergence.

## Models loaded from
./resnet50-11ad3fa6.pth

## Final Output Expectations

Once extended to training phase, you should get:

🔹 Training accuracy & loss curves
🔹 Validation accuracy trends
🔹 Test performance matrix

Typical evaluation metrics:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix


This work was built to experiment with transfer learning for vegetable image classification.

# Skin Disease Classification using ResNet

## Overview
This project trains a **ResNet** model for skin disease classification. The model is trained using PyTorch and torchvision, achieving high validation accuracy.

## Installation and Dependencies
Ensure you have the following dependencies installed:
```bash
pip install torch torchvision numpy matplotlib
```
Or, if using Google Colab, they are pre-installed.

## Model Training Process
- The **ResNet** model is loaded with pretrained weights.
- Training is performed for **10 epochs**.
- The model's accuracy and loss improve significantly over epochs.
- The trained model is saved as `model.pth`.
- Misclassified images from validation and test sets are saved in `misclassified` and `misclassified_test` folders.



## Training Results
| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
|-------|------------|---------------|---------|--------------|
| 1     | 1.0429     | 60.22%        | 0.9287  | 64.40%       |
| 2     | 0.8307     | 68.06%        | 0.7443  | 70.86%       |
| 3     | 0.7433     | 71.74%        | 0.7279  | 72.01%       |
| 4     | 0.6692     | 75.05%        | 0.5504  | 80.07%       |
| 5     | 0.5963     | 77.62%        | 0.4982  | 82.19%       |
| 6     | 0.5101     | 81.18%        | 0.3993  | 85.68%       |
| 7     | 0.4158     | 84.80%        | 0.2574  | 91.69%       |
| 8     | 0.3175     | 88.69%        | 0.2025  | 93.07%       |
| 9     | 0.2269     | 91.95%        | 0.1356  | 95.37%       |
| 10    | 0.1669     | 94.22%        | 0.1160  | 96.01%       |

## Model Evaluation
- The trained model is evaluated on the test set.
- Misclassified test samples are saved in `misclassified_test`.



## Future Improvements
- Fine-tune the model on a larger dataset.
- Implement more robust augmentation techniques.
- Use more advanced architectures like EfficientNet.

---
For any issues, feel free to raise a query!


# skin_diseases_prediction
This repository contains a skin diseases prediction using CNN (resnet)

# **data sourses:**
- https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset
- https://www.kaggle.com/datasets/xuannguyenuet2004/skin-disease-dataset
             

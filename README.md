# Human Emotion Detection from Face Images

## Overview
This project explores **Human Emotion Detection from Face Images** using deep learning techniques. Our research focuses on detecting emotions from facial images, utilizing several pretrained Convolutional Neural Network (CNN) models to identify expressions with high accuracy. Key components include data preprocessing, feature extraction, and model fine-tuning.

## Authors
- **Rohan Bali** - BTech Data Science, NIIT University
- **Shweta Singh** - BTech Artificial Intelligence, NIIT University
- **Tanu** - BTech Big Data Engineering, NIIT University

**Supervisor:** Dr. Santanu Roy

## Abstract
Emotion detection from facial expressions has applications in various fields like human-computer interaction, security, and psychological research. This study leverages CNN-based deep learning models trained on pre-existing image datasets to accurately classify emotions under challenging, real-world conditions. Using data augmentation and transfer learning techniques, we improve the model's generalization and accuracy.

## Datasets
We used two prominent datasets for this research:
1. **FER2013**: Contains 35,887 grayscale images (48x48 pixels) labeled into seven emotion categories: angry, disgust, fear, happy, sad, surprise, and neutral.
   - [FER2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

2. **Human Emotions Dataset**: Contains 9,077 grayscale images divided into three emotion categories: angry, happy, and sad.
   - [Human Emotions Dataset on Kaggle](https://www.kaggle.com/datasets/muhammadhananasghar/human-emotions-datasethes/data)

## Models and Results

The table below highlights the models used in this research along with their highest achieved accuracies on each dataset:

### FER2013 Dataset
| Model         | Loss  | Accuracy | Precision | Recall  | F1-Score |
|---------------|-------|----------|-----------|---------|----------|
| VGG-16        | 1.6618 | 53.68%  | 0.575     | 0.4939  | 0.4571   |
| MobileNet V2  | 0.9724 | 62.52%  | 0.6307    | 0.6252  | 0.6201   |
| ResNet-50     | 1.9174 | 57.06%  | 0.5798    | 0.5568  | 0.5777   |

### Human Emotion Dataset
| Model         | Loss   | Accuracy | Precision | Recall  | F1-Score |
|---------------|--------|----------|-----------|---------|----------|
| MobileNet V2  | 1.5631 | 75.33%   | 0.755     | 0.752   | 0.7414   |
| ResNet-50     | 0.4312 | 82.47%   | 0.8672    | 0.7814  | 0.8247   |
| Xception      | 0.3638 | 87.18%   | 0.8650    | 0.8563  | 0.8601   |

### Additional Experiments on Human Emotion Dataset (MobileNet V2)
| Configuration                    | Loss   | Accuracy | Precision | Recall  | F1-Score |
|----------------------------------|--------|----------|-----------|---------|----------|
| 50% Freezing, 50% Fine-Tuning, GAP | 0.5953 | 90.93%   | 0.9103    | 0.9082  | 0.9008   |
| 100% Fine-Tuning, GAP            | 1.5631 | 75.33%   | 0.755     | 0.752   | 0.7414   |
| 75% Freeze, GAP                  | 2.3372 | 77.44%   | 0.7746    | 0.774   | 0.7617   |

## Methodology
Our research employs a pre-trained CNN architecture for feature extraction and classification, utilizing:
- **Data Augmentation**: Enhances dataset variety with rotations, resizing, brightness changes, etc.
- **Transfer Learning**: Leverages knowledge from models pretrained on ImageNet to improve training speed and accuracy.
- **Hyperparameter Tuning**: Includes adjustments to batch normalization, layer freezing, dropout, and adaptive learning rate schedulers.

## Technology Stack
- **Deep Learning Framework**: Keras with TensorFlow backend
- **Models**: VGG-16, MobileNet V2, ResNet-50, Xception
- **Datasets**: FER2013, Human Emotions Dataset

## Future Scope
We aim to:
1. Implement advanced data augmentation techniques.
2. Experiment with novel loss functions and attention mechanisms.
3. Incorporate additional datasets to further enhance model accuracy and robustness.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **Kaggle** for providing the FER2013 and Human Emotions datasets.
- **NIIT University** for supporting this research project.


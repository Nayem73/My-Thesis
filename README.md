# How to Run

1. Download all the Eight Datasets from this link: https://drive.google.com/drive/folders/13_Fn7hBBSm-15inzodGu21oC3lULJ4Di?usp=sharing

2. Upload the datasets in this direction of google drive: Colab Notebooks/Thesis

3. upload the code in the above mentioned direction and open the code with Google Colab. Run all the cells of the code.


https://github.com/Nayem73/My-Thesis/assets/111779609/f7d9ca6a-a380-4824-98d3-54cdd0bcff70




# Brief

Remote sensing is a powerful technology that captures information about the Earth’s surface from a distance. Accurately classifying these vast amount of data in a relatively short time allows us to monitor changes in land use, assess the impact of natural disasters, and aid in urban planning and real time monitoring during emergencies.  

- We experimented on state-of-the-art popular deep learning CNNs, lightweight and not computationally demanding, with a variety of Remote Sensing image datasets.  
- We picked the top performing models and experimented further on how to Fine-Tune them with the Remote Sensing datasets.  
- We trained the most effective layers of the pre trained models with Remote Sensing datasets and evaluated their performance.  
- We picked the two most lightweight and top performing models among them and created a hybrid model to further improve the accuracy.  
- We prevented any overfitting issues by adding different combinations of end layers, which can further optimize the model.  
- For further optimization and assessing our model's capability, we experimented on Image Dimensionality Reduction methods and Image Compression techniques.  
- We used eight remote sensing image datasets of varying spatial resolutions to assess our model’s capability.  
- We merged all the datasets with a variety of preprocessing methods, resulted in a 64 class dataset and compared the results with different standalone and fine-tuned hybrid models with different assessment and measuring methods including train and test accuracy, precision, recall, f1-score, confusion matrix, loss-accuracy curve etc.  
- By comparing the results from both the individual and merged datasets with different standalone and fine-tuned hybrid models, we can decide which model can be used to accurately classify remote sensing images in a relatively shorter time duration.

# Abstract

Remote sensing is a powerful technology that captures information about the Earth’s
surface from a distance, enabling various applications such as environmental
management, urban planning, and disaster response. However, classifying remote
sensing images into meaningful categories is a challenging task, especially when
dealing with large-scale and diverse datasets. In this work, we present a hybrid model
designed for accurate classification across 64 categories of remotely sensed images,
addressing the complexity of large-scale remote sensing datasets.
The proposed model utilizes two pre-trained convolutional neural networks (CNNs),
EfficientNetB0, and ResNet50, to extract deep features from the remotely sensed land
use and land cover images.The deep features extracted from each CNN are then fused
using the concatenation technique to produce a more suitable feature set from which
the model can learn better about the dataset.
We merged four datasets (EuroSAT, UCMerced, NWPU-RESISC45, PatternNet) into
a 99,400-image, 64-class dataset. Data augmentation methods introduced variability,
enhancing image diversity and quantity, enabling intricate pattern recognition. Our
model's effectiveness was assessed by comparing it to pre-trained CNN models.
The proposed hybrid model achieves an accuracy of 95.37%, outperforming popular
pre-trained models with fine-tuning, namely EfficientNetB0 (95.02%), ResNet50
(93.90%), VGG16 (93.40%), VGG19 (92.91%), DenseNet121 (75.63%),
MobileNetV2 (61.10%). Additionally, we tested other hybrid architectures and
compared it with our proposed model. Our findings demonstrate that our proposed
model is the most optimal choice, offering superior accuracy without being
computationally intensive, compared to the other models we tested.
We fine-tuned EfficientNetB0 and ResNet50, with the last 50 and 20 layers trainable
respectively. Features extracted via global average pooling were concatenated.
Dropout layers (0.4 rate) and L2 regularization prevent overfitting. The model was
implemented on Google Colab with a T4 GPU. It uses a learning rate of 0.0001 for
the Adam optimizer and categorical cross-entropy as the loss function. Early stopping
based on validation accuracy (patience of 5 epochs) and a learning rate schedule were
used for better generalization and convergence. Despite performing image
compression using PCA and resizing images, our model’s accuracy compares well
with that of the original dataset. This reconfirms our model’s effectiveness on
large-scale remote sensing images.

![Image preview](https://raw.githubusercontent.com/Nayem73/My-Portfolio/main/assets/images/thesis.jpeg)

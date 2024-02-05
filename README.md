# Comparative Study of different CNN Architectures

Traffic Signs are easy to identify for humans. However for computer systems, its a challenging problem. Traffic Sign Classification is the process of automatically recognizing traffic signs along the road. In this study the performance of different CNN architectures is analyzed for Traffic sign classification against 3 datasets of varying classes and samples. Effects of different hyperparameters on model performance and its convergence is also studied. The trained models are evaluated using different performance metrics and a comprehensive comparison report is being generated. Careful consideration was given on the choice of datasets for the study to make sure they can represent signs of varied variety and real-world distortions and weather conditions. 

## Directory Structure

```
├── proposal/                   <- Project's Proposal
├── datasets/                   <- Placeholder for datasets used in this project
├── Traffic Sign Detection/     <- Source code for the project
├── models/                     <- Placeholder folder for trained models
├── deliverables/               <- All project deliverables
├── LICENSE                     <- Project's License
├── README.md                   <- The top-level README for developers using this project
```

## Dataset
We have choosen 3 types of traffic signs datasets (Tab. 1) that have traffic signs from varying countries to ensure that our models are robust. The
main concern while selecting the datasets was the number of images available per class as most datasets were highly skewed. traffic signs in the images exhibit limited variation or are relatively similar, we took precautions to address potential challenges or biases in the dataset. This will help reduce the time spent in the pre-processing stage. The dataset links are as follows:
- Dataset 1 - https://www.kaggle.com/datasets/sarangdilipjodh/indian-traffic-signs-prediction85-classes
- Dataset 2 - https://www.kaggle.com/datasets/saraparsaseresht/persian-traffic-sign-dataset-ptsd
- Dataset 3 - https://www.kaggle.com/datasets/daniildeltsov/traffic-signs-gtsrb-plus-162-custom-classes

| Dataset                 | No. of Images                | Classes | 
|-------------------------|------------------------------|---------|
| Dataset 1 (Indian TS)   | `2.1k:0.5k:0.7k`             | `15`    | 
| Dataset 2 (Persian TS)  | `6.3k:1.6k:1.2k`             | `12`    |
| Dataset 3 (German TS)   | `11.2k:2.8k:4.6k`            | `8`     |

<img width="500" alt="sign-samples" src="https://github.com/YuganshG/Comparative-Study-of-different-CNN-Architectures/assets/34838617/b2127f14-8c71-4a04-a9b9-31519fbdbe33">


## Methodology

In this study, 11 models, three models for each of the datasets trained from scratch and 2 models trained using transfer learning. The hyperparameters will be fixed across models to produce comparable results. Next, hyperparameters will be tuned to find the best model. Finally, the models will be visualized using t-SNE to explain model results. 

<ol type="A">
<li><b>Pre-processing & Data Augmentation</b></li>
Image samples are first preprocessed to 224x224 image size and applied with transforms of  ColorJitter with brightness = (0.5,1.2) ,  RandomHorizontalFlip, RandomAdjustSharpness and finally the image is normalized. Instead of doing weighted sampling of the images to handle the class imbalance, a weighted cost function is being used.

<li><b>Architectures</b></li>
Different backbone architectures were chosen to ensure that different types of Convolution blocks were tested for the data. 
Estimated computational complexity, top 1% accuracy, top 5% accuracy of some popular architectures trained mostly on ILSVRC 2012 were compared as follows: <br>

| Model               | GFLOPS  | Top 1% Accuracy | Top 5% Accuracy |
|---------------------|---------|------------------|-----------------|
| ShuffleNetV2 1.0x   | `0.149` | `69.55%`         | `88.92%`        |
| MobileNet V2        | `0.319` | `71.86%`         | `90.42%`        |
| ReXNet_1.0          | `0.4`   | `77.90%`         | `93.90%`        |
| AlexNet             | `0.727` | `63.30%`         | `84.60%`        |
| ResNet-18           | `1.82`  | `70.07%`         | `89.44%`        |
| VGG-11              | `7.63`  | `68.75%`         | `88.87%`        |
| RegNetX-12GF        | `12.15` | `79.67%`         | `95.03%`        |
| VGG-19              | `19.67` | `72.41%`         | `90.80%`        |
| ResNeXt-101 32x32d  | `174`   | `85.10%`         | `97.50%`        |

<b>AlexNet</b>, <b>VGG-11</b> and <b>ResNet-18 </b> were chosen finally.

| Architecture      | Learnable Params (Mil.) | Layers  | GFlops       |
|-------------------|-------------------------|---------|--------------|
| AlexNet           | `57.05±0.2`             | `8`     | `0.71`       |
| VGG-11            | `128.82±0.02`           | `11`    | `7.63`       |
| ResNet-18         | `11.18`                 | `18`    | `1.83`       |

</ol>

## Experimental Setup

Image samples are first preprocessed to 224x224 image size and applied with transforms of ColorJitter with brightness = (0.5,1.2) ,  RandomHorizontalFlip, RandomAdjustSharpness and finally the image is normalized.

The backbone architectures were obtained directly from the torchvision library and the final classification layer was modified for the selected datasets. For the models which had to be trained from scratch, the weights were randomly initialized and the entire model was trained for a total of 10 epochs each. For the transfer learning models, the weights were initialized with the IMAGENET1K_V2 weights but only the fully connected layers were fine-tuned.

The 3 chosen architectures were trained against each of the 3 chosen datasets without transfer learning to get a total of 9 models. On top of these 9 models 2 additional models were trained with transfer learning. Archtitectures picked for transfer learning were AlexNet and ResNet-18 on dataset 1. All of these models were trained the same and fixed hyperparameters to get comparable results. Performance results of these models are displayed below:

![9 Models Before Hyper-parameter tuning](https://github.com/YuganshG/Comparative-Study-of-different-CNN-Architectures/assets/34838617/1fe7788f-37f9-4dbe-a8bf-d52cf989d17c)

![Transfer Learning Before Hyper-Parameters Tuning](https://github.com/YuganshG/Comparative-Study-of-different-CNN-Architectures/assets/34838617/ac4d8806-e000-4cc1-a73e-77c261e2863d)

<table>
    <tr>
        <th>Model</th>
        <th colspan="3">AlexNet</th>
        <th colspan="3">VGG-11</th>
        <th colspan="3">ResNet-18</th>
    </tr>
    <tr>
        <th>Dataset</th>
        <th>Train Accuracy</th>
        <th>Test Accuracy</th>
        <th>Test F1 score</th>
        <th>Train Accuracy</th>
        <th>Test Accuracy</th>
        <th>Test F1 score</th>
        <th>Train Accuracy</th>
        <th>Test Accuracy</th>
        <th>Test F1 score</th>
    </tr>
    <tr>
        <th> Indian TSD </th>
        <td> 53.12% </td>
        <td> 46.90% </td>
        <td> 0.46 </td>
        <td> 65.62% </td>
        <td> 57.32% </td>
        <td> 0.57 </td>
        <td> 62.50% </td>
        <td> 53.09% </td>
        <td> 0.53 </td>
    </tr>
    <tr>
        <th> Persian TSD </th>
        <td> 84.38% </td>
        <td> 86.72% </td>
        <td> 0.86 </td>
        <td> 93.75% </td>
        <td> 85.26% </td>
        <td> 0.85 </td>
        <td> 90.62% </td>
        <td> 79.64% </td>
        <td> 0.79 </td>
    </tr>
    <tr>
        <th> German TSD </th>
        <td> 65% </td>
        <td> 75.14% </td>
        <td> 0.75 </td>
        <td> 84.3% </td>
        <td> 79.23% </td>
        <td> 0.79 </td>
        <td> 80% </td>
        <td> 61.80% </td>
        <td> 0.61 </td>
    </tr>
</table>


## Hyperparameter Tuning

Out of the 9 models the best performing model was then picked and hyperparameter tuning was performed on the learning rate within range of (0.01, 0.001, 0.0001) and evaluated them using mean loss and accuracy over the trainset set batches. The rationale behind the chosen range is that we are expecting a increase in model performance with a slightly higher learning rate because of our choice of using Adam as the optimizer. Results of the hyperparameter tuning phase are displayed below:

![Learning Rate Hyper Parameter](https://github.com/YuganshG/Comparative-Study-of-different-CNN-Architectures/assets/34838617/3bbc9a41-3695-4d59-b7db-abaf18c6ea05)

## Main Results

As observed in the loss and accuracy plot above, learning rate of 0.001 resulted in more accuracy and faster convergence for the best performing model AlexNet on dataset 2. 
This learning rate was then used to retrain all the 11 models and evaluated against the Accuracy, F1-score, precision, recall and AUC score metrics. Performance of the trained models on test set is shown below:

<table>
    <tr>
        <th>Model</th>
        <th colspan="5">AlexNet</th>
        <th colspan="5">VGG-11</th>
        <th colspan="5">ResNet-18</th>
    </tr>
    <tr>
        <th>On Test</th>
        <th>Accuracy</th>
        <th>F1 Score</th>
        <th>Precision</th>
        <th>Recall</th>
        <th>AUC</th>
        <th>Accuracy</th>
        <th>F1 Score</th>
        <th>Precision</th>
        <th>Recall</th>
        <th>AUC</th>
        <th>Accuracy</th>
        <th>F1 Score</th>
        <th>Precision</th>
        <th>Recall</th>
        <th>AUC</th>
    </tr>
    <tr>
        <th> Indian TSD </th>
<td>55.21%</td>
<td>0.55</td>
<td>0.63</td>
<td>0.59</td>
<td>0.07</td>
<td>11.4%</td>
<td>0.11</td>
<td>0.007</td>
<td>0.07</td>
<td>0.5</td>
<td>62.25%</td>
<td>0.62</td>
<td>0.66</td>
<td>0.65</td>
<td>0.10</td>
    </tr>
    <tr>
        <th> Persian TSD </th>
<td>91.78%</td>
<td>0.91</td>
<td>0.90</td>
<td>0.93</td>
<td>0.08</td>
<td>95.11%</td>
<td>0.95</td>
<td>0.94</td>
<td>0.95</td>
<td>0.06</td>
<td>96.42%</td>
<td>0.96</td>
<td>0.95</td>
<td>0.97</td>
<td>0.09</td>
    </tr>
    <tr>
        <th> German TSD </th>
<td>80.13%</td>
<td>0.80</td>
<td>0.79</td>
<td>0.79</td>
<td>0.28</td>
<td>91.98%</td>
<td>0.91</td>
<td>0.93</td>
<td>0.92</td>
<td>0.21</td>
<td>95.77%</td>
<td>0.95</td>
<td>0.96</td>
<td>0.95</td>
<td>0.19</td>
    </tr>
</table>

and for transfer learning: 

<table>
    <tr>
        <th>Model on Transfer Learning</th>
        <th colspan="5">AlexNet</th>
        <th colspan="5">ResNet-18</th>
    </tr>
    <tr>
        <th>On Test</th>
        <th>Accuracy</th>
        <th>F1 Score</th>
        <th>Precision</th>
        <th>Recall</th>
        <th>AUC</th>
        <th>Accuracy</th>
        <th>F1 Score</th>
        <th>Precision</th>
        <th>Recall</th>
        <th>AUC</th>
    </tr>
    <tr>
        <th> Indian TSD </th>
        <td>75.77%</td>
        <td>0.76</td>
        <td>0.85</td>
        <td>0.79</td>
        <td>0.05</td>
        <td>70%</td>
        <td>0.7</td>
        <td>0.74</td>
        <td>0.71</td>
        <td>0.19</td>    
</tr>
</table>
    
Plots for the model performance during the traning phase has been displayed below:

![9 Models After Hyper-parameter tuning](https://github.com/YuganshG/Comparative-Study-of-different-CNN-Architectures/assets/34838617/bd13862e-e893-4b82-a0a0-d0e7bce77bf3)

![Transfer Learning After Hyper-Parameters Tuning](https://github.com/YuganshG/Comparative-Study-of-different-CNN-Architectures/assets/34838617/42cc0aee-993f-43ae-92fd-d04ee44010c9)

## t-SNE Visualizations

t-SNE has been used to visualize 4 models to compare data separability : AlexNet with/without transfer learning on dataset 1, ResNet-18 without transfer learning on dataset 1 and 3. Results of these are as follows:

![TSNE Datasets Comparisons](https://github.com/YuganshG/Comparative-Study-of-different-CNN-Architectures/assets/34838617/99ba0d20-8f68-4955-b51d-24b8430650f8)

![TSNE Transfer Learning](https://github.com/YuganshG/Comparative-Study-of-different-CNN-Architectures/assets/34838617/361bbdcb-55dd-449f-9147-5fba5b18a0a8)

As per the observed increased in distance among classes in figure 1, we can say that transfer learning on AlexNet with Indian traffic sign dataset helped the model to predict classes with more certainity. Although the model still suffered to differentiate between a few classes mainly the speed limit traffic signs.
In Figure , the bigger dataset 3 seemed to have helped the ResNet-18 model to better differentiate between the classes as prominent by the big nicely separated clusters of same classes on the right. 

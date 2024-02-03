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
Estimated computational complexity, top 1% accuracy, top 5% accuracy of some popular architectures trained mostly on ILSVRC 2012 were compared as follows: 

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
         <td> <b> 0.46 </b> </td>
         <td> <b> 65.62% </b> </td>
         <td> <b> 57.32% </b> </td>
         <td> 0.57 </td>
         <td> 62.50% </td>
         <td> 53.09% </td>
         <td> 0.53 </td>
      </tr>
      <tr>
         <th> Persian TSD </th>
         <td> 84.38% </td>
         <td> 86.72% </td>
         <td> <b> 0.86 </b> </td>
         <td> <b> 93.75% </b> </td>
         <td> <b> 85.26% </b> </td>
         <td> 0.85 </td>
         <td> <b> 90.62% </b> </td>
         <td> 79.64% </td>
         <td> 0.79 </td>
      </tr>
      <tr>
         <th> German TSD </th>
         <td> 65% </td>
         <td> 75.14% </td>
         <td> <b> 0.75 </b> </td>
         <td> <b> 84.3% </b> </td>
         <td> <b> 79.23% </b> </td>
         <td> 0.79 </td>
         <td> <b> 80% </b> </td>
         <td> 61.80% </td>
         <td> 0.61 </td>
      </tr>
    </table>

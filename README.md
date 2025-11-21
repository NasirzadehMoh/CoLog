<h1 align="center"> CoLog: A Unified Framework for Detecting Point and Collective Anomalies in Operating System Logs via Collaborative Transformers </h1>

<div align="center">

<br>

<a href="https://github.com/NasirzadehMoh/CoLog/stargazers"><img src="https://img.shields.io/badge/0-NasirzadehMoh%2FCoLog?style=social&logo=github&label=Stars" alt="Stars Badge"/></a>
<a href="https://github.com/NasirzadehMoh/CoLog/network/members"><img src="https://img.shields.io/badge/0-NasirzadehMoh%2FCoLog?style=social&logo=github&label=Forks" alt="Forks Badge"/></a>
<a href="https://github.com/NasirzadehMoh/CoLog/pulls"><img src="https://img.shields.io/badge/0.0-NasirzadehMoh%2FCoLog?style=plastic&logo=github&label=Pull%20request&color=544aa0" alt="Pull Requests Badge"/></a>
<a href="https://github.com/NasirzadehMoh/CoLog/issues"><img src="https://img.shields.io/badge/0.0-NasirzadehMoh%2FCoLog?style=plastic&logo=github&label=Issues&color=f68b1f" alt="Issues Badge"/></a>
<a href="https://github.com/NasirzadehMoh/CoLog/graphs/contributors"><img src="https://img.shields.io/badge/2.0-NasirzadehMoh%2FCoLog?style=plastic&logo=github&label=Contributors&color=f2678e" alt="GitHub contributors"></a>
<a href="https://github.com/NasirzadehMoh/CoLog/blob/main/LICENSE"><img src="https://img.shields.io/badge/MIT-NasirzadehMoh%2FCoLog?style=plastic&logo=github&label=License&color=3d83bf" alt="License Badge"/></a>

</div>

<i>This repository is the official implementation of ["A Unified Framework for Detecting Point and Collective Anomalies in Operating System Logs via Collaborative Transformers"](https://arxiv.org) as well as its follow-ups. It currently includes:</i>

> **Source code and training scripts**: Included in this repo. See [get_started.md](get_started.md) for a quick start.

> **Pretrained models for all datasets**: See [Pretrained Models](https://github.com/NasirzadehMoh/CoLog).

> **Dataset preprocessing and split scripts**: See [Datasets](https://github.com/NasirzadehMoh/CoLog).

> **Configuration files (hyperparameters, seeds)**: See [Configuration Files](https://github.com/NasirzadehMoh/CoLog).

> **Instructions for environment setup and execution**: See [Environment Setup](https://github.com/NasirzadehMoh/CoLog).

## Introduction

Log anomaly detection is crucial for preserving the security of operating systems. Depending on the source of log data collection, various information is recorded in logs that can be considered log modalities. In light of this intuition, unimodal methods often struggle by ignoring the different modalities of log data. Meanwhile, multimodal methods fail to handle the interactions between these modalities. Applying multimodal sentiment analysis to log anomaly detection, we propose CoLog, a framework that collaboratively encodes logs utilizing various modalities. CoLog utilizes collaborative transformers and multi-head impressed attention to learn interactions among several modalities, ensuring comprehensive anomaly detection. To handle the heterogeneity caused by these interactions, CoLog incorporates a modality adaptation layer, which adapts the representations from different log modalities. This methodology enables CoLog to learn nuanced patterns and dependencies within the data, enhancing its anomaly detection capabilities. Extensive experiments demonstrate CoLog’s superiority over existing state-of-the-art methods. Furthermore, in detecting both point and collective anomalies, CoLog achieves a mean precision of 99.63%, a mean recall of 99.59%, and a mean F1 score of 99.61% across seven benchmark datasets for log-based anomaly detection. The comprehensive detection capabilities of CoLog make it highly suitable for cybersecurity, system monitoring, and operational efficiency. CoLog represents a significant advancement in log anomaly detection, providing a sophisticated and effective solution to point and collective anomaly detection through a unified framework and a solution to the complex challenges automatic log data analysis poses.

<div align="center">

<img alt="CoLog's Architecture" src="figures/architecture.svg"> </img>

</div>

## Main Results on Point Anomaly Detection
**Results on Casper Dataset**

| Anomaly Detection Technique |  Precision | Recall | F1-Score | Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 66.959 |  60.863 |  59.700 |  90.993 |
| Support Vector Machines | 58.614 | 60.757 | 59.482 |  90.967 |
| Decision Tree |  83.466 | 77.037 | 79.488 |  94.998 |
| Attentional BiLSTM |  99.766 | 99.872 |  99.819 | 99.834 |
| Convolutional Neural Network | 99.766 | 99.872 |  99.819 | 99.834 |
| pylogsentiment |  99.487 |  99.413 | 99.449 |  99.459 |
| Isolation Forest |  52.407 |  50.650 | 49.926 |  88.149 |
| Principal Component Analysis |  51.205 |  50.282 | 49.362 |  87.480 |
| LSTM | 97.973 | 98.843 |  98.380 |  98.505 |
| Transformer | 98.409 | 99.100 | 98.738 |  98.837 |
| CoLog | 100 | 100 | 100 | 100 |

**Results on Jhuisi Dataset**

| Anomaly Detection Technique |  Precision | Recall | F1-Score | Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 68.373 | 66.127 | 64.886 | 79.182 |
| Support Vector Machines | 63.643 | 66.506 | 64.051 | 80.914 |
| Decision Tree | 91.534 | 89.769 |  90.550 | 93.313 |
| Attentional BiLSTM | 95.123 | 97.270 | 96.169 | 99.341 |
| Convolutional Neural Network | 97.139 | 94.885 | 95.982 | 99.341 |
| pylogsentiment | 98.867 | 98.761 | 98.813 | 98.850 |
| Isolation Forest | 57.519 | 52.774 | 51.700 | 74.707 |
| Principal Component Analysis | 44.828 | 49.299 | 45.014 | 75.448 |
| LSTM | 96.879 | 92.385 | 94.508 | 99.121 |
| Transformer | 96.879 | 92.385 | 94.508 | 99.121 |
| CoLog | 100 | 100 | 100 | 100 |

**Results on Nssal Dataset**

| Anomaly Detection Technique |  Precision | Recall | F1-Score | Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 85.133 | 74.728 | 76.476 | 97.604 |
| Support Vector Machines | 80.206 | 74.935 | 76.474 | 97.655 |
| Decision Tree | 94.791 | 87.700 | 89.470 | 98.063 |
| Attentional BiLSTM | 96.750 | 98.805 | 97.754 | 99.813 |
| Convolutional Neural Network | 96.703 | 98.243 | 97.460 | 99.789 |
| pylogsentiment | 97.170 | 96.050 | 96.602 | 99.020 |
| Isolation Forest | 65.504 | 57.352 | 56.101 | 80.967 |
| Principal Component Analysis | 52.642 | 53.827 | 49.505 | 80.614 |
| LSTM |  96.148 | 97.669 | 96.896 | 99.742 |
| Transformer | 96.304 | 99.354 | 97.778 | 99.813 |
| CoLog |  99.955 | 99.915 | 99.935 | 99.967 |

**Results on Casper Dataset**

**Results on Casper Dataset**

**Results on Casper Dataset**

**Results on Casper Dataset**


## Main Results on Point and Collective Anomaly Detection

**COCO Object Detection (2017 val)**



<div align="center">

<h2 align="left"> For more information, please visit www.alarmif.com. </h2>

<img alt="Alarmif" src="assets/alarmif.gif"> </img>

</div>

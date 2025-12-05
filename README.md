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

<p align="center">
<i>Official implementation of "A Unified Framework for Detecting Point and Collective Anomalies in Operating System Logs via Collaborative Transformers"</i>
</p>

---

<div align="center">

## 📋 Table of Contents

</div>

- [Introduction](#-introduction)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Datasets](#-datasets)
- [Results](#-results)
  - [Point Anomaly Detection](#main-results-on-point-anomaly-detection)
  - [Point and Collective Anomaly Detection](#main-results-on-point-and-collective-anomaly-detection)
  - [Method Ranking](#ranking-colog-and-other-log-anomaly-detection-methods)
- [Citation](#-citing-colog)
- [License](#-license)
- [Contact](#-contact)

---

<div align="center">

## 🎯 Introduction

</div>

Log anomaly detection is crucial for preserving the security of operating systems. Depending on the source of log data collection, various information is recorded in logs that can be considered log modalities. In light of this intuition, unimodal methods often struggle by ignoring the different modalities of log data. Meanwhile, multimodal methods fail to handle the interactions between these modalities. Applying multimodal sentiment analysis to log anomaly detection, we propose CoLog, a framework that collaboratively encodes logs utilizing various modalities. CoLog utilizes collaborative transformers and multi-head impressed attention to learn interactions among several modalities, ensuring comprehensive anomaly detection. To handle the heterogeneity caused by these interactions, CoLog incorporates a modality adaptation layer, which adapts the representations from different log modalities. This methodology enables CoLog to learn nuanced patterns and dependencies within the data, enhancing its anomaly detection capabilities. Extensive experiments demonstrate CoLog’s superiority over existing state-of-the-art methods. Furthermore, in detecting both point and collective anomalies, CoLog achieves a mean precision of 99.63%, a mean recall of 99.59%, and a mean F1 score of 99.61% across seven benchmark datasets for log-based anomaly detection. The comprehensive detection capabilities of CoLog make it highly suitable for cybersecurity, system monitoring, and operational efficiency. CoLog represents a significant advancement in log anomaly detection, providing a sophisticated and effective solution to point and collective anomaly detection through a unified framework and a solution to the complex challenges automatic log data analysis poses.

<div align="center">

<br>

<img alt="CoLog's Architecture" src="figures/architecture.svg" width="90%"> </img>

<i>Figure 1. The overview of CoLog. Light green and gold colors demonstrate modality encoders. Each encoder in the collaborative transformer consists of MHIA, MLP, MAL, and LNs. MHIA and MAL are multi-head impressed attention and modality adaptation layer modules, respectively. The preprocess layer transforms unstructured logs into easily understandable data for the model. The purpose of the balancing layer is to regulate the influences of different modalities when calculating the final results.</i>

</div>

---

<div align="center">

## 🚀 Installation

</div>

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/NasirzadehMoh/CoLog.git
cd CoLog
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download spaCy language model**
```bash
python -m spacy download en_core_web_lg
```

---

<div align="center">

## 🎬 Quick Start

</div>

### Preprocessing

```bash
# Preprocess a specific dataset
python groundtruth/main.py \
    --dataset hadoop \
    --sequence-type context \
    --window-size 1 \
    --model all-MiniLM-L6-v2 \
    --batch-size 128 \
    --device auto \
    --resample \
    --resample-method TomekLinks \
    --verbose
```

### Training

```bash
# Train CoLog on a specific dataset
python train.py --dataset casper-rw --epochs 100 --batch_size 32
```

### Testing

```bash
# Test the trained model
python test.py --dataset casper-rw --model_path runs/logs/best_model.pth
```

### Using Jupyter Notebook

```bash
# Launch the interactive notebook
jupyter notebook run_colog.ipynb
```

---

<div align="center">

## 📊 Datasets

</div>

CoLog has been evaluated on seven benchmark datasets:

- **Casper-RW**: System logs from Casper RW environment
- **DFRWS 2009 Jhuisi**: Forensic challenge dataset
- **DFRWS 2009 Nssal**: Network security logs
- **Honeynet Challenge 7**: Honeypot network logs
- **Zookeeper**: Apache Zookeeper distributed system logs
- **Hadoop**: Apache Hadoop distributed computing logs
- **BlueGene/L (BGL)**: Supercomputer system logs

All datasets are located in the `datasets/` directory. For more information, see [datasets/README.md](datasets/README.md).

---

<div align="center">

<h2> Main Results on Point Anomaly Detection </h2>

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

<br>

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

<br>

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

<br>

**Results on Honey7 Dataset**

| Anomaly Detection Technique |  Precision | Recall | F1-Score | Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| Logistic Regression |  68.143 | 70.000 | 69.042 | 96.286 |
| Support Vector Machines | 68.143 | 70.000 | 69.042 | 96.286 |
| Decision Tree |  93.260 | 83.307 | 86.359 | 97.471 |
| Attentional BiLSTM | 100 | 100 | 100 | 100 |
| Convolutional Neural Network | 100 | 100 | 100 | 100 |
| pylogsentiment | 99.970 | 99.107 | 99.535 | 99.943 |
| Isolation Forest | 47.509 | 50.948 | 48.064 | 85.966 |
| Principal Component Analysis | 60.871 | 58.898 | 55.943 | 88.279 |
| LSTM |  96.212 | 98.810 | 97.429 | 98.155 |
| Transformer |  96.923 | 99.048 | 97.932 | 98.524 |
| CoLog |  100 | 100 | 100 | 100 |

<br>

**Results on Zookeeper Dataset**

| Anomaly Detection Technique |  Precision | Recall | F1-Score | Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| Logistic Regression |  98.369 | 98.464 | 98.416 | 98.562 |
| Support Vector Machines |  97.987 | 97.769 | 97.877 | 98.078 |
| Decision Tree |  98.599 | 98.880 | 98.737 | 98.851 |
| Attentional BiLSTM |  95.783 | 92.928 | 94.300 | 98.387 |
| Convolutional Neural Network | 95.121 | 92.662 | 93.850 | 98.252 |
| pylogsentiment | 99.722 | 99.898 | 99.810 | 99.973 |
| Isolation Forest |  32.607 | 50.000 | 39.473 | 65.215 |
| Principal Component Analysis | 79.011 | 51.209 | 42.121 | 66.021 |
| LSTM |   95.825 | 91.887 | 93.750 | 98.252 |
| Transformer |   99.890 | 99.416 | 99.652 | 99.361 |
| CoLog |  100 | 100 | 100 | 100 |

<br>

**Results on Hadoop Dataset**

| Anomaly Detection Technique |  Precision | Recall | F1-Score | Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| Logistic Regression |  48.523 | 50.000 | 49.250 | 97.046 |
| Support Vector Machines |  48.523 | 50.000 | 49.250 | 97.046 |
| Decision Tree |  98.599 | 48.523 | 50.000 | 49.250 | 97.046 |
| Attentional BiLSTM |   97.640 | 97.955 | 97.792 | 97.902 |
| Convolutional Neural Network | 99.719 | 99.847 | 99.783 | 99.955 |
| pylogsentiment |  99.886 | 99.732 | 99.809 | 99.905 |
| Isolation Forest |  47.702 | 50.000 | 48.824 | 54.034 |
| Principal Component Analysis |  49.995 | 49.996 | 49.996 | 58.214 |
| LSTM |    99.850 | 97.397 | 98.589 | 99.715 |
| Transformer |    97.280 | 99.833 | 98.518 | 99.685 |
| CoLog |   99.997 | 99.956 | 99.977 | 99.994 |

<br>

**Results on BlueGene/L Dataset**

| Anomaly Detection Technique |  Precision | Recall | F1-Score | Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| Logistic Regression |   54.028 | 51.852 | 52.092 | 90.368 |
| Support Vector Machines |  46.314 | 50.000 | 48.087 | 92.628 |
| Decision Tree |  60.576 | 50.998 | 50.303 | 92.348 |
| Attentional BiLSTM |    97.640 | 97.955 | 97.792 | 97.902 |
| Convolutional Neural Network |  97.640 | 97.955 | 97.792 | 97.902 |
| pylogsentiment |  99.892 | 99.963 | 99.928 | 99.980 |
| Isolation Forest |   53.081 | 50.047 | 51.519 | 47.389 |
| Principal Component Analysis |  51.168 | 54.260 | 38.970 | 48.487 |
| LSTM |  97.414 | 98.296 | 97.806 | 97.902 |
| Transformer | 97.640 | 97.955 | 97.792 | 97.902 |
| CoLog |  99.999 | 99.990 | 99.994 | 99.998 |

<br>

<h2 align="center"> Ranking CoLog and Other Log Anomaly Detection Methods </h2>

<p align="center"><i>Comparison of mean F1-scores across all benchmark datasets</i></p>

| Anomaly Detection Technique |  Mean F1-Score |
| :---: | :---: |
| Logistic Regression |   47.273 |
| Support Vector Machines |  49.372 |
| Decision Tree |  66.323 |
| Attentional BiLSTM | 67.123 |
| Convolutional Neural Network | 77.737 |
| pylogsentiment | 96.765 |
| Isolation Forest | 97.661 |
| Principal Component Analysis | 97.812 |
| LSTM |  97.845 |
| Transformer | 99.135 |
| CoLog |  99.987 |

<br>

<h2 align="center"> Main Results on Point and Collective Anomaly Detection </h2>

<p align="center"><i>CoLog's unified framework performance on detecting both anomaly types</i></p>

| Dataset |  Precision | Recall | F1-Score | Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| Casper |    99.43 | 99.40 | 99.41 | 99.64 |
| Jhuisi | 99.82 | 99.48 |  99.65 | 99.70 |
| Nssal |  99.32 | 99.70 | 99.51 |  99.91 |
| Honey7 |    99.77 |  98.90 | 99.33 | 99.77 |
| Zookeeper |  99.98 | 99.94 |  99.96 | 99.99 |
| Hadoop | 99.17 | 99.84 |  99.50 |  99.98 |
| BlueGene/L | 99.94 |  99.89 | 99.91 | 100 |

<br>

---

<div align="left">

<div align="center">

## 📝 Citing CoLog

</div>

If you use CoLog in your research, please cite our paper:

```bibtex
@inproceedings{CoLog2024,
  title={A Unified Framework for Detecting Point and Collective Anomalies in Operating System Logs via Collaborative Transformers},
  author={CoLog Authors},
  booktitle={Proceedings of [Conference Name]},
  year={2024},
  url={https://arxiv.org}
}
```

</div>

---

## 🤝 Contributing

<div align="left">

We welcome contributions to CoLog! If you'd like to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows the project's coding standards and includes appropriate tests.

</div>

---

## 📄 License

<div align="left">

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

</div>

---

## 📧 Contact

<div align="left">

For more information, questions, or collaboration opportunities:

- **Website**: [www.alarmif.com](https://www.alarmif.com)
- **Issues**: [GitHub Issues](https://github.com/NasirzadehMoh/CoLog/issues)
- **Pull Requests**: [GitHub Pull Requests](https://github.com/NasirzadehMoh/CoLog/pulls)
- **LinkedIn**: [Mohammad Nasirzadeh](https://www.linkedin.com/in/nasirzadehmoh/)
- **Email**: [Mohammad Nasirzadeh](mailto:nasirzadehmohammad1997@gmail.com)

</div>

<div align="center">

<br>

<img alt="Alarmif" src="assets/alarmif.gif" width="1691"> </img>

<br>

---

<p>Made with ❤️ by the CoLog Team</p>

</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub forks](https://img.shields.io/github/forks/tirthajyoti/Deep-Learning-with-Python.svg)](https://github.com/tirthajyoti/Deep-Learning-with-Python/network)
[![GitHub stars](https://img.shields.io/github/stars/tirthajyoti/Deep-Learning-with-Python.svg)](https://github.com/tirthajyoti/Deep-Learning-with-Python/stargazers)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/tirthajyoti/Deep-Learning-with-Python/pulls)

# Deep Learning with Python
Various deep-learning code examples, (Tutorial-style) Jupyter notebooks, and projects.

Many of the Jupyter notebooks are built on **Google Colab** and may employ special functions exclusive to Google Colab.

Authored and maintained by **Dr. Tirthajyoti Sarkar ([Website](https://tirthajyoti.github.io), [LinkedIn profile](https://www.linkedin.com/in/tirthajyoti-sarkar-2127aa7/))**

Here is the **[Github Repo](https://github.com/tirthajyoti/Deep-learning-with-Python)**. 

---

## Requirements
* Python 3.6+
* NumPy (`pip install numpy`)
* Pandas (`pip install pandas`)
* MatplotLib (`pip install matplotlib`)
* Tensorflow (`pip install tensorflow` or `pip install tensorflow-gpu`)
> Of course, to use a local GPU correctly, you need to do lot more work setting up proper GPU driver and CUDA installation. <br>
> If you are using Ubuntu 18.04, [here is a guide](https://mc.ai/tensorflow-gpu-installation-on-ubuntu-18-04/). <br>
> If you are on Windows 10, [here is a guide](https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781) <br>
> It is also highly recommended to **install GPU version in a separate virtual environment**, so as to not mess up the default system install.
* Keras (`pip install keras`)

**NOTE**: Most of the Jupyter notebooks in this repo are built on **[Google Colaboratory](https://colab.research.google.com/)** using **[Google GPU cluster](https://cloud.google.com/gpu/)** and a virtual machine. Therefore, you may not need to install these packages on your local machine if you also want to use Google colab. You can **directly launch the notebooks in your Google colab environment by clicking on the links provided in the notebooks** (of course, that makes a copy of my notebook on to your Google drive).

> For more information about using **Google Colab** for your deep learning work, [check their FAQ here](https://research.google.com/colaboratory/faq.html).

---

## Notebooks

### Simple Conv Net
* [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) image classification using densely connected network and 1/2/3 layer CNNs ([Here is the Notebook](https://github.com/tirthajyoti/Computer_vision/blob/master/Notebooks/Fashion_MNIST_using_CNN.ipynb)).

### Using Keras `ImageGenerator`
* _Horse or human_ image classification using Keras `ImageGenerator` and **Google colaboratory** platform ([Here is the Notebook](https://github.com/tirthajyoti/Computer_vision/blob/master/Notebooks/Horse_or_Human_with_ImageGenerator.ipynb))

### Simple demo of transfer learning
* Simple illustration of [transfer learning](https://machinelearningmastery.com/transfer-learning-for-deep-learning/) using CIFAR-10 dataset ([Here is the Notebook](https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/Transfer_learning_CIFAR.ipynb))

### Adding object-oriented programming style to deep learning workflow
* Adding simple [Object-oriented Programming (OOP)](https://realpython.com/python3-object-oriented-programming/) principle to your deep learning workflow ([Here is the Notebook](https://github.com/tirthajyoti/Computer_vision/blob/master/Notebooks/OOP_principle_deep_learning.ipynb)).

### Keras `Callbacks` using ResNet
* [ResNet](https://medium.com/@14prakash/understanding-and-implementing-architectures-of-resnet-and-resnext-for-state-of-the-art-image-cf51669e1624) on [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), showing how to use Keras Callbacks classes like `ModelCheckpoint`, `LearningRateScheduler`, and `ReduceLROnPlateau`. You can also change a single parameter to generate ResNet of various depths. ([Here is the Notebook](https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/ResNet-on-CIFAR10.ipynb)).

### Text generation using LSTM
* Automatic text generation (based on simple character vectors) using [LSTM network](https://colah.github.io/posts/2015-08-Understanding-LSTMs/). Play with character sequence length, LSTM architecture, and hyperparameters to generate synthetic texts based on a particular author's style! ([Here is the Notebook](https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/LSTM_text_gen_Dickens.ipynb)).

### Bi-directional LSTM for sentiment classification
* [Bi-directional LSTM with embedding](https://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python-keras/) applied to the IMDB sentiment classification task ([Here is the Notebook](https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/LSTM_bidirectional_IMDB_data.ipynb))

### Scikit-learn wrapper for Keras
* [Keras Scikit-learn wrapper](https://keras.io/scikit-learn-api/) example with 10-fold cross-validation and exhaustive grid search ([Here is the Notebook](https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/Keras_Scikit_Learn_wrapper.ipynb))

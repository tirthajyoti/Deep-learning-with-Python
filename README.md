# Deep Learning with Python
Various deep-learning code examples, (Tutorial-style) Jupyter notebooks, and projects.

Many of the Jupyter notebooks are built on **Google Colab** and may employ special functions exclusive to Google Colab.

Authored and maintained by Dr. Tirthajyoti Sarkar, Fremont, CA. 
<br>([Please feel free to add me on LinkedIn here](https://www.linkedin.com/in/tirthajyoti-sarkar-2127aa7))

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

* Fashion MNIST image classification using densely connected network and 1/2/3 layer CNNs ([Here is the Notebook](https://github.com/tirthajyoti/Computer_vision/blob/master/Notebooks/Fashion_MNIST_using_CNN.ipynb)).

* Horse or human image classification using Keras `ImageGenerator` and **Google colaboratory** platform ([Here is the Notebook](https://github.com/tirthajyoti/Computer_vision/blob/master/Notebooks/Horse_or_Human_with_ImageGenerator.ipynb))

* Adding simple OOP principle to your deep learning workflow ([Here is the Notebook](https://github.com/tirthajyoti/Computer_vision/blob/master/Notebooks/OOP_principle_deep_learning.ipynb))

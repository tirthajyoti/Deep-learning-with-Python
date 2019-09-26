# Jupyter Notebooks

### Deep learning vs. linear model
* We show a nonlinear function approximation task performed by linear model (polynomial degree) and a simple 1/2 hidden layer (densely connected) neural net to illustrate the difference and the capacity of deep neural nets to take advantage of larger datasets ([Here is the Notebook](https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/Function%20approximation%20by%20linear%20model%20and%20deep%20network.ipynb)).

### Simple Conv Net
* [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) image classification using densely connected network and 1/2/3 layer CNNs ([Here is the Notebook](https://github.com/tirthajyoti/Computer_vision/blob/master/Notebooks/Fashion_MNIST_using_CNN.ipynb)).

### Using Keras `ImageDataGenerator` and other utilities

* _Horse or human_ image classification using Keras `ImageDataGenerator` and **Google colaboratory** platform ([Here is the Notebook](https://github.com/tirthajyoti/Computer_vision/blob/master/Notebooks/Horse_or_Human_with_ImageGenerator.ipynb))

* Classification on the [flowers dataset](https://www.kaggle.com/alxmamaev/flowers-recognition) and the famous [Caltech-101 dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) using `fit_generator` and `flow_from_directory()` method of the `ImageDataGenerator`. Illustrates how to streamline CNN model building from a single storage of image data using these utility methods. ([Here is the Notebook](https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/Keras_flow_from_directory.ipynb))

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

### Generative adversarial network (GAN)
* Simple demo of building a GAN model from scratch using a one-dimensional algebraic function ([Here is the Notebook](https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/GAN_1D.ipynb))

### Scikit-learn wrapper for Keras
* [Keras Scikit-learn wrapper](https://keras.io/scikit-learn-api/) example with 10-fold cross-validation and exhaustive grid search ([Here is the Notebook](https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/Keras_Scikit_Learn_wrapper.ipynb))

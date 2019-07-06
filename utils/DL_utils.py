# # Inheritance, child class, and utility functions (OOP principles) for optimizing your deep learning work
# ### Dr. Tirthajyoti Sarkar, Fremont, CA 94536
# 
# In this notebook, we implement a simple callback feature by defining a special class which inherits from the superclass **`keras.callbacks.Callback`**. 
# 
# The callback checks the accuracy of the trained model at the end of every epoch, and stops the training when it reaches a desired threshold. Users can set the desired threshold while instantiating the class.
# 
# We also create simple utility functions like **`build_model`** and **`compile_train_model`** to generate and train the deep learning model from user inputs. These functions can later be called from a ***higher-order optimization loop*** or ***analytics script***.
# 
# Overall, this notebook aims to illustrate how by **mixing simple [Object-oriented programming good practices](https://realpython.com/python3-object-oriented-programming/), we can add immense value to our deep learning prototype work.**

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class myCallback(tf.keras.callbacks.Callback):
  """
  User can pass on the desired accuracy threshold while creating an instance of the class
  """
  def __init__(self,acc_threshold=0.9,print_msg=True):
    self.acc_threshold=acc_threshold
    self.print_msg = print_msg
    
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>self.acc_threshold):
      if self.print_msg:
        print("\nReached 90% accuracy so cancelling the training!")
      self.model.stop_training = True
    else:
      if self.print_msg:
        print("\nAccuracy not high enough. Starting another epoch...\n")


def build_model(num_layers=1, architecture=[32],act_func='relu', 
                input_shape=(28,28), output_class=10):
  """
  Builds a densely connected neural network model from user input
  num_layers: Number of hidden layers
  architecture: Architecture of the hidden layers (densely connected)
  act_func: Activation function. Could be 'relu', 'sigmoid', or 'tanh'.
  input_shape: Dimension of the input vector
  output_class: Number of classes in the output vector
  """
  layers=[tf.keras.layers.Flatten(input_shape=input_shape)]
  if act_func=='relu':
    activation=tf.nn.relu
  elif act_func=='sigmoid':
    activation=tf.nn.sigmoid
  elif act_func=='tanh':
    activation=tf.nn.tanh
    
  for i in range(num_layers):
    layers.append(tf.keras.layers.Dense(architecture[i], activation=tf.nn.relu))
  layers.append(tf.keras.layers.Dense(output_class, activation=tf.nn.softmax))
  
  model = tf.keras.models.Sequential(layers)
  return model


def compile_train_model(model,x_train, y_train, callbacks=None,
                        learning_rate=0.001,batch_size=1,epochs=10,verbose=0):
  """
  Compiles and trains a given Keras model with the given data. 
  Assumes Adam optimizer for this implementation.
  
  learning_rate: Learning rate for the optimizer Adam
  batch_size: Batch size for the mini-batch optimization
  epochs: Number of epochs to train
  verbose: Verbosity of the training process
  """
  
  model_copy = model
  model_copy.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
  
  model_copy.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                 callbacks=[callbacks],verbose=verbose)
  return model_copy



def plot_loss_acc(model,target_acc=0.9, title=None):
  """
  Takes a deep learning model and plots the loss ans accuracy over epochs
  Users can supply a title if needed
  target_acc: The desired/ target acc. This parameter is needed for this function to show a horizontal bar.
  """
  e=np.array(model.history.epoch)+1 # Add one to the list of epochs which is zero-indexed
  l=np.array(model.history.history['loss'])
  a=np.array(model.history.history['acc'])
  
  fig, ax1 = plt.subplots()

  color = 'tab:red'
  ax1.set_xlabel('Epochs',fontsize=15)
  ax1.set_ylabel('Loss', color=color,fontsize=15)
  ax1.plot(e, l, color=color,lw=2)
  ax1.tick_params(axis='y', labelcolor=color)
  ax1.grid(True)

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

  color = 'tab:blue'
  ax2.set_ylabel('Accuracy', color=color,fontsize=15)  # we already handled the x-label with ax1
  ax2.plot(e, a, color=color,lw=2)
  ax2.tick_params(axis='y', labelcolor=color)

  fig.tight_layout()  # otherwise the right y-label is slightly clipped
  if title!=None:
    plt.title(title)
  plt.hlines(y=target_acc,xmin=1,xmax=e.max(),colors='k', linestyles='dashed',lw=3)
  plt.show()
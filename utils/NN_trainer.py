class Trainer(object):
    """
    The core neural net trainer class
    Author: Dr. Tirthajyoti Sarkar
    Requirements: Numpy, Pandas, Matplotlib, Scikit-learn, Keras (TensorFlow)
    """

    def __init__(self):
        """
        Initialization
        """
        # Dataset
        self.df = None
        # Input and output columns (features)
        self.output_cols = None
        self.input_cols = None
        # Model
        self.model = None
        # Training and test sets
        self.train = None
        self.test = None
        # Scaling objects (a dictionary)
        self.scalers = None
        # Keras history object
        self.history = None
        # Binary status flags
        self.built_ = False
        self.fitted_ = False
        self.preparaed_ = False

    def __str__(self):
        """
        Printing description
        """
        return "Hello, I am a neural net trainer!"

    def __repr__(self):
        """
        Printing description
        """
        return "Hello, I am a neural net trainer!"

   def read_csv(self, file,**kargs):
        """
        Reads a CSV file into a Pandas DataFrame
        You can use all the usual read_csv keywords 
        e.g. reading only first few rows or choosing a specific delimiter
        """
        from pandas import read_csv

        df = read_csv(file,**kargs)
        self.df = df
        
        return self.df
    
    def read_html(self, url,**kargs):
        """
        Reads a HTML table into the internal (Pandas) DataFrame
        (For now) this only works for a single table on the HTML page.
        """
        from pandas import read_html

        df = read_html(url,**kargs)[0]
        if 'Unnamed: 0' in df.columns:
            df.drop('Unnamed: 0',axis=1,inplace=True)
        self.df = df
        
        return self.df

    def show_cols(self):
        """
        Shows the columns of the dataset
        """
        if self.df is None:
            print(
                "No dataset has been read yet! Try again after reading in a dataset (CSV file)"
            )
        else:
            cols_list = list(self.df.columns)
            print("Here are the features:", end=" ")
            for i in cols_list[:-1]:
                print(i, end=", ")
            print("and " + cols_list[-1], end=".")

    def drop_cols(self, cols_list=None):
        """
        Drops a given list of columns from the dataset.
        Used for cleaning up the dataset.
        This method does not return a DataFrame but modifies the internal DataFrame.
        """
        df1 = self.df
        if cols_list is not None:
            df2 = df1.drop(cols_list, axis=1)
            print("Dropped the following columns:", end=" ")
            for i in cols_list[:-1]:
                print(i, end=", ")
            print("and " + cols_list[-1], end=".")
        else:
            print("Nothing to be dropped")
            df2 = df1
        self.df = df2

    def drop_static_cols(self):
        """
        Drops columns which have no variation in data.
        Used for dropping 'un-interesting' features i.e. variables which should not be used in modeling.
        This method does not return a DataFrame but modifies the internal DataFrame.
        """
        cols_to_be_dropped = []
        df1 = self.df
        for c in df1.columns:
            if df1[c].nunique() == 1:
                cols_to_be_dropped.append(c)
        df2 = df1.drop(cols_to_be_dropped, axis=1)
        if len(cols_to_be_dropped) == 0:
            print("Nothing to be dropped")
        if len(cols_to_be_dropped) == 1:
            print("Dropped the following column:", cols_to_be_dropped[0])
        if len(cols_to_be_dropped) > 1:
            print("Dropped the following columns:", end=" ")
            for i in cols_to_be_dropped[:-1]:
                print(i, end=", ")
            print("and " + cols_to_be_dropped[-1], end=".")
            
            df2 = df1
        self.df = df2

    def show_missing(self):
        """
        Shows how many missing values are there
        """
        print("The following table shows the number of missing values in the dataset (by the features)\n")
        print(self.df.isna().sum())

    def drop_missing(self):
        """
        Drops rows of data which have missing values.
        This method does not return a DataFrame but modifies the internal DataFrame.
        """
        df1 = self.df
        df2 = df1.dropna(axis=0)

        self.df = df2

    def set_outputs(self, output_cols=None):
        """
        Sets the list the output columns (to be modeled)
        """
        if output_cols is not None:
            self.output_cols = output_cols
            assert (type(output_cols)) == list, print(
                "The argument must be a list (of features)"
            )
            if len(output_cols) > 1:
                print("Following features are set as output (to be modeled):", end=" ")
                for i in output_cols[:-1]:
                    print(i, end=", ")
                print("and " + output_cols[-1], end=".")
            else:
                print(
                    "Following feature is set as output (to be modeled):",
                    output_cols[0],
                )
        else:
            print("No list of features supplied to be set as output (to be modeled)")

    def set_single_output(self, output_var=None):
        """
        Sets a single variable column (to be modeled).
        Arguments:
            output_var: Name of the variable to be set as output. Must be a string.
        """
        if output_var is not None:
            self.output_var = output_var
            assert (type(output_var)) == str, print(
                "The argument must be a string (name of the variable)"
            )
            print("Following feature is set as output (to be modeled):", output_var)
        else:
            print("No feature supplied to be set as output (to be modeled)")
        
        if str(output_var) not in list(self.df.columns):
            print(f"Warning! The designated output variable {str(output_var)} is not in the list of columns of the dataset!")

    def set_inputs(self, input_cols=None):
        """
        Sets the list the input columns (to be used for modeling)
        """
        if input_cols is not None:
            self.input_cols = input_cols
            assert (type(input_cols)) == list, print(
                "The argument must be a list (of features)"
            )
            if len(input_cols) > 1:
                print("Following features are set as input:", end=" ")
                for i in input_cols[:-1]:
                    print(i, end=", ")
                print("and " + input_cols[-1], end=".")
            else:
                print("Following feature is set as input:", input_cols[0])
        else:
            print("No list of features supplied to be set as input")

    def plot_xy(self, x, y):
        """
        Pretty plot of variables x vs. y
        """
        import matplotlib.pyplot as plt

        data = self.df
        xlbl = str(x)
        ylbl = str(y)
        x = data[xlbl]
        y = data[ylbl]
        plt.figure(figsize=(7, 5))
        plt.title(f"Plot of {xlbl} vs. {ylbl}", fontsize=17)
        plt.scatter(x, y, c="orange", edgecolor="k", s=80)
        plt.grid(True)
        plt.xlabel(xlbl, fontsize=15)
        plt.ylabel(ylbl, fontsize=15)
        plt.show()

    def prepare_data(self, test_size=0.3, scaler_type="MinMax"):
        """
        Processes dataset to make it ready for neural net training
        Arguments
            test_size: Fraction of the test set size. Default is 0.3 or 30%.
            scaler_type: Type of scaling method used on the data. One of 'MinMax', 'Standard', or None. 
            This uses the MinMaxScaler or StandardScaler estimators of the Scikit-learn package to fit and transform the dataset.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

         # Checking if the data has been read
        if self.df is None:
            print("No data has been read yet. Cannot do processing!")
            return None
        
        df1 = self.df
        X = df1[self.input_cols]
        y = df1[str(self.output_var)]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        # Min-max scaler from Scikit-learn
        if scaler_type == "MinMax":
            scalerx = MinMaxScaler()
            scalery = MinMaxScaler()
            X_train_scaled = scalerx.fit_transform(X_train)
            X_test_scaled = scalerx.fit_transform(X_test)
            y_train_scaled = scalery.fit_transform(y_train.values.reshape(-1, 1))
            y_test_scaled = scalery.fit_transform(y_test.values.reshape(-1, 1))
        # Standard scaler from Scikit-learn
        elif scaler_type == "Standard":
            scalerx = StandardScaler()
            scalery = StandardScaler()
            X_train_scaled = scalerx.fit_transform(X_train)
            X_test_scaled = scalerx.fit_transform(X_test)
            y_train_scaled = scalery.fit_transform(y_train.values.reshape(-1, 1))
            y_test_scaled = scalery.fit_transform(y_test.values.reshape(-1, 1))
        # No scaling
        elif scaler_type is None:
            X_train_scaled = X_train
            X_test_scaled = X_test
            y_train_scaled = y_train
            y_test_scaled = y_test

        self.train, self.test = {}, {}
        self.train["X_train_scaled"] = X_train_scaled
        self.test["X_test_scaled"] = X_test_scaled
        self.train["y_train_scaled"] = y_train_scaled
        self.test["y_test_scaled"] = y_test_scaled
        self.train["X_train"] = X_train
        self.test["X_test"] = X_test
        self.train["y_train"] = y_train
        self.test["y_test"] = y_test

        self.scalers = {}
        self.scalers["scalerx"] = scalerx
        self.scalers["scalery"] = scalery

        self.preparaed_ = True

    def build_model(
        self,
        input_dim=None,
        n_hidden_layers=2,
        layer_architecture=[10, 10],
        activation_func="relu",
    ):
        """
        Builds and returns a feed-forward multi-layer neural network
        n_hidden_layers: Number of hidden layers
        layer_architecture: Nuerons per hidden layer, must be a list with neurons per layer
                            e.g. [10,5] and length must match the number of layers
        activation_func: The activation function used in the hidden layer, such as 'relu', 'sigmoid', or 'tanh'. Note that the output layer activation is linear for the regresstion task performed here.
        """
         # Checking if the data preparation has been done
        if self.preparaed_ is False:
            print("Data preparation not done yet. Should not build a model!")
            return None

        import keras
        from keras.models import Sequential
        from keras.layers import Dense, Dropout
       
        if input_dim is None:
            input_dim = len(self.input_cols)
        model1 = Sequential()
        model1.add(
            Dense(layer_architecture[0], input_shape=(input_dim,), activation="relu")
        )
        if len(layer_architecture) > 1:
            for i in layer_architecture[1:]:
                model1.add(Dense(i, activation="relu"))
        model1.add(Dense(1))
        self.model = model1
        self.built_ = True

        return model1

    def train_model(
        self, 
        model=None, 
        lr=0.001, 
        n_epochs=100, 
        batch_size=1, 
        verbose=1,
        ):
        """
        Trains the DNN model
        Arguments:
            lr = Learning rate
            n_epoch = Number of epochs for training
        """        
        # Checking if the model has been defined
        if self.built_ is False:
            print("Model has not been built yet. Cannot train!")
            return None
        
        # Checking if the data preparation has been done
        if self.preparaed_ is False:
            print("Data preparation not done yet. Cannot train!")
            return None
        
        from keras.optimizers import adam

        if model is None:
            model = self.model

        optimizer = adam(lr=lr)
        model.compile(loss="mse", optimizer=optimizer, metrics=["mse"])

        X_train_scaled = self.train["X_train_scaled"]
        y_train_scaled = self.train["y_train_scaled"]

        history1 = model.fit(
            X_train_scaled,
            y_train_scaled,
            epochs=n_epochs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.fitted_ = True
        self.history = history1

        return history1

    def plot_training_error(self):
        """
        Plots training error
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Checking if the model training has been done
        if self.fitted_ is False:
            print("Training has not been done yet. Cannot plot!")
            return None

        n_epochs = self.history.params["epochs"]
        x = np.arange(1, n_epochs + 1)
        y = np.sqrt(self.history.history["mean_squared_error"])

        plt.figure(figsize=(9, 5))
        plt.title(f"Training RMSE plot", fontsize=17)
        marker_style = dict(marker="o", markersize=12, markerfacecolor="tab:red")
        plt.plot(x, y, c="k", lw=2, **marker_style)
        plt.grid(True)
        plt.xlabel("Epochs", fontsize=15)
        plt.ylabel("RMSE error (training)", fontsize=15)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

    def predict_testset(self, rescale=True):
        """
        Test set prediction
        """

        # Checking if the model training has been done
        if self.fitted_ is False:
            print("Training has not been done yet. Cannot predict!")
            return None

        X_test_scaled = self.test["X_test_scaled"]
        if rescale:
            scalery = self.scalers["scalery"]
            preds_scaled = self.model.predict(X_test_scaled)
            preds = scalery.inverse_transform(preds_scaled)
        else:
            preds = self.model.predict(X_test_scaled)

        return preds

    def benchmark_test(self, rescale=True):
        """
        Benchmarks the predictions on the test set with the ground truth
        """
        import matplotlib.pyplot as plt

        # Checking if the model training has been done
        if self.fitted_ is False:
            print("Training has not been done yet. Cannot predict!")
            return None

        y_test_scaled = self.test["y_test_scaled"]
        if rescale:
            preds = self.predict_testset(rescale=True)
            scalery = self.scalers["scalery"]
            y_test = scalery.inverse_transform(y_test_scaled)

            var = str(self.output_var)
            plt.figure(figsize=(8, 5))
            plt.title(f"True {var} vs. predicted values for the test set", fontsize=18)
            plt.plot(y_test, y_test, color="k", linestyle="--", lw=2)
            plt.scatter(y_test, preds, color="red")
            plt.legend(["True values", "Predicted values (test set)"])
            plt.grid(True)
            plt.xlabel(f"True values of {var} in the test set", fontsize=15)
            plt.ylabel(f"Predicted values of {var} on the test set", fontsize=15)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.show()

    def predict(self, inputx, rounding=3):
        """
        Computes model prediction for a single input feature vector
        Arguments:
            inputx: The input feature vector. 
            The dimension must match that of the training vector, with which the model was trained.
            rounding: Number of digits to retain for the output.
        """
        import numpy as np

        # Checking if the model training has been done
        if self.fitted_ is False:
            print("Training has not been done yet. Cannot predict!")
            return None

        scalerx = self.scalers["scalerx"]
        scalery = self.scalers["scalery"]
        model = self.model

        transformed_array = scalerx.transform(np.array(inputx).reshape(1, -1))
        pred_scaled = model.predict(transformed_array)
        pred = scalery.inverse_transform(pred_scaled)

        return round(float(pred), rounding)

    def rmse_error(self, X_test, y_test, rescale=True):
        """
        Evaluates the RMSE (Root-mean-square-error) score of a given dataset against the model prediction
        Arguments:
            X_test: The input feature vector for which the error is sought
            y_test: Output vector (ground truth)
        """
        import numpy as np

        # Checking if the model training has been done
        if self.fitted_ is False:
            print("Training has not been done yet. Cannot predict!")
            return None

        scalerx = self.scalers["scalerx"]
        scalery = self.scalers["scalery"]
        y_test = y_test.values.reshape(-1, 1)

        X_test_scaled = scalerx.transform(X_test)
        preds_scaled = self.model.predict(X_test_scaled)
        preds = scalery.inverse_transform(preds_scaled)
        result = np.sqrt(np.mean((preds - y_test) ** 2))

        return result

    def mae_error(self, X_test, y_test, rescale=True):
        """
        Evaluates the MAE (Mean absolute error) acore of a given dataset against the model prediction
        Arguments:
            X_test: The input feature vector for which the error is sought
            y_test: Output vector (ground truth)
        """
        import numpy as np

        # Checking if the model training has been done
        if self.fitted_ is False:
            print("Training has not been done yet. Cannot predict!")
            return None

        scalerx = self.scalers["scalerx"]
        scalery = self.scalers["scalery"]
        y_test = y_test.values.reshape(-1, 1)

        X_test_scaled = scalerx.transform(X_test)
        preds_scaled = self.model.predict(X_test_scaled)
        preds = scalery.inverse_transform(preds_scaled)
        result = np.mean(np.abs(preds - y_test))

        return result

    def rmse_test(self):
        """
        Computes the test set RMSE
        """

        X_test = self.test['X_test']
        y_test = self.test['y_test']
        error = self.rmse_error(X_test,y_test)

        return round(error,3)

    def save_model(self,filename=None):
        """
        Saves the fitted model in a h5 file
        """
        if self.fitted_:
            model = self.model
            if filename is not None:
                filename = filename
            else:
                var = str(self.output_var)
                filename = "model_" + var + ".h5"
            model.save(filename)
        else:
            print("Nothing to be saved. Model not fitted yet!")

    def save_scalers(self):
        """
        Saves the scaling objects
        """
        import joblib

        if self.preparaed_:
            var = str(self.output_var)
            scalerx_filename = "scalerx_" + var + ".sav"
            scalery_filename = "scalery_" + var + ".sav"

            scalerx = self.scalers["scalerx"]
            scalery = self.scalers["scalery"]

            joblib.dump(scalerx, scalerx_filename)
            joblib.dump(scalery, scalery_filename)
        else:
            print("Nothing to be saved. Data not prepared yet!")

    def model_summary(self):
        """
        Prints a summary of the Keras model
        """
        if self.built_:
            print(self.model.summary())
        else:
            print("No model has been built yet!")

    def model_in_plain_english(self):
        """
        Summarizes the DNN model in plain English layer by layer.
        """
        if self.built_ is False:
            print("No model has been built yet!")
            return None

        config = self.model.get_config()
        if type(config)==dict:
            d = config['layers']
        else:
            d = config
        num_layers = len(d)
        total_param = self.model.count_params()
        print("Here is the model description")
        print("=" * 75)
        for i in range(num_layers):
            if i == num_layers - 1:
                print("Layer No.", i + 1, "(output layer)")
            print("Layer No.", i + 1)
            print("Number of neurons:", d[i]["config"]["units"])
            print("Activation function:", d[i]["config"]["activation"])
            print("-" * 40)
        print(f"In total, there are {total_param} parameters in this model!")
        print()

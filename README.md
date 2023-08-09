# Project Overview
... Some information to the goal of the project, the data sets provided and the result...


# Project Components
The project includes the following three components.

### 1. ETL Pipeline
In a Python script, process_data.py, a data cleaning pipeline is executed that:

Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database

### 2. ML Pipeline
In a Python script, train_classifier.py, a machine learning pipeline is executed that:

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file

### 3. Flask Web App
A Flask web app is provided, that can be executed with the run.py script. It includes an input field for a message that you would like to be classified with its results and it shows some visualizations to the training data.

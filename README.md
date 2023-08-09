# Project Overview
Using data engineering to construct ETL and NLP workflows and develop a Flask-based application for disaster relief.

In this project, I am using my expertise in data engineering and data science to explore disaster-related data from Figure 8. The goal is to use an API to create a model that can effectively categorize messages about disasters.

The project includes a dataset of authentic messages sent during various disaster events. I am in the process of developing a machine learning pipeline to classify these incidents and facilitate their targeted routing to the appropriate disaster response organizations.

The project includes the creation of a web application tailored to the needs of emergency personnel. This platform allows the entry of new reports, which are then subjected to classification according to several categories. In addition, the web application provides graphical representations of the data.

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

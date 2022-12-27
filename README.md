# Disaster Response Pipeline Project
## Table of Contents
1. [Introduction](https://github.com/flstahl/udacity-disaster-response-pipeline#introduction)
2. [File Descriptions](https://github.com/flstahl/udacity-disaster-response-pipeline#file-descriptions)
3. [Installation](https://github.com/flstahl/udacity-disaster-response-pipeline#installation)
4. [Instructions](https://github.com/flstahl/udacity-disaster-response-pipeline#instructions)


## Introduction
This project is my final project submission for the chapter Data Engineering of the Udacity Data Science Nano Degree Program.

In this project, we have built a web application allowing to classify messages sent during disasters into one of 36 categories. The application uses a Machine Learning model which was trained on a set of pre-labeled real-life examples. 
After a series of data engineering and cleaning steps, the multiclass-multioutput model is trained and stored. Through a web app, users can interact with the model and classify unseen messages. Furthermore, some visualizations of the underlying data can be found.




## File Descriptions
### Folder: app
**run.py** - python script to launch web application.<br/>
Folder: templates - web dependency files (go.html & master.html) required to run the web application.

### Folder: data
**disaster_messages.csv** - real messages sent during disaster events (provided by Figure Eight)<br/>
**disaster_categories.csv** - categories of the messages<br/>
**process_data.py** - ETL pipeline used to load, clean, extract feature and store data in SQLite database<br/>
**ETL Pipeline Preparation.ipynb** - Jupyter Notebook used to prepare ETL pipeline<br/>
**DisasterResponse.db** - cleaned data stored in SQlite database

### Folder: models
**train_classifier.py** - ML pipeline used to load cleaned data, train model and save trained model as pickle (.pkl) file for later use<br/>
**classifier.pkl** - pickle file contains trained model<br/>
**ML Pipeline Preparation.ipynb** - Jupyter Notebook used to prepare ML pipeline

## Installation
All required libraries are included in the Anaconda distribution.

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


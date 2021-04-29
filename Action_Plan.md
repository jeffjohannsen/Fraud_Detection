# Fraud Detection
# Project Outline and Action Plan

## Table of Contents
* [Project Goals](#Project-Goals)
    * [Central Questions](#Central-Questions)
    * [Methods and Concepts](#Methods-and-Concepts)
    * [Tech Used](#Tech-Used)
* [Current Focus](#Current-Focus)
* [Next Steps and Notes](#Next-Steps-and-Notes)
    * [Data Acquisition](#Data-Acquisition)
    * [Data Storage](#Data-Storage)
    * [Data Cleaning and Processing](#Data-Cleaning-and-Processing)
    * [Feature Engineering](#Feature-Engineering)
    * [Analysis and Visualization](#Analysis-and-Visualization)
    * [NLP](#NLP)
    * [Machine Learning](#Machine-Learning)
    * [Deployment and Production](#Deployment-and-Production)

# Project Goals

### Main Goal

The main goal of this project is to predict the likelihood of event data being fraud in order to flag potential fraudulent records for further investigation. This will improve the user experience and decrease the risk to the company of being responsible for fraudulent events that were purchased by users.

### Secondary Goals

* Present a full data lifecyle project from acquiring live streaming data thru to displaying realtime results in a dashboard/web app.
* Create an intuitive and functional UX/UI for fraud investigations personal to utilize.

## Central Questions

* How well can fraud be predicted in this situation?
* Which features are the most useful? Text or Other?
* Which model type performs the best in this situation?

## Methods and Concepts

Data Analysis
Machine Learning - Logistic Regression, Random Forest
Model Interpretation
Natural Language Processing
Dashboards
Web-Apps

## Tech Used

Code
* Python
    * Data Analysis - Pandas, Dataprep.eda, Geopandas, Wordcloud, geoplot, shapely
    * Visualization - Matplotlib, Seaborn, Folium
    * Machine Learning - Sklearn, SHAP, eli5
    * Other - Numpy, Beautiful Soup, Flask, Bootstrap, Plotly-Dash, Pickle
* SQL - Postgres, SQLAlchemy, psql
Other
* Google Data Studio
* AWS RDS, S3
* Git/Github

# Current Focus

1. Phase 2 - 2nd iteration from start to finish. Simplify, Clean, Test, Document

# Next Steps and Notes

## Data Acquisition

* Work on failsafes for server request failures.
* Try to determine server requests limits if any. 

## Data Storage

* Work plan for saving RDS to S3 during extended downtimes.
* SQL work for connecting DB to Dashboard and creating further metrics.

## Data Cleaning and Processing

* Mostly done. Clean up code.

## Feature Engineering

* Do another iteration but the focus here is more on text/NLP work.

## Analysis and Visualization

* Mostly done.
* Focus on dashboard and web app for further visuals.

## NLP

* Hopefully turn model into a fully text based feature set.
* Spacy linguistic concepts vectorized and evaluated. Dig Deeper here.
* Deep Learning

## Machine Learning

* Tune Logistic Regression and Random Forest more fully. Consider entire pipeline impacts.
* Try new models for comparison - XGBoost with Hyperparameter tuning
* Deep Learning for both main model and advanced NLP work.

## Deployment and Production

* Improve Dashboard
    * Plotly/Dash
* Run new record processing in the cloud.
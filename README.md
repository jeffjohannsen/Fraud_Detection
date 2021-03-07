<p align="center">
<img src="https://images-na.ssl-images-amazon.com/images/I/51ZrPDH14vL._AC_.jpg">
</p>

# Fraud Detection Project
*by Jeff Johannsen, Devon Silk, Jess Curley, Pedro Meyer*

Fraud is a major concern for any company. The goal of this project is to create a system that monitors event transactions to accurately detect fraudulent events and present the results of the fraud predictions in a easy to use user interface.

# Exploring the Data

The data provided by the company was reasonably complete, although complex. This data contained a mix of categorical and numerical data in varied formats including html, datetime objects, lists of dictionaries, along with normal text and numerical values.

The first step on our exploration was to split the data into fraudulent and not fraudulent transactions. The acct_type feature provided was condensed to give us a fraudulent record count of 1033 and a non-fraudulent record count of 10436 in our training data.

### Original Data

|    | acct_type   |   approx_payout_date |   body_length |   channels | country   | currency   |   delivery_method | description          | email_domain      |   event_created |   event_end |   event_published |   event_start |   fb_published |     gts |   has_analytics |   has_header |   has_logo | listed   | name                 |   name_length |   num_order |   num_payouts |   object_id | org_desc             |   org_facebook | org_name             |   org_twitter | payee_name   | payout_type   | previous_payouts     |   sale_duration |   sale_duration2 |   show_map | ticket_types         |   user_age |   user_created |   user_type | venue_address        | venue_country   |   venue_latitude |   venue_longitude | venue_name           | venue_state   |
|---:|:------------|---------------------:|--------------:|-----------:|:----------|:-----------|------------------:|:---------------------|:------------------|----------------:|------------:|------------------:|--------------:|---------------:|--------:|----------------:|-------------:|-----------:|:---------|:---------------------|--------------:|------------:|--------------:|------------:|:---------------------|---------------:|:---------------------|--------------:|:-------------|:--------------|:---------------------|----------------:|-----------------:|-----------:|:---------------------|-----------:|---------------:|------------:|:---------------------|:----------------|-----------------:|------------------:|:---------------------|:--------------|
|  0 | premium     |           1355097600 |         43227 |          8 | CA        | CAD        |                 1 | <p class="MsoNorm... | cedec.ca          |      1351629944 |  1354665600 |       1.35163e+09 |    1354654800 |              0 |   85.29 |               0 |            0 |          1 | y        | Ten Follow-Up Mis... |            46 |           5 |            31 |     4710425 | <p><strong>CEDEC ... |              9 | CEDEC Small Busin... |             9 |              | ACH           | [{'name': '', 'cr... |              35 |               35 |          1 | [{'event_id': 471... |        722 |     1289255732 |           3 | 475 rue Frontière    | CA              |          45.0475 |         -73.5845  | Café Hemmingford     | Quebec        |
|  1 | premium     |           1334775600 |           664 |          8 | US        | USD        |                 1 | Please join us fo... | yahoo.com         |      1328839445 |  1334343600 |       1.3291e+09  |    1334340000 |              0 |   50    |               0 |          nan |          0 | y        | Student Career Da... |            57 |           8 |             6 |     2930589 | <p>The IIDA Cleve... |             53 | Cleveland Akron I... |             0 |              | ACH           | [{'name': '', 'cr... |              61 |               64 |          1 | [{'event_id': 293... |        205 |     1311116195 |           3 | 1122 Prospect Avenue | US              |          41.4995 |         -81.6822  | Ohio Desk            | OH            |
|  2 | premium     |           1380684600 |           923 |         11 | US        | USD        |                 0 | <p style="text-al... | projectgradli.org |      1374514499 |  1380252600 |       1.37452e+09 |    1380243600 |              0 | 9550    |               0 |          nan |          1 | y        | DREAMING UNDER TH... |            24 |           5 |             0 |     7540419 | <p style="text-al... |              0 | Project GRAD Long... |             0 |              | ACH           | [{'name': '', 'cr... |              66 |               66 |          1 | [{'event_id': 754... |          0 |     1374514498 |           1 | 1 Davis Ave          | US              |          40.728  |         -73.6019  | The Cradle of Avi... | NY            |
|  3 | premium     |           1362142800 |          4417 |         11 | IE        | EUR        |                 0 | <p><strong>&nbsp;... | gmail.com         |      1360608512 |  1361710800 |       1.36061e+09 |    1361680200 |              0 | 1813.36 |               0 |          nan |          1 | n        | King of Ping         |            12 |          51 |             1 |     5481976 | <p>Mabos is a mul... |             27 | mabos                |            11 |              | ACH           | [{'name': '', 'cr... |              12 |               12 |          1 | [{'event_id': 548... |         50 |     1356308239 |           3 | 8 Hanover Quay       | IE              |          53.3438 |          -6.23214 | mabos                | County Dublin |
|  4 | premium     |           1358746200 |          2505 |          8 | US        | USD        |                 1 | <p style="text-al... | gmail.com         |      1353197931 |  1358314200 |       1.35339e+09 |    1358308800 |              0 |  105.44 |               0 |            0 |          1 | y        | Everyone Communic... |            50 |           1 |             1 |     4851467 | <p><span style="f... |             14 | Kim D. Moore - Co... |            10 | Kim D. Moore | CHECK         | [{'name': 'Kim D.... |              57 |               59 |          0 | [{'event_id': 485... |       1029 |     1264268063 |           4 |                      | None            |         nan      |         nan       | None                 | None          |

# Data Processing

![Data Cleaning Steps](images/data_cleaning_steps.png)

The data cleaning process consisted of five main steps:
1. Convert Datatypes
    * Convert unix timestamps to datetime objects
    * Convert yes/no (y/n) strings to boolean values (1, 0)
2. Deal With Nan/Null Values
    * This was a little more complicated than I thought at first glance. I tried to keep as much of the data as possible so I replaced nulls with either -1 or Unknown.
    * There was a lot of features that had empty strings which aren't immediately recognized as nan/null values. I located these and replace them with 'Unknown'.
    * The amount of nan/null values was a solid predictor of fraud so it was a good thing not to simply drop any data with nan/nulls.
3. Remove Unnecessary or Unusable Features
    * Some features provided no value or overlapped with other features.
    * Some features caused data leakage for the modeling process.
4. Condense and aggregate nested features (list of previous payouts and dicts of ticket information).
5. Convert html features into plain text using the Beautiful Soup library.

# Data Analysis

## Number of Missing Values

Exploration of the data showed that multiple features had different common values for fraudulent events than non-fraudulent events. One interesting marker of a fraudulent events is that the user that created the event provided less information about it. As you can see below fraudulent events are more likely to have more missing information.

![Number of Missing Values](images/total_empty_values_comparison.png)

# NLP Feature Engineering

![NLP Steps](images/nlp_steps.png)

A few of the original data features consisted of text that needed to be explored and processed into a usable format for modeling. These text features contained some of the most useful information so I utilized some NLP modeling techniques to turn the raw text into probabilities of an event being fraudulent or not. These probabilities are then combined with the other data features for use in the main machine learning model. 

Four text features were utilized:
1. Event Name
2. Event Description
3. Organization Name
4. Organization Description
As you can see from the below examples, there are some interesting differences between which words appear in fraudulent events compared to non-fraudulent events.

### Event Name

![Event Name Wordclouds](images/name_wordclouds.png)

### Event Description

![Event Description Wordclouds](images/description_wordclouds.png)

# Modeling

## Model Setup

In order to ensure that the modeling process provided accurate and relavent results the following model preparation steps were crucial to the process.

### Metric Selection

The current estimate is that only 10% of events are fraudulent. This creates a situation where it is fairly easy to get a high accuracy (>90%) but this does not necessarily mean that the system is successful. The main goal is to detect as many of the fraudulent events (true positives) as possible. With this in mind, recall is the metric that will be the main focus, though other metrics (Accuracy, Precision, F1-Score) will not be completely ignored. 

### Data Integrity

Multiple steps were taken to ensure that there is no data leakage throughout the system. A cross validation system was used for model tuning with the main working dataset. A separate dataset was used for testing. A third holdout dataset is also available for further testing to ensure data integrity. Each of these datasets were created using random sampling from the original data before any processing was completed.

### Model Tuning

A grid search cross validation process was used to locate the optimal hyperparameters for each of the model types.

## Model Selection, Results, and Important Features

Multiple models were tested to determine which provided the best predictions 

### Logistic Regression





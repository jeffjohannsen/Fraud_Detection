<p align="center">
<img src="https://images-na.ssl-images-amazon.com/images/I/51ZrPDH14vL._AC_.jpg">
</p>

# Fraud Detection Project
*by Devon Silk, Jeff Johannsen, Jess Curley, Pedro Meyer*

Fraud is a major concern for any company. We set out to find patterns on event purchases to develop a method to accurately spot fraud. At the end of this project, this project aims to be presented on a simple application software.

# Exploring the Data

The data given by the company was reasonably complete, although complex. This data contained a mix of categorical and numerical data in varied formats including html, datetime objects, lists of dictionaries, and normal text and numerical values.

The first step on our exploration was to split the data into fraudulent and not fraudulent transactions. The acct_type feature provided was condensed to give us a fraudulent record count of 1033 and a non-fraudulent record count of 10436.

### Original Data

|    | acct_type   |   approx_payout_date |   body_length |   channels | country   | currency   |   delivery_method | description          | email_domain      |   event_created |   event_end |   event_published |   event_start |   fb_published |     gts |   has_analytics |   has_header |   has_logo | listed   | name                 |   name_length |   num_order |   num_payouts |   object_id | org_desc             |   org_facebook | org_name             |   org_twitter | payee_name   | payout_type   | previous_payouts     |   sale_duration |   sale_duration2 |   show_map | ticket_types         |   user_age |   user_created |   user_type | venue_address        | venue_country   |   venue_latitude |   venue_longitude | venue_name           | venue_state   |
|---:|:------------|---------------------:|--------------:|-----------:|:----------|:-----------|------------------:|:---------------------|:------------------|----------------:|------------:|------------------:|--------------:|---------------:|--------:|----------------:|-------------:|-----------:|:---------|:---------------------|--------------:|------------:|--------------:|------------:|:---------------------|---------------:|:---------------------|--------------:|:-------------|:--------------|:---------------------|----------------:|-----------------:|-----------:|:---------------------|-----------:|---------------:|------------:|:---------------------|:----------------|-----------------:|------------------:|:---------------------|:--------------|
|  0 | premium     |           1355097600 |         43227 |          8 | CA        | CAD        |                 1 | <p class="MsoNorm... | cedec.ca          |      1351629944 |  1354665600 |       1.35163e+09 |    1354654800 |              0 |   85.29 |               0 |            0 |          1 | y        | Ten Follow-Up Mis... |            46 |           5 |            31 |     4710425 | <p><strong>CEDEC ... |              9 | CEDEC Small Busin... |             9 |              | ACH           | [{'name': '', 'cr... |              35 |               35 |          1 | [{'event_id': 471... |        722 |     1289255732 |           3 | 475 rue Frontière    | CA              |          45.0475 |         -73.5845  | Café Hemmingford     | Quebec        |
|  1 | premium     |           1334775600 |           664 |          8 | US        | USD        |                 1 | Please join us fo... | yahoo.com         |      1328839445 |  1334343600 |       1.3291e+09  |    1334340000 |              0 |   50    |               0 |          nan |          0 | y        | Student Career Da... |            57 |           8 |             6 |     2930589 | <p>The IIDA Cleve... |             53 | Cleveland Akron I... |             0 |              | ACH           | [{'name': '', 'cr... |              61 |               64 |          1 | [{'event_id': 293... |        205 |     1311116195 |           3 | 1122 Prospect Avenue | US              |          41.4995 |         -81.6822  | Ohio Desk            | OH            |
|  2 | premium     |           1380684600 |           923 |         11 | US        | USD        |                 0 | <p style="text-al... | projectgradli.org |      1374514499 |  1380252600 |       1.37452e+09 |    1380243600 |              0 | 9550    |               0 |          nan |          1 | y        | DREAMING UNDER TH... |            24 |           5 |             0 |     7540419 | <p style="text-al... |              0 | Project GRAD Long... |             0 |              | ACH           | [{'name': '', 'cr... |              66 |               66 |          1 | [{'event_id': 754... |          0 |     1374514498 |           1 | 1 Davis Ave          | US              |          40.728  |         -73.6019  | The Cradle of Avi... | NY            |
|  3 | premium     |           1362142800 |          4417 |         11 | IE        | EUR        |                 0 | <p><strong>&nbsp;... | gmail.com         |      1360608512 |  1361710800 |       1.36061e+09 |    1361680200 |              0 | 1813.36 |               0 |          nan |          1 | n        | King of Ping         |            12 |          51 |             1 |     5481976 | <p>Mabos is a mul... |             27 | mabos                |            11 |              | ACH           | [{'name': '', 'cr... |              12 |               12 |          1 | [{'event_id': 548... |         50 |     1356308239 |           3 | 8 Hanover Quay       | IE              |          53.3438 |          -6.23214 | mabos                | County Dublin |
|  4 | premium     |           1358746200 |          2505 |          8 | US        | USD        |                 1 | <p style="text-al... | gmail.com         |      1353197931 |  1358314200 |       1.35339e+09 |    1358308800 |              0 |  105.44 |               0 |            0 |          1 | y        | Everyone Communic... |            50 |           1 |             1 |     4851467 | <p><span style="f... |             14 | Kim D. Moore - Co... |            10 | Kim D. Moore | CHECK         | [{'name': 'Kim D.... |              57 |               59 |          0 | [{'event_id': 485... |       1029 |     1264268063 |           4 |                      | None            |         nan      |         nan       | None                 | None          |

# Data Cleaning

The data cleaning process consisted of three main steps.
1. Convert Datatypes
    * Convert unix timestamps to datetime objects
    * Convert yes/no (y/n) strings to boolean values (1, 0)
2. Deal With Nan/Null Values
    This was a little more complicated than I thought at first glance. I tried to keep as much of the data as possible so I replace nulls with either -1 or Unknown. There was a lot of features that had empty strings which aren't immediately recognized as nan/null values. I located these and replace them with 'Unknown'.
3. Remove Unnecessary of Unusable Features
    * Some features provided no value or overlapped with other features.
    * Some features caused data leakage for the modeling process.

# Feature Engineering

This step consisted of 3 main parts:
1. Featurize non-numerical and categorical features to make them easier to model.
    * Aggregate features created from list and dict type features.
2. Create composite features that are more informative than the originals.
    * 
3. Dig into text data using NLP tools.
    * 

These are two of the many features we compared:

![userage](images/user_agecomparisson.png)
<br>
![eventend](images/event_endcomparisson.png)

After going through this process with the other features, we created a dataframe of all the features we wanted to use, however, not all data types in this table could be used for our prediction model. Therefore, we had more cleanup to do.

We created a new column that correlates 'fraud values' to 1 and 'not fraud' values to 0.

We then turned all of our non-numerical values into numbers (featurized) so our prediction model can use them. 

For the "Event Description", we cleaned up the *html* format to get the text and found the most used words to get a sense for what kinds of events we were working with:

 |   | Top 10 Not Fraud Words|
----------|---------
  0 | {'getaway, charleston, winthrop, br, dinner, noon, winter, metro, waiver, strong'}                        |\n|  
  1 | {'developmental, desired, 2010, profile, results, 11pt, span, strong, li, training'}                      |\n|  
  2 | {'repeat, background, irishtabletennis, template, interface, sport, li, bullet, url, gif'}                |\n|  
  3 | {'silman, contracts, filmmakers, press, film, james, writers, rights, lawyer, morrow'}                    |\n|  
  4 | {'razorsharks, bluegrass, stallions, rochester, league, basketball, premier, come, join, donderdagavond'} |
<br>
<br>

 |  | Top 10 Fraud Words|
 ---:|:------------------
2 | {'party, gras, mardi, line, wear, dancing, shoes, paparazzi, umbrella, gold'}                  |\n|  
3 | {'span, exhibition, underline, london, decoration, properties, abroad, buying, text, style'}   |\n|  
4 | {'zumba, electric, eastside, easy, economy, education, effective, electrcity, electrically, ease'}|\n|  
5 | {'site, mowbray, melton, thrussington, service, food, strong, units, miles, surrounding'}       |\n|  
6 | {'213, dial, 226, 465905, et, 0400, scheduled, today, 30pm, number'}                           |\n|  
7 | {'atlanta, width, img, height, alt, nomarrow, stay, online, bone, marrow'}
<br>

Some of the _html_ functions leaked to our final result, however, those were mostly dealt with on our next step: the featurization process.

<br>
<br>

## Featurizing Categorical Data
The next step was to use a Naive Bayes Model to predict the probability of each transaction to be fraudulent or not. Those probabilities then were used as a feature for our final model.

For example, what is the probability of an event being fraud based on the above event descriptions?

![notproba](./images/notfraudproba.png)

This distribution raised a red flag: How come our model is predicting data disproportionately as "Not Fraud" than "Fraud"? 

The main possible reason is class imbalance, which we calculated to be around 95% of the data being "Not Fraud". In hindsight, this could be fixed with Cross Validation. To handle class imbalance, we used SMOTE to balance our training data that was input into the random forest (discussed later).

## Feature Engineering
<img src="https://github.com/JCurley10/fraud-detection-case-study/blob/jess/images/Screen%20Shot%202020-12-18%20at%202.58.57%20PM.png" width="500" height="250">

A few things came to mind when considering our own experiences with fraud or fraudulent behavior:
- Time between when the user created the account and when they created the event
- Time between when the user created an event and when the event started
- How many previous payouts did the creater have
- The average ticket cost
- The range of ticket prices

## Final Features: Predicting Fraud

After we finilized our final feature matrix, this is what we ended up with:


|    |   Event Created|   Event Start |   Time Between User Created and Event Created |   Time Between Create and Start Date |   Number of Previous Payouts |   User Age |   User Created (Date) |   Average Ticket Cost |   Ticket Cost Range |
|---:|----------------:|--------------:|------------------------------------:|------------------------:|------------------:|-----------:|---------------:|-----------:|-------------:|
|  0 |     1.26274e+09 |   1.26559e+09 |                         3.12576e+06 |             2.85469e+06 |                 0 |         36 |    1.25961e+09 |     208.33 |          525 |
|  1 |     1.29383e+09 |   1.29626e+09 |                         1.28899e+07 |             2.42293e+06 |                49 |        149 |    1.28094e+09 |      35    |            0 |
<br>

From here, we calculated the importance of each feature:

<br>

![importance](images/feature_importances.png)

<br>

![permutation](images/permutation_importances.png)

<br>
Based on those feature importances, we ran those into a Random Forest Model to then predict if an event had fraud or not. This is how our model did:
<br>

![modelresult](images/model_results.png)
The metrics we care about are recall and F1 score. A better recall score  minimizes false negatives, which means we minimize how often we predict it isn’t fraud when it is. The F1 is a combination of recall and precision, since we do also care about not having too many false positives, which could lower user confidence in this event platform.

<br>

## Predicting on Streaming Data 

Drumroll for our fancy fraud detection app being tested in 3...2...1.... 

This is just a proof of concept for now. We did not successfully reach the stage of running this over a flask app. We do have a baseline set up, but it is not connected to this model. 

## Conclusion and Future Work

Since we ultimately did not incorporate text data in our model, we want add our vectorized text as features in our model to test against the other important features. The next goal would be to run the vectorized text data through a Naive Bayes model, and use the predicted probability as new features in the dataset that is run through random forest model.
The final model was very successful in determining fraud. However, it was too successful. That leaves open oportunities for future fine tuning of our models and data.
Further testing for class imbalances and more a/b testing with different features could give us better results. 
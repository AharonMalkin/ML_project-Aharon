# ML_project
## instructions:
Build a system that predicts the chances of a particular user, under certain conditions, to make a purchase while browsing the site.
In the project we will deal with the problem of Binary Classification, in which you must classify records into two categories -
is a user likely to buy (1) or not (0), based on the number of features in the data set. Some of the features are known and some are anonymous.
 

## Data information:
The dataset is from https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset

The dataset depicts sessions of users in an E-Commerce site, when each session can end with a purchase or without. 

*Note: The dataset the lecturer used in the project has different feature names and some of the values are missing, to make things harder :)
The dataset is also splitted randomly to train and test sets, this splitted files are included in this repository*


The dataset consists of 22 features,
Each record in the database corresponds to one session on a page. Data was collected within one year.

### Feature information
**num_of_admin_pages, num_of_info_pages, num_of_product_pages** features, correspond to the number of different pages of a given category visited during the session.

**admin_page_duration, info_page_duration, and product_page_duration** is the total time spent on pages of a given category during the session, while **total_duration** is the sum of them.

**BounceRate** is the average GoogleAnalytics rate of pages visited during the session (the indicator itself is the ratio of sessions in which the user did not perform any action(such as opening another page or downloading a file) to all sessions starting on that page).

**ExitRate** is the average GoogleAnalytics indicator of pages visited during the session (the indicator itself is the ratio of sessions that ended on a given page to all sessions during which this page was visited).

**PageValue** is the average GoogleAnalytics indicator of pages visited during the session (the indicator itself is the share of the page in sessions, which resulted in a purchase multiplied by the purchase value)

**device, internet_browser, Region** are categorical values, each value specifies a different operating system, browser or  Unfortunately, we do not know what exactly the individual values are.

**user_type** specifies whether the user has visited the site before.

**Weekend** specifies whether the session is from a weekend

**Month** specifies the month of the session

**closeeness_to_holiday** is the distance to special days, such as Valentine's Day. 

**A, B, C, D** are Anonymous columns – might be helpful for prediction

**purchase** Label column- determines whether the session has resulted in a purchase.

# Machine Learning project 
### Aharon Malkin

*During my studies at Tel Aviv University, I took a Machine learning course. As part of it, I did a project, In which I planned and built a system that predicts the chance of a user to perform a purchase under certain conditions.
During this project, I worked with Sklearn, Pandas, and Numpy libraries, implemented the knowlege I gain during the course, and learned a lot about the different aspects of Machine learning algorithms.*

## Summary:
In this project, I received a database containing 23 features about sessions (10479 samples) of users on an
online shopping site. I was asked to build a system that predicts the chance of a user to perform a purchase under certain conditions while browsing the site.
First, I performed several steps to investigate and receive an initial understanding of the data. I learned some interesting insights about the correlative behaviour between the features, in addition, by visualizing the data, I discovered a few informative attributes about the features by checking how they distribute and making comparisons between the different features and their values. This step made the data much more comprehensible. 

Afterwards. I performed some pre-processing steps to the data - Such as handling and filling Null values, handling outliers, as well as different values transformations, to make the data much more usable. In addition, I performed dimension reduction and data normalization. During this step, I used different processing methods, which are tailored to each feature individually.

In the Models execution step, I ran 4 different models. The model that was selected is **Random Forest**, which was evaluated by a confusion matrix to examine different metrics of its quality. The model received a final AUC score of 0.931. 
Moreover, I analyzed the features that most significantly affected the purchase, where the most significant one was 'PageValues' (a Google Analytics feature).

## Working method:
### 1 . **Exploration** -
After a quick look at the data, I realized that it consists of both categorical features and numeric, and that there are
features that their values need to be converted from string to numeric. Given the high amount of features, I decided to perform
general visual presentation, and then delve into interesting features. I examined a number of elements:

  a. The correlation between features - as a preparation for the pre-processing phase, assuming that there are features with
     high correlation and it will be possible to unite them.
     
  b. How do the categorial features effect the label- For example, I checked by 'Month' 'Weekend' and 'user_type' features, that at November has the highest amount of      purchases, probably because of the Black Friday and Chinese singles day sales. I also found out that the region of the user doesn't effect his purchase tendency.
  
  c. 'Google Analytics' Feature analysis - I discovered that most of the purchases are occuring on early stages of the session (low 'total_duration'), and the               'PageValues' value during these sessions is very high. 
  
  
  
### 2 . **Pre-proccessing** -

In the given data, the challenge was to adjust all the features and observations to uniform, numeric, and relevant values, therefore I made the following actions: 

### a. Filling Null values:
The rule of thumb that guided me while filling in the missing values is the distinction between categorical features (such as
'User type', 'month', 'region', 'browser', etc.) to the quantitative and continuous ones.

The categorial, were filled according to frequency considerations and recurring values (such as filling the null months with 'May' that is the most frequent one) 
In addition, according to a proportional calculation I understood the distribution of the values in 'internet_browser' and 'device' features and filled in the missing values respectively (e.g. 50% of the data in the frequent browser, and 25%  for each of the second following browsers in their frequency).
In contrast, the numeric features were filled with their median value.

Some of the features were filled by mathematical manipulaton,  such as 'total duration' feature that were filled by sum of all the features that composed it:           admin duration + product duration + info duration.
      
###  b. Creating a new feature:
The motivation for building a new class is a similar distribution between 2 features (found by distplot visualization)
the same conclusion derived from the correlation table where I found that there is a very high correlative match of 0.91 between 'ExitRate' and 'BounceRate'           features. Another rationale for creating the feature is to prepare for the dimension reduction phase and unify features that contribute in the same way.

 ###  c. Handling Outliers:
 First, I checked the distributions of each features, and by using QQPLOT I determined whether it distributes normally. This step helped me identifing outliers afterwards easily.
	
I identified that there are many features that contain a numeric value starting with 0, which represents a quantity or time, and contain outlier values 
that are scattered along the axis. For them, and given that they are not normally distributed, I decided with the help of the **Windsorize** function to set the
outliers values to the 95th percentile of these features from the side of the exceptions, when the values from the other direction weren't cutted, as they aren't
representing exceptions.
      
As for the features that are normally distributed: the values that deviated by 3 segments from the mean of the this feature in both directions, were cut and
filled with the threshold values I set on the 2 sides. 
The outliers were handled carefully, and made it without deleting a single line in the data.
In addition, there are also features that have not been treated at all, which contained a number of Gaussians or an low amount of exceptions, so I refrained from corrupting them.

### d. Handling Categroails:
The motivation that guided me here is the optimization of the data which is many categorical values, meaning that for now, there is no statistical significance that exists for these features. Therfore, I used One-Hot-Encoding method, and created dummy values. By doing that, a 10 additional features that represent 'browser', 'user_type' and 'Month' were added to our dataset.

I assumed that a binary representation is better, so  all the data will exist under the same vector space. On the other hand, features that contained only 2
Categories , were converted to binary values, in order to reduce the amount of dummies in the data.

Another consideration that guided me, is to avoid overfeeting and the Curse of dimensionality before running the models. Therefore I united some of sub-values (such as different versions of the 'Chrome' browser) it helped decreasing the amount of unique features and helped reduce the dimensions later on. 

In addition, I decided to remove features that I found at the exploration step as insufficiently important, including Region. And 'device' - introduced a poor purchase rate for different types of parameters, or the unknown feature 'A', which contained 97 unique categorical values that required the creation of an unreasonable amount of dimensions.

In contrast, the unknown feature 'C' that contained (in my opinion) HTTP requests from the server was selected and reduced to only 2 unique values (error / not error, depending on the request code).

### e. Data Normalization:
Before reducing the dimensions I saw that the data is composed of different ranges, lengths and scales, and therefore we want to avoid unwanted effects and normalize the data.

Based on the qqplot visualization that was performed, I used the StandardScaler function to change the values in the normal distributed features to the same variance 1 and mean 0, and to the non normal features feature the MinMaxScaler function, to rescale them by converting their minimum and maximum values to be between 0 and 1.

### f. Splitting to Train/Validation and Dimension reduction:
After adding the categorical features (we now have 30 in total) I divided the set of the train to 80% / 20% for validation training. The current amount of features affects the uniqueness of each observation and significantly increases the complexity of our data.

Since my goal is to predict a new user's chance to make a purchase on the site, I must optimize the data as much as possible and avoid overfitting and an overly complex model on the one hand and underfitting- when the model fails to express the ratio of observations to the desired output and handle each sample the same.

We can identify that there is a dimensional problem because the number of features is now higher than in the beginning, and I also found quite a few features that haven't contributed too much to the data (Region, Device). Therefore, I decided to use PCA. The assumption taken in this method is the need to maintain as much as possible of the explained variance (99%) while reducing the dimensionality that significantly improves the learning process, while removing the categorical features from the calculation. We can see that the number of components to preserve 99% Of the variance is 9 features.

Since the new features created as a result of the running the PCA are a linear combination and has no bussineswise meaning, I searched for a tool to find the 9 best features that would explain the data most clearly. To do this I used a mutual info classifier that measures the level of dependence between the variables (entropy) in relation to our label. From this, the 9 features that will be run in the model phase were found.

### g. Pre-processing on the Test set:
The given set constitutes about 20% of the complete data and therefore I performed all the pre-processing operations that I made on the train set, also on the test set, while maintaining the focus on the relevant features selected to perform the models. The rule of thumb that guided me in the pre-processing of this set, based on the different parameters of the train set, as we do not want to use the distributions of the test set because in business reality we will not always be able to be aware of the distributions of the test set, and my assumption is that the train set represents the data optimally.

### 3. Running the models:
In this project I selected the two basic models - KNN and Logistic Regression and selected the two advanced models SVM and Random Forest and ran them on the given data.

To evaluate the quality of the model selection, I first created a function that performs K-Fold CV with 5 folds. The output is a graph and a roc auc score on the relevant data set (train / valid) in each model that is run. 
(The function performs a random split each time and therefore the results are reliable)

Selected models:
### a. Logistic Regression: 
First, I ran the GridSearchCV function so that we can use it to determine the best parameters according to standard data that I selected in advance.
By doing so, I found that the stop tolerance criterion would be 0.001 and the 'C' that defines the hyperparameter was set to 0.1, which means high regularization. 

However after running the model with C = 10 I got a similar auc score, so I chose this parameter that punishes less and simulates reality more reliably. The AUC scores obtained were very good for the validation set (0.903) In addition, I tested the Train set and found that the model was not overfitted. (Score 0.904)

### b. KNN:
Here I ran again GridSearchCV which helped to select 100 neighbors and set the main weight by distance, meaning to give meaning to big proximity between neighbors.
The distance calculated by the Manhattan method (absolute values) resulted in an AUC score of 0.9 in validation and a very high score of 0.995 in train, although there is a difference between the two sets, the high amount of neighbors and running the model with more neighbors convinced me that the model was not overfitted.

### c. Random Forest:
In this model, the final parameters set by the GridSearchCV were examined along with dependence between other parameters (such as punishment, and depth of tree versus number of trees) and their effect on the score.

Finally  the following balance is set: I chose forest with 1000 trees with a maximum depth of 8 for each tree (in order to avoid overfitting) and the minimum observations per leaf to 5 and each split to 2.

In addition as the criterion for the quality of the split I chose entropy. An interesting details that was discoverd, is that raising the depth of the tree improves the score to a depth of 8, and then lowers it. Finally we got decent results of 0.922 in the validation set, and 0.96 in the training set, which means the model is not too complicated.

### d. SVM:
On this model I ran Randomizedsearch due to runtime considerations, since the model includes heavy calculations.

In this model, a default penalty (C = 1) and a linear kernel (but 8-dimensional) was decided, and I also turned on the probabilty option in order to be able to calculate probabilities later on. The model gave us not such a good results: 0.89 on the validation set and a similar result on the training set, in addition, the high variance between the different fold indicates an unstable model, so I beleive this model is less suitable for our data type.

### 4. Model Evaluation:
Finally, I chose the Random Forest model that yielded the best results for the validation dataset. 

At this point, I built a Confusion Matrix based on this model, which examined the number of predictions that were properly classified, and those that were not. The upper right rectangle symbolizes the FP, meaning a user who predicted he would buy but did not actually buy, while the lower left rectangle symbolizes the users we thought would not buy, but actually bought. 

I selected 3 types of metrics to examine the quality of the model. 

Specificity- Examining among the users who did not buy, how many we were able to catch. Our model excels in this parameter, and catches about 97% of the users who did not buy. It indicates that the amount of FP that are False alarms is tiny. Business-wise, this figure is excellent, since we will probably invest effort and money in targeting a potential user that I expect from him to buy, so we will prefer a minimal amount of false alarms.

In contrast, the trade-off comes at the expense of its Recall score (about 60%), in which the model is flawed: It does not properly catch users who have actually made a purchase. Business-wise we can live with that in peace since an FN user is a user that is a "nice surprise" for us, meaning we havn't invested resources on him, but he still bought. 

In terms of Precision, the model was right about 77% of users predicted that they would actually make a purchase. In view of the high specificity and precision scores compared to other models, this model was chosen to make predictions in the end.


In addition, I conducted a feature importance test and as expected, 'PageValues', which previously had the highest positive correlation with the label, was the most significant feature and contributed almost 60% to the classification of observations in the model. This fact empthsaises the relevance of Google analytical features.

In order to illustrate its importance, I ran the same model with the same parameters but without the feature, and received a significantly lower score (0.75), which means that the feature's contribution to the data classification is significant.

## Conclusion:
Summary:
The model chosen gives us a high chance of predicting purchase performance on the site, and in particular manages to detect users who are not interested in buying (the model manages to identify 97 out of 100 users who haven't bought), so we can invest resources in users with high chances of buying.

The various work phases focused more and more on the features and details that should be addressed, and cleared a lot of background noise and features that haven't contribute much to the model. The visualization phase contributed to illustrating and understanding the data, while the pre-processing phase confronted me with various data processing attempts- that on the one hand, clears noise, and on the other hand maintains efficiency.

Thus, I managed with only 40% of the original features number (I started at 23 and finished at 9) to achieve good scores in the validation set, and a balanced model in terms of flexibility. The project taught me that there is not one right method necessarily, but many ways to reach the same goal, as long as it is done according to the rules learned in the course, and maintains Bias / variance Tradeoff.

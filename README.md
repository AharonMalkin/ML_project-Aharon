# Machine learning project 
## Machine learning course, Tel Aviv University
During my studies at Tel Aviv university, I took the Machine learning course,as part of it, I did a project that contained building a system that predicts the chance of a user to perform a purchase under certain conditions.
During this project, I worked with Sklearn, Pandas, and Numpy libraries, and learned a lot about the different aspects of Machine learning algorithms. 

### Summary:
In this project I received a database containing 23 features about sessions (10479 samples) of users on an
online shopping site. I was asked to build a system that predicts the chance of a user to perform a purchase under certain conditions while browsing in the site.
First, I performed a number of steps to investigate and to receive an initial understanding of the data. I learned some interesting insights about the correlative behavior between the features, in addition, by visulaizing the data, I discovered few informative attributes about the features by checking how do they distribute and making comparisons between the different features and their values. This step made the data much more comprehensible. 
Afterwards. I performed a number of pre-process steps to the data - Such as handling and filling Null values, handling outliers, as well as different transformations, to make the data much more usable. In addition, I performed a dimension reduction and data normalization. During this step I used different processing methods, which are  tailored to each feature individually.
In the Models execution step, I ran 4 different models. The model that was selected is **Random Forest**, that was evaluated by confusion matrix to examine different metrics of its quality. The model received a final auc score of 0.931. 
Moreover, I analyzed the features that most significantly affected the purchase, where the most significant one was 'PageValues' (A Google Analytics feature).


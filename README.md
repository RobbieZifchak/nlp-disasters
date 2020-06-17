# Twitter Disasters

## Utilizing NLP to classify tweets as real or fake disasters


### Objective

* Can NLP be used to differentiate a real "disaster" scenario from a false alarm?

* Can we build a classification model that might be able to tell one from the other?

### The Data

* Source: Kaggle

* 7613 observations of tweets

* Tweets are labaled as 'Disaster' and 'Not Disaster'

    * Disaster: “Police officer wounded suspect dead after exchanging shots: RICHMOND Va. (AP) ÛÓ A Richmond police officer wa... http://t.co/Y0qQS2L7bS”
	
    * Not Disaster: “FedEx no longer will transport bioterror germs http://t.co/qfjjDxes7G via @USATODAY”
       
        
### Analyzing Text Data / Initial EDA

* Upon analyzing tweets:

    * Many tweets contained  irregular characters and hyperlinks

    * Sentence length for ‘fake disaster’ scenarios tended to be slightly greater than that of actual disasters

    * Data contains no class imbalance

### Data Cleaning Process

* Get rid of hyperlinks and punctuations except the hashtag

* Lowercase all words

* Remove stopwords and irrelevant words such as 'airplane mode', 'co', 'http',       

* Tokenize all tweets

* Lemmatize all tokens

* Create separate dataframes for real disaster tweets and fake disaster tweets

* Analyze tokens by frequencies from both sets

### Model Selection

* Models attempted include:

   * Random Forest Classification with both Count Vectorizer and Tfidf Vectorizer
   * Logistic Regression with Count Vectorizer
   * XGBoost with both Count Vectorizer and Tfidf Vectorizer
   * Naive Bayes with both Count Vectorizer and Tfidf Vectorizer

	   * Random Forest Classification gave us highest accuracy with both Count Vectorizer and Tfidf Vectorizer

	   * Both Vectorizers gave us the same accuracy score of 78%, with differences in precision and recall


### Final Model Results

* Random Forest Classifier - TF-IDF Vectorizer
        
	* We chose tfidf vectorizer because we thought recall was more important that precision in this case. We want to correctly identify more tweets pertaining to actual disaster scenarios. 

           
### Conclusions and Next steps

* Real disaster tweets and fake disaster tweets had different tokens that were more prevalent

* Real disaster tweets had more words that referred to actual disasters such as fire, storm, accident, and police

* Fake disaster tweets had more words that were general such as like, time, day, video, and people

* Of all our models to categorize and predict if tweets were referring to actually real disasters, the Random Forest Classifier with tfidf vectorization was most accurate

* Explore Neural Network models to see if we can achieve a higher accuracy.

* Look at words as a grouping to see if certain words together can give us a better prediction

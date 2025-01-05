# MIE1626 Data Science Methods and Statistical Learning
# Text and Sentiment Analysis for Canadian Election Data

--- Python Data Science Package
Numpy;  
Pandas;  
Seaborn;
Matplotlib;  
wordcloud: Wordcloud  
collections: defaultdict
gensim.models: Word2Vec  
transformers: BertTokenizer, BertModel  
Sklearn:
  -	sklearn.feature_extraction.text: CountVectorizer, TfidfVectorizer
  -	sklearn.decomposition: TruncatedSVD
  -	sklearn.model_selection; train_test_split  
  

--- Python Data Cleaning Package  
Re: re.complie (compile a regular expression pattern into regular)  
Html: html.unescape (Strip HTML)  
nltk:  
-	nltk.tokenize: word_tokenize
-	nltk.corpus: stopwords
-	nltk.stem: PorterStemmer

Part 1 – Data Cleaning
-	Import two datasets sentiment_analysis.csv & Canadian_elections_2021.csv
-	Clean tweets data according to the following:
  - Remove emojis and special characters.
  - Remove html tags and attributes (i.e., /<[^>]+>/).
  -	Html character codes (i.e., &...;) are replaced with an ASCII equivalent.
  -	All URLs are removed.
  - Lowercase text.
  -	All stop words are removed. (stop = stopwords.words('English))
  -	Preserve if a tweet is empty after pre-processing.
- Tokenize the sentences and join the tokens back to strings for next steps’ analysis.

Part 2 – EDA
-	Design a simple procedure that determines the political party according to the chosen related words being observed from the given data.
-	Draw wordcloud graphs to show vital words recognized.
-	Draw histograms of the top words used

Part 3 – Model Preparation
-	Prepare the data to try seven classification algorithms – logistic regression, k-NN, Naive Bayes, SVM, decision trees, Random Forest and XGBoost.
-	Using five different types of features, Bag of Words (word frequency), TF-IDF, Word2Vec embedding, BERT [cls] embedding, and N-grams on all 7 models.

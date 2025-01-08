# MIE1626 Data Science Methods and Statistical Learning
# Text and Sentiment Analysis for Canadian Election Data

Python Data Science Package
Numpy; Pandas; Seaborn; Matplotlib    
Sklearn:
-	sklearn.feature_extraction.text: CountVectorizer, TfidfVectorizer
-	sklearn.decomposition: TruncatedSVD
-	sklearn.model_selection; train_test_split
-	sklearn.linear_model: LogisticRegression
-	sklearn.neighbors: KNeighborsClassifier
-	sklearn.naive_bayes: GaussianNB, MultinomialNB
-	sklearn.svm: LinearSVC
-	sklearn.tree: DecisionTreeClassifier
-	sklearn.ensemble: RandomForestClassifier
-	xgboost: XGBClassifier
-	sklearn.model_selection: GridSearchCV
-	sklearn.metrics: accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

wordcloud: Wordcloud  
collections: defaultdict  
gensim.models: Word2Vec  
transformers: BertTokenizer, BertModel  

Python Data Cleaning Package
Re: re.complie (compile a regular expression pattern into regular)
Html: html.unescape (Strip HTML)
nltk: 
-	nltk.tokenize: word_tokenize
-	nltk.corpus: stopwords
-	nltk.stem: PorterStemmer

Part 1 – Data Cleaning
- Import two datasets sentiment_analysis.csv & Canadian_elections_2021.csv
- Data cleaning
  - Remove emojis and special characters.  
  - Remove html tags and attributes (i.e., /<[^>]+>/).  
  - Html character codes (i.e., &...;) are replaced with an ASCII equivalent.  
  -	All URLs are removed.
  -	Lowercase text.
  -	All stop words are removed. (stop = stopwords.words('english'))
  -	Preserve a tweet is empty after pre-processing.
-	Tokenize the sentences and join the tokens back to strings for next steps’ analysis.

Part 2 – EDA
-	Design a simple procedure that determines the political party according to the chosen related words being observed from the given data.
-	Draw wordcloud graphs to show vital words recognized.
-	Draw histograms of the top words used

Part 3 – Model Preparation
-	Using five different types of features, Bag of Words (word frequency), TF-IDF, Word2Vec embedding, BERT [cls] embedding, and N-grams on 7 models.
-	Perform data splitting to training and testing. The next-steps’ tokenization apply to both testing and training.
-	Prepare for BoW: Using ‘CountVectorizer’ function, to creates a dictionary (vocabulary) of all unique tokens in the dataset. It doesn't consider the importance of words in the context of the entire corpus.
-	Prepare for TF-IDF: Using TfidfVectorizer function, it assigns higher weight to words that are frequent in a document but rare in the corpus. Reduces the impact of common but less meaningful words like "the" or "and". Highlights unique and important words in the context of the document.
-	Prepare for Word2Vec: After splitting text into a list of words, applying the Word2Vec model. Computing the average Word2Vec vector of all words in the document, skipping the words that are not in the Word2Vec vocabulary (assigns a zero vector for unknown words)
-	Prepare for BERT: Load pre-trained BERT tokenizer and model, (Forward pass through the model*) tokenize and extract [CLS] token embeddings. It has significantly advanced the field of Natural Language Processing (NLP)
-	Prepare for N-grams: Using ‘CountVectorizer’ function with ‘ngram_range’ to specify the range of n-grams that will be extracted from the text during feature extraction. Captures local context when n>1, e.g., "New York" as a bigram is more meaningful than "New" and "York" separately.
-	I comment on a bunch of code as they are potentially bugging. However, these code were used to perform SVD and did dimensional reduction by keeping n significant components with 95% variance.

Part 4 - Model implementation and tuning
-	Build a function to return the accuracy information of the model
-	Apply the five types of features on seven classification algorithms – logistic regression, K-NN, Naive Bayes, SVM, decision trees, Random Forest and XGBoost.
-	Train a random forest classification model to predict the reason for the negative tweets data.
-	Use a combination of prediction models instead of using just one single model. Stacking is an ensemble learning technique where predictions from multiple base models (level-1 learners) are combined using a meta-model (level-2 learner) to improve classification performance.
-	Also consider to combine negative reasons into broader categories.

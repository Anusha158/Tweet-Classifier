# Twitter Sentiment Classification

This project classifies tweets into **political (1)** or **non-political (0)** categories using **natural language processing** and **Logistic Regression**. The dataset was provided by the US Naval Academy and includes two sets of tweets: political keyword-based and general.

---

## Dataset

- **[keyword-tweets.txt](https://www.usna.edu/Users/cs/nchamber/data/twitter/keyword-tweets.txt)** – Labeled as Political (POLIT)
- **[general-tweets.txt](https://www.usna.edu/Users/cs/nchamber/data/twitter/general-tweets.txt)** – Labeled as Not Political (NOT)

Loaded with `pandas.read_csv()` using tab (`\t`) delimiter and concatenated into a single DataFrame called `LabeledTweets`.

---

##  Data Preprocessing

Steps taken to clean the tweet text:

- Removed tokens containing `@` and `http`
- Replaced punctuation with space
- Replaced all numbers and non-ASCII characters with space
- Converted to lowercase and stripped extra whitespace
- Applied lemmatization using `WordNetLemmatizer`
- **Stopwords were not removed** (handled by TfidfVectorizer)

---

## Feature Engineering

Used `TfidfVectorizer` from `sklearn` to convert text into numerical features.

- Tried multiple values for `max_features`: `5`, `50`, `500`, `5000`, `50000`
- Vectorizer automatically removed stopwords

---

##  Model Training

Used `LogisticRegression` from `sklearn`:

- Data split into **75% training** and **25% testing**
- Trained on TF-IDF features for each max feature set

---

##  Evaluation Metrics

For each vectorizer size, evaluated:

- **Training Accuracy**
- **Test Accuracy**
- **Baseline Accuracy** (majority class)



## Requirements

```bash
pandas
scikit-learn
nltk

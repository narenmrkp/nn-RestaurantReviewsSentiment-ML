import pandas as pd
import re
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Download once
nltk.download('stopwords')

data = pd.read_csv("Restaurant_Reviews.tsv", sep="\t")

custom_stopwords = {
    'don',"don't",'ain','aren',"aren't",
    'no','nor','not'
}

ps = PorterStemmer()
stop_words = set(stopwords.words("english")) - custom_stopwords

corpus = []

for i in range(len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data['Review'][i])
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    review = " ".join(review)
    corpus.append(review)

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = data['Liked']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "model/Restaurant_review_model.pkl")
joblib.dump(cv, "model/count_v_res.pkl")

print("Model Saved Successfully")

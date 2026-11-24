"""Lab 6: Sentiment analysis with a small bag-of-words model."""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

TRAIN = [
    ("I love this movie it was wonderful and fun", "positive"),
    ("Amazing acting and a great story", "positive"),
    ("I really enjoyed the film highly recommend", "positive"),
    ("Best experience I have had in months", "positive"),
    ("So happy with the result it is excellent", "positive"),
    ("The food was delicious and the staff were kind", "positive"),
    ("Terrible movie I hated every minute", "negative"),
    ("Awful service and the food was bad", "negative"),
    ("Worst product I have ever bought do not buy", "negative"),
    ("I am disappointed and angry about this", "negative"),
    ("Boring plot and poor acting", "negative"),
    ("This is horrible and a complete waste", "negative"),
]
TEST = [
    ("I love the story it was wonderful", "positive"),
    ("Great and fun highly recommend", "positive"),
    ("I hated this awful boring movie", "negative"),
    ("Worst and terrible experience", "negative"),
    ("The staff were kind and it was excellent", "positive"),
]

def main():
    texts, labels = zip(*TRAIN)
    vec = CountVectorizer()
    X = vec.fit_transform(texts)
    clf = MultinomialNB()
    clf.fit(X, labels)

    test_texts, test_labels = zip(*TEST)
    pred = clf.predict(vec.transform(test_texts))

    print("=== Program 06: Sentiment Analysis (NLP) ===")
    print("Method: bag-of-words + Naive Bayes")
    print(f"Training sentences: {len(TRAIN)}")
    print(f"Test accuracy: {accuracy_score(test_labels, pred):.3f}")
    print()
    print(f"{'Sentence':48} {'Actual':10} {'Predicted'}")
    for text, actual, got in zip(test_texts, test_labels, pred):
        short = text if len(text) <= 46 else text[:43] + "..."
        print(f"{short:48} {actual:10} {got}")
    print()
    extra = "The product is wonderful and I love it"
    print("New sentence:", extra)
    print("Predicted sentiment:", clf.predict(vec.transform([extra]))[0])
    print("Program completed successfully.")

if __name__ == "__main__":
    main()

#Download dataset of product reviews 
#Preprocess data - clean stopwords, whitespace and punctuations
#Create function for sentiment analysis
#Test the model on sample prodeuct reviews
import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import string       #added so can remove punctuations


#Load the English language model
nlp = spacy.load("en_core_web_sm")

# Add TextBlob capabilities to the spaCy pipeline
nlp.add_pipe("spacytextblob")

#Load dataset
dataframe =pd.read_csv('dataset/amazon_product_reviews.csv', delimiter=",")

#Preprocess the text data
#Selecting  reviews.text column from the data frame
reviews_data = dataframe["reviews.text"]

#Remove all missing values from this column
clean_data = dataframe.dropna(subset=['reviews.text'])

#Creating a function to remove stopwords and combine tokens back after removal
def remove_stopwords(text):
    doc = nlp(text)     #Tokenisation
    return ' '.join(token.text for token in doc if not token.is_stop)   #join after taking off stopwords

#Removing punctuation,white space and converting to lowercase
def clean_text(text):
    #taking off punctuation
    text = text.translate(str.maketrans('','',string.punctuation))  
    #taking off white
    text = text.strip()     
    #convert text to lowercase
    text = text.lower()
    return text

#applying the function to remove stopwords on the reviews.text column
clean_data["reviews.text"] = clean_data["reviews.text"].apply(remove_stopwords)

#applying clean_text function to review.text column
clean_data["reviews.text"] = clean_data["reviews.text"].apply(clean_text)

#Function for sentiment analysis
def analyze_sentiment(clean_data):
    doc = nlp(clean_data)
    polarity = doc._.blob.polarity      #measures the sentiment of the review
    sentiment = doc._.blob.sentiment    #measures subjectivity of the review
    if polarity > 0:
        return 'positive', polarity     #postive polarity score means positive sentiment
    elif polarity < 0:
        return 'negative', polarity     #negative polarity score means negative sentiment
    else:
        return 'neutral', polarity      #neutral polarity score means neutral sentiment

#Testing the model on sample products
sample_reviews = clean_data['reviews.text'].head().tolist()

print("Sentiment Analysis Results:")
for review in sample_reviews:
    sentiment, polarity = analyze_sentiment(review)
    print(f"Review: '{review}'\nSentiment: {sentiment.capitalize()}, Polarity Score: {polarity:.2f}\n")

#Conducting a similarity function to compare the reviews
# Selecting a review of choice
my_review_of_choice = clean_data['reviews.text'][0]

# Selecting two product reviews for comparison
review1 = clean_data['reviews.text'][0]  # Selecting the first review
review2 = clean_data['reviews.text'][1]  # Selecting the second review

# Calculating similarity between the two reviews
similarity_score = nlp(review1).similarity(nlp(review2))

print("Similarity between the two reviews and the chosen review:")
print(f"Review of Choice: '{my_review_of_choice}'")
print(f"Review 1: '{review1}'")
print(f"Review 2: '{review2}'")
print(f"Similarity Score between Review 1 and Review 2: {similarity_score:.2f}")


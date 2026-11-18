
import nltk
import re

# Download required NLTK datasets
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def process_review(review):

    # Convert review into lowercase
    review_lower = review.lower()

    # Remove punctuation and numbers
    review_puncs_nums_removed = re.sub(
        r"[^a-z\s]", "", review_lower
    )

    # Tokenization: Convert sentence into words
    review_tokens = word_tokenize(review_puncs_nums_removed)

    # Remove stop words
    stop_words = stopwords.words('english')

    review_tokens_stopwords_removed = [
        token for token in review_tokens
        if token not in stop_words
    ]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()

    lemmatized_tokens = [
        lemmatizer.lemmatize(token, pos='n')
        for token in review_tokens_stopwords_removed
    ]

    # Convert tokens into a single string
    return " ".join(lemmatized_tokens)


# Main program
if __name__ == "__main__":

    review = "The product was really nice!!!"

    print(process_review(review))

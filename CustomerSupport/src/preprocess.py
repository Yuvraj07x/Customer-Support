import re
import string
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()



def clean_text(text):

    if pd.isna(text):
        return ""


    text = text.lower()


    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )


    text = re.sub(
        r"\S+@\S+",
        "",
        text
    )


    text = re.sub(
        r"\d+",
        "",
        text
    )


    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )


    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()


    words = word_tokenize(text)


    words = [
        word
        for word in words
        if word not in stop_words
    ]


    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]


    return " ".join(words)




def preprocess_dataset(input_path):


    df = pd.read_csv(input_path)


    print("="*60)
    print("ORIGINAL DATASET")
    print("="*60)

    print(df.head())



    # Use customer query

    df["clean_text"] = df["instruction"].apply(
        clean_text
    )



    # Keep required columns

    df = df[
        [
            "clean_text",
            "category",
            "intent",
            "response"
        ]
    ]



    print("\n")
    print("="*60)
    print("PROCESSED DATASET")
    print("="*60)

    print(df.head())


    print("\nShape:")
    print(df.shape)


    return df




if __name__ == "__main__":


    dataset = preprocess_dataset(
        "data/bitext_customer_support.csv"
    )


    dataset.to_csv(
        "data/customer_support_cleaned.csv",
        index=False
    )


    print(
        "\nDataset saved successfully!"
    )
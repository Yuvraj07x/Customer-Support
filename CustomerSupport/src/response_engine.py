import pandas as pd
import random


df = pd.read_csv(
    "data/customer_support_cleaned.csv"
)



def get_response(intent):

    responses = df[
        df["intent"] == intent
    ]["response"].tolist()


    if len(responses) > 0:

        return random.choice(
            responses
        )


    return (
        "Thank you for contacting customer support. "
        "Our team will assist you shortly."
    )
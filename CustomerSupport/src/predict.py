import joblib

from src.preprocess import clean_text
from src.response_engine import get_response


# ==========================================================
# Load Model Files
# ==========================================================

model = joblib.load(
    "models/classifier.pkl"
)

vectorizer = joblib.load(
    "models/vectorizer.pkl"
)

label_encoder = joblib.load(
    "models/label_encoder.pkl"
)



# ==========================================================
# Prediction Function
# ==========================================================

def predict_ticket(ticket):


    # ------------------------------------------------------
    # Clean User Input
    # ------------------------------------------------------

    cleaned = clean_text(ticket)



    # ------------------------------------------------------
    # Convert Text to TF-IDF
    # ------------------------------------------------------

    vector = vectorizer.transform(
        [cleaned]
    )



    # ------------------------------------------------------
    # Predict Intent
    # ------------------------------------------------------

    prediction = model.predict(
        vector
    )[0]



    intent = label_encoder.inverse_transform(
        [prediction]
    )[0]



    # ------------------------------------------------------
    # Confidence Score
    # Linear SVM uses decision_function
    # ------------------------------------------------------

    confidence = None


    if hasattr(model, "decision_function"):

        scores = model.decision_function(
            vector
        )


        confidence = (
            max(scores[0])
            - min(scores[0])
        )



    # ------------------------------------------------------
    # Generate Response
    # ------------------------------------------------------

    response = get_response(
        intent
    )



    # ------------------------------------------------------
    # Return Result
    # ------------------------------------------------------

    return {


        "Original Ticket":
            ticket,


        "Cleaned Ticket":
            cleaned,


        "Predicted Intent":
            intent,


        "Confidence Score":
            round(confidence, 2)
            if confidence is not None
            else "N/A",


        "Suggested Response":
            response

    }




# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":


    while True:


        print("\n")
        print("=" * 60)
        print("CUSTOMER SUPPORT AI ASSISTANT")
        print("=" * 60)



        ticket = input(
            "Enter customer issue (type 'exit' to quit): "
        )



        if ticket.lower() == "exit":

            break



        result = predict_ticket(
            ticket
        )



        print("\nPrediction Result")
        print("-" * 60)



        for key, value in result.items():

            print(
                f"{key}: {value}"
            )
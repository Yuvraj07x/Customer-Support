import streamlit as st

from src.predict import predict_ticket


# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Customer Support AI Assistant",
    page_icon="🤖",
    layout="centered"
)


# =====================================================
# Header
# =====================================================

st.title("🤖 Customer Support AI Assistant")

st.write(
    "Enter a customer issue and the AI will predict the intent "
    "and generate a support response."
)


st.divider()


# =====================================================
# User Input
# =====================================================

ticket = st.text_area(
    "Customer Issue",
    placeholder="Example: I want to cancel my order"
)



# =====================================================
# Prediction Button
# =====================================================

if st.button("Analyze Ticket"):


    if ticket.strip() == "":

        st.warning(
            "Please enter a customer issue."
        )


    else:

        with st.spinner(
            "Analyzing customer request..."
        ):


            result = predict_ticket(
                ticket
            )


        st.success(
            "Analysis Completed"
        )


        st.divider()


        # ------------------------------------------------
        # Prediction Details
        # ------------------------------------------------

        st.subheader(
            "Prediction Result"
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Detected Intent",
                result["Predicted Intent"]
            )


        with col2:

            st.metric(
                "Confidence Score",
                result["Confidence Score"]
            )


        st.divider()


        # ------------------------------------------------
        # Response
        # ------------------------------------------------

        st.subheader(
            "AI Suggested Response"
        )


        st.info(
            result["Suggested Response"]
        )


        st.divider()


        # ------------------------------------------------
        # Debug Information
        # ------------------------------------------------

        with st.expander(
            "View Processing Details"
        ):

            st.write(
                "Original Ticket:"
            )

            st.write(
                result["Original Ticket"]
            )


            st.write(
                "Cleaned Text:"
            )

            st.write(
                result["Cleaned Ticket"]
            )
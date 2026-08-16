import os
import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best_model.joblib"
)

model = joblib.load(MODEL_PATH)

for root, dirs, files in os.walk("/content"):
    if "data_register.py" in files:
        print(os.path.join(root, "data_register.py"))
# ---------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Visit with Us - Wellness Tourism",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Visit with Us")
st.subheader("Wellness Tourism Package Purchase Prediction")

st.write(
    "Enter the customer details below to predict whether "
    "the customer is likely to purchase the Wellness Tourism Package."
)


# ---------------------------------------------------------
# Collect customer inputs
# ---------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    Age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    TypeofContact = st.selectbox(
        "Type of Contact",
        ["Company Invited", "Self Inquiry"]
    )

    CityTier = st.selectbox(
        "City Tier",
        [1, 2, 3]
    )

    DurationOfPitch = st.number_input(
        "Duration of Pitch",
        min_value=0.0,
        value=15.0
    )

    Occupation = st.selectbox(
        "Occupation",
        [
            "Salaried",
            "Small Business",
            "Large Business",
            "Free Lancer"
        ]
    )

    Gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    NumberOfPersonVisiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        value=2
    )

    NumberOfFollowups = st.number_input(
        "Number of Followups",
        min_value=0,
        value=3
    )

    ProductPitched = st.selectbox(
        "Product Pitched",
        [
            "Basic",
            "Deluxe",
            "Standard",
            "Super Deluxe",
            "King"
        ]
    )


with col2:

    PreferredPropertyStar = st.selectbox(
        "Preferred Property Star",
        [3, 4, 5]
    )

    MaritalStatus = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )

    NumberOfTrips = st.number_input(
        "Number of Trips",
        min_value=0.0,
        value=3.0
    )

    Passport = st.selectbox(
        "Passport",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    PitchSatisfactionScore = st.selectbox(
        "Pitch Satisfaction Score",
        [1, 2, 3, 4, 5]
    )

    OwnCar = st.selectbox(
        "Own Car",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    NumberOfChildrenVisiting = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        value=1
    )

    Designation = st.selectbox(
        "Designation",
        [
            "AVP",
            "Executive",
            "Manager",
            "Senior Manager",
            "VP"
        ]
    )

    MonthlyIncome = st.number_input(
        "Monthly Income",
        min_value=0.0,
        value=25000.0
    )


# ---------------------------------------------------------
# Make prediction
# ---------------------------------------------------------

if st.button("Predict Purchase", type="primary"):

    # Create DataFrame with exactly the features used during training
    input_data = pd.DataFrame([{
        "Age": Age,
        "TypeofContact": TypeofContact,
        "CityTier": CityTier,
        "DurationOfPitch": DurationOfPitch,
        "Occupation": Occupation,
        "Gender": Gender,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "NumberOfFollowups": NumberOfFollowups,
        "ProductPitched": ProductPitched,
        "PreferredPropertyStar": PreferredPropertyStar,
        "MaritalStatus": MaritalStatus,
        "NumberOfTrips": NumberOfTrips,
        "Passport": Passport,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "OwnCar": OwnCar,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "Designation": Designation,
        "MonthlyIncome": MonthlyIncome
    }])

    # Display the input DataFrame
    st.write("### Customer Input")
    st.dataframe(input_data)

    # Prediction
    prediction = int(model.predict(input_data)[0])

    # Probability
    probability = float(
        model.predict_proba(input_data)[0][1]
    )

    # Display result
    st.write("### Prediction")

    if prediction == 1:
        st.success(
            f"🎉 Customer is likely to purchase the Wellness Tourism Package."
        )
    else:
        st.warning(
            f"Customer is unlikely to purchase the Wellness Tourism Package."
        )

    st.metric(
        "Purchase Probability",
        f"{probability:.2%}"
    )

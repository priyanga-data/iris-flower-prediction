import streamlit as st
st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌷",
    layout="centered"
)
import joblib
import pandas as pd
loaded_model = joblib.load('random_forest_model.pkl')
st.title("🌷Iris flower prediction🌷")
st.write("predict the flower species" 
         "using random forest algorithm.")
st.divider()
st.subheader("Enter Flower Measurements")
sepal_length=st.number_input(
    "sepal length(cm)",
    min_value=0.0,
    max_value=10.0,
    value = 5.1

)
sepal_width=st.number_input(
    "sepal width(cm)",
    min_value=0.0,
    max_value=10.0,
    value=3.5
)
petal_length = st.number_input(
    "petal length(cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.4 
)
petal_width=st.number_input(
    "petal width(cm)",
    min_value=0.0,
    max_value=10.0,
    value=0.2
)

st.divider() 

input_data=pd.DataFrame(
    [[sepal_length,sepal_width,petal_length,petal_width]],
     columns=loaded_model.feature_names_in_
)
if st.button("predicted flower"):
    prediction = loaded_model.predict(input_data)
    flower_names=["setosa","vericolor","virginica"]
    predicted_index = int(prediction[0])
    predicted_flower = flower_names[predicted_index]
    probabilities = loaded_model.predict_proba(input_data)[0]
    confidence=probabilities[predicted_index]*100
    st.success(f"predicted species:{predicted_flower}")
    st.metric(
        label="prediction probability",
        value=f"{confidence:.2f}%"
    )
    st.subheader(" Flower probability Distribution")

    for i,flower in enumerate(flower_names):
        st.write(f"{flower} : {probabilities[i]*100 :.2f}%")
        st.progress(float(probabilities[i]))

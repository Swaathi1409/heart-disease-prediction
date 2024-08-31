import numpy as np
import pickle
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(open('C://Users//Admin//Pictures//SNU//First Year//2nd Semester//Foundations of Data Science//Project//trained_model.sav','rb'))

# Creating a function for prediction
def heart_disease_pred(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    input_data = np.array((age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal), dtype=float)
    input_data_reshaped = input_data.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return 'Person is healthy. Does not have heart disease'
    else:
        return 'Person is having heart disease.'

def main():
    # Title for our webpage
    st.title("Heart Disease Prediction")

    # Getting input data from the user
    age = st.text_input("Age")
    sex = st.text_input("Gender(1 : Male, 0 : Female)")
    cp = st.text_input("Chest pain type: (0 - 3) (Value 0: typical angina, Value 1: atypical angina, Value 2: non-anginal pain, Value 3: asymptomatic)")
    trestbps = st.text_input("Resting Blood pressure (in mm Hg)")
    chol = st.text_input("Serum cholestrol (mg/dl)")
    fbs = st.text_input("Fasting blood sugar >120mg/dl (1 : yes, 0 : no) ")
    restecg = st.text_input("Resting electrocardiographic results:(0 - 2)(Value 0: normal, Value 1: have ST-T wave abnormality, Value 2: showing probable or definite left ventricular hypertrophy by Estes'criteria)")
    thalach = st.text_input("Max heart rate")
    exang = st.text_input("Exercise induced angina (1 : yes,0 : no)")
    oldpeak = st.text_input("Old Peak")
    slope = st.text_input("Slope of peak exercise ST segment (0 : Up sloping, 1 : Flat, 3 : Down sloping)")
    ca = st.text_input("Number of major vessels colored by fluorosopy(0-3)")
    thal = st.text_input("Thal (Normal=0, Fixed=1, Reversible defect=2)")

    # Code for prediction
    diagnosis = ''

    # Creating a button for prediction
    if st.button('Heart Disease Test Result'):
        diagnosis = heart_disease_pred(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)

    st.success(diagnosis)

if __name__ == '__main__':
    main()

    
    

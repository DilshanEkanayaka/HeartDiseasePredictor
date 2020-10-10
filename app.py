import pickle
import pandas as pd
import urllib.request
import json
import ast
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

with open('framingham_classifier_Logistic_regression_new.pkl', 'rb') as f:
    model = pickle.load(f)


def main():
    st.title(" Disease Predictor")
    st.sidebar.header('Patient Details')
    age = st.sidebar.number_input("Age (years)", 0, 200, 2)
    sysBP = st.sidebar.number_input("systolic blood pressure(mmHg)", 0, 200, 2)
    diaBP = st.sidebar.number_input("diastolic blood pressure(mmHg)", 0, 200, 2)
    glucose = st.sidebar.number_input("glucose level", 0, 200, 2)
    # diabetes = st.sidebar.number_input('diabetes', 0, 200, 2)
    option = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    if option == 'Male':
        male = 1
    elif option == 'Female':
        male = 0

    option2 = st.sidebar.selectbox('Blood Pressure medications', ('Yes', 'No'))
    if option2 == 'Yes':
        BPMeds = 1
    elif option2 == 'No':
        BPMeds = 0

    totChol = st.sidebar.number_input("total cholesterol level(mg/dL)", 0, 200, 2)
    BMI = st.sidebar.number_input("BMI(Body Mass Index )", 0, 200, 2)
    option3 = st.sidebar.selectbox('prevalentStroke', ('Yes', 'No'))
    if option3 == 'Yes':
        prevalentStroke = 1
    elif option3 == 'No':
        prevalentStroke = 0

    option4 = st.sidebar.selectbox('prevalentHyp', ('Yes', 'No'))
    if option4 == 'Yes':
        prevalentHyp = 1
    elif option4 == 'No':
        prevalentHyp = 0

    pregnantNo = st.sidebar.number_input("pregnantNo", 0, 200, 2)
    plasmaGlucoseConc = st.sidebar.number_input("plasmaGlucoseConc", 0, 200, 2)
    tricepsThickness = st.sidebar.number_input("tricepsThickness", 0, 200, 2)
    SerumInsulin = st.sidebar.number_input("SerumInsulin", 0, 200, 2)
    diabPedigreeFunc = st.sidebar.number_input("diabPedigreeFunc", 0, 200, 2)

    data1 = {
        "Inputs": {
            "input1":
                [
                    {
                        'Number of times pregnant': pregnantNo,
                        'Plasma glucose concentration a 2 hours in an oral glucose tolerance test': plasmaGlucoseConc,
                        'Diastolic blood pressure (mm Hg)': diaBP,
                        'Triceps skin fold thickness (mm)': tricepsThickness,
                        '2-Hour serum insulin (mu U/ml)': SerumInsulin,
                        'Body mass index (weight in kg/(height in m)^2)': BMI,
                        'Diabetes pedigree function': diabPedigreeFunc,
                        'Age (years)': age,
                        'Class variable (0 or 1)': "0",
                    }
                ],
        },
        "GlobalParameters": {}
    }
    body = str.encode(json.dumps(data1))

    url = 'https://ussouthcentral.services.azureml.net/workspaces/13c077d4051e4e1088654297b2bbcb04/services/934466005a2243948e5d6b46d9cdec64/execute?api-version=2.0&format=swagger'
    api_key = 'u4bfO9QM3gPLQ4nbSXiFNXP/h4B3yO0QE1lQy0/GOSqPwgOTFwAyWr4WXEYKj4tfrvZ/mIvRZpH2b5bn9QxHgg=='  # Replace this with the API key for the web service
    headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        my_json = result.decode('utf8').replace("'", '"')
        data = json.loads(my_json)
        s = json.dumps(data, indent=4, sort_keys=True)
        FinalData = data["Results"]['output1']
        res = str(FinalData)[1:-1]
        json_data = ast.literal_eval(res)
        FinalOutputAzure = json_data["Scored Labels"]
        NewDiabetesColumn = json_data["Scored Labels"]

    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(json.loads(error.read().decode("utf8", 'ignore')))

    input_variables = pd.DataFrame(
        [[age, sysBP, diaBP, glucose, NewDiabetesColumn, male, BPMeds, totChol, BMI, prevalentStroke, prevalentHyp]],
        columns=['age', 'sysBP', 'diaBP', 'glucose', 'diabetes', 'male', 'BPMeds', 'totChol', 'BMI',
                 'prevalentStroke', 'prevalentHyp'],
        dtype=float)
    result2 = ""

    azureresult = int(FinalOutputAzure)

    # if st.sidebar.button("Predict"):
    result2 = model.predict(input_variables)[0]
    if result2 == 1:
        result2 = 'positive'
    elif result2 == 0:
        result2 = 'negative'

    if azureresult == 1:
        azureresult = 'positive'
    elif azureresult == 0:
        azureresult = 'negative'

    st.subheader("Predicted result for Coronary Heart Diseases in next 10 years:")
    st.success(result2)

    st.subheader("Predicted result for diabetes from AzureML")
    st.success(azureresult)

    heart_raw = pd.read_csv('Preprocessed_framingham.csv')
    heart_pro = heart_raw.drop(columns=['TenYearCHD'])
    df = pd.DataFrame(heart_pro)
    df1 = df[["sysBP", "diaBP"]]

    normal_up = [4, 5, 6, 4, 8, 6, 5, 9, 4, 8, 9]
    normal_down = [1, 2, 1, 2, 1, 1, 3, 2, 1, 1, 2]
    current = [sysBP, diaBP, glucose, BPMeds, totChol, BMI, prevalentStroke, plasmaGlucoseConc, tricepsThickness,
               SerumInsulin, diabPedigreeFunc]

    names = ['sysBP', 'diaBP', 'glucose', 'BPMeds', 'totChol', 'BMI', 'prevalentStroke', 'plasmaGlucoseConc',
             'tricepsThickness',
             'SerumInsulin', 'diabPedigreeFunc']

    li = [normal_up, normal_down, current]
    chart_data = pd.DataFrame({'Upper Limit': normal_up,
                               'Lower Limit': normal_down,
                               'Current Position': current})

    st.subheader('')

    fig = go.Figure(data=[
        go.Bar(name='Upper Limit', x=names, y=normal_up),
        go.Bar(name='Lower Limit', x=names, y=normal_down),
        go.Bar(name='Current Position', x=names, y=current)])
    fig.update_layout(title={
        'text': "Range  of Safty ",
        'y': 0.9,
        'x': 0.4,
        'xanchor': 'center',
        'yanchor': 'top'}, font=dict(
        family="Courier New, monospace",
        size=10,
        color="black"
    ))
    st.plotly_chart(fig)

    labe = ['Male', 'Female']
    values = df['male']

    s1 = df['male'].value_counts()[1]
    s2 = df['male'].value_counts()[0]
    s3 = [s1, s2]

    fig = go.Figure(data=[go.Pie(labels=labe, values=s3, textinfo='label+percent',
                                 insidetextorientation='radial', hole=.3)])

    fig.update_layout(title={
        'text': "Gender Distribution ",
        'y': 0.9,
        'x': 0.44,
        'xanchor': 'center',
        'yanchor': 'top'}, font=dict(
        family="Courier New, monospace",
        size=18,
        color="black"
    ))
    st.plotly_chart(fig)

    fig = px.scatter(df.query("diabetes==0"), x="totChol", y="age",
                     size="age", color="age",
                     hover_name="age", log_x=True, size_max=10)
    st.plotly_chart(fig)


if __name__ == '__main__':
    main()

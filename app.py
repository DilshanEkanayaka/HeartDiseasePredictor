import pickle
import pandas as pd
import urllib.request
import json
import ast
import streamlit as st


# app = flask.Flask(__name__, template_folder='templates')

with open('framingham_classifier_Logistic_regression_new.pkl', 'rb') as f:
    model = pickle.load(f)

# app = flask.Flask(__name__, template_folder='templates')


# @app.route('/', methods=['POST', 'GET'])
st.title("Heart Disease Prediction Model")


def main():
    st.sidebar.header('Heart Details')
    age = st.sidebar.number_input("Age (years)", 0, 200, 2)
    sysBP = st.sidebar.number_input("sysBP", 0, 200, 2)
    diaBP = st.sidebar.number_input("diaBP", 0, 200, 2)
    glucose = st.sidebar.number_input("glucose", 0, 200, 2)
    # diabetes = st.sidebar.number_input('diabetes', 0, 200, 2)
    male = st.sidebar.number_input("male", 0, 2, 2)
    BPMeds = st.sidebar.number_input("BPMeds", 0, 200, 2)
    totChol = st.sidebar.number_input("totChol", 0, 200, 2)
    BMI = st.sidebar.number_input("BMI", 0, 200, 2)
    prevalentStroke = st.sidebar.number_input("prevalentStroke", 0, 200, 2)
    prevalentHyp = st.sidebar.number_input("prevalentHyp", 0, 200, 2)
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

    # original_input = ""
    # if st.button("Predict"):
    #     original_input = {'Age': age,
    #                       'Systolic BP': sysBP,
    #                       'Diastolic BP': diaBP,
    #                       'Glucose': glucose,
    #                       'Diabetes': NewDiabetesColumn,
    #                       'Gender': male,
    #                       'BP Medication': BPMeds,
    #                       'Total Cholesterol': totChol,
    #                       'BMI': BMI,
    #                       'Prevalent Stroke': prevalentStroke,
    #                       'Prevalent Hypertension': prevalentHyp},
    #     result = prediction,
    #     azureresult = FinalOutputAzure
    # return original_input

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

    st.bar_chart(df1)

    normal_up = [4, 5, 6, 4, 8, 6, 5, 9, 4, 8, 9]
    normal_down = [1, 2, 1, 2, 1, 1, 3, 2, 1, 1, 2]
    current = [sysBP, diaBP, glucose, BPMeds, totChol, BMI, prevalentStroke, plasmaGlucoseConc, tricepsThickness,
               SerumInsulin, diabPedigreeFunc]

    #st.line_chart(normal_up, normal_down, current)

    # chart_data = pd.DataFrame(normal_up, normal_down, current,columns=['a', 'b', 'c'])
    # st.line_chart(chart_data)

    # chart_data = pd.DataFrame(
    #     np.random.randn(20, 3),
    #     columns=['a', 'b', 'c'])
    li =[normal_up, normal_down, current]
    chart_data = pd.DataFrame({'colA': normal_up,
                               'colB': normal_down,
                               'colC': current})
    print(chart_data)



    st.line_chart(chart_data)

# return flask.render_template('main.html',
#                                      original_input={'Age': age,
#                                                      'Systolic BP': sysBP,
#                                                      'Diastolic BP': diaBP,
#                                                      'Glucose': glucose,
#                                                      'Diabetes': NewDiabetesColumn,
#                                                      'Gender': male,
#                                                      'BP Medication': BPMeds,
#                                                      'Total Cholesterol': totChol,
#                                                      'BMI': BMI,
#                                                      'Prevalent Stroke': prevalentStroke,
#                                                      'Prevalent Hypertension': prevalentHyp},
#                                      result=prediction,
#                                      azureresult=FinalOutputAzure,
#                                      )


if __name__ == '__main__':
    main()

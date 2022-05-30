import pickle
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# instantiating the class Flask with app obj, name = current module (app in this case)
app = Flask(__name__)
# opening the heart pickle file that contains the trained LR model and saving it in model var
file = open("Heart/heart_model.pkl", "rb")
model = pickle.load(file)

file_diabetes = open("Diabetes/diabetes_model.pkl", "rb")
model_diabetes = pickle.load(file_diabetes)
diabetes_df = pd.read_csv('Diabetes/Diabetes.csv')
diabetes_X = diabetes_df.drop(columns='Outcome', axis=1)
diabetes_Y = diabetes_df['Outcome']



# decorator used to map the URL with the given fun so the fun ouput is dis when the user goes to the specified route
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/index.html')
def index():
    return render_template("index.html")

@app.route('/heart.html')
def heart():
    return render_template("heart.html")

@app.route('/diabetes.html')
def diabetes():
    return render_template("diabetes.html")







@app.route('/heartpredict', methods=['POST'])
def heartpredict():
    float_features = [float(x) for x in request.form.values()]

    features = np.asarray(float_features)

    reshaped_features = features.reshape(1, -1)
    
    predict = model.predict(reshaped_features)
    if predict == 1:
        return render_template("heartpredict.html", prediction="There is a high chance that you have a heart disease. Consult your Doctor.")
    else:
        return render_template("heartpredict.html", prediction="You don't seem to have a heart disease, but still take care of your heart and do regular check-ups.")

@app.route('/diabetespredict', methods=['POST'])
def diabetespredict():
    float_features = [float(y) for y in request.form.values()]

    features = np.asarray(float_features)

    reshaped_features = features.reshape(1, -1)

    # Splitting the data to training data & Test data
    X_train, X_test, Y_train, Y_test = train_test_split(diabetes_X, diabetes_Y, test_size=0.2, stratify=diabetes_Y, random_state=2)

    # standardize the input data
    sc1 = StandardScaler()
    sc1.fit(X_train)
    std_data = sc1.transform(reshaped_features)

    predict = model_diabetes.predict(std_data)
    if predict == 1:
        return render_template("diabetespredict.html", prediction="There is a high chance that you have Diabetes. Consult your Doctor.")
    else:
        return render_template("diabetespredict.html", prediction="You don't seem to have Diabetes, but still take care of your health and do regular check-ups.")

# if true then run the current app
if __name__ == "__main__":
    app.run(debug=True)

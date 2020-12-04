import numpy as np
import math
from flask import Flask,request,render_template
import joblib
app=Flask(__name__)
model=joblib.load('student_mark_predictor.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features=[float(x) for x in request.form.values()]
    features_value=np.array(input_features)

    output=model.predict([features_value])[0][0].round(2)

    return render_template('index.html',prediction_text="You will get {}% marks".format(math.floor(output)))





if __name__ == "__main__":
    app.run(debug=True)

from flask import render_template, request, url_for, redirect, flash, Flask
import pandas as pd
import joblib

app = Flask(__name__)

@app.route('/')
def init():
    return render_template("index.html")  


@app.route('/', methods=['GET', 'POST'])
def index():

    age = request.form.get("age")
    weight = request.form.get("weight")


    clf = joblib.load("regr.pkl")
    x = pd.DataFrame([[age, weight]], columns=["Age", "Weight"])
    prediction = clf.predict(x)[0]
    return render_template("index.html", msg = prediction)  


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=5000)

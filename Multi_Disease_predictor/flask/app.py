
import numpy as np
from flask import Flask, request, render_template
import joblib as jb

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')


@app.route('/heart')
def heart():
    return render_template('heart.html')


@app.route('/kidney')
def kidney():
    return render_template('kidney.html')


@app.route('/stroke')
def stroke():
    return render_template('stroke.html')


@app.route('/covid')
def covid():
    return "<h2>Covid Prediction Page</h2>"


@app.route('/liver')
def liver():
    return "<h2> Liver Page</h2>"


def predict(values, dic):
    if len(values) == 4:
        model = jb.load('diabetes.joblib')
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    if len(values) == 18:
        model = jb.load('kidney.joblib')
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    if len(values) == 13:
        model = jb.load('heart.joblib')
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    if len(values) == 10:
        model = jb.load('stroke.joblib')
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]


@app.route("/predict", methods=['POST', 'GET'])
def predictPage():
    if request.method == 'POST':
        to_predict_dict = request.form.to_dict()
        to_predict_list = list(map(float, list(to_predict_dict.values())))
        pred = predict(to_predict_list, to_predict_dict)
        if pred == 1:
            pred = '---- You have diabetes ----'
        elif pred == 0:
            pred = "---- You don't have diabetes ----"
    return render_template('diabetes.html', pred=pred)


@app.route('/predictkidney', methods=['GET', 'POST'])
def predictkidney():
    if request.method == 'POST':
        to_predict_dict = request.form.to_dict()
        to_predict_list = list(map(float, list(to_predict_dict.values())))
        pred = predict(to_predict_list, to_predict_dict)
        if pred == 1:
            pred = '---- You have kidney disease ----'
        elif pred == 0:
            pred = "---- You don't have kidney disease ----"
    return render_template('kidney.html', predicted=pred)


@app.route('/predictheart', methods=['GET', 'POST'])
def predictheart():
    if request.method == 'POST':
        to_predict_dict = request.form.to_dict()
        to_predict_list = list(map(float, list(to_predict_dict.values())))
        pred = predict(to_predict_list, to_predict_dict)
        if pred == 1:
            pred = '---- You have Heart disease ----'
        elif pred == 0:
            pred = "---- You don't have Heart disease ----"
    return render_template('heart.html', predicted=pred)


@app.route('/predictstroke', methods=['GET', 'POST'])
def predictstroke():
    if request.method == 'POST':
        to_predict_dict = request.form.to_dict()
        to_predict_list = list(map(float, list(to_predict_dict.values())))
        pred = predict(to_predict_list, to_predict_dict)
        if pred == 1:
            pred = '---- Highly possible to get stroke ----'
        elif pred == 0:
            pred = "---- Low possiblility to get stroke ----"
    return render_template('stroke.html', predicted=pred)


if __name__ == "__main__":
    app.run(debug=True)

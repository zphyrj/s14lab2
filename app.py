from flask import Flask, render_template
import joblib
app = Flask(__name__)
model = joblib.load('./regression.pkl')
treeModel = joblib.load('./regressionWithTree.pkl')
# Make prediction -       ['BEDS', 'BATHS', 'SQFT', 'AGE', 'LOTSIZE', 'GARAGE']
linPrediction = model.predict([[12, 4.0, 4772, 50, 8350.0, 2]])[0][0].round(1)
linPrediction = str(linPrediction)
treePredict = treeModel.predict([[12, 4.0, 4772, 50, 8350.0, 0]])[0].round(1)
treePredict = str(treePredict)
@app.route('/')
def displayPredictions():
    return 'Linear prediction: ' + linPrediction + '\n Tree prediction: ' + treePredict
@app.route('/hello/<name>')
def index():
    bestClassEver = 'Best Class Ever'
    return render_template('index.html', bCE=bestClassEver)
@app.route('/world')
def hello_world():
    return 'hello world!'
@app.route('/<you>')
def hello_you(you):
    return f'Hello, {you}!'
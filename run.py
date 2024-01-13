import numpy as np
import pandas as pd
import model
from flask import Flask,request,jsonify,render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return 'Hello, World!'
@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/predict',methods=['POST'])
def postInput():
    insertValues = request.form.to_dict()
    data = pd.DataFrame([insertValues])
    result = model.predict(data)
    dia = 'Benign' if str(result[0]) == 'B' else 'Malignant'
    return (f'Result : {dia}')

if __name__ == '__main__':
    app.run()

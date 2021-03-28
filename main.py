import pickle
import numpy as np
from flask import Flask, request, jsonify
#from model_files.predict import predict


##creating a flask app and naming it "app"
app = Flask('app')

@app.route('/test', methods=['GET'])
def test():
    return 'Pinging Model Application!!'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print("****************************************************************************",data)
    with open('./model_files/model.pkl', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()
    predictions = model.predict(data)

    result = {'prediction': predictions}
    return jsonify(result)



#if __name__ == '__main__':
#    app.run(debug=True, host='127.0.0.1', port=5000)
    
    
    
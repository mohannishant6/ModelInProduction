from flask import Flask, request, jsonify, render_template
import requests
import numpy as np
import pickle
import pandas as pd


app = Flask(__name__)

# with open('Weather_Prediction.pkl', 'rb') as read:
#     model = pickle.load(read)

with open('model/encoder.pkl', 'rb') as f_in:
    encoderzz = pickle.load(f_in)
    f_in.close()
with open('model/model1.pkl', 'rb') as f_in:
    clf1 = pickle.load(f_in)
    f_in.close()
with open('model/model2.pkl', 'rb') as f_in:
    clf2 = pickle.load(f_in)
    f_in.close()
with open('model/model3.pkl', 'rb') as f_in:
    clf3 = pickle.load(f_in)
    f_in.close()
with open('model/model4.pkl', 'rb') as f_in:
    clf4 = pickle.load(f_in)
    f_in.close()
with open('model/model5.pkl', 'rb') as f_in:
    clf5 = pickle.load(f_in)
    f_in.close()


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    features = request.form["user"]
    feature_list=[]
    for x in features[1:len(features)-1].split(','):
        try:
            if x=='':
                x=np.nan
            else:
                try:
                    x=float(x)
                except:
                    x=str(x).strip()
        except: pass
        feature_list.append(x)
    
    print(feature_list, type(feature_list))
    data = pd.DataFrame(feature_list).T
    
    data.fillna(-1,inplace=True)
    print(data)

    inp = encoderzz.transform(data).values

    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')



    final_pred_sum = clf1.predict_proba(inp)[:, 1] +\
        clf2.predict_proba(inp)[:, 1] +\
        clf3.predict_proba(inp)[:, 1] +\
        clf4.predict_proba(inp)[:, 1] +\
        clf5.predict_proba(inp)[:, 1]
    final_pred_sum1 = pd.Series(final_pred_sum).map(
        lambda x: 1 if x/5 >= 0.5 else 0).values
    print("result prediction: ", final_pred_sum1[0])

   # prediction = model.predict([userInput])
    # output = round(predictio n[0], 2)
    # print(output)
    return render_template('index.html', prediction_text='Prediction: Default= {}'.format(final_pred_sum1[0]))


# @app.route('/results', methods=['POST'])
# def results():
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])
#     output = prediction[0]
#     return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)

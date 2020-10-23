#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Dependencies
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.externals import joblib
import traceback
import pandas as pd
import os


# In[ ]:



# Your API definition
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if classifier:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            print(query)
            prediction = classifier.predict(query)
            
            confidence = classifier.predict_proba(query)
            return jsonify({'prediction': str(prediction[0]), 'confidence': [(round((confidence[0][0]*100),3)),(round((confidence[0][1]*100),3))]})
           
        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    classifier = joblib.load("classification_model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')
    app.run(host='127.0.0.1', use_reloader=False,port=8080, debug=True)


# In[ ]:





# In[ ]:





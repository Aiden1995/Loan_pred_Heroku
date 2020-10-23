#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Dependencies
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import pandas as pd
import joblib
import os
classifier = joblib.load('classification_model.pkl') # Load "model.pkl"
print ('Model loaded')
model_columns = joblib.load('model_columns.pkl') # Load "model_columns.pkl"
print ('Model columns loaded')


# In[ ]:



# Your API definition
app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict():
    if classifier:
        try:
            data = request.get_json(force=True)
            data.update((x, [y]) for x, y in data.items())
            data_df = pd.DataFrame.from_dict(data)
            #json_ = request.json
            #print(json_)
            #query = pd.get_dummies(pd.DataFrame(json_))
            data_df = data_df.reindex(columns=model_columns, fill_value=0)
            print(data_df)
            prediction = classifier.predict(data_df)
            
            confidence = classifier.predict_proba(data_df)
            return jsonify({'prediction': str(prediction[0]), 'confidence': [(round((confidence[0][0]*100),3)),(round((confidence[0][1]*100),3))]})
           
        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(port = port,use_reloader=False, debug=True)
    #app.run(host='127.0.0.1', use_reloader=False,port=8080, debug=True)


# In[ ]:





# In[ ]:





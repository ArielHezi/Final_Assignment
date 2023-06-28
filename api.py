import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template
import pickle
import os
import locale
from madlan_data_prep import prepare_data
locale.setlocale(locale.LC_ALL, 'he_IL.UTF-8')


####### Functions:########


def check_keywords(text):
    num=100
    try:
        keywords = ["אדריכלי","יוקרתי","שדות","מפואר","אור","חצר","מיקום","חדר כושר","פארק","בריכה","בריכת","גדול","יוקרתי","גינה" "יחידת הורים","נוף" ,"ים","מוטפחת","קרוב"]
        for keyword in keywords:
            if keyword in text:
                num = num+20
                    
        keywords = ["ארנונה","ילדיים","חינוך","להשקעה","מתווכים" ,"פינוי בינוי","תמא","ללא תיווך","מושכרת","מחולקת","חרדי","משקיעים","מפוצלת","תשואה","לא בשבת","אוטובוס","טובה","כנסת","פוטנציאל","בתי","מקרר","ריצוף","יחידות","קטן"]
        for keyword in keywords:
            if keyword in text:
                num = num-20
        return num
    except:
        return num


app = Flask(__name__)
rf_model = pickle.load(open('trained_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    feature_names = ['City', 'type', 'room_number', 'Area', 'Street', 'city_area', 'floor','total_floors',
                     'hasElevator', 'hasParking', 'hasBars', 'hasStorage',
                     'hasBalcony', 'hasMamad', 'entranceDate', 'description']

    
    features = [request.form.get(name) for name in feature_names]
    features_df = pd.DataFrame([features], columns=feature_names)
    features_df['description'] = features_df['description'].apply(check_keywords)


    prediction = rf_model.predict(features_df)[0]
    output = locale.currency(prediction, symbol=True, grouping=True)

    return render_template('index.html', prediction_text=output)



if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)



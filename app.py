from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Memuat model dan kolom fitur saat server dimulai
model = joblib.load('model/fraud_detection_smote.pkl')
columns = joblib.load('model/columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check_fraud', methods=['POST'])
def check_fraud():
    transaction_data = request.form.to_dict()
    print("Received transaction data:", transaction_data)  # Logging data yang diterima
    transaction_df = pd.DataFrame([transaction_data])
    
    # Melakukan encoding dan pemrosesan data yang diperlukan
    transaction_df = pd.get_dummies(transaction_df, columns=['type'])
    
    # Mengisi kolom yang hilang setelah one-hot encoding
    for col in set(columns) - set(transaction_df.columns):
        transaction_df[col] = 0
    
    # Mengurutkan kolom sesuai dengan kolom yang digunakan selama pelatihan model
    transaction_df = transaction_df[columns]
    
    # Prediksi menggunakan model
    prediction = model.predict(transaction_df)
    print("Prediction result:", prediction)  # Logging hasil prediksi
    result = 'Fraudulent' if prediction[0] else 'Non-Fraudulent'
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
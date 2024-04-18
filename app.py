from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

selected_feature = pd.read_csv("selected_feature.csv")
tobe_scaled = pd.read_csv("tobe_scaled.csv")

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/service')
def service():
    return render_template('service.html')

@app.route('/predict')
def prediction():
    return render_template('prediction.html')


@app.route("/prediction", methods = ["GET","POST"])
def predict():
        if request.method == 'POST':
             
             income = float(request.form['income'])
             bnkacc = request.form["bnkacc"]
             creditcard = request.form["creditcard"]
             delaypay = request.form["delaypay"]
             creditinq = request.form["creditinq"]
             cmix = request.form["cmix"]
             odebt = float(request.form["odebt"])
             curatio = float(request.form["curatio"])
             chage = request.form["chage"]
             paymin = request.form["paymin"]
            
        



             credit_prediction = {
                     "Annual_Income":income,
                     "Num_Bank_Accounts":bnkacc,
                     "Num_Credit_Card":creditcard,
                     "Num_of_Delayed_Payment":delaypay,
                     "Num_Credit_Inquiries":creditinq,
                     "Credit_Mix":cmix,
                     "Outstanding_Debt":odebt,
                     "Credit_Utilization_Ratio":curatio,
                     "Credit_History_Age":chage,
                     "Payment_of_Min_Amount":paymin
                     }
        
             credit_prediction_df = pd.DataFrame([credit_prediction])

             #encoding
             encoder = pickle.load(open("encoder_le.pkl","rb"))

             encoder.fit_transform(selected_feature["Credit_Mix"])
             credit_prediction_df["Credit_Mix"] = encoder.transform(credit_prediction_df["Credit_Mix"])

             encoder.fit_transform(selected_feature["Payment_of_Min_Amount"])
             credit_prediction_df["Payment_of_Min_Amount"] = encoder.transform(credit_prediction_df["Payment_of_Min_Amount"])


             #scaling
             scaler = pickle.load(open("scaler.pkl","rb"))

             scaler.fit_transform(tobe_scaled)

             credit_prediction_scaled =  scaler.transform(credit_prediction_df)

             #modeling
             pickled_model = pickle.load(open("xg_tuned.pkl","rb"))

             results = pickled_model.predict(credit_prediction_scaled)

             print(results)

        
     
        return render_template("credit_result.html",income = income,
                           bnkacc = bnkacc,
                           creditcard = creditcard,
                           delaypay = delaypay,
                           creditinq = creditinq,
                           cmix = cmix,
                           odebt = odebt,
                           curatio = curatio,
                           chage = chage,
                           paymin = paymin,
                           result = results )
    
 
    

if __name__ == "__main__":
    app.run(port=5588)
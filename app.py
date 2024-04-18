from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

selected_feature = pd.read_csv("selected_feature.csv")
#tobe_scaled = pd.read_csv("tobe_scaled.csv")

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
             oneh_encoder = pickle.load(open("onehot.pkl","rb"))

             # encoding creditmix

             credit_mix = selected_feature["Credit_Mix"].values.reshape(-1,1)
             en_cmix= oneh_encoder.fit_transform(credit_mix).toarray()
             #encoded_cmix = pd.DataFrame(en_cmix, columns = oneh_encoder.get_feature_names_out(['Creditmix_']))

             creditmix = credit_prediction_df["Credit_Mix"]

             cmix_df = pd.DataFrame(creditmix)
             one_cmix = oneh_encoder.transform(cmix_df).toarray()
             one_cmix_df = pd.DataFrame(one_cmix, columns=oneh_encoder.get_feature_names_out(["Creditmix_"]))

             credit_df = pd.concat([credit_prediction_df, one_cmix_df], axis = 1)

             #encoding paymin_amount
             pay_min = selected_feature["Payment_of_Min_Amount"].values.reshape(-1,1)
             en_paymin = oneh_encoder.fit_transform(pay_min).toarray()
             #encoded_paymin = pd.DataFrame(en_paymin, columns=oneh_encoder.get_feature_names_out(['Paymin_']))

             payminamt = credit_prediction_df["Payment_of_Min_Amount"]

             pay_min_df = pd.DataFrame(payminamt)
             onh_paymin = oneh_encoder.transform(pay_min_df).toarray()
             onh_paymin_df = pd.DataFrame(onh_paymin, columns=oneh_encoder.get_feature_names_out(['Paymin_']))

             credit_df = pd.concat([credit_df, onh_paymin_df], axis=1)

             credit_df.drop(columns=["Credit_Mix","Payment_of_Min_Amount"], inplace=True)

             
             #scaling
             scaler = pickle.load(open("scaler.pkl","rb"))

             #scaler.fit_transform(tobe_scaled)

             credit_prediction_scaled =  scaler.fit_transform(credit_df)

             #modeling
             pickled_model = pickle.load(open("xg_one.pkl","rb"))

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
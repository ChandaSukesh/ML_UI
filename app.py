from flask import Flask,render_template,request

import pandas as pd

app = Flask(__name__) #creating the Flask class object   
 
@app.route("/") #decorator drfines the   
def index():  
    return render_template("Uploading.html")


@app.route('/textfile', methods = ['GET','POST'])
def textfile():
    filename=request.form.get("filename")
    df = pd.read_csv(filename,header=None)
    return render_template('ClassReg.html',reads=df.values.tolist())

@app.route('/MainFunction', methods = ['GET','POST'])
def MainFunction():
    return render_template('ClassReg.html')

if __name__ =='__main__':  
    app.run(debug = True)

from flask import Flask,render_template,request,session

import pandas as pd

app = Flask(__name__) #creating the Flask class object 
app.secret_key = "any random string"
 
@app.route("/") #decorator drfines the   
def index():  
    return render_template("Uploading.html")


@app.route('/textfile', methods = ['GET','POST'])
def textfile():
    filename=request.form.get("filename")
    df = pd.read_csv(filename,header=None)
    session["df"]=df.values.tolist()
    return render_template('ClassReg.html',reads=df.values.tolist())

@app.route('/MainFunction', methods = ['GET','POST'])
def MainFunction():
    df=session["df"]
    print(df)
    # filename=request.form.get("")
    return render_template('BasicFunctions.html')

@app.route('/showData', methods = ['GET','POST'])
def showData():
    type=request.form.get("id")
    df=session["df"]
    # converting list to dataframe
    df=pd.DataFrame(df) 
    if(type=="Head"):
        content=df.head()
        message="The below table displays the top 5 rows in a given dataset"
        flag=1
    if(type=="Describe"):
        content=df.describe()
        message="The below table displays entire rows in a given dataset"
        flag=1
    if(type=="Predict"):
        message="Select the Learning rate and Epochs values of your choice"
        flag=2
        content=df
    return render_template('BasicFunctions.html', h=content.values.tolist(),message=message,flag=flag)
    # 

if __name__ =='__main__':  
    app.run(debug = True)

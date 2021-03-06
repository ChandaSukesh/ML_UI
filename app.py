from flask import Flask,render_template,request,session, sessions,url_for

import pandas as pd
import numpy as np
import matplotlib.pyplot as mat
from sklearn.model_selection import train_test_split



app = Flask(__name__) #creating the Flask class object 
app.secret_key = "String dummy"
 
@app.route("/") #decorator drfines the   
def index():  
    return render_template("Uploading.html")


@app.route('/textfile', methods = ['GET','POST'])
def textfile():
    filename=request.form.get("filename")
    df = pd.read_csv(filename,header=None)
    
    
    session["df"]=df.values.tolist()
    # session["dataframe"]=df
    # print(type(session["df"]))
    return render_template('ClassReg.html',reads=df.values.tolist())

@app.route('/MainFunction', methods = ['GET','POST'])
def MainFunction():
    typename=request.form.get("typename")
    # print(typename)
    df=session["df"]
    if typename=="Click2":
        val=2
    else:
        val=1
    session["val"]=val
    if(val==2):
        if len(df[0])>2:
            modelName="Multi Variate"
        else:
            modelName="Uni Variate"
    else:
        modelName="Binomial"
    return render_template('BasicFunctions.html',modelName=modelName,val=val)


@app.route('/showData', methods = ['GET','POST'])
def showData():
    type=request.form.get("id")
    df=session["df"]
    # converting list to dataframe
    df=pd.DataFrame(df) 

    if session["val"] == 1:
        typename=1
    else:
        typename=2
    if(type=="Head"):
        content=df.head()
        message="The below table displays the top 5 rows in a given dataset"
        flag=1
    if(type=="Describe"):
        content=df.describe()
        message="The below table displays entire rows in a given dataset"
        flag=1
    if(type=="PlotGraph"):
        message="Enter the column number to which you want to plot the Graph for X axis"
        flag=3
        content=df
    if(type=="Predict"):
        message="Choose a test file with which you want to predict"
        flag=4
        content=df
    return render_template('BasicFunctions.html', h=content.values.tolist(),message=message,flag=flag,typename=typename)
    

@app.route('/Graph', methods = ['GET','POST'])
def Graph():
    df=session["df"]
    df=pd.DataFrame(df)
    
    # if :
    n=len(df.columns)
    x=request.form.get("xcol")

    xval=df[int(x)]
    print(xval)
    y=df[n-1]
    mat.scatter(xval, y,c='blue')
    mat.show()
    print(y)
    # else :
    #     X = df.iloc[:, :-1].values
    #     y = df.iloc[:, -1].values
    #     pos, neg = (y == 1).reshape(len(df.index), 1), (y == 0).reshape(len(df.index), 1)

    #     mat.scatter(X[pos[:, 0], 0], X[pos[:, 0], 1], c="r", marker="+")
    #     mat.scatter(X[neg[:, 0], 0], X[neg[:, 0], 1], marker="o", s=10)
    #     mat.xlabel("X - axis")
    #     mat.ylabel("Y - axis")
    #     mat.legend(["Positive", "Negative"], loc=0)
    #     mat.show()
    #     print(pos)
    # return render_template('graph.html')
   

@app.route('/linearregression', methods = ['GET','POST'])
def linearregression():
    data=session["df"]
    data=pd.DataFrame(data)
    
    filename=request.form.get("filename")
    col_length=len(data.columns)
    if(col_length>2):
        model= 2      #multivariate
    else:
        model= 1   #univariate
    
    test_data  = pd.read_csv(filename,header=None)
    learning_rate= float(request.form.get("fname1"))    
    epochs=int(request.form.get("fname2"))
    
    theta_values, J_history, X, y = procedure(data, learning_rate, epochs, model)
    
    dftest = test_data.values
    len_of_testdata = dftest[:, 0].size
    hypo=printHypothesis(theta_values,model)
    if model == 1:
        test = np.append(np.ones((len_of_testdata, 1)), dftest[:, 0].reshape(len_of_testdata, 1),
                            axis=1)
    else:
        test = dftest[:, 0:model].reshape(len_of_testdata, 2)
        test, mean_test, std_test = featureNormalization(test)
        test = np.append(np.ones((len_of_testdata, 1)), test, axis=1)
    predict1 = predict(test, theta_values)

    return render_template('linear.html',predict1=predict1,hypo=hypo)

@app.route('/logisticregression', methods = ['GET','POST'])
def logisticregression():
    split_input=request.form.get("id")
    if split_input==1:
        split_ratio = 0.4
    elif split_input == 2:
        split_ratio = 0.3
    elif split_input == 3:
        split_ratio = 0.2
    data=session["df"]
    data=pd.DataFrame(data)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=split_ratio,random_state=1)
    return render_template("graph.html")
    

def featureNormalization(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    X_norm = (X - mean) / std

    return X_norm, mean, std

def predict(x, thetaP):
    predictionvalues = np.dot(x,thetaP)
    return predictionvalues

def procedure(data_frame,learning_rate,epochs,model):
    data_frame = data_frame.sample(frac=1)
    if model == 1:
        dftrain = data_frame[(len(data_frame) // 5):]
        dftrain = dftrain.values
        len_of_traindata = dftrain[:,0].size
        x_training = np.append(np.ones((len_of_traindata, 1)), dftrain[:,0].reshape(len_of_traindata,1), axis=1)
        y_training = dftrain[:,1].reshape(len_of_traindata,1)
        theta = np.zeros((2, 1))
        theta_values, J_history = gradientDescent(x_training, y_training, theta, learning_rate, epochs)
        return theta_values, J_history, x_training, y_training
    elif model > 1:
        data_n2 = data_frame.values
        m2 = len(data_n2[:, -1])
        x_training = data_n2[:, 0:model].reshape(m2, 2)
        x_training, mean_x_training, std_x_training = featureNormalization(x_training)
        x_training = np.append(np.ones((m2, 1)), x_training, axis=1)
        y_training = data_n2[:, -1].reshape(m2, 1)
        theta = np.zeros((model+1, 1))
        theta_values, J_history = gradientDescent(x_training, y_training, theta, learning_rate, epochs)

        return theta_values, J_history, x_training, y_training
 
def gradientDescent(X, y, thetaG, alpha, num_iters):
    """
    Take in numpy array X, y and theta and update theta by taking num_iters gradient steps
    with learning rate of alpha
    return theta and the list of the cost of theta during each iteration
    """

    m = len(y)
    J_history = []

    for i in range(num_iters):
        predictions = X.dot(thetaG)
        error = np.dot(X.transpose(), (predictions - y))
        descent = alpha * 1 / m * error
        thetaG -= descent
        J_history.append(computeCost(X, y, thetaG))

    return thetaG, J_history

def computeCost(X, y, thetaC):
    """
    Take in a numpy array X,y, theta and generate the cost function of using theta as parameter
    in a linear regression model
    """
    m = len(y)
    predictions = X.dot(thetaC)
    square_err = (predictions - y) ** 2

    return 1 / (2 * m) * np.sum(square_err)

def printHypothesis(theta_values,model):
    string = "h(x) ="+str(round(theta_values[0,0],2))
    for i in range(1,model+1):
        string += (" + "+str(round(theta_values[i,0],2))+"x"+str(i))
    return string

# @app.route("/demo")
# def demo():
#     x = np.linspace(0,20,100)
#     y=np.sin(x)
#     labels = [s for s in x]
#     values = [v for v in y]
#     return render_template("index.html",x=labels,y=values)

if __name__ =='__main__':  
    app.run(debug = True)

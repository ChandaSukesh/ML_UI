from flask import Flask,render_template,request,session, sessions

import pandas as pd
import numpy as np

app = Flask(__name__) #creating the Flask class object 
app.secret_key = "any random string"
 
@app.route("/") #decorator drfines the   
def index():  
    return render_template("Uploading.html")


@app.route('/textfile', methods = ['GET','POST'])
def textfile():
    filename=request.form.get("filename")
    df = pd.read_csv(filename,header=None)
    # session["data"]=df
    session["df"]=df.values.tolist()
    print(type(session["df"]))
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
        flag=True
    if(type=="Describe"):
        content=df.describe()
        message="The below table displays entire rows in a given dataset"
        flag=True
    if(type=="Predict"):
        message="Choose a test file with which you want to predict"
        flag=False
        content=df
    return render_template('BasicFunctions.html', h=content.values.tolist(),message=message,flag=flag)
    # 

@app.route('/linearregression', methods = ['GET','POST'])
def linearregression():
    data=session["df"]
    data=pd.DataFrame(data)
    # print(type(data))

    filename=request.form.get("filename")
    
    model=1 # change it to dynamically
    test_data  = pd.read_csv(filename,header=None)
    learning_rate= float(request.form.get("fname1"))    
    epochs=int(request.form.get("fname2"))
    theta_values, J_history, X, y,Original_mean,Original_std = procedure(data, learning_rate, epochs,model)
    dftest = test_data.values
    len_of_testdata = dftest[:, 0].size
    if model == 0:
        xCount = 1
    else:
        xCount = len(test_data.columns)
    
    testX=dftest[:, 0: len(test_data.columns) ].reshape(len_of_testdata, xCount)
    if model == 1:
        testX = (testX - Original_mean) / Original_std
    test = np.append(np.ones((len_of_testdata, 1)), testX,axis=1)
    predict1 = predict(test, theta_values)
    # print(predict1)

    return render_template('linear.html',predict1=predict1)
    # ,predict1=predict1 
























    # # print(epochs)
    # # print("**************")
    # # print("chanda")
    # theta_values, J_history, X, y,Original_mean,Original_std = procedure(test_data, learning_rate, epochs,model)
    # print(theta_values)
    # dftest = test_data.values
    # len_of_testdata = dftest[:, 0].size
    
    # # if model == 1:
    # #     test = np.append(np.ones((len_of_testdata, 1)), dftest[:, 0].reshape(len_of_testdata, 1),axis=1)
    # # else:
    # #     test = dftest[:, 0:model].reshape(len_of_testdata, 2)
    # #     test, mean_test, std_test = featureNormalization(test)
    # #     test = np.append(np.ones((len_of_testdata, 1)), test, axis=1)

    # if model == 0:
    #     xCount = 1
    # else:
    #     xCount = len(test_data.columns)
    # testX=dftest[:, 0: len(test_data.columns) ].reshape(len_of_testdata, xCount)
    # if model == 1:
    #     testX = (testX - Original_mean) / Original_std
    # test = np.append(np.ones((len_of_testdata, 1)), testX,axis=1)
    # predict1 = predict(test, theta_values)
    # print(predict1)
    # # predict1 = predict(test, theta_values)
    # return render_template('linear.html',predict1=predict1 )

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
    dftrain = data_frame[(len(data_frame) // 5):]
    dftrain = dftrain.values
    if model==0:
        xCount=1
    else:
        xCount = len(data_frame.columns) - 1

    thetaCount = len(data_frame.columns) - 1
    len_of_traindata= dftrain[:, -1].size
    print(len_of_traindata)
    # data_n2[:,0:2].reshape(m2,2)
    X2=dftrain[:,0:len(data_frame.columns)-1].reshape(len_of_traindata,xCount)
    if model==1:
        X2, mean_X2, std_X2 = featureNormalization(X2)
    else:
        mean_X2=0
        std_X2=0
    x_training = np.append(np.ones((len_of_traindata, 1)),X2 , axis=1)

    y_training = dftrain[:,1].reshape(len_of_traindata,1)

    theta = np.zeros((thetaCount+1, 1))

    theta_values, J_history = gradientDescent(x_training, y_training, theta, learning_rate, epochs)
    # print("h(x) ="+str(round(theta_values[0,0],2))+" + "+str(round(theta_values[1,0],2))+"x1")
    return theta_values, J_history, x_training, y_training,mean_X2,std_X2

 
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

if __name__ =='__main__':  
    app.run(debug = True)

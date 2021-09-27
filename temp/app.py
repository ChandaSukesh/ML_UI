from flask import Flask,render_template
# import pandas as pd
app = Flask(__name__) #creating the Flask class object   
 
@app.route("/") #decorator drfines the   
def index():  
    return render_template("Uploading.html")


# @app.route('/textfile', methods = ['GET','POST'])
# def textfile():
#     df = pd.read_csv(request.form.get("myFile"),header=None)
#     return render_template('New.html')

if __name__ =='__main__':  
    app.run(debug = True)

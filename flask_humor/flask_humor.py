import numpy as np
from flask import Flask, render_template
import pickle 

app = Flask(__name__) 
svc = pickle.load(open("HumorDetection.pkl", "rb"))

@app.route("/")
@app.route("home")
def home(): 
    return render_template('home.html')

#checks if running on local host 
if __name__ == '__main__': 
    app.run(debug=True)
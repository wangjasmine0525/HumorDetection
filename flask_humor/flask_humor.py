from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
svc = pickle.load(open("HumorDetection.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

@app.route("/", methods=['POST', 'GET'])
def home():
    if request.method =='POST':
        joke = request.form['content'] 
        transformed = tfidf.transform([joke])
        funny = svc.predict(transformed.toarray())
        return render_template('results.html', joke=joke, funny=funny)
    else:
        return render_template('home.html')

#checks if running on local host
if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
svc = pickle.load(open("HumorDetection.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

@app.route("/", methods=['POST', 'GET'])
def home():
    laughs = ["You're quite the jokester!",
              "You should be a comedian!",
              "That's a good one!"]
    lames = ["You're lame.",
             "Nice try.",
             "Seems like you're the only one laughing."]

    if request.method =='POST':
        rand_num = np.random.randint(3)
        joke = request.form['content']
        transformed = tfidf.transform([joke])
        funny = svc.predict(transformed.toarray())
        if funny == 1:
            return render_template('results.html', joke=joke, funny=funny, message=laughs[rand_num])
        elif funny == 0:
            return render_template('results.html', joke=joke, funny=funny, message=lames[rand_num])
    else:
        return render_template('home.html')

#checks if running on local host
if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request
app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def home():
    if request.method =='POST':
        joke = request.form['content']
        return render_template('results.html', joke=joke, funny=0)
    else:
        return render_template('home.html')

#checks if running on local host
if __name__ == '__main__':
    app.run(debug=True)

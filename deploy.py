from flask import Flask, render_template, request, session, redirect
from model.predict import prediction

app = Flask(__name__)
 
@app.route('/')
def index():
  return render_template('index.html')

@app.route('/translation', methods=['POST', 'GET'])
def result():
  if request.method == 'POST':
    sentence = request.form['text']
  
  if sentence == '':
    answer="Please provide a sentence."
  else:
    # answer='Translated sentence.'
    answer = prediction(sentence)
  return render_template("translation.html", result= answer)

if __name__ == '__main__':
  app.run()
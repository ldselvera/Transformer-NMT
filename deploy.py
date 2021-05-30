from flask import Flask, render_template, request, session, redirect
# from model.predict import prediction
# from model.load import load_info

app = Flask(__name__)

# global model
# model, tokenizer = load_info()
 
@app.route('/')
def index():
  return render_template('index.html')

@app.route('/translation', methods=['POST', 'GET'])
def result():
  if request.method == 'POST':
    question = request.form['text']
  
  if question=='':
    answer="Please provide a sentence."
  else:
    answer='Translated sentence.'
    # answer = prediction(question, model, tokenizer)
  return render_template("translation.html", result= answer)

if __name__ == '__main__':
  app.run()
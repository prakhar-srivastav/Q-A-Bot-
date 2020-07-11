from flask import Flask,render_template,session,url_for,redirect,request,jsonify
import numpy as np 
import tensorflow as tf
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
import joblib 
from keras.preprocessing.sequence import pad_sequences
app = Flask(__name__)


sess = tf.Session()
graph = tf.get_default_graph()

set_session(sess)
my_model = load_model("chatbot_120_epochs.h5")
word_index,max_story_len,max_question_len = joblib.load("word_index.pkl")

app.config['SECRET_KEY'] = 'mysecretkey'

class Form1(FlaskForm):
    story = TextField('story')
    question = TextField('question')
    #pet_len = TextField('Petal Length')
    #pet_wid = TextField('Petal Width')

    submit = SubmitField('Analyze')


def model_predict(x,y, model,max_question_len=max_question_len,max_story_len=max_story_len,word_index=word_index):
    
    X = []
    Y = []
    x.split(' ')
    y.split(' ')
  
    x = [word_index[word.lower()] for word in x.split(' ')]
    y = [word_index[word.lower()] for word in y.split(' ')]  

        
    X.append(x)
    Y.append(y)
        
    X=pad_sequences(X, maxlen=max_story_len)
    Y=pad_sequences(Y, maxlen=max_question_len)
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        pred_results = model.predict( ([X,Y]) )
    
    val_max = np.argmax(pred_results)
    max_val=max([data for data in pred_results])
    for key, val in word_index.items():
        if val == val_max:
            k = key
    return k

@app.route('/', methods=['GET','POST'])
def index():
    form = Form1()
    if form.validate_on_submit():
        # Grab the data from the breed on the form.

        session['story'] = form.story.data
        session['question'] = form.question.data
        return redirect(url_for("prediction"))
    return render_template('home.html', form=form)


@app.route('/prediction')
def prediction():
    #data = request.get_json()
    #print(word_index)
    x=session['story']
    y=session['question']
    #ata = request.get_json()
    #x=data['story']
    #y=data['question']
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        pred=model_predict(x,y, my_model,max_question_len=max_question_len,max_story_len=max_story_len,word_index=word_index)
    
    #print(pred)
    # returning the predictions as json
    return render_template('prediction.html',results=pred)
 
if __name__ == '__main__':
	app.run()

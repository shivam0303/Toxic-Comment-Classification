from flask import Flask, render_template, Response, request, jsonify, json
import os,inspect,sys
from scrape_twitter import scrape_twitter
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from utils import *

template_dir = 'views/'
app = Flask(__name__, template_folder=template_dir, static_url_path='/static')
app.config['model_url'] = "saved_models/"

print("Loading the Vectorizer...")  
app.config['vectorizer'] = joblib.load(app.config['model_url'] + 'vectorizer/vectorizer.sav')
print("Done loading the Vectorizer...")

@app.route('/predict', methods=['GET'])
def apicall():
    #print(request.args.get('id'))
    if request.args.get('id') is None:
        test_data = [request.args.get('comment')]
        test_df = pd.DataFrame(test_data,dtype="str",columns=['comment_text'])
        test_obj = test(test_df, app.config['model_url'])

        json_response,status_code = test_obj.get_predictions(app.config['vectorizer'])
        print(json_response)
        ## FOR WORD CLOUD ##
        text = " ".join(i for i in test_obj.df['preprocessed_text'])
        stopwords = set(STOPWORDS)
        wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
        plt.figure( figsize=(15,10))
        plt.axis("off")
        plt.imshow(wordcloud)
        plt.savefig("api/static/wc.jpg")
        ####################
        
        responses = app.response_class( response=json.dumps(json_response),
                                            status=status_code,
                                            mimetype='application/json')
        return (responses)
    else:
        return api_call_twitter()

def api_call_twitter():
    if request.args.get('id') is not None:
        id = request.args.get('id')

        scrape_twitter_obj = scrape_twitter()
        test_df = scrape_twitter_obj.scrape_tweets_by_user(id,100)
        test_obj = test(test_df, app.config['model_url'])

        json_response,status_code = test_obj.get_predictions(app.config['vectorizer'])
       
        ## FOR WORD CLOUD ##
        text = " ".join(i for i in test_obj.df['preprocessed_text'])
        stopwords = set(STOPWORDS)
        wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
        plt.figure( figsize=(15,10))
        plt.axis("off")
        plt.imshow(wordcloud)
        plt.savefig("api/static/wc.jpg")
        ####################
        
        responses = app.response_class( response=json.dumps(json_response),
                                            status=status_code,
                                            mimetype='application/json')
        return (responses)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)    
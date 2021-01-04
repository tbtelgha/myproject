from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
from sklearn.svm import LinearSVC
#from sklearn.metrics import  accuracy_score
#from sklearn import metrics
#import pandas as pd
import pickle
import json
from flask import Flask , render_template , request , Response


vectorizer = pickle.load(open(r"data\vectorizer.pickle", "rb"))
tfidf = pickle.load(open(r"data\tfidf.pickle", "rb"))
train_mat = pickle.load(open(r"data\train_mat.pickle", "rb"))
train_tfmat = pickle.load(open(r"data\train_tfmat.pickle", "rb"))
test_mat = pickle.load(open(r"data\test_mat.pickle", "rb"))
test_tfmat = pickle.load(open(r"data\test_tfmat.pickle", "rb"))
lsvm = pickle.load(open(r"data\lsvm.pickle", "rb"))
y_pred_lsvm = pickle.load(open(r"data\y_pred_lsvm.pickle", "rb"))
train_lbl = pickle.load(open(r"data\train_lbl.pickle", "rb"))
test_lbl = pickle.load(open(r"data\test_lbl.pickle", "rb"))


app = Flask(__name__)

@app.route('/process/<rawtext>',methods=["GET"])
def process(rawtext):
	if request.method == 'GET':
		phrase=rawtext
		arr = phrase.split()
		y=[]
		token=[]
		for x in arr:
			x=[x]
			test_str = vectorizer.transform(x)
			test_tfstr = tfidf.transform(test_str)
			test_tfstr.shape
			token.append(x)
			y.append(lsvm.predict(test_tfstr.toarray())[0])

			#result=dict(zip([''.join(ele) for ele in token],y))
			result=str(zip([''.join(ele) for ele in token],y))

		#return json.dumps(result, ensure_ascii=False).encode('utf8')
		return result
		

if __name__ == '__main__':
	app.run(debug=True)

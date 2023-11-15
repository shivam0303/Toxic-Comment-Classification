from utils import *

class test:
    def __init__(self,df,url):
        self.model_names = glob.glob(url + "*.sav")
        self.all_models = dict()
        for i in self.model_names:
            model_name = i.split('.')[0]
            model_name = model_name.split('\\')[1]
            self.all_models[model_name] = joblib.load(i)
        self.df = df
        self._preprocessing()
    def _preprocessing(self):
        preprocessing_obj = preprocessing(self.df)
        self.df = preprocessing_obj.fit_transform()
        print(self.df['preprocessed_text'])
    def get_predictions(self,vectorizer):
        predictions = dict()
        if len(self.df) > 0:
            tfs = vectorizer.transform(self.df['preprocessed_text'])
            for i in self.all_models.keys():
                y_pred_proba = self.all_models[i].predict_proba(tfs)
                p = 0
                if(len(y_pred_proba[:][0]) == 2):
                    p = np.argmax(y_pred_proba[:][0])*9
                else:
                    p = np.argmax(y_pred_proba[:][0])

                score = 0.1 * (float(p) + float(y_pred_proba[:,np.argmax(y_pred_proba[:][0])][0]))
                predictions[i] = score
                #print(predictions[i])
            data = dict()
            for i in self.model_names:
                model_name = i.split('.')[0]
                model_name = model_name.split('\\')[1]
                data[model_name] = predictions[model_name]
            return data,200
        else:
            return data,400
        
        
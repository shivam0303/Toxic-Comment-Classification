from utils import *
class save_models():
    def __init__(self, p_models,p_vectorizer):
        self.m_models = p_models
        self.m_vectorizer = p_vectorizer
    
    def save(self):
        os.makedirs("saved_models/", exist_ok=True)
        os.makedirs("saved_models/vectorizer/", exist_ok=True)
        for i,class_ in enumerate(self.m_models.keys()):
            joblib.dump(self.m_models[class_], 'saved_models/'+class_+'.sav')
        joblib.dump(self.m_vectorizer, 'saved_models/vectorizer/vectorizer.sav')

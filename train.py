from utils import *

class train_models():

    def __init__(self, p_datasets, p_proc_dataset):
        self.m_classes = ['toxicity','severe_toxicity','obscene','threat','insult','sexual_explicit']
        self.m_datasets = p_datasets
        self.m_proc_dataset = p_proc_dataset
        self.m_vectorizer = self._init_vectorizer()
    
    def _init_vectorizer(self):
        vectorizer = TfidfVectorizer(
            analyzer = 'word',
            stop_words = 'english',
            ngram_range = (1, 3),
            max_features = 70000
        )
        print("Vectorizing the dataset...")
        vectorizer.fit(self.m_proc_dataset['preprocessed_text'])
        return vectorizer

    def _prep_train(self, X):
        X_tf = X.to_numpy()
        X_tfidf = self.m_vectorizer.transform(X_tf)

        return  X_tfidf

    def _create_tfidf_datasets(self):
        X_tfidfs = list()
        y_tfidfs = list()

        for i in self.m_classes:
            X = self._prep_train(self.m_datasets[i]['preprocessed_text'])
            y = np.array(self.m_datasets[i].drop(columns=['preprocessed_text'],axis=1))

            X_tfidfs.append(X)
            y_tfidfs.append(y)

        for p in range(6):
            y_tfidfs[p] = [int(i) for i in y_tfidfs[p][:]]
        
        return X_tfidfs, y_tfidfs

    def fit(self):
        X, y = self._create_tfidf_datasets()
        models_ = dict()
        print("Training the Models...")
        for i,class_ in tqdm(enumerate(self.m_classes)):
            model = LogisticRegression(C=5, random_state=2020, max_iter=5000)
            model.fit(X[i],y[i])
           
            y_pred = model.predict(X[i])
           
            print()
            print(metrics.classification_report(y_pred, y[i]))
            models_[class_] = model
        
        return models_
from utils import *
class preprocessing():
  def __init__(self,df):
    self.df = df

  # Web Based Data
  def _html_remover(self,data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)

  def _url_remover(self,data):
    return re.sub(r'https\S','',data)

  # Noise Removal
  def _remove_brackets(self,data):
    return re.sub('\(.*?\)','',data)
  def _remove_punc(self,data):
    trans = str.maketrans('','', string.punctuation)
    return data.translate(trans)
  def _white_space(self,data):
    return ' '.join(data.split())
  def _remove_emojis(self,data):
    emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        "]+", flags=re.UNICODE)
    return emoji_pattern.sub('', data) # no emoji

  # Text Normalization
  def _text_lower(self,data):
    return data.lower()
  def _contraction_replace(self,data):
    return contractions.fix(data)
  def _remove_num(self,data):
    return re.sub("(\s\d+)","",data) 

  # End Process 
  def _replace_words(self,data):
    data = data.split(" ")
    WORDS_REPLACER = {
        "sh*t": "shit",
        "s**t": "shit",
        "f*ck": "fuck",
        "fu*k": "fuck",
        "f**k": "fuck",
        "f*****g": "fucking",
        "f***ing": "fucking",
        "f**king": "fucking",
        "p*ssy": "pussy",
        "p***y": "pussy",
        "pu**y": "pussy",
        "p*ss": "piss",
        "b*tch": "bitch",
        "bit*h": "bitch",
        "h*ll": "hell",
        "h**l": "hell",
        "cr*p": "crap",
        "d*mn": "damn",
        "stu*pid": "stupid",
        "st*pid": "stupid",
        "n*gger": "nigger",
        "n***ga": "nigger",
        "f*ggot": "faggot",
        "scr*w": "screw",
        "pr*ck": "prick",
        "g*d": "god",
        "s*x": "sex",
        "a*s": "ass",
        "a**hole": "asshole",
        "a***ole": "asshole",
        "a**": "ass",
    }
    clean = []
    for i in data:
      if i in WORDS_REPLACER.keys():
        clean.append(WORDS_REPLACER[i])
      else:
        clean.append(i)
    s = str()
    for i,text in enumerate(clean):
      if(i!=0):
        s+=" "
      s+=str(text)
    return s 
  def _stopword(self,data):
    data = data.split(" ")
    stop_words = set(stopwords.words('english'))
    clean = []
    for i in data:
      if i not in stop_words:
        clean.append(i)
    return clean
  def _lemmatization(self,data):
    lemma = WordNetLemmatizer()
    lemmas = []
    for i in data:
      lem = lemma.lemmatize(i, pos='v')
      lemmas.append(lem)
    return lemmas 
   
  def fit_transform(self):
    self.df['preprocessed_text'] = self.df['comment_text'].apply(self.entire_process)
    return self.df
  def entire_process(self,text):
    new_text = text
    new_text = self._html_remover(new_text)
    new_text = self._url_remover(new_text)
    new_text = self._remove_brackets(new_text)
    
    new_text = self._remove_emojis(new_text)
    new_text = self._white_space(new_text)
    new_text = self._contraction_replace(new_text)
    new_text = self._text_lower(new_text)
    new_text = self._remove_num(new_text)

    new_text = self._replace_words(new_text)
    new_text = self._remove_punc(new_text)
    new_text = self._stopword(new_text)
    #new_text = self._stemming(new_text)
    #print(new_text)
    new_text = self._lemmatization(new_text)
    s = str()
    for i,text in enumerate(new_text):
      if(i!=0):
        s+=" "
      s+=str(text)
    return s
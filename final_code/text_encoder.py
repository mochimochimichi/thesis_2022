# Databricks notebook source
class tfidf_encoder():
  def __init__(self, df):
    super().__init__()
    self.vectorizer = TfidfVectorizer(stop_words=stopwords.words("english"), smooth_idf=True, use_idf=True)
    self.embeddings = self.vectorizer.fit_transform(df).toarray().astype("float32")
    self.feature_names = self.vectorizer.get_feature_names()
  
  def encode_text(self, doc):
    text_vector = self.vectorizer.transform(doc)
    text_vector = np.array(text_vector.todense()).reshape(self.embeddings.shape[1]).astype("float32")
    return text_vector
  
  
class distilbert_encoder():
  def __init__(self, df):
    super().__init__()
    self.vectorizer = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    if torch.cuda.is_available():
       self.vectorizer = self.vectorizer.to(torch.device("cuda"))
    self.embeddings = self.vectorizer.encode(df, show_progress_bar=True).astype("float32")
  
  def encode_text(self, doc):
    text_vector = self.vectorizer.encode(doc)
    text_vector = text_vector.reshape(text_vector.shape[1]).astype("float32")
    return text_vector
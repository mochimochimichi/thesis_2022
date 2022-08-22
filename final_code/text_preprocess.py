# Databricks notebook source
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

def remove_number(s):
  s = re.sub(r'[0-9]+', '', s)
  return s

def remove_punctuation(s):
  s = ''.join([c if c not in frozenset(string.punctuation) else " " for c in s ])
  return s

def remove_nonalphabet(s):
  s = re.sub('[^a-zA-Z]+', ' ', s)
  return s

# COMMAND ----------

def get_brand_names(df):
  brand_names = " ".join(df['brand_slug'].unique().tolist()).replace('-', " ").split()
  return brand_names

# COMMAND ----------

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

# COMMAND ----------

def clean_text(df):
  
  brand_names = get_brand_names(df)
  
  for a in df.columns:
    df[a] = df[a].fillna(" ")\
                 .apply(remove_punctuation)\
                 .apply(remove_number)\
                 .apply(remove_nonalphabet)

  # join all attributes and save to a new column
  df['concat_text'] = df[df.columns].apply(lambda row: ' '.join(row.values), axis=1)  
  # apply tokenizer
  df['tokenized_text'] = df.apply(lambda row: RegexpTokenizer('\s+', gaps = True).tokenize(row['concat_text'].lower()), axis=1)
  # remove english stopwords
  df['tokenized_text'] = df.apply(lambda row: [w for w in row['tokenized_text'] if w not in stopwords.words("english")], axis=1)
  #lemmantization
  df['tokenized_text'] = df.apply(lambda row: [WordNetLemmatizer().lemmatize(w, get_wordnet_pos(nltk.pos_tag([w])[0][1])) for w in row['tokenized_text'] if get_wordnet_pos(nltk.pos_tag([w])[0][1])!=""], axis=1)
  # remove single letters 
  df['tokenized_text'] = df.apply(lambda row: [w for w in row['tokenized_text'] if len(w)>2], axis = 1)
  # remove retailer names
  df['tokenized_text'] = df.apply(lambda row: [w for w in row['tokenized_text'] if w not in list(df_retailer['retailers'])], axis = 1)
  # remove fashion stopwords
  df['tokenized_text'] = df.apply(lambda row: [w for w in row['tokenized_text'] if w not in list(df_fashion_stopword['stopword'])], axis = 1)
  # remove brand stopwords
  df['tokenized_text_no_brand'] = df.apply(lambda row: [w for w in row['tokenized_text'] if w not in brand_names], axis = 1) 
  # replace with cleaned concat text, no retailer, no stopwords
  df['concat_text'] = df.apply(lambda row: ' '.join(row['tokenized_text']), axis=1)
  # clean concat text, no brand
  df['concat_text_no_brand'] = df.apply(lambda row: ' '.join(row['tokenized_text_no_brand']), axis=1) 
  # unique tokenized text
  df['tokenized_text_unique'] = df.apply(lambda row: list(set(row['tokenized_text'])), axis = 1)
  
  # #stemming
  # from nltk.stem.porter import PorterStemmer
  # df['tokenized_text'] = df.apply(lambda row: [PorterStemmer().stem(w) for w in row['tokenized_text']], axis=1)
  # # stemmed_words = [PorterStemmer().stem(w) for w in words]

  # ##remove non-english word
  # ##but will remove brand names
  # # words = set(nltk.corpus.words.words())
  # # df['tokenized_text'] = df.apply(lambda row: [w for w in row['tokenized_text'] if w in words], axis = 1)

  return df

# COMMAND ----------

# build vocab
def build_vocab(token_column):

  flattened = []
  for tokens in token_column:
    for t in tokens:
      flattened.append(t)

  vocab = []
  for token in flattened:
    if not token in vocab:
      vocab.append(token)

  print("flattened: ", len(flattened), "vocab: ", len(vocab))
  
  return flattened, vocab

# COMMAND ----------

# 'brand_slug',  #tbd
#                   'name',
#                   'gender',
#                   'colour',
#                   'colour_name',
#                   'predominant_colour',
#                   'predominant_pattern',
#                   'product_details',
#                   'category',
#                   'product_searches',
#                   'care',
#                   'description',

# COMMAND ----------

def preprocess_text(df, top_n_word = 800):
  
  df[['category','care','description','merch_info','product_searches']] = df[['category','care','description','merch_info','product_searches']].astype(str)
  
  text_attribute_list = ['brand_slug',  #tbd
                  'name',
                  'gender',
                  'colour',
                  'colour_name',
                  'predominant_colour',
                  'predominant_pattern',
                  'category',
                  'product_searches',
                  'care',
                  'description'
                 ]
  
  df= df[text_attribute_list]
  brand_names = get_brand_names(df)
  print("0. prepared dataset and brand names \n")
  
  df_clean = clean_text(df)
  print("1. cleaned all text columns \n")
  
  flattened, vocab = build_vocab(df_clean['tokenized_text_no_brand'])
  print("2. built vocabulary \n")
  
  # build frequency table
  pdf_frequency = pd.DataFrame(pd.Series(flattened),columns = ['flat_token'])
  pdf_frequency_s = pdf_frequency.groupby(by = ['flat_token'])['flat_token'].count().sort_values(ascending = False).reset_index(name="count")
  top_words = list(pdf_frequency_s[:top_n_word]['flat_token'])
  print("3. extracted Top{} frequentist words \n".format(top_n_word))

  # obatin a csv file of words to take out common but unrelated words
  pdf_frequency_s.to_csv('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/word_frequency.csv', header = True, index = False)
  # save brand names to csv for query text preprocessing
  pdf_brand_names = pd.DataFrame(pd.Series(brand_names),columns = ['brand_names'])
  pdf_brand_names.to_csv('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/brand_name.csv', header = True, index = False)
  # save top_words to csv for query text preprocessing
  pdf_top_words = pd.DataFrame(pd.Series(top_words),columns = ['top_words'])
  pdf_top_words.to_csv('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/top_word.csv', header = True, index = False)
  print("4. saved vocab_frequency/brand_names/top{} words TABLES \n".format(top_n_word))
  
  df_clean['top_token_no_brand'] = df_clean.apply(lambda row: [w for w in row['tokenized_text_no_brand'] if w in top_words], axis=1)
  df_clean['top_token_concat_text_no_brand'] = df_clean.apply(lambda row: ' '.join(row['top_token_no_brand']), axis=1)
  df_clean['top_token_concat_text_unique_no_brand'] = df_clean.apply(lambda row: ' '.join(list(set(row['top_token_no_brand']))), axis = 1)
  print("5. ALL columns built successfully \n")
  
  return df_clean

# COMMAND ----------

def clean_text_query(query_text):
  
  df = pd.DataFrame()
  df['query'] = [query_text]
  
  brand_names = df_brand_name['brand_names']
  
  for a in df.columns:
    df[a] = df[a].fillna(" ")\
                 .apply(remove_punctuation)\
                 .apply(remove_number)\
                 .apply(remove_nonalphabet)

  # join all attributes and save to a new column
  df['concat_text'] = df[df.columns].apply(lambda row: ' '.join(row.values), axis=1)  
  # apply tokenizer
  df['tokenized_text'] = df.apply(lambda row: RegexpTokenizer('\s+', gaps = True).tokenize(row['concat_text'].lower()), axis=1)
  # remove english stopwords
  df['tokenized_text'] = df.apply(lambda row: [w for w in row['tokenized_text'] if w not in stopwords.words("english")], axis=1)
  #lemmantization
  df['tokenized_text'] = df.apply(lambda row: [WordNetLemmatizer().lemmatize(w, get_wordnet_pos(nltk.pos_tag([w])[0][1])) for w in row['tokenized_text'] if get_wordnet_pos(nltk.pos_tag([w])[0][1])!=""], axis=1)
  # remove double letters 
  df['tokenized_text'] = df.apply(lambda row: [w for w in row['tokenized_text'] if len(w)>2], axis = 1)
  # remove retailer names
  df['tokenized_text'] = df.apply(lambda row: [w for w in row['tokenized_text'] if w not in list(df_retailer['retailers'])], axis = 1)
  # remove fashion stopwords
  df['tokenized_text'] = df.apply(lambda row: [w for w in row['tokenized_text'] if w not in list(df_fashion_stopword['stopword'])], axis = 1)
  # remove brand stopwords
  df['tokenized_text_no_brand'] = df.apply(lambda row: [w for w in row['tokenized_text'] if w not in brand_names], axis = 1) 
  # replace with cleaned concat text, no retailer, no stopwords
  df['concat_text'] = df.apply(lambda row: ' '.join(row['tokenized_text']), axis=1)
  # clean concat text, no brand
  df['concat_text_no_brand'] = df.apply(lambda row: ' '.join(row['tokenized_text_no_brand']), axis=1) 
  # unique tokenized text
  df['tokenized_text_unique'] = df.apply(lambda row: list(set(row['tokenized_text'])), axis = 1)
  df['concat_token_text_unique'] = df.apply(lambda row: ' '.join(row['tokenized_text_unique']), axis = 1)
  
  # # #stemming
  # # from nltk.stem.porter import PorterStemmer
  # # df['tokenized_text'] = df.apply(lambda row: [PorterStemmer().stem(w) for w in row['tokenized_text']], axis=1)
  # # # stemmed_words = [PorterStemmer().stem(w) for w in words]

  # ##remove non-english word
  # ##but will remove brand names
  # # words = set(nltk.corpus.words.words())
  # # df['tokenized_text'] = df.apply(lambda row: [w for w in row['tokenized_text'] if w in words], axis = 1)

  return df

# COMMAND ----------

def sort_coo(coo_matrix):
    """Sort a dict with highest score"""
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(np.round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature, score
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results
  
def get_keywords(vectorizer, feature_names, doc, TOP_K_KEYWORDS):
    """Return top k keywords from a doc using TF-IDF method"""

    #generate tf-idf for the given document
    tf_idf_vector = vectorizer.transform([doc])
    
    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())

    #extract only TOP_K_KEYWORDS
    keywords=extract_topn_from_vector(feature_names,sorted_items,TOP_K_KEYWORDS)
    
    return list(keywords.keys())
  
def get_top_k_keywords(vectorizer, feature_names, df, TOP_K_KEYWORDS = 20):
  keywords = []  
  for doc in df:
    keywords.append(get_keywords(vectorizer, feature_names, doc, TOP_K_KEYWORDS = 20))
    
  return keywords
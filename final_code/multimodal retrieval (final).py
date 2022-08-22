# Databricks notebook source
# MAGIC %run "/Users/Sifang.Du@burberry.com/project_code/project_utils"

# COMMAND ----------

# MAGIC %run "/Users/Sifang.Du@burberry.com/project_code/project_configs"

# COMMAND ----------

# MAGIC %run "/Users/Sifang.Du@burberry.com/project_code/text_preprocess"

# COMMAND ----------

# MAGIC %run "/Users/Sifang.Du@burberry.com/project_code/text_encoder"

# COMMAND ----------

# MAGIC %run "/Users/Sifang.Du@burberry.com/project_code/image_encoder"

# COMMAND ----------

set_seed(42)

# COMMAND ----------

# main data
pdf_edited = pd.read_csv('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/pdf_hash_5year_sampled_v3.csv')
pdf_edited_p = preprocess_text(pdf_edited,500)

color_embeddings = np.load('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/color_embeddings_pdf_hash_5year_sampled_v3_16612.npy')
image_embeddings = np.load('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/image_embeddings_v12.npy')


# COMMAND ----------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("{} is avaliable".format(device))

# COMMAND ----------

# a dictionary of all trained models' pathes
image_embeddings = np.load('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/image_embeddings_v12.npy')#10,12,16

torch.manual_seed(42)                  
trained_model_dict = {'keyword_prediction_nn': '/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/vgg16_keyword_nn_v5_final_12_epoches.pth',
                      'class_prediction_nn': '/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/vgg16_b_bag_v4.pth'}

SavedFilePath = trained_model_dict['class_prediction_nn']
trained_model = load_checkpoint(SavedFilePath)
img_encoder = CreateImageEmbedding(trained_model,2)

# COMMAND ----------

df_sample = pd.read_csv('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/sample_query.csv')

# COMMAND ----------

df_sample

# COMMAND ----------

id=0
b_test_1_doc_raw = df_sample.loc[id,"text"]
b_test_1 = df_sample.loc[id,"image_url"]

cleaned_text = clean_text_query(b_test_1_doc_raw)
b_test_1_doc = [cleaned_text['concat_text_no_brand'][0]]
print(b_test_1_doc)

# COMMAND ----------

display_img(b_test_1)

# COMMAND ----------

display_n = pdf_edited.shape[0]
ails_k = 30

# COMMAND ----------

# MAGIC %md ### Image only

# COMMAND ----------

b_img_embed = get_img_feature_vector(b_test_1, img_encoder)
i_idx = faiss_retrieve_img(b_img_embed, image_embeddings,display_n)
display_img(idx_to_path(i_idx[:10],pdf_edited))

# COMMAND ----------

b_img_embed = get_img_feature_vector(b_test_1, img_encoder)
i_idx = faiss_retrieve_img(b_img_embed, image_embeddings,display_n, metric = "ed")
b_img_embed = get_img_feature_vector(b_test_1, img_encoder)
j_idx = faiss_retrieve_img(b_img_embed, image_embeddings,display_n, metric = "cs")

for k in range(pdf_edited.shape[0]):
  if np.sum(i_idx[:k] == j_idx[:k]) == k:
    k=k+1
  else:
    print("Top",k,"results are the same")
    break

# COMMAND ----------

# MAGIC %md ### Text only

# COMMAND ----------

top_encoder = tfidf_encoder(pdf_edited_p['top_token_concat_text_no_brand'])
b_test = top_encoder.encode_text(b_test_1_doc)
t_idx = faiss_retrieve_text(b_test, top_encoder.embeddings, display_n,metric = "ed")
display_img(idx_to_path(t_idx[:10], pdf_edited))

# COMMAND ----------

top_encoder = tfidf_encoder(pdf_edited_p['top_token_concat_text_no_brand'])
b_test = top_encoder.encode_text(b_test_1_doc)
i_idx = faiss_retrieve_text(b_test, top_encoder.embeddings, display_n,metric = "ed")
b_test = top_encoder.encode_text(b_test_1_doc)
j_idx = faiss_retrieve_text(b_test, top_encoder.embeddings, display_n,metric = "cs")

for k in range(pdf_edited.shape[0]):
  if np.sum(i_idx[:k] == j_idx[:k]) == k:
    k=k+1
  else:
    print("Top",k,"results are the same")
    break

# COMMAND ----------

# MAGIC %md ### bert encoder

# COMMAND ----------

bert_encoder = distilbert_encoder(pdf_edited_p['top_token_concat_text_no_brand'])
# bert_encoder = distilbert_encoder(pdf_edited_p['concat_text']) 

# COMMAND ----------

# MAGIC %md ### Combined 

# COMMAND ----------

import time
t,i,c = 15,1,0.05

composer = embedding_composer(img_encoder, top_encoder)
database = embedding_database(image_embeddings,top_encoder.embeddings,color_embeddings)

b_all = composer.concat_vectors(b_test_1,b_test_1_doc,t,i,c)
all_embeddings = database.concat_embeddings(t,i,c)

assert b_all.shape[1] == all_embeddings.shape[1]
start = time.time()
idx = faiss_retrieve_text(np.array(b_all).reshape(b_all.shape[1]), all_embeddings, 10)
end = time.time()
print("The time of execution of above program is :", end-start)
# display_img(idx_to_path(idx[:10], pdf_edited))

# COMMAND ----------

from math import comb
#0.07,1,0.04
t,i,c = 1,1,0.2
brand_list = []
img_AILS_k = []
txt_AILS_k = []
for ails_k in [10,20,30,40,50]:#10,20,30,40,50,60,70,80,90,100
  img_dist = []
  txt_dist = []
  for id in range(5):#range(5)
    b_test_1_doc_raw = df_sample.loc[id,"text"]
    b_test_1 = df_sample.loc[id,"image_url"]

    cleaned_text = clean_text_query(b_test_1_doc_raw)
    b_test_1_doc = [cleaned_text['concat_text_no_brand'][0]]

    text_encoder = bert_encoder #top_encoder #bert_encoder
    composer = embedding_composer(img_encoder,text_encoder)
    database = embedding_database(image_embeddings,text_encoder.embeddings,color_embeddings)

    b_all = composer.concat_vectors(b_test_1,b_test_1_doc,t,i,c)
    all_embeddings = database.concat_embeddings(t,i,c)

    assert b_all.shape[1] == all_embeddings.shape[1]

    idx = faiss_retrieve_text(np.array(b_all).reshape(b_all.shape[1]), all_embeddings,display_n, metric ="ed")
    brand_list.append(pdf_edited.loc[idx[:ails_k], 'brand_slug'].values)
    display_img(idx_to_path(idx[:10], pdf_edited))
    idx = idx[:ails_k]

    dist = np.zeros((len(idx),len(idx)))
    for x in range(len(idx)):
      for y in range(len(idx)):
        dist[x,y] = np.linalg.norm(image_embeddings[idx[x]]-image_embeddings[idx[y]])
    img_dist.append(dist.sum()/comb(ails_k,2))

    d_lst = [pdf_edited_p.loc[i, "top_token_concat_text_unique_no_brand"].split(" ") for i in idx]
    mtx = np.array([len(set(d_lst[i]).intersection(d_lst[j])) for j in range(len(d_lst)) for i in range(len(d_lst))]).reshape(len(d_lst),len(d_lst))
    np.fill_diagonal(mtx,0)
    txt_dist.append(mtx.sum()/comb(ails_k,2))
    
    print("image distance",dist.sum()/comb(ails_k,2))
    print("text similarity",mtx.sum()/comb(ails_k,2))
    
    
  img_AILS_k.append(np.mean(img_dist))
  txt_AILS_k.append(np.mean(txt_dist))

# COMMAND ----------

flattened, vocab = build_vocab(brand_list)
pdf_frequency = pd.DataFrame(pd.Series(flattened),columns = ['flat_token'])
pdf_frequency_s = pdf_frequency.groupby(by = ['flat_token'])['flat_token'].count().sort_values(ascending = False).reset_index(name="count")
# pdf_frequency_s.style.background_gradient()

# COMMAND ----------

class_dist_non_still_life = pdf_frequency_s

class_dist_non_still_life.columns = ['brand', 'number_of_matches']

g = sns.catplot(data=class_dist_non_still_life, kind="bar", x=class_dist_non_still_life.brand, y=class_dist_non_still_life.number_of_matches, orient="v", height = 5, aspect = 3)
ax = g.facet_axis(0, 0)
for c in ax.containers:
    labels = [f'{(v.get_height()):.0f}' for v in c]
    ax.bar_label(c, labels=labels, label_type='edge')   
plt.xticks(rotation=45)
plt.ylabel('Number of matches',fontsize = 12)
plt.xlabel('Brand',fontsize = 12)
plt.title('Number of matches from each brand in the Top 10 retrieval results', fontsize = 20, fontweight='bold')


# COMMAND ----------

# img_1 = [img_AILS_k, txt_AILS_k]
# img_2 = [img_AILS_k, txt_AILS_k]
# img_3 = [img_AILS_k, txt_AILS_k]

# tf_only = [img_AILS_k,txt_AILS_k]
# bert_only = [img_AILS_k,txt_AILS_k]
# original_only = [img_AILS_k,txt_AILS_k]

# tf_classifier = [img_AILS_k,txt_AILS_k]
# bert_classifier = [img_AILS_k,txt_AILS_k]
# bert_original = [img_AILS_k,txt_AILS_k]
# tf_original = [img_AILS_k,txt_AILS_k]

# COMMAND ----------

# plt.figure(figsize = (15,10))

plt.title('Text AILS score vs. Number of results in retrieved list')
plt.xlabel('Top k results')
plt.ylabel('Text AILS score')
bars = ('k=10', 'k=20', 'k=30', 'k=40', 'k=50')
x_pos = np.arange(len(bars))
plt.xticks(x_pos, bars)

plt.plot(img_1[1])
plt.plot(img_2[1])
plt.plot(img_3[1])

plt.legend(['original', 'bag classifier', 'multi-label'], loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=5)

# COMMAND ----------

plt.title('Image AILS score vs. Number of results in retrieved list')
plt.xlabel('Top k results')
plt.ylabel('Image AILS score')
bars = ('k=10', 'k=20', 'k=30', 'k=40', 'k=50')
x_pos = np.arange(len(bars))
plt.xticks(x_pos, bars)

plt.plot(img_1[0])
plt.plot(img_2[0])
plt.plot(img_3[0])
plt.legend(['original', 'bag classifier', 'multi-label'], loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=5)

# COMMAND ----------

# def make_word_grid(pdf_edited, image_embeddings, lb, ub, n_components=300, perplexity=30):

#   features = np.array(image_embeddings[lb:ub])
#   images = list(pdf_edited["first_image"][lb:ub].values)
#   word_labels = list(pdf_edited["brand_slug"][lb:ub].values)

#   pca = PCA(n_components,random_state = 42)
#   pca.fit(features)

#   pca_features = pca.transform(features)

#   X = np.array(pca_features)
#   tsne = TSNE(n_components=2, learning_rate=150, perplexity = perplexity, angle=0.2, verbose=2, random_state = 42).fit_transform(X)

#   tx, ty = tsne[:,0]*300, tsne[:,1]*300
# #   tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
# #   ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))
#   x_coords,y_coords = tx, ty
  
#   max = __builtins__.max

#   plt.scatter(x_coords, y_coords)

#   for label, x, y in zip(word_labels, x_coords, y_coords):
#       plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
#   plt.xlim(x_coords.min()-50, x_coords.max()+50)
#   plt.ylim(y_coords.min()-50, y_coords.max()+50)
  
#   plt.show()
#   plt.figure(figsize = (40,40))
    
  
#   return

# COMMAND ----------

# make_word_grid(pdf_edited[tote_bool], top_encoder.embeddings[tote_bool], 1000,1500, n_components=499, perplexity=30)

# COMMAND ----------

# plt.title('Text AILS score vs. Number of results in retrieved list')
# plt.xlabel('Top k results')
# plt.ylabel('Text AILS score')
# bars = ('k=10', 'k=20', 'k=30', 'k=40', 'k=50')
# x_pos = np.arange(len(bars))
# plt.xticks(x_pos, bars)
# plt.plot(bert_original[1])
# plt.plot(tf_original[1])
# plt.legend(['DistilBERT', 'TF-IDF'])

# COMMAND ----------

# MAGIC %md ### sparse rate and matrix mean 

# COMMAND ----------

text_embed_sparse_rate = np.sum(text_encoder.embeddings == 0)/(text_encoder.embeddings.shape[0]*text_encoder.embeddings.shape[1])
img_embed_sparse_rate = np.sum(image_embeddings == 0)/(image_embeddings.shape[0]*image_embeddings.shape[1])

print(text_embed_sparse_rate,img_embed_sparse_rate)
print(text_encoder.embeddings.mean(),image_embeddings.mean(),color_embeddings.mean())
# t,i,c = 300,1,0.5
# text_encoder.embeddings.mean()*t,image_embeddings.mean()*i,color_embeddings.mean()*c

# COMMAND ----------

# MAGIC %md ### competitors in top 10

# COMMAND ----------

# df_brand_rank = pdf_edited.loc[idx[:10]][['brand_slug']]
# df_brand_rank.reset_index(drop = True, inplace = True)
# df_brand_rank['rank'] = df_brand_rank.index
# cov = df_brand_rank.groupby(by = "brand_slug")["rank"].mean().values / df_brand_rank.groupby(by = "brand_slug")["rank"].std().values
# df_brand_rank.groupby(by = "brand_slug")["rank"].mean().index[np.argsort(cov)]

pdf_edited.loc[idx[:10], 'brand_slug'].values

# COMMAND ----------

pd.read_csv("/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/word_frequency.csv")[:500].values

# COMMAND ----------

# MAGIC %md ### make t-SNE map

# COMMAND ----------

tote_bool = pdf_edited_p['top_token_concat_text_no_brand'].str.contains('tote')
# tote_bool = [True for i in range(len(image_embeddings))]

# COMMAND ----------

make_grid(pdf_edited[tote_bool], all_embeddings[tote_bool], 0, 500, n_components = 499, perplexity=22, mode = "cloud")

# COMMAND ----------

make_grid(pdf_edited[tote_bool], all_embeddings[tote_bool], 0, 500, n_components = 499, perplexity=22, mode = "grid")

# COMMAND ----------

# MAGIC %md ### Hard blender

# COMMAND ----------

#get colour index
c_idx = faiss_retrieve_text(np.array(composer.color_vector).astype("float32"), np.array(color_embeddings).astype("float32"), display_n)

# get individual feature ranking
image_rank = pd.DataFrame(i_idx, columns = ['image_id'])
image_rank["id"] = image_rank['image_id']
image_rank.reset_index(inplace = True)
image_rank.rename({'index': 'image_rank'}, axis=1,inplace = True)

text_rank = pd.DataFrame(t_idx, columns = ['text_id'])
text_rank["id"] = text_rank['text_id']
text_rank.reset_index(inplace = True)
text_rank.rename({'index': 'text_rank'}, axis=1,inplace = True)

color_rank = pd.DataFrame(c_idx, columns = ['color_id'])
color_rank["id"] = color_rank['color_id']
color_rank.reset_index(inplace = True)
color_rank.rename({'index': 'color_rank'}, axis=1,inplace = True)

#merge rankings
merge_rank = pd.merge(image_rank, text_rank, on='id')
merge_rank['merge_rank'] = merge_rank['image_rank']+merge_rank['text_rank']
merge_rank = pd.merge(merge_rank, color_rank, on='id')
merge_rank['merge_rank'] = merge_rank['merge_rank']+merge_rank['color_rank']*0.2
merge_rank.sort_values(by = 'merge_rank',inplace = True)
merge_rank.reset_index(inplace = True, drop=True)

#display results
idx = merge_rank['id']
display_img(idx_to_path(idx[:10], pdf_edited))
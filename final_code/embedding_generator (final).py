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

# MAGIC %md ## Reduce Dataset
# MAGIC 
# MAGIC -remove highly similar images and keep the one with longest description

# COMMAND ----------

def faiss_retrieve(q_img_vector, img_embed, topk):
  dim = img_embed.shape[1]
  index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
  assert index.is_trained == True
  faiss.normalize_L2(img_embed)
  index.add(img_embed)
  D, I = index.search(np.array([q_img_vector]), topk)  # search

  return D[0], I[0] 

# COMMAND ----------

def find_distinct_images(pdf_edited, image_embeddings,pdf_edited_p):

  similars = []
  for n in tqdm(range(len(pdf_edited))):
    d,i = faiss_retrieve(image_embeddings[n],image_embeddings,5)
    keep_idx = [x.tolist()[0] for x in np.argwhere(i!=n)]
    i = i[keep_idx]
    d = d[keep_idx]
    similars.append(i[np.argwhere(d >= 0.9)])
    
  # convert list of array to 1D list
  l_similars = [[y[0] for y in x.tolist()] for x in similars]

  #assign similar indexes
  print(l_similars)
  pdf_edited['similars'] = l_similars

  #calculate token length
  pdf_edited["length"] = [len(x) for x in pdf_edited_p['top_token_concat_text_no_brand']]

  #find token length for similar products
  pdf_edited['similars_length'] = [[pdf_edited['length'][j] for j in pdf_edited.loc[i,"similars"]] if len(pdf_edited.loc[i,"similars"]) > 0 else [pdf_edited.loc[i,'length']] for i in range(len(pdf_edited))]

  # select index with the longest token list as the root product
  pdf_edited['root_product'] = [pdf_edited.loc[i,"similars"][np.argmax(pdf_edited.loc[i,"similars_length"])] if np.asarray(pdf_edited.loc[i,"similars_length"]).max() != pdf_edited.loc[i,'length'] else i for i in range(len(pdf_edited))]

  # mark row to keep if row is root product
  distinct_product_bool = np.asarray([pdf_edited.loc[i,'root_product'] == i for i in range(len(pdf_edited))])
#     np.save('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/distinct_product_bool_{}.npy'.format(distinct_product_bool.shape[0]), distinct_product_bool)
    
  return distinct_product_bool
    

# COMMAND ----------

# MAGIC %md ## Validate Image urls 
# MAGIC 
# MAGIC -remove rows with faulty urls that throws 404 error

# COMMAND ----------

# check image source validity
# empty image array([[11592],[16726]])
def check_url(pdf_edited):
  validity=[]
  for url in tqdm(pdf_edited['first_image']):
    try:
      req = urllib.request.urlopen(url)
      validity.append(1)
    except:
      validity.append(0)

  arr = np.asarray(validity)
  zero_index = np.argwhere(arr == 0)
  zero_index = [x[0] for x in zero_index]
  
  print("------ Removed invalid indexes: {} ------ \n".format(zero_index))
  pdf_edited.drop(zero_index, inplace = True)
  pdf_edited.reset_index(drop = True, inplace = True)
  
  return pdf_edited

# COMMAND ----------

# MAGIC %md ## Text embeddings

# COMMAND ----------

#preprocess text attributes for all data
keep_top_n = 800
pdf_edited_p = preprocess_text(pdf_edited, keep_top_n)

# COMMAND ----------

save pre-processed text data for later calling
pdf_edited_p.to_csv('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/processed_{}_{}_{}.csv'.format(keep_top_n, pdf_name, len(pdf_edited)), header = True, index = False)

# COMMAND ----------

#load pre-processed text data
pdf_edited_p = pd.read_csv('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/processed_{}_{}_{}.csv')

# COMMAND ----------

# MAGIC %md ## Color embeddings

# COMMAND ----------

# create color embeddings for all data
def get_color_embeddings(pdf_edited):
  color_embeds = []
  for url in tqdm(pdf_edited['first_image']):
    color_embeds.append(color_moments(url))

  #save as array for later calling
  arr_color_embeds = np.asarray(color_embeds)
  return arr_color_embeds

# COMMAND ----------

# np.save('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/color_embeddings_{}_d{}.npy'.format(pdf_name,arr_color_embeds.shape[1]), arr_color_embeds)

# COMMAND ----------

# MAGIC %md ## Image embeddings

# COMMAND ----------

# a dictionary of all trained models' pathes

saved_model = {'class_prediction_nn': '/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/vgg16_b_bag_4.pth',
               'keyword_prediction_nn': '/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/vgg16_keyword_nn_v5_1_epoch_8.pth'}

# COMMAND ----------

pdf_edited = pd.read_csv('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/pdf_hash_5year_sampled_v3.csv')

# COMMAND ----------

# MAGIC %md ### class_prediction_nn 
# MAGIC - Burberry data

# COMMAND ----------

#set seed for Reproducibility 
torch.manual_seed(42)

#load feature extraction model
SavedFilePath = saved_model['class_prediction_nn']
trained_model = load_checkpoint(SavedFilePath)

#set up feature extraction network
remove_n_layer = 2
img_encoder = image_encoder(pdf_edited,trained_model,remove_n_layer)

#start image encoding
image_embeddings = img_encoder.encode()

# COMMAND ----------

#save as array for later calling
np.save('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/image_embeddings_v14.npy', image_embeddings)

# COMMAND ----------

# load saved image embeddings
image_embeddings = np.load('/dbfs/mnt/personal/sdu/image_embeddings_v5.npy')

# COMMAND ----------

# MAGIC %md ### keyword_prediction_nn 
# MAGIC 
# MAGIC - edited_unique_hash_sampled data

# COMMAND ----------

#set seed for Reproducibility 
torch.manual_seed(42)

#load feature extraction model

SavedFilePath = saved_model['keyword_prediction_nn']

checkpoint = torch.load(SavedFilePath)
model = models.vgg16(pretrained = True)
model.classifier = nn.Sequential(nn.Linear(25088,1568),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(1568,512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512,499))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.load_state_dict(checkpoint['state_dict'])

#set up feature extraction network
remove_n_layer = 1
img_encoder = image_encoder(pdf_edited,model,remove_n_layer)

#start image encoding
image_embeddings = img_encoder.encode()

# COMMAND ----------

# save image embeddings
np.save('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/image_embeddings_v16.npy', image_embeddings)

# COMMAND ----------

def create_database(path):
  
  pdf_edited = pd.read_csv(path)
  pdf_name = path[-path[::-1].find("/"):-path[::-1].find(".")-1]

  #check gpu avaliability before extracting image features
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("{} is avaliable".format(device))
    
  #check url validity
  pdf_edited = check_url(pdf_edited)
  
#   zero_index = [808, 1522, 4876, 5577, 11537, 11798, 12421, 12777]
#   pdf_edited.drop(zero_index, inplace = True)
#   pdf_edited.reset_index(drop = True, inplace = True)
  
  print("Stage 0 ------ remove invalid URLs DONE ------ \n")
  path = '/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/'+pdf_name+'_cleaned.csv'
  pdf_edited.to_csv(path, header = True, index = False)
  pdf_name = pdf_name + '_cleaned'
  
  #-------------------Color-------------------
  #get_color_embeddings
  arr_color_embeds = get_color_embeddings(pdf_edited)
  print("Stage 1 ------ color embeddings DONE ------ \n")
  np.save('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/color_embeddings_{}_d{}.npy'.format(pdf_name,arr_color_embeds.shape[1]), arr_color_embeds)
  
  #-------------------Image-------------------
  #set seed for Reproducibility 
  torch.manual_seed(42)
  
  #load feature extraction model
  trained_model = load_checkpoint('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/vgg16_b_bag_v2.pth')

  #set up feature extraction network
  remove_n_layer = 999 # use raw embeddings
  img_encoder = image_encoder(pdf_edited,trained_model,remove_n_layer)

  #start image encoding
  image_embeddings = img_encoder.encode()
  print("Stage 2 ------ raw vgg16 image embeddings DONE ------ \n")
  np.save('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/image_embeddings_{}_d{}.npy'.format(pdf_name,image_embeddings.shape[1]), image_embeddings)
  
  #-------------------Text-------------------
  keep_top_n = 800
  pdf_edited_p = preprocess_text(pdf_edited, keep_top_n)
  pdf_edited_p.to_csv('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/processed_{}_{}_{}.csv'.format(keep_top_n, pdf_name, len(pdf_edited)), header = True, index = False)
  
  #-------------------find_distinct-------------------
  distinct_product_bool = find_distinct_images(pdf_edited, image_embeddings,pdf_edited_p)
  print("Stage 3 ------ find distinct product images DONE ------ \n")
  np.save('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/distinct_product_bool_{}_d{}.npy'.format(pdf_name, distinct_product_bool.shape[0]), distinct_product_bool)
  

  return

# COMMAND ----------

path = '/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/pdf_hash_5_year_first_party.csv'

# COMMAND ----------


# Databricks notebook source
# MAGIC %run "/Users/Sifang.Du@burberry.com/project_code/project_utils"

# COMMAND ----------

# MAGIC %run "/Users/Sifang.Du@burberry.com/project_code/project_configs"

# COMMAND ----------

# MAGIC %run "/Users/Sifang.Du@burberry.com/project_code/text_preprocess"

# COMMAND ----------

# MAGIC 
# MAGIC %run "/Users/Sifang.Du@burberry.com/project_code/text_encoder"

# COMMAND ----------

# MAGIC %run "/Users/Sifang.Du@burberry.com/project_code/image_encoder"

# COMMAND ----------

set_seed(42)

# COMMAND ----------

# pdf_edited = pd.read_csv('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/edited_bag_unique_hash_sampled.csv')
pdf_edited = pd.read_csv('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/pdf_hash_5year_sampled_v3.csv')
# pdf_edited_p = preprocess_text(pdf_edited, 500)
pdf_edited_p = pd.read_csv('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/processed_500_sample_v3_final.csv')

# COMMAND ----------

torch.manual_seed(42)

# COMMAND ----------

trained_model_dict = {'keyword_prediction_nn': '/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/vgg16_keyword_nn_v2.pth',
                      'class_prediction_nn': '/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/vgg16_b_bag_v2.pth'}

SavedFilePath = trained_model_dict['keyword_prediction_nn']
trained_model = load_checkpoint(SavedFilePath)
img_encoder = CreateImageEmbedding(trained_model,0)

b_test_1 = "https://assets.burberry.com/is/image/Burberryltd/DA129E6A-C053-4265-8A55-4EED6FE9A39F?$BBY_V2_SL_1x1$&wid=1876&hei=1876"
get_img_feature_vector(b_test_1, img_encoder)

# COMMAND ----------

# extract keywords from cleaned text data
top_encoder = tfidf_encoder(pdf_edited_p['top_token_concat_text_no_brand'])
keywords = get_top_k_keywords(top_encoder.vectorizer, top_encoder.feature_names, pdf_edited_p['top_token_concat_text_no_brand'], 20)
pdf_edited_p['keywords'] = keywords
f, v = build_vocab(keywords)
del keywords

from sklearn import preprocessing
df_vocab = pd.DataFrame(v, columns = ['keyword'])
encoder = preprocessing.LabelEncoder()
df_vocab['label'] = encoder.fit_transform(df_vocab['keyword'])
keyword_to_idx = df_vocab.set_index('keyword').label.to_dict()

# convert keywords to target vectors

target_vectors = []
for lst in pdf_edited_p['keywords']:
  vec = np.zeros(len(df_vocab),int)
  for w in lst:
    vec[keyword_to_idx[w]] = 1
  target_vectors.append(vec)    
  
pdf_edited_p['target_vectors'] = target_vectors
del target_vectors

#select random negative indexes for negative sampling
negative_indexes = []
for xs in pdf_edited_p['target_vectors']:
  indexes = []
  for i,x in enumerate(xs):
    if x == 0:
      indexes.append(i)
  negative_indexes.append(indexes)
  
n_negative =20
chosen_neg_indexes = []

for i in range(len(pdf_edited_p)):
  chosen_index = get_rand_idx(size = n_negative, max_idx = len(negative_indexes[i]))
  chosen_neg_indexes.append(chosen_index)   
  
pdf_edited_p['chosen_neg_indexes'] = chosen_neg_indexes

del chosen_neg_indexes
del negative_indexes

pdf_edited[['keywords','label','chosen_neg_indexes']] = pdf_edited_p[['keywords','target_vectors','chosen_neg_indexes']].copy()

# COMMAND ----------

# save keyword index conversion in case of inference
df_vocab.to_csv("/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/keyword_to_index.csv", index = False, header = True)

# COMMAND ----------

# add multilabel sampler into customized dataloader

def create_dataloader(image_datasets, batch_size):
  
  if len(image_datasets)>1:
    all_loader = torch.utils.data.DataLoader(image_datasets["all"], batch_size=batch_size)
    train_loader = torch.utils.data.DataLoader(image_datasets["train"], batch_size=batch_size, shuffle=False, sampler = MultilabelBalancedRandomSampler(np.asarray([x for x in image_datasets["train"].image_labels['label']]), class_choice="cycle"))
    valid_loader = torch.utils.data.DataLoader(image_datasets["valid"], batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(image_datasets["test"], batch_size=batch_size)

    dataloaders = {"all": all_loader,
                   "train": train_loader,
                   "valid": valid_loader,
                   "test": test_loader,
                   } 
    return dataloaders
                                           
  else:
    all_loader = torch.utils.data.DataLoader(image_datasets["all"], batch_size=batch_size)
    dataloaders = {"all": all_loader}
    return dataloaders

# COMMAND ----------

pdf_edited['label'] = [x.astype("float32") for x in pdf_edited['label']]
# pdf_edited['label'] = [x for x in pdf_edited['label']]
df_train, df_valid, df_test = train_valid_test_split(pdf_edited,0.3,0.2,0.5,stratify_required = False)
image_datasets = load_transform(pdf_edited, df_train, df_valid, df_test, img_size = 224)
dataloaders = create_dataloader(image_datasets, 64)

# COMMAND ----------

# check if all labels are present in the training set 
final = []
for i,x in enumerate(df_train["keywords"]):
  final = final + df_train.loc[i, "keywords"]
  i = i+1
  
assert len(list(set(final))) == len(v), "missing label in training set"

# COMMAND ----------

del df_train, df_valid, df_test

# COMMAND ----------

# load checkpoint information
model = models.vgg16(pretrained = True)
checkpoint = torch.load("/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/vgg16_keyword_nn_v5_final_12_epoches.pth") 

# COMMAND ----------

# change parameters
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(25088,1568),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(1568,512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512,499))


criterion = nn.BCELoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003) #SGD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])

# COMMAND ----------

# training model and validate every few batches 
print("Running on {}".format(device))

epochs = 15
steps = 0
running_loss = 0
print_every = 5


for epoch in range(epochs):
  
    for inputs, labels in tqdm(dataloaders['train']):
        steps += 1
      
#       Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model.forward(inputs)
        output = torch.sigmoid(output)
#       calculate loss and update weights
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            valid_loss = 0
            valid_accuracy = 0
            valid_precision = 0
            valid_recall = 0
            
            model.eval()
            with torch.no_grad():
                for inputs, labels in dataloaders['valid']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    output = model.forward(inputs)
                    output = torch.sigmoid(output)
                    
                    batch_loss = criterion(output, labels)
                    output = np.round(output.cpu().numpy())
                    labels = labels.cpu().numpy()
                    
                    batch_accuracy = accuracy_score(labels[0], output[0])
                    batch_precision = precision_score(labels[0], output[0])
                    batch_recall = recall_score(labels[0], output[0])
                    
                    valid_loss += batch_loss.item()
                    valid_accuracy += batch_accuracy.item()
                    valid_precision += batch_precision.item()
                    valid_recall += batch_recall.item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Valid loss: {valid_loss/len(dataloaders['valid']):.3f}.. ",
                  f"Valid accuracy: {valid_accuracy/len(dataloaders['valid']):.3f}.. ",
                  f"Valid precision: {valid_precision/len(dataloaders['valid']):.3f}.. ",
                  f"Valid recall: {valid_recall/len(dataloaders['valid']):.3f}.. ")
            running_loss = 0
            model.train()
            
  #model snapshot
    if ((epoch+1) % 1 ==0) and (epoch!=0):
      checkpoint = {'epochs': epoch,
            'arch': "vgg16",
            'classifier': model.classifier,
            'optimizer': optimizer.state_dict(),
            'state_dict': model.state_dict()
             }

      SavedFilePath = '/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/vgg16_keyword_nn_v6_epoch_{}.pth'.format(epoch+1)
      torch.save(checkpoint, SavedFilePath)

# COMMAND ----------

# running on test set

model.eval()
test_loss = 0
with torch.no_grad():
  for inputs, labels in tqdm(dataloaders['test']):
    inputs, labels = inputs.to(device), labels.to(device)
    output = model.forward(inputs)
    output = torch.sigmoid(output)
    batch_loss = criterion(output, labels)
    test_loss += batch_loss.item()
            
print(f"Valid loss: {test_loss/len(dataloaders['test']):.3f}.. ")

# COMMAND ----------


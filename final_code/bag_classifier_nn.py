# Databricks notebook source
# MAGIC %run "/Users/Sifang.Du@burberry.com/project_code/project_utils"

# COMMAND ----------

# MAGIC %run "/Users/Sifang.Du@burberry.com/project_code/project_configs"

# COMMAND ----------

set_seed(42)

# COMMAND ----------

df_burberry_bag = spark.table("sdu.burberry_bag_uniform__still_life_unique_material")
get_shape(df_burberry_bag)
pdf_burberry_bag = df_burberry_bag.toPandas()

# COMMAND ----------

len(pdf_burberry_bag['material'].unique())

# COMMAND ----------

pdf_bag = pdf_burberry_bag.copy()

# COMMAND ----------

pdf_burberry_bag.head(3)

# COMMAND ----------

for c in pdf_burberry_bag.columns:
  print(pdf_burberry_bag[c].describe())

# COMMAND ----------

pdf_bag = pdf_burberry_bag[["material", "alt_l5_desc", "content"]]

# COMMAND ----------

# pdf_bag = pdf_bag[pdf_bag.columns[2:]]

# COMMAND ----------

pdf_bag['class'] = [val[2:] for val in pdf_bag['alt_l5_desc'].values]
pdf_bag.drop(pdf_bag[pdf_bag["class"] == "SLING"].index, inplace = True)
pdf_bag = pdf_bag.reset_index(drop = True)

# COMMAND ----------

encoder = preprocessing.LabelEncoder()
pdf_bag["label"] = encoder.fit_transform(pdf_bag['class'])
pdf_bag.columns = ["material", "alt_l5_desc", "image", "class", "label"]

# COMMAND ----------

len(pdf_bag)

# COMMAND ----------

len(pdf_bag.label.unique())

# COMMAND ----------

df_idx_to_class = pd.DataFrame(encoder.classes_)
df_idx_to_class.columns = ["class"]
df_idx_to_class = df_idx_to_class.reset_index()
df_idx_to_class

# COMMAND ----------

sdf_idx_to_class =  spark.createDataFrame(df_idx_to_class)
sdf_idx_to_class.write.mode('overwrite').saveAsTable('sdu.burberry_bag_idx_to_class')

# COMMAND ----------

pdf_bag.head(3)

# COMMAND ----------

display_img(list(pdf_bag['image'][:30]))

# COMMAND ----------

class_dist_non_still_life = pdf_bag.groupby(by = "class")[['label']].count()
class_dist_non_still_life = class_dist_non_still_life.sort_values(by = "label", ascending = False)
class_dist_non_still_life

# COMMAND ----------

sns.catplot(data=class_dist_non_still_life, kind="bar", y=class_dist_non_still_life.label, x=class_dist_non_still_life.index, orient="v", height = 5, aspect = 3)
sns.lineplot(data=class_dist_non_still_life)

# COMMAND ----------

s_pdf_bag = pdf_bag.sample(frac=1)

# COMMAND ----------

df_train, df_valid, df_test = train_valid_test_split(s_pdf_bag, 0.6, 0.2, 0.2, True)
image_datasets = load_transform(s_pdf_bag, df_train, df_valid, df_test, img_size = 224)
dataloaders = create_dataloader(image_datasets, 64)

# COMMAND ----------

# import model
model = models.vgg16(pretrained = True)

#change parameters

for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(25088,1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(1024,256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256,12),
                                 nn.LogSoftmax(dim=1))


criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# COMMAND ----------

# CUDA_LAUNCH_BLOCKING=1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print("Running on {}".format(device))

epochs = 5
steps = 0
running_loss = 0
print_every = 5

for epoch in range(epochs):
    for inputs, labels in tqdm(dataloaders['train']):
        steps += 1

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in dataloaders['valid']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    valid_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Valid loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                  f"Valid accuracy: {accuracy/len(dataloaders['valid']):.3f}")
            running_loss = 0
            model.train()

# COMMAND ----------

# Save the checkpoint
checkpoint = {'epochs': epochs,
              'arch': "vgg16",
              'classifier': model.classifier,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'loss': loss,
              'idx_to_class':df_idx_to_class
             }  

SavedFilePath = '/dbfs/mnt/personal/sdu/vgg16_b_bag_v5.pth'
torch.save(checkpoint, SavedFilePath)

# COMMAND ----------

#load saved model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("{} is avaliable".format(device))
# SavedFilePath = '/dbfs/mnt/personal/sdu/vgg16_b_bag_2.pth'
model = load_checkpoint(SavedFilePath)
model = model.to(device)

# COMMAND ----------

# validation on the test set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_accuracy = 0

model.eval()

with torch.no_grad():
            for inputs, labels in tqdm(dataloaders['test']):
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

model.train()
            
print("Test Accuracy: {:.3f}".format(test_accuracy/len(dataloaders['test'])))

# COMMAND ----------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    model.eval()
    if type(image_path) == str:
      image = Image.open(requests.get(image_path, stream=True).raw)
    if type(image_path) == bytes:
      image = Image.open(io.BytesIO(image_path))
      
    image = test_transforms(image)
    image.unsqueeze_(0)
    image = image.to(device)
    
    
    with torch.no_grad():
        output = model.forward(image)
        
    ps = torch.exp(output)
    probs, indexes = ps.topk(topk)
    probs, indexes = probs.cpu().numpy(), indexes.cpu().numpy()
    
    classes = [pdf_idx_to_class.loc[i, 'class'] for i in indexes]
    return probs[0], classes[0]

# COMMAND ----------

test1_img = pdf_bag['image'][0]
test1_label = pdf_bag['class'][0]
display_img(test1_img)

# COMMAND ----------

print(test1_label)
predict(test1_img, model, topk=5)

# COMMAND ----------

# unseen data
c1 = "https://product-images.edited.com/6a876058eaad22cadd56a8e1/6ae2a940a4/full.jpg"
c2 = "https://i1.adis.ws/i/tom_ford/H0460T-TNY005_U9000_APPENDGRID?$listing_grid$"
c3 = "https://www.charleskeith.co.uk/dw/image/v2/BCWJ_PRD/on/demandware.static/-/Sites-ck-products/default/dw73d6821c/images/hi-res/2022-L2-CK2-10840452-1-01-1.jpg?sw=580&sh=774"
c4 = "https://img1.cohimg.net/is/image/Coach/c3890_b4nq4_a3?fmt=jpg&wid=680&hei=885&bgc=f0f0f0&fit=vfit&qlt=75"
c5 = "https://cdn.shopify.com/s/files/1/0332/0620/6508/products/FW21-Purity-AdelMicro-Meadow-1_2890x.jpg?v=1649784695"
c6 = "https://img.ltwebstatic.com/images3_pi/2021/07/09/1625816423faafe6120a70074a07aa6a0c2b7b44a3_thumbnail_900x.jpg"
c7 = "https://cdn.shopify.com/s/files/1/0003/2535/3534/products/Stone_Biscuit_5000x.jpg?v=1621430176"
c8 = "https://images.selfridges.com/is/image/selfridges/R03790826_TAN_M?$PDP_M_ZOOM$"
c9 = "https://cdn.shopify.com/s/files/1/0561/7297/0150/products/J45428_BR028_1_2800x.jpg?v=1644941147"


# COMMAND ----------

cs = [c1,c2,c3,c4,c5,c6,c7,c8,c9]

for c in cs:
  display_img(c)
#   print(predict(c, model, topk=5))
  probs, classes = predict(c, model, topk=5)
  plot_prediction(probs, classes)

# COMMAND ----------

idxs = get_rand_idx(5,4000)
ls = list(pdf_bag.loc[idxs,'class'])
cs = list(pdf_bag.loc[idxs,'image'])

for i,c in enumerate(cs):
  display_img(c)
  probs, classes = predict(c, model, topk=5)
  plot_prediction(probs, classes)
  

# COMMAND ----------


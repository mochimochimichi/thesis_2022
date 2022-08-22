# Databricks notebook source
# image encoder v2 remove last classification layer

class image_encoder():
  def __init__(self, df, saved_model, remove_last_n_layer):
    super().__init__()
    torch.manual_seed(42)
    model = saved_model  
    self.remove_last_n_layer = remove_last_n_layer
    if self.remove_last_n_layer == 999:
      model.avgpool = nn.AdaptiveMaxPool2d(output_size = (1,1))
      model = nn.Sequential(*list(model.children())[:-1])
    else:
      model.classifier = model.classifier[:-self.remove_last_n_layer]

    model.eval()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = model
    self.model = self.model.to(self.device)    
    self.image_datasets = load_transform(df, img_size = 224)
    self.dataloaders = create_dataloader(self.image_datasets, 32)
    
  def forward(self, image):
    output = self.model(image)
    return output
  
  def encode(self):
    image_feature = []
    with torch.no_grad():
      for inputs, labels in tqdm(self.dataloaders['all']):
        if self.device.type == "cuda":
          inputs, labels = inputs.to(self.device), labels.to(self.device)
          
        feature = self.model(inputs)
        
        if self.remove_last_n_layer == 999:
          assert feature.shape[1] == 512, "shape does not match"
        else:  
          assert feature.shape[1] == self.model.classifier[3].out_features, "shape does not match"

        feature = feature.reshape(feature.shape[0], feature.shape[1])
        feature = feature.detach().cpu().numpy()

        image_feature.append(feature)

#     model = model.to("cpu")
    image_embeddings = np.vstack(image_feature)
  
    return image_embeddings

# COMMAND ----------

# feature extraction network v2
class CreateImageEmbedding(nn.Module):
  def __init__(self, trained_model,remove_last_n_layer):
    super().__init__()
    torch.manual_seed(42)
    self.remove_last_n_layer = remove_last_n_layer
    model = trained_model
  
    if self.remove_last_n_layer == 999:
      model.avgpool = nn.AdaptiveMaxPool2d(output_size = (1,1))
      model = nn.Sequential(*list(model.children())[:-1])
    else:
      model.classifier = model.classifier[:-self.remove_last_n_layer]
    model.eval()
    self.model = model
    
  def forward(self, image):
    output = self.model(image)
    if self.remove_last_n_layer == 999:
      assert output.shape[1] == 512, "shape does not match"
    else:
      assert output.shape[1] == self.model.classifier[3].out_features, "shape does not match"
    
    return output.numpy()

# COMMAND ----------

def get_img_feature_vector(image_path, model):
  torch.manual_seed(42)
  model.eval()
  if type(image_path) == str:
    image = Image.open(requests.get(image_path, stream=True).raw)
  if type(image_path) == bytes:
    image = Image.open(io.BytesIO(image_path))
    
  image = train_transforms(image) #from train to standard_transforms
  image.unsqueeze_(0)
  
  with torch.no_grad():
    feature_vector = model.forward(image)
    
  feature_vector = feature_vector.reshape(feature_vector.shape[0], feature_vector.shape[1])
  
  return feature_vector

# COMMAND ----------

# SavedFilePath = '/dbfs/mnt/personal/sdu/vgg16_b_bag_v2.pth'
# trained_model = load_checkpoint(SavedFilePath)
# img_encoder = image_encoder(pdf,trained_model)
# image_embeddings = img_encoder.encode()
# np.save('/dbfs/mnt/personal/sdu/image_embeddings_v4.npy', image_embeddings)
# image_embeddings = np.load('/dbfs/mnt/personal/sdu/image_embeddings_v4.npy')

# COMMAND ----------


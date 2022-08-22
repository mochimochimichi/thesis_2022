# Databricks notebook source
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import json
import random
from PIL import Image
import requests
import cv2
cv2.__version__
from tqdm import tqdm
import urllib
import io
import string
import re
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import gc
import faiss
from matplotlib.pyplot import imshow
import rasterfairy

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
from sentence_transformers import SentenceTransformer
from nltk import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
# from numpy import array, log

from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn import preprocessing
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
# from efficientnet.pytorch import EfficientNet
from transformers import AutoTokenizer
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torchvision



# COMMAND ----------

def set_seed(seed):
  """
  Description: set seed for multiple packages for reproducibility
  
  Input: integer
  
  Output: 
  """
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  return


# COMMAND ----------

# check dataframe size
def get_shape(df):
  print("row:\n {} \n column:\n {}".format(df.count(), len(df.columns)))
  return

# COMMAND ----------

def get_rand_idx(size = 5, max_idx = 100):
  
  """
  Description: get a list of random indexes for sanity checking
  
  Input: size, (integer), length of index list
         max_idx, (integer), max range for each random index
  
  Output: a list of random indexes
  """
  
  rand_idx = []
  for i in range(0,size):
    n = random.randint(0,max_idx)
    rand_idx.append(n)
  return rand_idx

# COMMAND ----------

def train_valid_test_split(df,train_size, valid_size, test_size, stratify_required = True):
  
  
  if stratify_required == True:
    stratify_info_df = df['label']
    stratify_info_df_v = df_valid['label']
  else:
    stratify_info_df, stratify_info_df_v = None, None
    
  df_train ,df_valid = train_test_split(df,test_size = (1-train_size) , shuffle = True, stratify = stratify_info_df, random_state = 42)
  df_valid ,df_test = train_test_split(df_valid,test_size = test_size/(1-train_size), shuffle = True, stratify = None, random_state = 42)
  
  df_train = df_train.reset_index(drop = True)
  df_valid = df_valid.reset_index(drop = True)
  df_test = df_test.reset_index(drop = True)
  
  return df_train, df_valid, df_test

# COMMAND ----------

def display_img(img_path):
  
  """
  Description: display a single image or list of images using subplots, 5 images per row
  
  Input: img_path, (list or str or bytes or ndarray)
  
  Output: 
  """
  
  if type(img_path) == list:
    no_row = np.round(len(img_path)/5+0.5)
    fig = plt.figure(figsize=(15, 7.5))

    for i,img in enumerate(img_path):
      
      if type(img) == str:
        image = Image.open(requests.get(img, stream=True).raw)
      if type(img) == bytes:
        image = Image.open(io.BytesIO(img))
      if type(img) == np.ndarray:
        image = Image.open(requests.get(img[0], stream=True).raw)
        
      a = fig.add_subplot(no_row,5,i+1)
      plt.axis('off')
      a.imshow(image)
    return
  
  if type(img_path) == str:
    image = Image.open(requests.get(img_path, stream=True).raw)
  if type(img_path) == bytes:
    image = Image.open(io.BytesIO(img_path))
      
  fig, ax = plt.subplots();    
  plt.axis('off')
  ax.imshow(image)
  return

# COMMAND ----------

# define image datasets

class ImageDataset(Dataset):
  """
  Description: customized dataset loading
  Input: file_path, (pandas dataframe : e.g. pdf_edited)
  Output: 
  """
  def __init__(self, file_path, transform=None, target_transform=None):
      file_path.rename(columns = {'first_image':'image'}, inplace = True)
      if "label" not in file_path.columns:
        file_path['label'] = file_path.index
      self.image_labels = file_path#pd.read_csv(file_path)
      self.image_paths = file_path#pd.read_csv(file_path)
      self.transform = transform
      self.target_transform = target_transform

  def __len__(self):
      return len(self.image_labels)

  def __getitem__(self, idx):
      image_path = self.image_paths.loc[idx, 'image']

        #if image is in url
      if isinstance(image_path, str):
        image = Image.open(requests.get(image_path, stream=True).raw)
        #if image is in bytes
      elif isinstance(image_path, bytes):
        image = Image.open(io.BytesIO(image_path))
      else:#if image is in array/series/list
        image = Image.open(requests.get(image_path[0], stream=True).raw)

      label = self.image_labels.loc[idx,'label']

      if self.transform:
          image = self.transform(image)
      if self.target_transform:
          label = np.array(label)
          label = self.target_transform(label)
      return image, label

# COMMAND ----------

# vgg16 transforms

standard_transforms = transforms.Compose([
                                       transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
train_transforms = transforms.Compose([
                                       # add saturation augmentation
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(0.5),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# COMMAND ----------

#v2
def load_transform(df_all, df_train = None, df_valid = None, df_test = None, img_size = 224):
  
  standard_transforms = transforms.Compose([
                                       transforms.Resize(255),
                                       transforms.CenterCrop(img_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
  train_transforms = transforms.Compose([
                                       transforms.RandomResizedCrop(img_size),
                                       transforms.RandomHorizontalFlip(0.5),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
  valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(img_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
  test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(img_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
  
  if df_train is None: 
    all_data = ImageDataset(df_all, transform=standard_transforms)
    image_datasets = {"all": all_data}
    return image_datasets
  
  else:
    
    all_data = ImageDataset(df_all, transform=standard_transforms)
    train_data = ImageDataset(df_train, transform=train_transforms)
    valid_data = ImageDataset(df_valid, transform=valid_transforms)
    test_data = ImageDataset(df_test, transform=test_transforms)

    image_datasets = {"all": all_data,
                      "train": train_data,
                      "valid": valid_data,
                      "test": test_data
                      }
  
    return image_datasets


#v2
def create_dataloader(image_datasets, batch_size):
  
  if len(image_datasets)>1:
    all_loader = torch.utils.data.DataLoader(image_datasets["all"], batch_size=batch_size)
    train_loader = torch.utils.data.DataLoader(image_datasets["train"], batch_size=batch_size, shuffle=True)
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

def load_checkpoint(filepath):
  
  
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
    if device == "cuda":
      checkpoint = torch.load(filepath)
    else:
      checkpoint = torch.load(filepath, map_location=torch.device("cpu"))
      
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
  
    return model

# COMMAND ----------

def plot_prediction(probs, classes):
  plt.rcdefaults()
  fig = plt.figure(figsize=(8, 6), dpi=80)
  fig, ax = plt.subplots()
  y_pos = np.arange(5)

  ax.barh(y_pos, probs, align='center')
  ax.set_yticks(y_pos)
  ax.set_yticklabels(list(classes))
  ax.invert_yaxis() 
  plt.show()
  return

# COMMAND ----------

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    
    if type(image_path) == str:
      image = Image.open(requests.get(image_path, stream=True).raw)
    if type(image_path) == bytes:
      image = Image.open(io.BytesIO(image_path))
      
    image = test_transforms(image)
    image.unsqueeze_(0)
    image = image.to(device)
    model = model.to(device)
    
    with torch.no_grad():
        output = model.forward(image)
        
    ps = torch.exp(output)
    probs, indexes = ps.topk(topk)
    probs, indexes = probs.cpu().numpy(), indexes.cpu().numpy()
    
    classes = [pdf_idx_to_class.loc[i, 'class'] for i in indexes]
    return probs[0], classes[0]

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

def faiss_retrieve_text(q_text_vector, text_embed, topk, metric = "ed"):
  
  #use eucliean distance
  if metric == "ed":
    dim = text_embed.shape[1]
    index = faiss.IndexFlatL2(dim)
    assert index.is_trained == True
    index.add(text_embed)
    D, I = index.search(np.array([q_text_vector]), k=topk)
    index_to_retrieve = I[0]

  
  #use cosine similarities 
  if metric == "cs":
    dim = text_embed.shape[1]
    index = faiss.IndexFlatIP(dim)
    assert index.is_trained == True
    faiss.normalize_L2(text_embed)
    index.add(text_embed)
    q_text_vector = np.array([q_text_vector])
    faiss.normalize_L2(q_text_vector)
    D, I = index.search(q_text_vector, k=topk)
    index_to_retrieve = I[0]
    
  return index_to_retrieve

# COMMAND ----------

def faiss_retrieve_img(q_img_vector, img_embed, topk, metric = "ed"):

  #use eucliean distance
  if metric == "ed":
    dim = img_embed.shape[1]
    index = faiss.IndexFlatL2(dim)
    assert index.is_trained == True
    index.add(img_embed)
    D, I = index.search(q_img_vector, topk)  # search
    index_to_retrieve = I[0]

  #use cosine similarities 
  if metric == "cs":
    dim = img_embed.shape[1]
    index = faiss.IndexFlatIP(dim)
    assert index.is_trained == True
    faiss.normalize_L2(img_embed)
    index.add(img_embed)
    faiss.normalize_L2(q_img_vector)
    D, I = index.search(q_img_vector, topk)  # search
    index_to_retrieve = I[0]
  
  return index_to_retrieve   

# COMMAND ----------

def color_moments(filename):
  
    req = urllib.request.urlopen(filename)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1) # 'Load it as it is'
      
    if img is None:
        return 1
    # Convert BGR to HSV colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Split the channels - h,s,v
    h, s, v = cv2.split(hsv)
    # Initialize the color feature
    color_feature = []
    # N = h.shape[0] * h.shape[1]
    # The first central moment - average
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    color_feature.extend([h_mean, s_mean, v_mean])
    # The second central moment - standard deviation
    h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
    color_feature.extend([h_std, s_std, v_std])
    # The third central moment - the third root of the skewness
    h_skewness = np.mean(np.abs(h - h.mean())**3)
    s_skewness = np.mean(np.abs(s - s.mean())**3)
    v_skewness = np.mean(np.abs(v - v.mean())**3)
    h_thirdMoment = h_skewness**(1./3)
    s_thirdMoment = s_skewness**(1./3)
    v_thirdMoment = v_skewness**(1./3)
    color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])
    return color_feature

# COMMAND ----------

def idx_to_path(idx, df_base):
  similar_results = [df_base.loc[x, 'first_image'] for x in idx]
  return similar_results

# COMMAND ----------

class embedding_composer():
  def __init__(self,img_encoder, text_encoder):
    super().__init__()
    torch.manual_seed(42)
    self.img_encoder = img_encoder
    self.text_encoder = text_encoder
    self.color_encoder = color_moments
    

  def concat_vectors(self,img_path,text, text_weight=1, img_weight=1, color_weight=0.3):
    self.img_vector = get_img_feature_vector(img_path, self.img_encoder)
    self.text_vector = self.text_encoder.encode_text(text)
    self.color_vector = self.color_encoder(img_path)
    
    all_embed = np.concatenate((np.atleast_2d(self.text_vector),self.img_vector,np.atleast_2d(self.color_vector)*color_weight), axis = 1).astype("float32")
    
    return all_embed
  
  
class embedding_database():
    def __init__(self,img_embeddings, text_embeddings, color_embeddings):
      super().__init__()
      self.img_embeddings = img_embeddings
      self.text_embeddings = text_embeddings
      self.color_embeddings = color_embeddings
                                            
    def concat_embeddings(self,text_weight=2000, img_weight=0.8, color_weight=0.3):
      all_embeddings = np.concatenate((self.text_embeddings*text_weight,self.img_embeddings*img_weight, self.color_embeddings*color_weight), axis = 1).astype("float32")
      
      return all_embeddings

# COMMAND ----------

def make_grid(pdf_edited, image_embeddings, lb, ub, n_components=300, perplexity=30, mode = "grid"):

  features = np.array(image_embeddings[lb:ub])
  images = list(pdf_edited["first_image"][lb:ub].values)

  pca = PCA(n_components,random_state = 42)
  pca.fit(features)

  pca_features = pca.transform(features)

  X = np.array(pca_features)
  tsne = TSNE(n_components = 2, learning_rate=150, perplexity = perplexity, angle=0.2, verbose=2, random_state = 42).fit_transform(X)

  tx, ty = tsne[:,0], tsne[:,1]
  tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
  ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

  max = __builtins__.max

  # image cloud
  if mode == "cloud":
    width = 4000
    height = 3000
    max_dim = 150

    full_image = Image.new('RGBA', (width, height))
    for img, x, y in tqdm(zip(images, tx, ty)):
        tile = Image.open(requests.get(img, stream=True).raw)
        tile.resize((224,224))
        rs = max(1, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

    plt.figure(figsize = (16,12))
    imshow(full_image)
    return

  # image grid
  if mode == "grid":
    # nx * ny = 1000, the number of images
    ny = 25
    nx = np.int(len(images)/ny)

    # assign to grid
    grid_assignment = rasterfairy.transformPointCloud2D(tsne, target=(nx, ny))

    tile_width = 72
    tile_height = 56

    full_width = tile_width * nx
    full_height = tile_height * ny
    aspect_ratio = float(tile_width) / tile_height

    grid_image = Image.new('RGB', (full_width, full_height))

    for img, grid_pos in tqdm(zip(images, grid_assignment[0])):
        idx_x, idx_y = grid_pos
        x, y = tile_width * idx_x, tile_height * idx_y
        tile = Image.open(requests.get(img, stream=True).raw)
        tile_ar = float(tile.width) / tile.height  # center-crop the tile to match aspect_ratio
        if (tile_ar > aspect_ratio):
            margin = 0.5 * (tile.width - aspect_ratio * tile.height)
            tile = tile.crop((margin, 0, margin + aspect_ratio * tile.height, tile.height))
        else:
            margin = 0.5 * (tile.height - float(tile.width) / aspect_ratio)
            tile = tile.crop((0, margin, tile.width, margin + float(tile.width) / aspect_ratio))
        tile = tile.resize((tile_width, tile_height), Image.ANTIALIAS)
        grid_image.paste(tile, (int(x), int(y)))

    plt.figure(figsize = (26,22))
    imshow(grid_image)
    return




# COMMAND ----------

# source: https://github.com/issamemari/pytorch-multilabel-balanced-sampler

class MultilabelBalancedRandomSampler(Sampler):
    """
    MultilabelBalancedRandomSampler: Given a multilabel dataset of length n_samples and
    number of classes n_classes, samples from the data with equal probability per class
    effectively oversampling minority classes and undersampling majority classes at the
    same time. Note that using this sampler does not guarantee that the distribution of
    classes in the output samples will be uniform, since the dataset is multilabel and
    sampling is based on a single class. This does however guarantee that all classes
    will have at least batch_size / n_classes samples as batch_size approaches infinity
    """

    def __init__(self, labels, indices=None, class_choice="least_sampled"):
        """
        Parameters:
        -----------
            labels: a multi-hot encoding numpy array of shape (n_samples, n_classes)
            indices: an arbitrary-length 1-dimensional numpy array representing a list
            of indices to sample only from
            class_choice: a string indicating how class will be selected for every
            sample:
                "least_sampled": class with the least number of sampled labels so far
                "random": class is chosen uniformly at random
                "cycle": the sampler cycles through the classes sequentially
        """
        self.labels = labels
        self.indices = indices
        if self.indices is None:
            self.indices = range(len(labels))

        self.num_classes = self.labels.shape[1]

        # List of lists of example indices per class
        self.class_indices = []
        for class_ in range(self.num_classes):
            lst = np.where(self.labels[:, class_] == 1)[0]
            lst = lst[np.isin(lst, self.indices)]
            self.class_indices.append(lst)

        self.counts = [0] * self.num_classes

        assert class_choice in ["least_sampled", "random", "cycle"]
        self.class_choice = class_choice
        self.current_class = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count >= len(self.indices):
            raise StopIteration
        self.count += 1
        return self.sample()

    def sample(self):
        class_ = self.get_class()
        class_indices = self.class_indices[class_]
        chosen_index = np.random.choice(class_indices)
        if self.class_choice == "least_sampled":
            for class_, indicator in enumerate(self.labels[chosen_index]):
                if indicator == 1:
                    self.counts[class_] += 1
        return chosen_index

    def get_class(self):
        if self.class_choice == "random":
            class_ = random.randint(0, self.labels.shape[1] - 1)
        elif self.class_choice == "cycle":
            class_ = self.current_class
            self.current_class = (self.current_class + 1) % self.labels.shape[1]
        elif self.class_choice == "least_sampled":
            min_count = self.counts[0]
            min_classes = [0]
            for class_ in range(1, self.num_classes):
                if self.counts[class_] < min_count:
                    min_count = self.counts[class_]
                    min_classes = [class_]
                if self.counts[class_] == min_count:
                    min_classes.append(class_)
            class_ = np.random.choice(min_classes)
        return class_

    def __len__(self):
        return len(self.indices)
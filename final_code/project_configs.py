# Databricks notebook source
df_retailer = pd.read_csv('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/cleaned_retailer_list.csv')
df_fashion_stopword = pd.read_csv('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/fashion_stopword_602.csv')
df_brand_name = pd.read_csv('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/brand_name.csv')
df_top_word = pd.read_csv('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/top_word.csv')
df_word_frequency = pd.read_csv('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/word_frequency.csv')

# COMMAND ----------

attribute_list = ['option_id',
                  'product_hash',
                  'brand_slug',  #tbd
                  'category',
                  'cs_grp',
                  'cs_style',
                  'cs_subcategory',
                  'colour_name',
                  'colours',
                  'product_details',
                  'gender',
                  'retailer', #tbd
                  'season', #tbd
                  'care',
                  'description',
                  'merch_info',
                  'name',
                  'predominant_colour',
                  'predominant_pattern',
                  'product_searches',
                  'image_urls'
                 ]

text_attribute_list = ['brand_slug',  #tbd
                'name',
                'gender',
                'colour',
                'colour_name',
                'predominant_colour',
                'predominant_pattern',
                'product_details',
                'category',
                'product_searches',
                'care',
                'description',
                'merch_info'
               ]

# COMMAND ----------


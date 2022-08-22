# Databricks notebook source
# MAGIC %run "/Users/Sifang.Du@burberry.com/project_code/project_utils"

# COMMAND ----------

# MAGIC %md ## Train data

# COMMAND ----------

# MAGIC %md ### all_selected_party

# COMMAND ----------

pdf_select_party = pd.read_csv("/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/selected_party_v12.csv", header = None)
select_party = pdf_select_party[0].tolist()

pdf_select_brand = pd.read_csv("/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/df_hash_5year_brand_slug.csv", header = None)
select_brand = pdf_select_brand[0].tolist()

# COMMAND ----------

select_party = ['alexandermcqueen-us',
 'balenciaga-us',
 'balmain-us',
 'bottegaveneta-us',
 'celine-us',
 'christianlouboutin-us',
 'dior-us',
 'fendi-us',
 'givenchy-us',
 'gucci-us',
 'hermes-us',
 'jimmychoo-us',
 'loewe-us',
 'moncler-us',
 'prada-us',
 'redvalentino-us',
 'saintlaurent-us',
 'salvatoreferragamo-us',
 'tomford-us',
 'valentino-us',
 'versace-us',
 'louisvuitton-us',
 'farfetch',
 'mytheresa',
 'netaporter',
 'chanel-us']

attribute_list = ['option_id',
                  'product_hash',
                  'brand_slug',  #tbd
                  'retailer_brand_slug',
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

# COMMAND ----------

# select attributes needed for project
df_edited_bag = (spark.table('hive_metastore.edited.options')).select(attribute_list)\
                              .filter(col("cs_grp") == "accessories")\
                              .filter(col("gender") == "women")\
                              .filter((col("cs_subcategory").like("%bag%")))\
                              .filter("brand_slug not like '%burberry%'")\
                              .filter(col("brand_slug") != "burberry")\
                              .filter(size(col("image_urls")) > 0)\
                              .filter(length(col("name")) > 0)\
                              .withColumn("first_image", expr("image_urls[0]"))\
                              .withColumn("new_cs_style", expr("substring(cs_style,7, length(cs_style))"))\
                              .withColumn("new_cs_subcategory", expr("substring(cs_subcategory,9, length(cs_subcategory))"))\
                              .withColumn("colour", expr("colours.basic_name[0]"))\
                              .withColumn("year", expr("substring(season,3, length(season))"))\
                              .withColumn("len_of_description",length(col("description")[0]))\
                              .filter(col("year") > 17)\
                              .drop("cs_style")\
                              .drop("cs_subcategory")\
                              .drop("colours")\
                              .where((col("brand_slug").isin(select_brand)))

w1 = Window.partitionBy("product_hash").orderBy(col("len_of_description").desc())
df_edited_bag_hash = df_edited_bag.withColumn("row",row_number().over(w1)).filter(col("row") == 1).drop("row")

w2 = Window.partitionBy("name","colour").orderBy(col("len_of_description").desc())
df_hash_5_year = df_edited_bag_hash.withColumn("row",row_number().over(w2)).filter(col("row") == 1).drop("row")

# COMMAND ----------

get_shape(df_hash_5_year)

# COMMAND ----------

# df_hash_5_year.write.mode('overwrite').saveAsTable('sdu.edited_unique_hash_5year_selected_party_2')

# COMMAND ----------

# df_hash_5_year = spark.table('sdu.edited_unique_hash_5year_selected_party_2')
# get_shape(df_hash_5_year)

# COMMAND ----------

df_hash_5_year.groupby('retailer').count().orderBy(col("count")).display()

# COMMAND ----------

df_brand_count = df_hash_5_year.groupby('brand_slug').count().orderBy(col("count"))
pdf_brand_count = df_brand_count.toPandas()

# COMMAND ----------

pdf_brand_count = pdf_brand_count.sort_values(by = 'count', ascending = False)
pdf_brand_count.set_index(["brand_slug"], inplace = True)
pdf_brand_count.columns = ["cnt"]

g = sns.catplot(data=pdf_brand_count, kind="bar", y=pdf_brand_count.cnt,x=pdf_brand_count.index, orient="v", height = 5, aspect = 5)
ax = g.facet_axis(0, 0)
for c in ax.containers:
    labels = [f'{(v.get_height()):.0f}' for v in c]
    ax.bar_label(c, labels=labels, label_type='edge')   
plt.xticks(rotation=45)
plt.title('Number of product records under each brand - Before limiting retailers and selling markets ', fontsize = 20, fontweight='bold')
plt.ylabel('Number of product records');
plt.xlabel('Brand');

# COMMAND ----------

# MAGIC %md ### first_party_only

# COMMAND ----------

select_first_party = ['alexandermcqueen-us',
 'balenciaga-us',
 'balmain-us',
 'bottegaveneta-us',
 'celine-us',
 'christianlouboutin-us',
 'dior-us',
 'fendi-us',
 'givenchy-us',
 'gucci-us',
 'hermes-us',
 'jimmychoo-us',
 'loewe-us',
 'moncler-us',
 'prada-us',
 'redvalentino-us',
 'saintlaurent-us',
 'salvatoreferragamo-us',
 'tomford-us',
 'valentino-us',
 'versace-us',
 'louisvuitton-us',
 'chanel-us']

# COMMAND ----------

df_hash_5_year_first_party = df_hash_5_year.where((col("retailer").isin (select_first_party)))

# COMMAND ----------

get_shape(df_hash_5_year_first_party)

# COMMAND ----------

df_hash_5_year_first_party.groupby('brand_slug').count().orderBy(col("count")).display()

# COMMAND ----------

# add id in case of sampling
df_hash_5_year_first_party = df_hash_5_year_first_party.withColumn("id",row_number().over(Window.orderBy(monotonically_increasing_id()))-1)

# COMMAND ----------

# MAGIC %md ### first_party_data to csv

# COMMAND ----------

# split dataframe to batches 
length = df_hash_5_year_first_party.count()
batch_size =4000
n_of_batch = int((length/batch_size)+0.5)
data_dict = {}
for i in range(n_of_batch):
  data_dict[i] = df_hash_5_year_first_party.orderBy(col("id").asc()).filter((col("id")>= batch_size*i) & (col("id")<batch_size*(i+1)))

# COMMAND ----------

#convert each batch to pandas dataframe then concatenate to one fil
pdf_lst = []
for i in range(n_of_batch):
  pdf_batch = data_dict[i].toPandas()
  pdf_lst.append(pdf_batch)
  
pdf_hash_5_year_first_party = pd.concat(pdf_lst)

# COMMAND ----------

# save full dataframe csv file
pdf_hash_5_year_first_party.to_csv('/dbfs/FileStore/shared_uploads/Sifang.Du@burberry.com/pdf_hash_5_year_first_party.csv', header = True, index = False)

# COMMAND ----------

df_hash_5_year_first_party.write.mode('overwrite').saveAsTable('sdu.edited_unique_hash_5year_first_party')
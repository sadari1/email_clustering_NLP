# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import pandas as pd 
import numpy as np
import time


# %%

# Read all of the emails we extracted
emails = pd.read_csv("../output/enron_from_to_largeset.csv")


# %%

# Filter this sender out, too many spam emails.
emails = emails[emails.From != '40enron@enron.com' ]


# %%

from_emails = np.unique(emails.From)

# %%

# Starts a job to search for all from-to pairs.
from_to_pairs = []

tic = time.time()
for f in from_emails:

    view = emails[emails.From == f]
    if np.array(view).shape[0] != 0:
        for g in range(len(view)):
            from_to_pairs.append([f, view.iloc[g].To])

toc = time.time()

print(f"Job took {toc-tic} seconds")


# %%

documents = np.array(from_to_pairs)#np.concatenate((np.array(emails.From),np.array(emails.To)))

# %%

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, StructField, StructType, StringType, IntegerType
from functional import seq
import time
import re
# old pyspark version is 2.4.5


# %%
import pyspark
pyspark.__version__


# %%
os.environ["SPARK_LOCAL_IP"]='127.0.0.1'
spark = SparkSession.builder.master("local[*]").config("spark.driver.memory", '1g').config("spark.executor.memory", '1g').config("spark.network.timeout", "300").config("spark.executor.pyspark.memory", "1g").config("spark.executor.memoryOverhead", "1g").config("spark.memory.fraction", "0.8").config('spark.jars.packages', 'graphframes:graphframes:0.8.0-spark3.0-s_2.12').getOrCreate()
        

spark.sparkContext._conf.getAll()


# %%

from graphframes import *
def getClusterId(spark, df,ouput_loc):
    v = df.select(df.src.alias("id")).union(df.select(df.dst).alias("id")).distinct()
   
    e = df
    g = GraphFrame(v, e)
    spark.sparkContext.setCheckpointDir("../../checkpoints")
    result = g.connectedComponents()
    res_ord = result.orderBy("component")
    res_ord.repartition(10).write.csv(ouput_loc)


# %%
# From and two pairs array created to be loaded into a spark df.
from_and_to = [(str(documents[f,0]), str(documents[f, 1])) for f in range(len(documents))]


# %%
from_to_df = spark.createDataFrame(from_and_to, ["src", "dst"])


# %%

# Passe sthe df to the graph frames function to get the clusters.
tic = time.time()
getClusterId(spark, from_to_df, f"../output/from_to_clusters.csv")
toc = time.time()

print(f"Job took {toc-tic} seconds")


# %%

# Reads the cluster csv  just created. 
df = spark.read.csv(f"../output/from_to_clusters.csv").withColumnRenamed("_c0","To" ).withColumnRenamed("_c1","Cluster" )
email_clusters = df.toPandas()


# %%

# Matches all of the emails to these clusters. 
tic = time.time()
clusters = ['0' for f in range(emails.shape[0])]

clustered_emails = emails
clustered_emails["Cluster"] = clusters

for g in range(email_clusters.shape[0]):

    clustered_emails.loc[clustered_emails.To == email_clusters.iloc[g].To, "Cluster"] = email_clusters.iloc[g].Cluster

toc = time.time()

print(f"Job took {toc-tic} seconds")


# %%

# Saves the clustered set to a csv.
clustered_emails.to_csv("../output/enron_clustered.csv", index=False)



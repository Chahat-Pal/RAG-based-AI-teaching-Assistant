import requests
import os
import json
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def create_embedding(text_list):
    r = requests.post( "http://localhost:11434/api/embed", json = {
            "model" : "bge-m3",
            "input" : text_list
        })
    
    embedding = r.json()["embeddings"]
    return embedding
#a=create_embedding("I'm Chahat")
#print(a)
jsons = os.listdir("jsons")
my_list=[]
chunk_id=0
for jsons_file in jsons:
    with open(f"jsons/{jsons_file}",'r') as f:
        content=json.load(f)
    embeddings=create_embedding([c['text'] for c in content['chunks']])
    for i,chunk in enumerate(content["chunks"]):
        chunk['chunk_id']=chunk_id
        chunk['embedding']=embeddings[i]
        chunk_id+=1
        my_list.append(chunk)
        
#print(f"LIST : {my_list}")
#a=create_embedding("I'm Chahat Pal")
#print(a)
df=pd.DataFrame.from_records(my_list)
#print(df)
# save df with the help of joblib so that we can use it whenever we want :-
joblib.dump(df, "embeddings.joblib")

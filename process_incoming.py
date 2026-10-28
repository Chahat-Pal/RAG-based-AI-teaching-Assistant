import pandas as pd 
import numpy as np
import joblib
import json
import requests
import os
from sklearn.metrics.pairwise import cosine_similarity

def create_embedding(text_list):
    r = requests.post( "http://localhost:11434/api/embed", json = {
            "model" : "bge-m3",
            "input" : text_list
        })
    
    embedding = r.json()["embeddings"]
    return embedding

def inferrence(prompt):
    r = requests.post( "http://localhost:11434/api/generate", json = {
                "model" : "deepseek-r1",
                "prompt" : prompt,
                "stream" : False
            })
    response=r.json
    print(response)
    return response

df = joblib.load("embeddings.joblib")

incoming_query=input("ask a question : ")
question_embedding=create_embedding([incoming_query])[0]

#find similarities of question embeddings with other embeddings
#print(np.vstack(df['embedding'].values))
#print(np.vstack(df['embedding']).shape)
similarities=cosine_similarity(np.vstack(df['embedding'].values), [question_embedding]).flatten()
print(similarities)
top_results=3
max_indx=similarities.argsort()[::-1][0:top_results]
print(max_indx)
new_df=df.loc[max_indx]
print(new_df[["text"]])
prompt=f''' so basically this is my video in which I've given Introduction about myself , so these are the timestamps (including start time of the video chunks and the end time) and the "text" of the video 
{new_df[["start","end","text"]].to_json()}
and 
the user asked this question related to the video  chunks {incoming_query}, you have tell the anwser of this question which is given in the video so read the embeddings and identify the correct answer and then tell him the correct answera nd also time stamp odf the video where the answer is given, if user ask a question not related to the video chunks then tell him that please ask question only from the video'''
with open("prompt.txt", "w") as f:
    f.write(prompt)

response=inferrence(prompt)
print(response)

with open("response.txt", "w") as f:
    f.write(response)
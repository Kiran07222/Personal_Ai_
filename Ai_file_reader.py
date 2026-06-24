
from groq import Groq
import os 
from sentence_transformers import SentenceTransformer
import re
import numpy as np
import faiss
client = Groq(api_key=os.getenv("groq_api_key"))

prompt = input("Nani: ")
model= SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

with open("C:/Users/nani3/Desktop/sample.txt", "r") as f:
    file_content = f.read()
file_content=file_content.lower()
file_content=re.sub(r'[^a-z\s]','',file_content)

chunks=file_content.split("\n")

embdding=model.encode(chunks)

dimension=embdding.shape[1]
index=faiss.IndexFlatL2(dimension)
index.add(np.array(embdding,dtype="float32"))




while(1):
    q_embdding=model.encode([prompt])
    distance ,indeces= index.search(np.array(q_embdding,dtype="float32"),3)
    llm_text=""
    for i in indeces[0]:
        llm_text+=chunks[i]+"\n"


    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
            "role": "system",
            "content": "Your name is Nani and you are giving answers to your friends."
            },
            {
            "role": "system",
            "content":llm_text
            },
            {
            "role": "user",
            "content": prompt
            }
        ],
        max_completion_tokens=1000
    )
    print(response.choices[0].message.content)
    n=int(input(" 0 to stop: "))
    
    if n==0:
        break
    else:
        prompt=input("enter the message :")



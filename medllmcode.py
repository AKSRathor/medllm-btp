from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import faiss
from flask_cors import CORS 
from sentence_transformers import SentenceTransformer
import pickle
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

llm = ChatGroq(
    model_name="llama-3.1-70b-versatile",
    temperature=0,
    groq_api_key = "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"
)

app = Flask(__name__)
CORS(app)  
model = SentenceTransformer('all-MiniLM-L6-v2')

index = faiss.read_index('./Medllm_vector/faiss_index.index')
with open('./Medllm_vector/embeddings.pkl', 'rb') as f:
    embeddings_np = pickle.load(f)
df = pd.read_pickle('./Medllm_vector/dataframe.pkl')

def do_predict(query, instruction, I):
    prompt_extract = PromptTemplate.from_template(
        """
        ### Give the Following Question asked by user:
        {query}
        ### INSTRUCTION:
        {instruction}
        and the context of the document is: {document}: {context}
        and the another context of the document is: {document2}: {context2}
        ### (NO PREAMBLE):    
        #Please provide the valid response to the user's question in html inline css all in white text with no html tag, based on the context of the document and the instruction given and don't say about less context given.
        """
    )
    print("Prediction for query: ", query)
    chain_extract = prompt_extract | llm 
    res = chain_extract.invoke(input={'instruction':instruction, 'query':query, 'context':df.iloc[I[0][0]]['context'], 'document':df.iloc[I[0][0]]['question'], 'document2':df.iloc[I[0][1]]['question'], 'context2':df.iloc[I[0][1]]['context'],})
    print("Predicted is ", res.content)
    return res.content

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    
    if data and 'query' in data:
        query_param = data['query']
        query =query_param
        query_embedding = model.encode(query, convert_to_tensor=True)
        instruction = ""
        query_embedding_np = query_embedding.cpu().detach().numpy().reshape(1, -1)
        k = 3
        D, I = index.search(query_embedding_np, k) 
        myres = do_predict(query, instruction, I)
        print(myres)
        response = {
            # 'query': query_param,
            'response': myres,
            "success":True
        }
    else:
        response = {
            'error': 'No query parameter provided in the body',
            "success":False
        }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=4000)

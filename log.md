## LOG
Problem: Uploading duplicates
05/04 hash value to prevent duplicates documenting

Problem: LLM slow quering
paper: Feasibility_of_Wrist-Worn_Real-Time_Hand_and_Surface_Gesture_Recognition_via_sEMG_and_IMU_Sensing.pdf
query: According to this paper, how was gesture recognition testing performed?
runtime: 8 min
06/04 flash attention only works on linux, lowering max_new_token from 512 to 150
is the process of retrieving taking much time or generating response?

200 max_new_token, the question isnt fully answer yet...
INFO:app.api:Retrieval took 0.72 seconds
INFO:app.api:Generation took 1062.92 seconds

07/04
-Changed to Ollama instead of using transformers
Generation took only 40 sec
-Normalize embeddings before storing in ChromaDB
ChromaDB's default "hnsw:space": "cosine" assumes normalized vectors.
-Introduce RecursiveCharacterTextSplitter from langchain

08/04
-benchmark QA pair for RAG evaluation from hugging face
-deepeval metric use llm judge (目的是做超参数微调)
ollama3:8b variate score
deepseek-r1:14b all the relevancy score is 1.0

09/04
-通过ollama调用本地的llm时，模型越大，评估时间越长
-分两步测，
测llm超参数
测retrieval超参数 chunk_size, chunk_overlap

for faithfulness and answer relevancy, clear retriever for every example runned
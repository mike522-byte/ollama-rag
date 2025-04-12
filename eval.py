from datasets import load_dataset
from app.llm import LocalLLM
from app.retriever import Retriever
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRecallMetric
from deepeval import evaluate
from app.document_parser import DocumentParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd

dataset = load_dataset('FreedomIntelligence/RAG-Instruct',split='train')
first_50 = dataset.shuffle(seed=42).select(range(50))

llm = LocalLLM(model_name="mistral")
retriever = Retriever()
doc_parser = DocumentParser()

# define llm based metric
answer_relevancy = AnswerRelevancyMetric(threshold = 0.5)
faithfulness = FaithfulnessMetric(threshold = 0.5)
context_precision = ContextualPrecisionMetric(threshold = 0.5)
context_recall = ContextualRecallMetric(threshold = 0.5)


test_cases = []
for example in first_50:
    retriever.clear()
    query = example['question']
    expected = example['answer']
    documents = example['documents']

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ". ", ""]
        )
    chunks = text_splitter.create_documents(documents)

    DOCS = [{'content': chunk.page_content,'metadata': {'source': 'eval'} } for chunk in chunks]

    retriever.add_documents(DOCS)

    context_chunks = retriever.retrieve(query, rerank_k = 5)
    response = llm(query, context_chunks)

    test_case = LLMTestCase(
        input=query,
        actual_output=response['answer'],
        expected_output=expected,
        retrieval_context=[c['content'] for c in context_chunks]
    )
    
    test_cases.append(test_case)

evaluate(test_cases = test_cases, metrics=[answer_relevancy, faithfulness])




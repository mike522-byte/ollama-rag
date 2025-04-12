import requests
from typing import List, Dict
import logging

class LocalLLM:
    def __init__(self, model_name: str = 'mistral', top_p: float = 0.05, temperature: float = 0.5):
        self.model_name = model_name
        self.base_url = "http://localhost:11434/api/generate"
        self.logger = logging.getLogger(__name__)
        self.top_p = top_p
        self.temperature = temperature
    
    def _create_prompt(self, query: str, context: List[Dict]) -> str:
        """Create a prompt from the query and context."""
        context_str = "\n".join([chunk['content'] for chunk in context])
        
        prompt = f"""<s>[INST] <<SYS>>
You are a helpful AI assistant. Answer the question based on the provided context.
If you cannot answer based on the context, say "I cannot answer this question based on the provided context."
<</SYS>>

Context:
{context_str}

Question: {query} [/INST]"""
        
        return prompt

    def generate_response(self, query: str, relevant_chunks: List[Dict]) -> Dict:
        """Generate a response using the Ollama Rest API"""
        try:
            # Create the prompt
            prompt = self._create_prompt(query, relevant_chunks)
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "option": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": 512
                }
            }
            
            response = requests.post(self.base_url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            return {
                "answer": result.get("response", "").strip(),
                "sources": relevant_chunks
            }
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return {
                "answer": "Sorry, I encountered an error while generating the response.",
                "sources": []
            }

    def __call__(self, query: str, relevant_chunks: List[Dict]) -> Dict:
        """Make the class callable."""
        return self.generate_response(query, relevant_chunks)
    
if __name__ == "__main__":
    llm = LocalLLM(model_name="mistral")
    test_query = "What is the capital of France?"
    test_chunks = [{"content": "France is a country in Europe. Its capital is Paris."}]
    print(llm(test_query, test_chunks))
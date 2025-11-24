import os
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Any, Set
import pytest

# --- Part 1: Index Creation (Slightly Modified to save chunks) ---

def create_and_save_faiss_index(
    chunks: List[str],
    metadata: List[Dict[str, Any]],
    model_name: str = 'all-MiniLM-L6-v2',
    base_path: str = 'test_data' # Use a dedicated test directory
) -> None:
    """
    Encodes text chunks, builds a FAISS index, and saves all necessary artifacts.
    """
    # Define paths
    models_path = os.path.join(base_path, 'models')
    data_path = os.path.join(base_path, 'data')
    embeddings_path = os.path.join(models_path, 'embeddings.pkl')
    faiss_index_path = os.path.join(models_path, 'faiss.index')
    metadata_path = os.path.join(data_path, 'metadata.pkl')
    chunks_path = os.path.join(data_path, 'chunks.pkl') # Added chunks path

    # Create directories
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)

    print(f"Encoding {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=False)
    embeddings_np = np.array(embeddings).astype('float32')

    d = embeddings_np.shape[1]
    faiss.normalize_L2(embeddings_np)

    print("Building FAISS Index...")
    index = faiss.IndexFlatL2(d)
    index.add(x=embeddings_np)

    print("Saving files...")
    with open(embeddings_path, 'wb') as f:
        pickle.dump(embeddings_np, f)
    faiss.write_index(index, faiss_index_path)
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    # Save the raw chunks as well
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
    print("All files saved successfully.")


# --- Part 2: Retrieval Function and Class ---

class FaissRetriever:
    """
    A class to handle loading FAISS artifacts and performing retrieval.
    """
    def __init__(self, base_path: str = 'test_data'):
        # Define paths
        models_path = os.path.join(base_path, 'models')
        data_path = os.path.join(base_path, 'data')
        faiss_index_path = os.path.join(models_path, 'faiss.index')
        metadata_path = os.path.join(data_path, 'metadata.pkl')
        chunks_path = os.path.join(data_path, 'chunks.pkl')

        # Load artifacts
        print("Loading retrieval artifacts...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.read_index(faiss_index_path)
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        print("Artifacts loaded.")

    def retrieve(self, query: str, k: int = 3, score_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Embeds a query, searches the FAISS index, and returns the top k results
        that meet the score threshold, along with their metadata.

        Args:
            query (str): The user's query string.
            k (int): The number of top results to retrieve.
            score_threshold (float): The minimum cosine similarity score for a result to be included.

        Returns:
            List[Dict[str, Any]]: A list of result dictionaries, each containing the
                                  chunk text, metadata, and similarity score.
        """
        # 1. Embed the query
        query_embedding = self.model.encode([query], convert_to_tensor=False).astype('float32')
        faiss.normalize_L2(query_embedding) # Normalize the query embedding for cosine similarity search

        # 2. Search FAISS
        # The `search` method returns L2 distances and the indices of the vectors
        distances, indices = self.index.search(query_embedding, k)

        results = []
        # 3. Process results and ensure metadata retrieval
        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = distances[0][i]

            # For normalized vectors, Cosine Similarity = 1 - (L2_distance^2) / 2
            score = 1 - (dist**2) / 2

            # 4. Add ranking score threshold
            if score >= score_threshold:
                results.append({
                    "chunk": self.chunks[idx],
                    "score": score,
                    "metadata": self.metadata.iloc[idx].to_dict(), # Change this line
                })

        # 5. Return k best chunks (that passed the threshold)
        return results

# --- Part 4: Prompt Engineering ---

# 1. Create system prompt template
SYSTEM_PROMPT_TEMPLATE = """You are a helpful Q&A assistant. Your task is to answer the user's question based *only* on the context provided below.

If the context does not contain the answer, state that you cannot answer the question with the given information. Do not use any external knowledge.

---
CONTEXT:
{context}
---

QUESTION:
{question}

ANSWER:
"""

def build_prompt(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Injects retrieved context chunks and a question into a prompt template.

    Args:
        question (str): The user's question.
        context_chunks (List[Dict[str, Any]]): A list of retrieved chunk dictionaries
                                                from the FaissRetriever.

    Returns:
        str: A formatted prompt ready for a language model.
    """
    # 2. Format context chunks and inject into the template
    context_str = "\n\n".join([chunk['chunk'] for chunk in context_chunks])
    prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context_str, question=question)
    return prompt

# --- Part 5: Generation ---

class Generator:
    """
    A class to handle loading a generative model and producing an answer.
    """
    def __init__(self, model_name: str = 'google/flan-t5-small'):
        """
        Initializes the tokenizer and model.
        """
        # 1. Choose model
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        print(f"Loading generative model: {model_name}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        print("Generative model loaded.")

    def generate_answer(
        self,
        prompt: str,
        context_chunks: List[Dict[str, Any]],
        max_new_tokens: int = 100
    ) -> Dict[str, Any]:
        """
        Generates an answer using the provided prompt and context.

        Args:
            prompt (str): The full prompt including context and question.
            context_chunks (List[Dict[str, Any]]): The context chunks used in the prompt.
            max_new_tokens (int): The maximum number of tokens to generate for the answer.

        Returns:
            Dict[str, Any]: A dictionary containing the generated 'answer' and a list of 'sources'.
        """
        # 2. Write generation function
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        # 3. Handle max token length
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True, # Use sampling
            temperature=0.7 # Control randomness
        )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 5. Return answer + sources
        sources = list(set(chunk['metadata'].get('source', 'Unknown') for chunk in context_chunks))
        return {"answer": answer, "sources": sources}

# --- Part 3: Unit Tests for Retrieval ---

# Setup function for pytest: creates dummy files before tests run
@pytest.fixture(scope="module")
def setup_retriever():
    """
    A pytest fixture to set up the necessary files and initialize the FaissRetriever.
    This runs once per test module.
    """
    base_path = 'test_data'
    sample_chunks = [
        "Artificial intelligence is transforming many industries.", # 0
        "Machine learning is a subset of AI.", # 1
        "Python is a popular programming language for data science.", # 2
        "The quick brown fox jumps over the lazy dog.", # 3
    ]
    sample_metadata = [
        {"id": 1, "source": "tech_news"}, # 0
        {"id": 2, "source": "tech_news"}, # 1
        {"id": 3, "source": "programming_guide"}, # 2
        {"id": 4, "source": "animal_facts"}, # 3
    ]
    # Create the index and other files
    create_and_save_faiss_index(chunks=sample_chunks, metadata=sample_metadata, base_path=base_path)
    
    # Initialize the retriever
    retriever = FaissRetriever(base_path=base_path)
    yield retriever
    
    # Teardown: clean up created files (optional, but good practice)
    import shutil
    if os.path.exists(base_path):
        shutil.rmtree(base_path)

def test_retrieval_basic(setup_retriever):
    """Tests basic retrieval and metadata correctness."""
    retriever = setup_retriever
    query = "What is AI?"
    results = retriever.retrieve(query, k=1)
    
    assert len(results) >= 1
    # The first result should be one of the known AI-related chunks.
    # The test is now more robust and doesn't depend on which one is ranked higher.
    expected_chunks = ["Artificial intelligence is transforming many industries.", "Machine learning is a subset of AI."]
    assert results[0]["chunk"] in expected_chunks
    # Ensure metadata retrieval works
    assert results[0]["metadata"]["source"] == "tech_news"
    assert "score" in results[0]
    print(f"\nTest Basic: Passed. Top result score: {results[0]['score']:.4f}")

def test_retrieval_k_parameter(setup_retriever):
    """Tests if the 'k' parameter correctly limits the number of results."""
    retriever = setup_retriever
    query = "Tell me about machine intelligence"
    results_k2 = retriever.retrieve(query, k=2)
    results_k1 = retriever.retrieve(query, k=1)
    
    assert len(results_k2) == 2
    assert len(results_k1) == 1
    print("Test K Param: Passed.")

def test_retrieval_score_threshold(setup_retriever):
    """Tests if the score threshold correctly filters results."""
    retriever = setup_retriever
    query = "AI and ML"
    # A high threshold should filter out less relevant results
    results_high_thresh = retriever.retrieve(query, k=3, score_threshold=0.9)
    # A low threshold should include more results
    results_low_thresh = retriever.retrieve(query, k=3, score_threshold=0.5)
    
    assert len(results_high_thresh) < len(results_low_thresh)
    print(f"Test Threshold: Passed. Found {len(results_high_thresh)} with >0.9 threshold vs {len(results_low_thresh)} with >0.5.")

def test_retrieval_no_relevant_results(setup_retriever):
    """Tests that a completely irrelevant query returns no results with a reasonable threshold."""
    retriever = setup_retriever
    query = "What is the best recipe for pizza?"
    results = retriever.retrieve(query, k=3, score_threshold=0.6)
    
    assert len(results) == 0
    print("Test No Results: Passed.")

def test_build_prompt():
    """Tests that the prompt builder correctly formats the text block."""
    question = "What is the best language for data science?"
    context_chunks = [
        {'chunk': 'Python is a popular programming language for data science.', 'metadata': {}, 'score': 0.9},
        {'chunk': 'R is also widely used in statistics.', 'metadata': {}, 'score': 0.8}
    ]

    prompt = build_prompt(question, context_chunks)

    # 3. Ensure it's a clean text block with all components
    assert "QUESTION:\nWhat is the best language for data science?" in prompt
    assert "Python is a popular programming language for data science." in prompt
    assert "R is also widely used in statistics." in prompt
    assert prompt.startswith("You are a helpful Q&A assistant.")
    print("\nTest Build Prompt: Passed.")

def test_generation():
    """Tests the answer generation and source extraction."""
    # This test is slower as it loads a transformer model.
    generator = Generator()
    question = "What is machine learning?"
    context_chunks = [
        {'chunk': 'Machine learning is a subset of AI.', 'metadata': {'source': 'tech_news'}, 'score': 0.95}
    ]
    prompt = build_prompt(question, context_chunks)

    result = generator.generate_answer(prompt, context_chunks)

    assert "answer" in result
    assert "sources" in result
    assert len(result["answer"]) > 0
    # 4. Strip hallucinations: The prompt instructs the model to only use the context.
    # A simple check is to see if the answer reflects the context.
    assert "subset of AI" in result["answer"]
    assert result["sources"] == ["tech_news"]
    print("\nTest Generation: Passed.")

# To run the tests, save the code as a Python file (e.g., `retrieval.py`)
# and run `pytest` in your terminal in the same directory.
# The output will show the print statements and test results.

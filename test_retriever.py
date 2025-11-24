import unittest
import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import pickle

# Important: We need to import the class we want to test
from retrieval import Retriever

class TestRetriever(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up a temporary environment for testing. This runs once before all tests.
        """
        print("\nSetting up test environment...")
        # Create dummy directories
        os.makedirs('test_models', exist_ok=True)
        os.makedirs('test_data', exist_ok=True)

        # Create dummy metadata
        cls.test_metadata = pd.DataFrame([
            {'chunk_id': '0_0', 'context_chunk': 'The Normans were a people in France.'},
            {'chunk_id': '0_1', 'context_chunk': 'Their leader was a man named Rollo.'},
            {'chunk_id': '1_0', 'context_chunk': 'The kilogram-force is a unit of force.'}
        ])
        with open('test_data/metadata.pkl', 'wb') as f:
            pickle.dump(cls.test_metadata, f)

        # Create dummy embeddings
        model = SentenceTransformer('all-MiniLM-L6-v2')
        dummy_embeddings = model.encode(cls.test_metadata['context_chunk'].tolist(), normalize_embeddings=True)
        
        # Create and save a dummy FAISS index
        embedding_dimension = dummy_embeddings.shape[1]
        index = faiss.IndexFlatIP(dummy_embeddings.shape[1])

        # Explicitly normalize the embeddings to unit length
        faiss.normalize_L2(dummy_embeddings)

        # Ensure the array is C-contiguous and of type float32 for FAISS
        contiguous_embeddings = np.ascontiguousarray(dummy_embeddings.astype(np.float32))
        index.add(contiguous_embeddings)

        faiss.write_index(index, 'test_models/faiss.index')
        
        # Monkey-patch the file paths in the Retriever class for testing
        cls.original_paths = ('models/faiss.index', 'data/metadata.pkl')
        Retriever.__init__ = lambda self, model_name='all-MiniLM-L6-v2', device='cpu': cls.mock_init(self, model_name, device)
        
        # Instantiate the retriever with our mocked paths
        cls.retriever = Retriever()
        print("Test environment ready.")

    @classmethod
    def mock_init(cls, self, model_name, device):
        """A mock __init__ to load from our test directories."""
        self.model = SentenceTransformer(model_name, device=device)
        self.index = faiss.read_index('test_models/faiss.index')
        with open('test_data/metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)

    def test_search_finds_correct_chunk(self):
        """Ensure search returns the most relevant chunk."""
        query = "Who was the leader of the Normans?"
        results = self.retriever.search(query, k=1)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['chunk_id'], '0_1')
        self.assertIn("Rollo", results[0]['context_chunk'])

    def test_k_parameter_works(self):
        """Ensure the 'k' parameter returns the correct number of results."""
        query = "normans"
        results = self.retriever.search(query, k=2)
        self.assertEqual(len(results), 2)

    def test_score_threshold_works(self):
        """Ensure the score threshold correctly filters results."""
        # This query is irrelevant, so its score should be low
        query = "What is the capital of Japan?"
        # Set a high threshold that should filter everything out
        results = self.retriever.search(query, k=3, score_threshold=0.8)
        self.assertEqual(len(results), 0)

    def test_metadata_retrieval_is_correct(self):
        """Verify that the text returned matches the metadata."""
        query = "unit of force"
        results = self.retriever.search(query, k=1)
        
        self.assertEqual(len(results), 1)
        # Check if the returned text is exactly what's in our dummy metadata
        expected_text = self.test_metadata[self.test_metadata['chunk_id'] == '1_0']['context_chunk'].iloc[0]
        self.assertEqual(results[0]['context_chunk'], expected_text)

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary files and directories. This runs once after all tests."""
        print("\nTearing down test environment...")
        os.remove('test_models/faiss.index')
        os.remove('test_data/metadata.pkl')
        os.rmdir('test_models')
        os.rmdir('test_data')
        
        # Restore original __init__ if necessary, though not critical for this script
        print("Cleanup complete.")

# To run the tests from your terminal:
# python -m unittest test_retriever.py

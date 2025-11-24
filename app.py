import gradio as gr
from retrieval import FaissRetriever, Generator, build_prompt, create_and_save_faiss_index
import os
import csv
import pickle
# --- 1. Setup: Create dummy data and index if they don't exist ---
# This part is for demonstration purposes so the app can run out-of-the-box.
# In a real application, you would have a pre-built index.
def setup_environment(base_path='app_data'):
    """Creates a sample FAISS index if it doesn't already exist."""
    if not os.path.exists(base_path):
        print("First-time setup: Creating FAISS index for the app...")
        
        context_chunks = []
        with open("output_pandas.csv", 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row:  # Ensure the row is not empty
                    context_chunks.append(row[0])  # Append the first (and only) element
    
        file_path = 'data/metadata.pkl'

        with open(file_path, 'rb') as file:
            metadata = pickle.load(file)
        
        create_and_save_faiss_index(chunks=context_chunks, metadata=metadata, base_path=base_path)
        print("Setup complete.")
    return base_path

# --- 2. Load Models and Set Up Pipeline ---
print("Setting up the application environment...")
APP_DATA_PATH = setup_environment()

print("Loading models... This may take a moment.")
RETRIEVER = FaissRetriever(base_path=APP_DATA_PATH)
GENERATOR = Generator()
print("Models loaded successfully.")

def answer_question(question, top_k):
    """
    The main pipeline function that orchestrates retrieval, prompt building, and generation.
    """
    print(f"Received question: '{question}', top_k: {top_k}")

    # 1. Retrieve context chunks
    context_chunks = RETRIEVER.retrieve(question, k=int(top_k), score_threshold=0.1)

    if not context_chunks:
        return "I could not find any relevant information to answer your question.", "No sources found."

    # 2. Build the prompt
    prompt = build_prompt(question, context_chunks)

    # 3. Generate the answer
    result = GENERATOR.generate_answer(prompt, context_chunks)

    # 4. Show retrieved answer and sources
    answer = result.get("answer", "Failed to generate an answer.")
    sources = "\n".join(f"- {s}" for s in result.get("sources", []))

    return answer, sources

# --- 3. Create and Polish Gradio UI ---
with gr.Blocks(title="RAG Q&A Bot") as demo:
    gr.Markdown(
        """
        # Retrieval-Augmented Generation (RAG) Q&A Bot
        Ask a question, and the system will retrieve relevant information to generate a fact-based answer.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            # Input: question text
            question_box = gr.Textbox(label="Question", placeholder="e.g., What is machine learning?")
            # Add configurable top-k
            top_k_slider = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Top-K Chunks to Retrieve")
            submit_btn = gr.Button("Get Answer", variant="primary")

        with gr.Column(scale=3):
            # Show retrieved answer and sources
            answer_box = gr.Textbox(label="Answer", lines=4, interactive=False)
            sources_box = gr.Textbox(label="Sources", lines=2, interactive=False)

    # Define interactions
    submit_btn.click(
        fn=answer_question,
        inputs=[question_box, top_k_slider],
        outputs=[answer_box, sources_box]
    )

    gr.Examples(
        [["What is FAISS used for?"], ["Tell me about Python."]],
        inputs=[question_box]
    )

if __name__ == "__main__":
    # 5. Test UI locally
    demo.launch()
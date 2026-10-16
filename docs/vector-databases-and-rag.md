# Vector databases and RAG

The core retrieval path vectorizes knowledge documents with TF-IDF, projects them into a dense latent space with truncated SVD, normalizes vectors and ranks results by cosine similarity. Euclidean distance is included for comparison.

The local joblib index is appropriate for a small corpus. Chroma and Qdrant examples show where persistent vector stores fit. The deterministic RAG function returns source-labelled evidence and does not claim to be a generative model. The Ollama example can pass the same context to a local GPT-style model.

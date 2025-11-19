#!/usr/bin/env python3
"""
Debug script to test semantic search retrieval.
Tests why job-related queries aren't finding stored information.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.database import AssistantDatabase
from src.retrieval import ContextRetriever
from src.semantic_search import generate_embedding, bytes_to_embedding, cosine_similarity

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

def main():
    print("=" * 80)
    print("SEMANTIC SEARCH DEBUG TEST")
    print("=" * 80)

    # Initialize database and retriever
    db = AssistantDatabase()
    retriever = ContextRetriever(db)

    # Test query
    query = "what is my current company name at job?"
    print(f"\nQuery: {query}")
    print("-" * 80)

    # Get all embeddings from database
    stored_embeddings = db.get_embeddings_for_search()
    print(f"\nTotal embeddings in database: {len(stored_embeddings)}")

    if not stored_embeddings:
        print("ERROR: No embeddings found in database!")
        return

    # Generate embedding for query
    print("\nGenerating query embedding...")
    query_embedding = generate_embedding(query)
    print(f"Query embedding shape: {query_embedding.shape}")

    # Calculate similarity with ALL stored embeddings
    print("\nCalculating similarity with ALL conversations:")
    print("-" * 80)

    similarities = []
    for conv_id, embedding_bytes in stored_embeddings:
        stored_embedding = bytes_to_embedding(embedding_bytes)
        similarity = cosine_similarity(query_embedding, stored_embedding)

        # Get conversation content
        conv = db.get_conversation_with_metadata(conv_id)
        content = conv.get('content', '')[:100] if conv else ''

        similarities.append((conv_id, similarity, content))
        print(f"Conv #{conv_id:3d} | Similarity: {similarity:.4f} | {content}...")

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 80)
    print("TOP 5 MATCHES:")
    print("=" * 80)
    for conv_id, similarity, content in similarities[:5]:
        conv = db.get_conversation_with_metadata(conv_id)
        print(f"\nConv #{conv_id} | Similarity: {similarity:.4f}")
        print(f"Content: {conv.get('content', '')}")
        print(f"People: {conv.get('people', [])}")
        print(f"Topics: {conv.get('topics', [])}")
        print("-" * 40)

    # Test with retriever
    print("\n" + "=" * 80)
    print("TESTING RETRIEVER.SMART_RETRIEVE():")
    print("=" * 80)

    results = retriever.smart_retrieve(query, top_k=5)
    print(f"\nRetrieved {len(results)} results")

    if not results:
        print("WARNING: Retriever returned NO results!")
        print("This explains why the LLM is hallucinating!")
    else:
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Conv #{result.get('id')} | Score: {result.get('final_score', 0):.4f}")
            print(f"   Content: {result.get('content', '')[:100]}...")
            print(f"   Similarity: {result.get('similarity', 0):.4f}")

    # Check specific conversations that should match
    print("\n" + "=" * 80)
    print("CHECKING SPECIFIC JOB-RELATED CONVERSATIONS:")
    print("=" * 80)

    job_convs = [10, 20, 22]  # Known conversations with job info
    for conv_id in job_convs:
        conv = db.get_conversation_with_metadata(conv_id)
        if conv:
            print(f"\nConv #{conv_id}:")
            print(f"Content: {conv.get('content', '')}")

            # Find its similarity score
            for cid, sim, _ in similarities:
                if cid == conv_id:
                    print(f"Similarity: {sim:.4f}")
                    if sim < 0.3:
                        print("⚠️  BELOW THRESHOLD (0.3) - This is why it's not being retrieved!")
                    break

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Integration test for the Personal Assistant system.

Tests the complete flow: database, embeddings, extraction, retrieval.
"""

import logging
import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from database import AssistantDatabase
from semantic_search import (
    generate_embedding,
    embedding_to_bytes,
    bytes_to_embedding,
    cosine_similarity,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_database():
    """Test database operations."""
    logger.info("Testing database...")

    # Use a test database
    db_path = "test_assistant.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    db = AssistantDatabase(db_path)

    # Test adding conversations
    conv1 = db.add_conversation("user", "I have a meeting with Matthew tomorrow")
    conv2 = db.add_conversation("assistant", "Good luck with your meeting!")
    conv3 = db.add_conversation(
        "user", "Matthew was sad last time we talked about the project"
    )

    assert conv1 > 0, "Failed to add conversation 1"
    assert conv2 > 0, "Failed to add conversation 2"
    assert conv3 > 0, "Failed to add conversation 3"

    # Test adding metadata
    meta1 = db.add_metadata(
        conversation_id=conv1,
        people=["Matthew"],
        topics=["meeting"],
        sentiment="neutral",
        category="work",
    )

    meta3 = db.add_metadata(
        conversation_id=conv3,
        people=["Matthew"],
        topics=["project", "discussion"],
        sentiment="concerned",
        category="work",
    )

    assert meta1 > 0, "Failed to add metadata 1"
    assert meta3 > 0, "Failed to add metadata 3"

    # Test retrieval
    conversations = db.get_conversations(limit=10)
    assert len(conversations) == 3, f"Expected 3 conversations, got {len(conversations)}"

    # Test search by people
    matthew_convs = db.search_by_people("Matthew", limit=10)
    assert (
        len(matthew_convs) == 2
    ), f"Expected 2 conversations with Matthew, got {len(matthew_convs)}"

    # Test search by topic
    meeting_convs = db.search_by_topic("meeting", limit=10)
    assert (
        len(meeting_convs) >= 1
    ), f"Expected at least 1 meeting conversation, got {len(meeting_convs)}"

    # Test stats
    stats = db.get_stats()
    assert stats["total_conversations"] == 3
    assert stats["user_messages"] == 2
    assert stats["assistant_messages"] == 1
    assert "Matthew" in db.get_all_people()
    assert "meeting" in db.get_all_topics()

    logger.info("✓ Database tests passed")

    # Cleanup
    db.close()
    os.remove(db_path)

    return True


def test_embeddings():
    """Test embedding generation and similarity."""
    logger.info("Testing embeddings...")

    # Generate embeddings
    text1 = "I have a meeting with Matthew tomorrow"
    text2 = "Matthew and I will meet tomorrow"
    text3 = "The weather is nice today"

    emb1 = generate_embedding(text1)
    emb2 = generate_embedding(text2)
    emb3 = generate_embedding(text3)

    assert emb1.shape[0] == 384, f"Expected 384-dim embedding, got {emb1.shape[0]}"

    # Test similarity
    sim_12 = cosine_similarity(emb1, emb2)
    sim_13 = cosine_similarity(emb1, emb3)

    logger.info(f"Similarity (meeting texts): {sim_12:.3f}")
    logger.info(f"Similarity (meeting vs weather): {sim_13:.3f}")

    # Similar texts should have higher similarity
    assert (
        sim_12 > sim_13
    ), f"Expected meeting texts to be more similar: {sim_12} vs {sim_13}"
    assert sim_12 > 0.5, f"Expected high similarity for similar texts: {sim_12}"

    # Test byte conversion
    emb_bytes = embedding_to_bytes(emb1)
    emb_restored = bytes_to_embedding(emb_bytes)

    sim_original_restored = cosine_similarity(emb1, emb_restored)
    assert (
        sim_original_restored > 0.99
    ), f"Byte conversion should preserve embedding: {sim_original_restored}"

    logger.info("✓ Embedding tests passed")

    return True


def test_integrated_search():
    """Test integrated semantic search with database."""
    logger.info("Testing integrated search...")

    # Create test database
    db_path = "test_search.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    db = AssistantDatabase(db_path)

    # Add test conversations with embeddings
    messages = [
        "I have a pair programming session with Matthew today. He was sad last time.",
        "Tomorrow I need to review Sarah's code for the authentication feature.",
        "The team meeting went well. Everyone seemed happy with the progress.",
        "Matthew asked about the deadline for the API project.",
        "I need to buy groceries after work.",
    ]

    for msg in messages:
        # Add conversation
        conv_id = db.add_conversation("user", msg)

        # Generate and add embedding
        embedding = generate_embedding(msg)
        emb_bytes = embedding_to_bytes(embedding)

        # Add metadata with embedding
        db.add_metadata(conversation_id=conv_id, embedding=emb_bytes)

    # Test search
    query = "What do you know about Matthew?"
    stored_embeddings = db.get_embeddings_for_search()

    from semantic_search import search_conversations

    results = search_conversations(query, stored_embeddings, top_k=3, min_similarity=0.2)

    logger.info(f"Search query: {query}")
    logger.info(f"Found {len(results)} results")

    for conv_id, similarity in results:
        conv = db.get_conversation_with_metadata(conv_id)
        logger.info(
            f"  [{similarity:.3f}] {conv['content'][:80]}..."
            if len(conv["content"]) > 80
            else f"  [{similarity:.3f}] {conv['content']}"
        )

    # Should find Matthew-related conversations
    assert len(results) > 0, "Should find at least one result"

    # Top result should be about Matthew
    top_conv_id, top_sim = results[0]
    top_conv = db.get_conversation_with_metadata(top_conv_id)
    assert (
        "matthew" in top_conv["content"].lower()
    ), "Top result should mention Matthew"

    logger.info("✓ Integrated search tests passed")

    # Cleanup
    db.close()
    os.remove(db_path)

    return True


def run_all_tests():
    """Run all integration tests."""
    logger.info("=" * 60)
    logger.info("Personal Assistant Integration Tests")
    logger.info("=" * 60)

    tests = [
        ("Database Operations", test_database),
        ("Embedding Generation & Similarity", test_embeddings),
        ("Integrated Semantic Search", test_integrated_search),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            logger.info(f"\n{test_name}...")
            if test_func():
                passed += 1
        except Exception as e:
            logger.error(f"✗ {test_name} failed: {e}", exc_info=True)
            failed += 1

    logger.info("\n" + "=" * 60)
    logger.info(f"Test Results: {passed} passed, {failed} failed")
    logger.info("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)


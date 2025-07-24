# Verschoben aus dem Hauptverzeichnis
#!/usr/bin/env python3
"""
Test script to demonstrate the memory search behavior change.
This script tests that memory search now ignores user_id and session_id filters.
"""

import asyncio
import os
import sys
from typing import Dict, Any

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.memory_qdrant import ConversationMemory


async def test_memory_search_behavior():
    """Test that memory search now ignores user_id and session_id filters."""
    
    print("🪪 Testing Memory Search Behavior Changes")
    print("=" * 50)
    
    # Create a test memory instance
    memory = ConversationMemory(collection_name="test_search_behavior")
    
    # Store some test data with different user_ids and session_ids
    test_data = [
        {"user_id": "user1", "session_id": "session1", "text": "I love programming with Python"},
        {"user_id": "user2", "session_id": "session2", "text": "Python is my favorite programming language"},
        {"user_id": "user3", "session_id": "session1", "text": "I enjoy coding in Java"},
        {"user_id": "user1", "session_id": "session3", "text": "Machine learning with Python is fascinating"},
    ]
    
    print("📝 Storing test memories...")
    for i, data in enumerate(test_data):
        success = await memory.store_conversation_turn(
            user_id=data["user_id"],
            user_message=data["text"],
            assistant_response=f"Response to: {data['text']}",
            session_id=data["session_id"]
        )
        if success:
            print(f"   ✅ Stored memory {i+1}: {data['user_id']}/{data['session_id']}")
        else:
            print(f"   ❌ Failed to store memory {i+1}")
    
    print("\n🔍 Testing search behavior...")
    
    # Test 1: Search for "Python" as user1/session1 - should find ALL Python-related memories
    print("\nTest 1: Searching for 'Python' as user1/session1")
    print("Expected: Should find ALL Python-related memories (not just user1/session1)")
    
    results = await memory.search_memory(
        user_id="user1",
        query_text="Python programming",
        session_id="session1",
        limit=10,
        threshold=0.1  # Low threshold to get more results
    )
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"   {i+1}. Text: {result['text'][:50]}...")
        print(f"      User: {result.get('user_id', 'N/A')}, Session: {result.get('session_id', 'N/A')}")
        print(f"      Score: {result['score']:.3f}")
    
    # Count how many different users/sessions were found
    unique_users = set(result.get('user_id', '') for result in results)
    unique_sessions = set(result.get('session_id', '') for result in results)
    
    print(f"\n📊 Analysis:")
    print(f"   Unique users found: {len(unique_users)} - {unique_users}")
    print(f"   Unique sessions found: {len(unique_sessions)} - {unique_sessions}")
    
    # Verify the change worked
    if len(unique_users) > 1:
        print("   ✅ SUCCESS: Search found memories from multiple users!")
        print("   ✅ The search is now ignoring user_id filters as intended.")
    else:
        print("   ❌ ISSUE: Search only found memories from one user.")
        print("   ❌ The user_id filter might still be active.")
    
    # Test 2: Search for "Java" as user1 - should find Java memory from user3
    print("\n" + "="*50)
    print("Test 2: Searching for 'Java' as user1")
    print("Expected: Should find Java memory from user3 (cross-user search)")
    
    java_results = await memory.search_memory(
        user_id="user1",
        query_text="Java programming",
        limit=10,
        threshold=0.1
    )
    
    print(f"Found {len(java_results)} results:")
    for i, result in enumerate(java_results):
        print(f"   {i+1}. Text: {result['text']}")
        print(f"      User: {result.get('user_id', 'N/A')}, Session: {result.get('session_id', 'N/A')}")
        print(f"      Score: {result['score']:.3f}")
    
    # Check if we found the Java memory from user3
    found_java_from_user3 = any(
        result.get('user_id') == 'user3' and 'Java' in result.get('text', '')
        for result in java_results
    )
    
    if found_java_from_user3:
        print("   ✅ SUCCESS: Found Java memory from user3 while searching as user1!")
        print("   ✅ Cross-user search is working correctly.")
    else:
        print("   ❌ ISSUE: Did not find Java memory from user3.")
        print("   ❌ Cross-user search might not be working.")
    
    print("\n" + "="*50)
    print("🎯 Summary:")
    if len(unique_users) > 1 and found_java_from_user3:
        print("   ✅ Memory search successfully ignores user_id and session_id filters!")
        print("   ✅ The changes are working as expected.")
    else:
        print("   ⚠️  Memory search behavior needs further investigation.")
    
    print("\n🧹 Cleaning up test collection...")
    # Note: In a real scenario, you might want to clean up the test collection
    # For now, we'll leave it as the collection is separate for testing


if __name__ == "__main__":
    print("Starting memory search behavior test...")
    try:
        asyncio.run(test_memory_search_behavior())
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

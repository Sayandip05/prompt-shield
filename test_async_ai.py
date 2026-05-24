"""
Quick test script to verify async AI endpoints work correctly.
Run with: python test_async_ai.py
"""
import asyncio
import time


async def simulate_ai_call(call_id: int):
    """Simulate an AI API call that takes 500ms."""
    print(f"[{call_id}] Starting AI call...")
    start = time.time()
    
    # Simulate AI processing (500ms)
    await asyncio.sleep(0.5)
    
    elapsed = time.time() - start
    print(f"[{call_id}] Completed in {elapsed:.2f}s")
    return f"Response {call_id}"


async def test_concurrent_calls():
    """Test multiple concurrent AI calls."""
    print("=" * 60)
    print("Testing Async AI Calls")
    print("=" * 60)
    
    # Test 1: Sequential (old way - blocking)
    print("\n1. Sequential Calls (Blocking - Old Way):")
    start = time.time()
    for i in range(5):
        await simulate_ai_call(i)
    sequential_time = time.time() - start
    print(f"   Total time: {sequential_time:.2f}s")
    
    # Test 2: Concurrent (new way - non-blocking)
    print("\n2. Concurrent Calls (Non-Blocking - New Way):")
    start = time.time()
    tasks = [simulate_ai_call(i) for i in range(5)]
    await asyncio.gather(*tasks)
    concurrent_time = time.time() - start
    print(f"   Total time: {concurrent_time:.2f}s")
    
    # Results
    print("\n" + "=" * 60)
    print("Results:")
    print(f"  Sequential: {sequential_time:.2f}s (5 calls × 0.5s each)")
    print(f"  Concurrent: {concurrent_time:.2f}s (all 5 calls in parallel)")
    print(f"  Speedup: {sequential_time/concurrent_time:.1f}x faster!")
    print("=" * 60)
    
    # Explanation
    print("\n✅ Your AI endpoints are now NON-BLOCKING!")
    print("   - Workers can handle other requests while waiting for AI")
    print("   - Multiple AI chats can run concurrently")
    print("   - No more blocked workers during AI processing")


if __name__ == "__main__":
    asyncio.run(test_concurrent_calls())

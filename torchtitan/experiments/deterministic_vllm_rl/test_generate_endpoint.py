#!/usr/bin/env python3
"""
Test script for the /generate endpoint on the TorchTitan inference server.

Usage:
    python test_generate_endpoint.py [--url URL] [--prompt PROMPT]

Example:
    python test_generate_endpoint.py --url http://localhost:9000
"""

import argparse
import json
import requests
from transformers import AutoTokenizer


def test_health(base_url: str) -> bool:
    """Test the /health endpoint."""
    print(f"\n=== Testing /health at {base_url}/health ===")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_generate_with_text(base_url: str, prompt: str, tokenizer_name: str = "Qwen/Qwen2.5-7B") -> bool:
    """Test /generate with a text prompt."""
    print(f"\n=== Testing /generate with text prompt ===")
    print(f"Prompt: {prompt[:100]}...")
    
    request_data = {
        "prompt": prompt,
        "max_tokens": 50,
        "temperature": 0.7,
        "n": 2,
        "logprobs": 0,
    }
    
    try:
        response = requests.post(
            f"{base_url}/generate",
            json=request_data,
            timeout=120,
        )
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nResponse keys: {list(result.keys())}")
            print(f"Number of completions: {len(result.get('text', []))}")
            print(f"Finish reasons: {result.get('finish_reasons', [])}")
            
            for i, text in enumerate(result.get("text", [])):
                print(f"\n--- Completion {i+1} ---")
                print(f"Text: {text[:200]}...")
                
                logprobs = result.get("logprobs", [[]])[i] if result.get("logprobs") else []
                if logprobs:
                    print(f"Logprobs (first 5): {logprobs[:5]}")
            
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generate_with_token_ids(base_url: str, prompt: str, tokenizer_name: str = "Qwen/Qwen2.5-7B") -> bool:
    """Test /generate with prompt_token_ids (the format Atropos uses)."""
    print(f"\n=== Testing /generate with prompt_token_ids (Atropos format) ===")
    
    # Load tokenizer to convert prompt to token IDs
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    prompt_token_ids = tokenizer.encode(prompt)
    print(f"Prompt: {prompt[:100]}...")
    print(f"Token IDs (first 10): {prompt_token_ids[:10]}...")
    print(f"Total prompt tokens: {len(prompt_token_ids)}")
    
    # This is the exact format Atropos VLLMServer sends
    request_data = {
        "prompt": {"prompt_token_ids": prompt_token_ids},
        "max_tokens": 50,
        "temperature": 0.7,
        "n": 4,  # Multiple samples per prompt
        "logprobs": 0,  # Return logprobs
        "stop": ["</answer>", "\n\n"],  # Stop sequences
    }
    
    try:
        response = requests.post(
            f"{base_url}/generate",
            json=request_data,
            timeout=120,
        )
        print(f"\nStatus: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response keys: {list(result.keys())}")
            print(f"Number of completions: {len(result.get('text', []))}")
            print(f"Finish reasons: {result.get('finish_reasons', [])}")
            
            # Verify the response format matches what Atropos expects
            logprobs = result.get("logprobs", [])
            if logprobs and len(logprobs) > 0:
                first_sample = logprobs[0]
                if first_sample and len(first_sample) > 0:
                    first_position = first_sample[0]
                    print(f"\nLogprobs format check:")
                    print(f"  - logprobs is list: {isinstance(logprobs, list)}")
                    print(f"  - First sample is list: {isinstance(first_sample, list)}")
                    print(f"  - First position is list: {isinstance(first_position, list)}")
                    if isinstance(first_position, list) and len(first_position) > 0:
                        print(f"  - First position[0] is dict: {isinstance(first_position[0], dict)}")
                        print(f"  - Format correct for Atropos: YES ✓")
            
            for i, text in enumerate(result.get("text", [])[:2]):  # Show first 2
                print(f"\n--- Completion {i+1} ---")
                print(f"Text: {text[:200]}...")
                
                # Decode to verify
                if logprobs and i < len(logprobs):
                    sample_logprobs = logprobs[i]
                    if sample_logprobs:
                        # Extract token IDs from logprobs (Atropos does this)
                        output_ids = [
                            int(list(item[0].keys())[0]) for item in sample_logprobs
                        ]
                        output_logprobs = [
                            list(item[0].values())[0] for item in sample_logprobs
                        ]
                        print(f"Output token IDs: {output_ids[:10]}...")
                        print(f"Output logprobs: {output_logprobs[:5]}...")
                        decoded = tokenizer.decode(output_ids)
                        print(f"Decoded from IDs: {decoded[:200]}...")
            
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test /generate endpoint")
    parser.add_argument("--url", default="http://localhost:9000", help="Server URL")
    parser.add_argument("--prompt", default="What is 2 + 2? Think step by step.\n\n", help="Test prompt")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B", help="Tokenizer name")
    args = parser.parse_args()
    
    print("=" * 60)
    print("TorchTitan /generate Endpoint Test")
    print("=" * 60)
    print(f"Server URL: {args.url}")
    print(f"Tokenizer: {args.tokenizer}")
    
    # Test 1: Health check
    health_ok = test_health(args.url)
    if not health_ok:
        print("\n❌ Health check failed. Is the server running?")
        return
    
    # Test 2: Generate with text prompt
    text_ok = test_generate_with_text(args.url, args.prompt, args.tokenizer)
    
    # Test 3: Generate with token IDs (Atropos format)
    token_ok = test_generate_with_token_ids(args.url, args.prompt, args.tokenizer)
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    print(f"  Health check:     {'✓ PASS' if health_ok else '✗ FAIL'}")
    print(f"  Text prompt:      {'✓ PASS' if text_ok else '✗ FAIL'}")
    print(f"  Token IDs prompt: {'✓ PASS' if token_ok else '✗ FAIL'}")
    
    if health_ok and text_ok and token_ok:
        print("\n✓ All tests passed! Server is ready for Atropos.")
    else:
        print("\n✗ Some tests failed. Check the output above.")


if __name__ == "__main__":
    main()

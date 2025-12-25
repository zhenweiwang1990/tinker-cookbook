#!/usr/bin/env python3
"""
Test script for Tinker's OpenAI-compatible API.

This script tests:
1. Basic text-only messages
2. Multimodal messages with images (if supported)
3. Different message formats
"""

import asyncio
import base64
import os
from pathlib import Path

from openai import AsyncOpenAI


async def test_text_only():
    """Test basic text-only message."""
    print("\n" + "="*60)
    print("Test 1: Text-only message")
    print("="*60)
    
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        raise ValueError("TINKER_API_KEY environment variable is required")
    
    # You'll need to provide a valid Tinker model path
    model_path = os.getenv("TINKER_MODEL_PATH", "tinker://test/test/sampler_weights/000080")
    
    base_url = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
    
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    
    try:
        response = await client.chat.completions.create(
            model=model_path,
            messages=[
                {
                    "role": "user",
                    "content": "Hello, what is 2+2?"
                }
            ],
            max_tokens=50,
        )
        print(f"✅ Success! Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_multimodal_list_format():
    """Test multimodal message with list format (standard OpenAI format)."""
    print("\n" + "="*60)
    print("Test 2: Multimodal message (list format: text + image_url)")
    print("="*60)
    
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        raise ValueError("TINKER_API_KEY environment variable is required")
    
    model_path = os.getenv("TINKER_MODEL_PATH", "tinker://test/test/sampler_weights/000080")
    base_url = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
    
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    
    # Create a simple test image (1x1 red pixel)
    # In real usage, this would be a screenshot from GBox
    test_image_data = base64.b64encode(
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82'
    ).decode()
    image_data_uri = f"data:image/png;base64,{test_image_data}"
    
    try:
        response = await client.chat.completions.create(
            model=model_path,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What do you see in this image?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_uri
                            }
                        }
                    ]
                }
            ],
            max_tokens=50,
        )
        print(f"✅ Success! Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"   Error type: {type(e).__name__}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"   Response: {e.response.text}")
        return False


async def test_multimodal_string_format():
    """Test if Tinker accepts string content only (no multimodal)."""
    print("\n" + "="*60)
    print("Test 3: String content only (no multimodal)")
    print("="*60)
    
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        raise ValueError("TINKER_API_KEY environment variable is required")
    
    model_path = os.getenv("TINKER_MODEL_PATH", "tinker://test/test/sampler_weights/000080")
    base_url = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
    
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    
    try:
        response = await client.chat.completions.create(
            model=model_path,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Describe what you would see in a screenshot."
                }
            ],
            max_tokens=50,
        )
        print(f"✅ Success! Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_responses_api_format():
    """Test Responses API format (input_text/input_image) - should fail."""
    print("\n" + "="*60)
    print("Test 4: Responses API format (input_text/input_image)")
    print("="*60)
    
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        raise ValueError("TINKER_API_KEY environment variable is required")
    
    model_path = os.getenv("TINKER_MODEL_PATH", "tinker://test/test/sampler_weights/000080")
    base_url = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
    
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    
    test_image_data = base64.b64encode(
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82'
    ).decode()
    image_data_uri = f"data:image/png;base64,{test_image_data}"
    
    try:
        # Try Responses API format (this should fail with Tinker)
        response = await client.chat.completions.create(
            model=model_path,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "What do you see?"
                        },
                        {
                            "type": "input_image",
                            "image_url": image_data_uri
                        }
                    ]
                }
            ],
            max_tokens=50,
        )
        print(f"✅ Success! (Unexpected - Responses API format worked)")
        print(f"   Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ Error (Expected): {e}")
        print(f"   This confirms Tinker doesn't support Responses API format")
        return False


async def main():
    """Run all tests."""
    print("Testing Tinker's OpenAI-compatible API")
    print("="*60)
    print(f"Base URL: https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1")
    print(f"Model: {os.getenv('TINKER_MODEL_PATH', 'tinker://test/test/sampler_weights/000080')}")
    
    results = []
    
    # Test 1: Text only
    results.append(("Text only", await test_text_only()))
    
    # Test 2: Multimodal (list format)
    results.append(("Multimodal (list)", await test_multimodal_list_format()))
    
    # Test 3: String content only
    results.append(("String content", await test_multimodal_string_format()))
    
    # Test 4: Responses API format
    results.append(("Responses API format", await test_responses_api_format()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
    
    print("\n" + "="*60)
    print("Conclusion:")
    print("="*60)
    multimodal_supported = results[1][1]  # Test 2 result
    if multimodal_supported:
        print("✅ Tinker API supports multimodal messages (list format)")
        print("   GBox Agent can use standard OpenAI format (text/image_url)")
    else:
        print("❌ Tinker API does NOT support multimodal messages (list format)")
        print("   GBox Agent needs to use alternative approach for images")


if __name__ == "__main__":
    asyncio.run(main())


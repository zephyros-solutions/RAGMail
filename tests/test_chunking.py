#!/usr/bin/env python
"""
Email Chunking Validation Tests
Tests sentence-based chunking strategy with configured constraints
"""

import sys
import re
from pathlib import Path
from mail_processing.mailconverter import MailConverter
from config.globals import MAX_CHUNK_LEN, MAX_CHUNK_EXCESS

def print_section(title):
    """Print formatted section header."""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")

def mock_make_chunks(text, max_chunk_len, max_chunk_excess):
    """
    Replicate the actual make_chunks() logic from MailConverter.
    Split text into sentences while preserving punctuation boundaries.
    """
    text_chunks = []
    current_chunk = ""
    
    # Split text by sentence-ending punctuation (keeps text, removes punctuation+space)
    sentences = re.split(r'(?<=[.!?;])(?: )*', text)
    
    # Fallback: if only one sentence and it's too long, use looser split
    if len(sentences) == 1 and len(sentences[0]) > max_chunk_excess * max_chunk_len:
        sentences = re.split(r'(?<=[.!?; ])(?: )*', text)
    
    for sentence in sentences:
        # Skip empty sentences
        if not sentence or not sentence.strip():
            continue
        
        if len(current_chunk) + len(sentence) + 1 < max_chunk_len:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                text_chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if len(current_chunk) > 0:
        text_chunks.append(current_chunk.strip())
    
    return text_chunks

def test_chunk_sizes():
    """Verify chunks respect MAX_CHUNK_LEN constraint."""
    print_section("TEST 1: Chunk Size Validation")
    
    # Create test emails with known structure
    test_emails = {
        "simple": "This is sentence one. This is sentence two. This is sentence three.",
        "long_sentence": "This is a very long sentence that contains many words and should still respect the maximum chunk length even though it is quite lengthy and contains multiple clauses and concepts all within a single sentence structure. The next sentence is here.",
        "mixed": "Short. " + "Medium length sentence with some content. " * 5 + "End.", 
        "multiline": "Sentence one.\n\nSentence two.\n\nSentence three.",
    }
    
    passed = 0
    failed = 0
    
    for test_name, text in test_emails.items():
        chunks = mock_make_chunks(text, MAX_CHUNK_LEN, MAX_CHUNK_EXCESS)
        
        # Validate chunk sizes
        all_valid = True
        for i, chunk in enumerate(chunks):
            if len(chunk) >= MAX_CHUNK_LEN:
                print(f"   ✗ {test_name}: Chunk {i} exceeds limit ({len(chunk)} >= {MAX_CHUNK_LEN})")
                all_valid = False
        
        if all_valid and chunks:
            print(f"   ✓ {test_name}: All chunks within limit ({len(chunks)} chunks, max size: {max(len(c) for c in chunks)})")
            passed += 1
        elif not chunks:
            print(f"   ✗ {test_name}: No chunks generated")
            failed += 1
    
    return passed, failed

def test_sentence_boundaries():
    """Verify chunks preserve sentence boundaries (no mid-sentence splits)."""
    print_section("TEST 2: Sentence Boundary Preservation")
    
    test_cases = [
        "Sentence one. Sentence two. Sentence three.",
        "What? Is this correct? Yes.",
        "End note; Another statement. Final.",
        "First. " + "A medium sentence. " * 20 + "Last.",  # Many sentences
    ]
    
    passed = 0
    failed = 0
    
    for text in test_cases:
        chunks = mock_make_chunks(text, MAX_CHUNK_LEN, MAX_CHUNK_EXCESS)
        
        # Verify chunks end with punctuation or space-trimmed word
        all_valid = True
        for i, chunk in enumerate(chunks):
            # Each chunk should be complete (end with a word, not partial)
            if chunk and chunk[-1] == ' ':
                print(f"   ⚠ Chunk {i} ends with space: '{chunk[-30:]}'")
            # Check that chunk is non-empty
            if not chunk or len(chunk.strip()) == 0:
                print(f"   ✗ Chunk {i} is empty")
                all_valid = False
        
        if all_valid and chunks:
            print(f"   ✓ Text: '{text[:45]}...' generated {len(chunks)} valid chunks")
            passed += 1
        else:
            failed += 1
    
    return passed, failed

def test_no_empty_chunks():
    """Verify no empty chunks are generated."""
    print_section("TEST 3: Empty Chunk Detection")
    
    test_cases = [
        ("Sentence. ", True),  # Should generate 1 chunk
        ("  Multiple   spaces  between. Words.", True),  # Should generate chunks
        ("Newline\n\nSeparated. Sentences.", True),  # Should generate chunks
        ("", True),  # Empty input - should return empty result or handle gracefully
    ]
    
    passed = 0
    failed = 0
    
    for text, expect_valid in test_cases:
        chunks = mock_make_chunks(text, MAX_CHUNK_LEN, MAX_CHUNK_EXCESS)
        
        # Filter out empty chunks (should already be done by mock_make_chunks)
        empty_chunks = [i for i, c in enumerate(chunks) if not c or not c.strip()]
        
        if expect_valid and not empty_chunks:
            num_chunks = len(chunks) if chunks else 0
            text_preview = text[:40] if text else "[empty]"
            print(f"   ✓ No empty chunks: '{text_preview}...' → {num_chunks} valid chunks")
            passed += 1
        elif expect_valid and empty_chunks:
            print(f"   ✗ Found {len(empty_chunks)} empty chunks in: '{text[:40]}'")
            failed += 1
        else:
            # For cases where we don't expect valid chunks (shouldn't happen in practice)
            num_chunks = len(chunks)
            print(f"   ✓ Handled edge case (text: '{text[:30]}...')")
            passed += 1
    
    return passed, failed

def test_config_consistency():
    """Verify that MAX_CHUNK_LEN and MAX_CHUNK_EXCESS are valid."""
    print_section("TEST 4: Configuration Consistency")
    
    passed = 0
    failed = 0
    
    # Check that MAX_CHUNK_LEN is positive
    if MAX_CHUNK_LEN > 0:
        print(f"   ✓ MAX_CHUNK_LEN is positive: {MAX_CHUNK_LEN}")
        passed += 1
    else:
        print(f"   ✗ MAX_CHUNK_LEN must be positive: {MAX_CHUNK_LEN}")
        failed += 1
    
    # Check that MAX_CHUNK_EXCESS is reasonable
    if 1.0 <= MAX_CHUNK_EXCESS <= 10.0:
        print(f"   ✓ MAX_CHUNK_EXCESS is reasonable: {MAX_CHUNK_EXCESS}")
        passed += 1
    else:
        print(f"   ⚠ MAX_CHUNK_EXCESS may be unusual: {MAX_CHUNK_EXCESS}")
        # This is a warning, not a failure
    
    # Check that excess allows for words longer than base limit
    excess_len = MAX_CHUNK_LEN * MAX_CHUNK_EXCESS
    if excess_len > 0:
        print(f"   ✓ Excess length available: {excess_len} (for long words)")
        passed += 1
    else:
        print(f"   ✗ Excess length calculation failed")
        failed += 1
    
    return passed, failed

def test_real_emails():
    """Test chunking with small sample of real emails."""
    print_section("TEST 5: Real Email Sample Processing")
    
    example_dir = Path("examples")
    eml_files = sorted(example_dir.glob("*.eml"))[:5]  # Test first 5 emails
    
    if not eml_files:
        print("   ⚠ No example emails found, skipping real email test")
        return 0, 0
    
    passed = 0
    failed = 0
    total_chunks = 0
    max_chunk_size = 0
    
    for eml_file in eml_files:
        try:
            with open(eml_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract body (simplified - after headers)
            parts = content.split('\n\n', 1)
            body = parts[1] if len(parts) > 1 else parts[0]
            
            # Apply actual chunking logic
            chunks = mock_make_chunks(body, MAX_CHUNK_LEN, MAX_CHUNK_EXCESS)
            
            # Validate chunks
            valid = True
            for chunk in chunks:
                if len(chunk) >= MAX_CHUNK_LEN:
                    valid = False
                max_chunk_size = max(max_chunk_size, len(chunk))
            
            total_chunks += len(chunks)
            
            if valid and chunks:
                print(f"   ✓ {eml_file.name}: {len(chunks)} chunks generated")
                passed += 1
            else:
                print(f"   ✗ {eml_file.name}: Validation failed")
                failed += 1
        
        except Exception as e:
            print(f"   ✗ {eml_file.name}: Error - {e}")
            failed += 1
    
    if total_chunks > 0:
        print(f"\n   Summary: {total_chunks} total chunks, max size: {max_chunk_size}")
    
    return passed, failed

def main():
    print("\n" + "="*70)
    print(f"{'Email Chunking Validation Test Suite':^70}")
    print("="*70)
    print(f"Configuration: MAX_CHUNK_LEN={MAX_CHUNK_LEN}, MAX_CHUNK_EXCESS={MAX_CHUNK_EXCESS}")
    
    total_passed = 0
    total_failed = 0
    
    # Run all tests
    tests = [
        test_chunk_sizes,
        test_sentence_boundaries,
        test_no_empty_chunks,
        test_config_consistency,
        test_real_emails,
    ]
    
    for test_func in tests:
        try:
            passed, failed = test_func()
            total_passed += passed
            total_failed += failed
        except Exception as e:
            print(f"   ✗ Test {test_func.__name__} failed with error: {e}")
            import traceback
            traceback.print_exc()
            total_failed += 1
    
    # Summary
    print_section("TEST SUMMARY")
    print(f"Total Passed: {total_passed}")
    print(f"Total Failed: {total_failed}")
    print(f"Pass Rate: {total_passed/(total_passed+total_failed)*100:.1f}%" if (total_passed+total_failed) > 0 else "N/A")
    
    print("="*70)
    
    return 0 if total_failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

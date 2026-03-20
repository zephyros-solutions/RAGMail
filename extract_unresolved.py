#!/usr/bin/env python
"""Extract all unresolved people from corpus for manual alias addition."""

import sys
from pathlib import Path
from mail import Mail
from alias import alias
from collections import defaultdict
import re

def sort_alias():
    # Sort alias by key
    sorted_alias = dict(sorted(alias.items(), key=lambda x: x[0].casefold()))
    
    # Print sorted alias
    print(f"{'='*80}")
    print(f"SORTED ALIAS (Total: {len(sorted_alias)})")
    print(f"{'='*80}\n")
    
    for i, (person, variants) in enumerate(sorted_alias.items(), 1):
        sorted_alias[person] = sorted(set(variants), key=str.casefold)  # Sort variants and remove duplicates
        variants_str = ", ".join(f"'{v}'" for v in sorted_alias[person]) if sorted_alias[person] else ""
        print(f"'{person}': [{variants_str}],")

def run_tests():
    example_dir = Path("examples")
    email_files = sorted(example_dir.glob("*.eml"))

    # Collect unresolved people
    unresolved_variants = defaultdict(int)  # person -> count

    print(f"Scanning {len(email_files)} emails for unresolved people...\n")

    for i, email_file in enumerate(email_files):
        if i % 300 == 0 and i > 0:
            print(f"  Processed {i}/{len(email_files)}...")
        
        try:
            with open(email_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Extract From/To/Cc headers
            for line in lines[:50]:
                people_in_line = []
                
                if line.startswith('From:'):
                    people_in_line = [line.replace('From:', '').strip()]
                elif line.startswith(('To:', 'Cc:', 'Bcc:')):
                    recipients_str = line.split(':', 1)[1].strip()
                    people_in_line = [r.strip() for r in recipients_str.split(',') if r.strip()]
                
                # Check if each person is resolved
                for person_str in people_in_line:
                    if not person_str:
                        continue
                
                    resolved_name = Mail.norm_mailer(person_str)
                    
                    # Check if it's actually unresolved (not in alias)
                    if resolved_name not in alias and resolved_name != "(No Recipient)":
                        # Clean up the person string for display
                        display_str = person_str.replace('<', '').replace('>', '')
                        if display_str and len(display_str) > 3:  # Skip short invalid strings
                            unresolved_variants[display_str] += 1
        
        except Exception as e:
            pass

    # Sort by frequency
    sorted_unresolved = sorted(unresolved_variants.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'='*80}")
    print(f"UNRESOLVED PEOPLE (Total: {len(sorted_unresolved)})")
    print(f"{'='*80}\n")

    # Group by canonical form (extract email domain/name)
    grouped = defaultdict(list)
    for person, count in sorted_unresolved:
        # Try to extract canonical name
        clean = re.sub(r'<[^>]+>', '', person).strip()
        clean = re.sub(r'\([^)]+\)', '', clean).strip()
        
        # Use the count-weighted name as key
        grouped[clean].append((person, count))

    # Print with frequency
    for i, (person, count) in enumerate(sorted_unresolved, 1):
        print(f"{i:3d}. [{count:3d}x] {person[:70]}")
        
        # Extract email if present
        email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', person)
        if email_match:
            print(f"     Email: {email_match.group()}")

    print(f"\n{'='*80}")
    print("\nRECOMMENDED ADDITIONS TO alias.py:\n")

    # Generate Python code for adding to alias.py
    added = set()
    for person, count in sorted_unresolved[:30]:  # Top 30
        # Extract canonical name
        clean = re.sub(r'<[^>]+>', '', person).strip()
        clean = re.sub(r'\s+', ' ', clean).strip()
        clean = re.sub(r'["\']', '', clean)
        
        if clean and len(clean) > 3 and clean not in added:
            # Extract variants (email addresses)
            variants = []
            email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+', person)
            if email_match:
                variants.append(f"'{email_match.group()}'")
            
            # Also add lowercased version
            if clean != clean.lower():
                variants.append(f"'{clean.lower()}'")
            
            variants_str = ", ".join(variants) if variants else ""
            print(f"'{clean}': [{variants_str}],")
            added.add(clean)

    print(f"\n{'='*80}")

if __name__ == "__main__":
    run_tests()

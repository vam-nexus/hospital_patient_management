#!/usr/bin/env python3
"""
Test script for the Health Gateway BC website scraper function.
"""

import sys
import os

# Add the parent directory to the path so we can import rag_utils
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rag.rag_utils import load_healthbc_website, load_healthbc_website_with_delay


def test_healthbc_scraper():
    """Test the Health Gateway BC website scraper."""
    print("Testing Health Gateway BC website scraper...")
    print("=" * 50)

    # Test the basic function
    content = load_healthbc_website_with_delay()

    if content:
        print("✅ Successfully scraped content from Health Gateway BC website!")
        print(f"Content preview (first 500 characters):")
        print("-" * 30)
        print(content[:500] + "..." if len(content) > 500 else content)
        print("-" * 30)

        # Check if file was created
        if os.path.exists("data/healthbc.txt"):
            print("✅ File successfully saved to data/healthbc.txt")

            # Get file size
            file_size = os.path.getsize("data/healthbc.txt")
            print(f"File size: {file_size} bytes")
        else:
            print("❌ File was not created")
    else:
        print("❌ Failed to scrape content")


if __name__ == "__main__":
    test_healthbc_scraper()

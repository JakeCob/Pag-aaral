# CharacterTextSplitter with List of Separators - Detailed Explanation

"""
Understanding the difference between:
1. CharacterTextSplitter with single separator
2. CharacterTextSplitter with list of separators (less common)
3. RecursiveCharacterTextSplitter with list of separators (most common)
"""

from typing import List, Union


# ==============================================================================
# VERSION 1: CharacterTextSplitter with SINGLE Separator (Original)
# ==============================================================================

class CharacterTextSplitter_Single:
    """
    Original CharacterTextSplitter - uses ONE separator to split text.
    This is the standard implementation in LangChain.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, 
                 separator: str = "\n\n"):
        """
        Args:
            separator: SINGLE string to split on (e.g., "\n\n" for paragraphs)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator  # Single separator
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text using a SINGLE separator.
        
        Algorithm:
        1. Split text by separator
        2. Merge splits into chunks of appropriate size
        3. Add overlap between chunks
        """
        # Step 1: Split by the separator
        if self.separator:
            splits = text.split(self.separator)
        else:
            # Empty separator means character-by-character
            splits = list(text)
        
        # Step 2 & 3: Merge splits into chunks with overlap
        return self._merge_splits(splits)
    
    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Merge splits into chunks, respecting chunk_size and overlap."""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_with_sep = split + self.separator if self.separator else split
            split_len = len(split_with_sep)
            
            # Check if adding this split exceeds chunk_size
            if current_length + split_len > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = self.separator.join(current_chunk) if self.separator else ''.join(current_chunk)
                chunks.append(chunk_text)
                
                # Calculate overlap for next chunk
                overlap_splits = []
                overlap_length = 0
                
                # Take splits from the end until we reach desired overlap
                for i in range(len(current_chunk) - 1, -1, -1):
                    test_length = overlap_length + len(current_chunk[i] + self.separator)
                    if test_length <= self.chunk_overlap:
                        overlap_splits.insert(0, current_chunk[i])
                        overlap_length = test_length
                    else:
                        break
                
                current_chunk = overlap_splits
                current_length = overlap_length
            
            current_chunk.append(split)
            current_length += split_len
        
        # Add final chunk
        if current_chunk:
            chunk_text = self.separator.join(current_chunk) if self.separator else ''.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks


# ==============================================================================
# VERSION 2: CharacterTextSplitter with LIST of Separators (Non-Recursive)
# ==============================================================================

class CharacterTextSplitter_MultiSeparator:
    """
    CharacterTextSplitter that accepts a LIST of separators.
    Unlike RecursiveCharacterTextSplitter, this tries each separator independently
    and picks the best one (non-recursive approach).
    
    This is less common but useful to understand.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200,
                 separators: Union[str, List[str]] = "\n\n"):
        """
        Args:
            separators: Can be a single string OR a list of strings
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Convert single separator to list for uniform handling
        if isinstance(separators, str):
            self.separators = [separators]
        else:
            self.separators = separators
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text using the FIRST separator from the list that exists in text.
        This is a non-recursive approach.
        
        Algorithm:
        1. Try each separator in order
        2. Use the FIRST one that exists in the text
        3. Split by that separator
        4. Merge into chunks
        """
        # Find the first separator that exists in the text
        chosen_separator = ""
        for sep in self.separators:
            if sep in text or sep == "":
                chosen_separator = sep
                break
        
        # Split by chosen separator
        if chosen_separator:
            splits = text.split(chosen_separator)
        else:
            # No separator found, treat as single piece
            splits = [text]
        
        # Merge splits into chunks
        return self._merge_splits(splits, chosen_separator)
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge splits into chunks."""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_with_sep = split + separator if separator else split
            split_len = len(split_with_sep)
            
            if current_length + split_len > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = separator.join(current_chunk) if separator else ''.join(current_chunk)
                chunks.append(chunk_text)
                
                # Add overlap
                overlap_splits = []
                overlap_length = 0
                for i in range(len(current_chunk) - 1, -1, -1):
                    test_length = overlap_length + len(current_chunk[i] + separator)
                    if test_length <= self.chunk_overlap:
                        overlap_splits.insert(0, current_chunk[i])
                        overlap_length = test_length
                    else:
                        break
                
                current_chunk = overlap_splits
                current_length = overlap_length
            
            current_chunk.append(split)
            current_length += split_len
        
        if current_chunk:
            chunk_text = separator.join(current_chunk) if separator else ''.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks


# ==============================================================================
# VERSION 3: RecursiveCharacterTextSplitter (For Comparison)
# ==============================================================================

class RecursiveCharacterTextSplitter:
    """
    This is the RECURSIVE version that tries separators hierarchically.
    Most sophisticated and commonly used approach.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200,
                 separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """
        RECURSIVELY split text using separator hierarchy.
        
        Key difference: If a split is too large, recursively split it
        using the NEXT separator in the list.
        """
        return self._split_recursive(text, self.separators)
    
    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursive splitting logic."""
        final_chunks = []
        
        # Choose separator
        separator = separators[-1]  # Default to last (smallest) separator
        new_separators = []
        
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                separator = sep
                new_separators = separators[i + 1:]  # Remaining separators for recursion
                break
        
        # Split by chosen separator
        splits = text.split(separator) if separator else list(text)
        
        # Process each split
        good_splits = []
        for split in splits:
            if not split:
                continue
                
            if len(split) <= self.chunk_size:
                # This split is small enough
                good_splits.append(split)
            else:
                # Split is too large - need to split further
                if good_splits:
                    # First, merge accumulated good splits
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                
                # RECURSIVELY split this large piece with next separator
                if new_separators:
                    recursive_chunks = self._split_recursive(split, new_separators)
                    final_chunks.extend(recursive_chunks)
                else:
                    # No more separators, just add as-is
                    final_chunks.append(split)
        
        # Merge any remaining good splits
        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)
        
        return final_chunks
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge small splits into chunks."""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_len = len(split)
            
            if current_length + split_len > self.chunk_size and current_chunk:
                chunks.append(separator.join(current_chunk))
                
                # Add overlap
                overlap_text = separator.join(current_chunk)
                if len(overlap_text) > self.chunk_overlap:
                    overlap_text = overlap_text[-self.chunk_overlap:]
                
                current_chunk = [overlap_text] if overlap_text else []
                current_length = len(overlap_text)
            
            current_chunk.append(split)
            current_length += split_len + len(separator)
        
        if current_chunk:
            chunks.append(separator.join(current_chunk))
        
        return chunks


# ==============================================================================
# COMPARISON EXAMPLES
# ==============================================================================

def compare_all_approaches():
    """Show the differences between all three approaches."""
    
    text = """This is paragraph one.
It has multiple sentences.

This is paragraph two.
It also has multiple sentences.
And it keeps going.

This is paragraph three.
Short and sweet."""
    
    chunk_size = 60
    chunk_overlap = 10
    separators = ["\n\n", "\n", " ", ""]
    
    print("=" * 80)
    print("ORIGINAL TEXT:")
    print("=" * 80)
    print(text)
    print()
    
    # Approach 1: Single separator
    print("=" * 80)
    print("APPROACH 1: CharacterTextSplitter with SINGLE separator ('\n\n')")
    print("=" * 80)
    splitter1 = CharacterTextSplitter_Single(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n\n"
    )
    chunks1 = splitter1.split_text(text)
    for i, chunk in enumerate(chunks1, 1):
        print(f"Chunk {i} ({len(chunk)} chars):")
        print(repr(chunk))
        print()
    
    # Approach 2: Multiple separators (non-recursive)
    print("=" * 80)
    print("APPROACH 2: CharacterTextSplitter with LIST of separators (non-recursive)")
    print("Uses FIRST separator that exists: '\n\n'")
    print("=" * 80)
    splitter2 = CharacterTextSplitter_MultiSeparator(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    chunks2 = splitter2.split_text(text)
    for i, chunk in enumerate(chunks2, 1):
        print(f"Chunk {i} ({len(chunk)} chars):")
        print(repr(chunk))
        print()
    
    # Approach 3: Recursive
    print("=" * 80)
    print("APPROACH 3: RecursiveCharacterTextSplitter (RECURSIVE)")
    print("Tries separators hierarchically: ['\n\n', '\n', ' ', '']")
    print("=" * 80)
    splitter3 = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    chunks3 = splitter3.split_text(text)
    for i, chunk in enumerate(chunks3, 1):
        print(f"Chunk {i} ({len(chunk)} chars):")
        print(repr(chunk))
        print()


# ==============================================================================
# DETAILED EXPLANATION WITH VISUAL EXAMPLES
# ==============================================================================

def visual_explanation():
    """Visual explanation of how each approach works."""
    
    text = "A\n\nB B B\n\nC C C C C\n\nD"
    
    print("=" * 80)
    print("VISUAL EXPLANATION")
    print("=" * 80)
    print(f"Text: {repr(text)}")
    print(f"Chunk size: 10 characters")
    print(f"Separators: ['\\n\\n', ' ']")
    print()
    
    # Show what happens with each approach
    print("Text structure:")
    print("  A")
    print("  [\\n\\n]")
    print("  B B B")
    print("  [\\n\\n]")
    print("  C C C C C")
    print("  [\\n\\n]")
    print("  D")
    print()
    
    print("-" * 80)
    print("SINGLE SEPARATOR (separator='\\n\\n'):")
    print("-" * 80)
    print("Step 1: Split by '\\n\\n'")
    print("  → ['A', 'B B B', 'C C C C C', 'D']")
    print()
    print("Step 2: Merge into chunks (size <= 10)")
    print("  → Chunk 1: 'A\\n\\nB B B' (9 chars) ✓")
    print("  → Chunk 2: 'C C C C C' (9 chars) ✓")
    print("  → Chunk 3: 'D' (1 char) ✓")
    print()
    
    print("-" * 80)
    print("MULTI-SEPARATOR NON-RECURSIVE (separators=['\\n\\n', ' ']):")
    print("-" * 80)
    print("Step 1: Find first separator that exists")
    print("  → '\\n\\n' exists in text, so use it")
    print()
    print("Step 2: Split by '\\n\\n'")
    print("  → ['A', 'B B B', 'C C C C C', 'D']")
    print()
    print("Step 3: Merge into chunks")
    print("  → Same result as single separator!")
    print("  → Note: Did NOT recursively use ' ' separator")
    print()
    
    print("-" * 80)
    print("RECURSIVE (separators=['\\n\\n', ' ']):")
    print("-" * 80)
    print("Step 1: Split by '\\n\\n'")
    print("  → ['A', 'B B B', 'C C C C C', 'D']")
    print()
    print("Step 2: Check each split")
    print("  → 'A' (1 char) ✓ small enough")
    print("  → 'B B B' (5 chars) ✓ small enough")
    print("  → 'C C C C C' (9 chars) ✓ small enough")
    print("  → 'D' (1 char) ✓ small enough")
    print()
    print("Step 3: Merge good splits")
    print("  → Chunk 1: 'A\\n\\nB B B' (9 chars) ✓")
    print("  → Chunk 2: 'C C C C C' (9 chars) ✓")
    print("  → Chunk 3: 'D' (1 char) ✓")
    print()
    print("In this case, no recursion needed since all splits fit!")
    print()
    
    # Now show case where recursion IS needed
    text2 = "A\n\nB B B B B B B B B B"
    print("=" * 80)
    print("CASE WHERE RECURSION IS NEEDED:")
    print("=" * 80)
    print(f"Text: {repr(text2)}")
    print(f"Chunk size: 10 characters")
    print()
    
    print("Non-Recursive approach:")
    print("  1. Split by '\\n\\n' → ['A', 'B B B B B B B B B B']")
    print("  2. Second split is 21 chars (too large!)")
    print("  3. Can't split further → returns ['A', 'B B B B B B B B B B']")
    print("  4. ❌ Chunk 2 violates chunk_size limit!")
    print()
    
    print("Recursive approach:")
    print("  1. Split by '\\n\\n' → ['A', 'B B B B B B B B B B']")
    print("  2. 'A' (1 char) ✓ good")
    print("  3. 'B B B B B B B B B B' (21 chars) ✗ too large")
    print("  4. RECURSE: split 'B B B B B B B B B B' by ' '")
    print("     → ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B']")
    print("  5. Merge these B's into chunks of size 10")
    print("     → 'B B B B B' (9 chars)")
    print("     → 'B B B B B' (9 chars)")
    print("  6. Final result:")
    print("     ✓ Chunk 1: 'A'")
    print("     ✓ Chunk 2: 'B B B B B'")
    print("     ✓ Chunk 3: 'B B B B B'")
    print()


# ==============================================================================
# KEY TAKEAWAYS
# ==============================================================================

def print_key_takeaways():
    """Summary of key differences."""
    print("=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print()
    print("1. CharacterTextSplitter with SINGLE separator:")
    print("   - Most basic approach")
    print("   - Uses ONE separator to split, then merges")
    print("   - Fast and simple")
    print("   - Problem: Can't handle splits that are too large")
    print()
    print("2. CharacterTextSplitter with LIST of separators (non-recursive):")
    print("   - Tries separators in order")
    print("   - Uses FIRST one that exists")
    print("   - Does NOT recursively try other separators")
    print("   - Same problem: Can't handle large splits")
    print()
    print("3. RecursiveCharacterTextSplitter:")
    print("   - Tries separators hierarchically")
    print("   - If a split is too large, RECURSES with next separator")
    print("   - Most sophisticated and flexible")
    print("   - Handles all edge cases")
    print("   - This is the RECOMMENDED approach")
    print()
    print("INTERVIEW TIP:")
    print("When asked about CharacterTextSplitter with list of separators,")
    print("clarify whether they mean:")
    print("  a) Non-recursive (pick first separator)")
    print("  b) Recursive (RecursiveCharacterTextSplitter)")
    print()
    print("In practice, RecursiveCharacterTextSplitter is what you want!")
    print("=" * 80)


# ==============================================================================
# RUN ALL EXAMPLES
# ==============================================================================

if __name__ == "__main__":
    print("\n")
    visual_explanation()
    print("\n\n")
    compare_all_approaches()
    print("\n\n")
    print_key_takeaways()
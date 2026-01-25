"""
Library of Congress Call Number Order Validator

This script validates if books are shelved in proper LC classification order
and identifies which books need to be swapped to correct the order.

LC Call Number Format:
- Class letters (1-3): A, QA, PS, etc.
- Class number: 123, 76.73, etc.
- Cutter number: .M87, .A123, etc.
- Additional elements: year, volume, etc.

Example: PS 3551 .L3 1998
"""

import json
import re
from typing import List, Tuple, Dict, Optional

class LCCallNumber:
    """Represents a Library of Congress call number"""
    
    def __init__(self, call_number_str: str):
        self.original = call_number_str.strip().upper()
        self.class_letters = ""
        self.class_number = 0.0
        self.class_decimal = ""
        self.cutter = ""
        self.year = None
        self.additional = []
        self.parse_errors = []
        
        self._parse()
    
    def _parse(self):
        """Parse the call number into components"""
        if not self.original or self.original == "UNKNOWN":
            self.parse_errors.append("Empty or unknown call number")
            return
        
        # Clean up the string
        text = self.original.strip()

        tokens = text.split()

        # Find the first token that looks like LC class letters AND is followed by a number token
        start_idx = None
        for i in range(len(tokens) - 1):
            t0 = tokens[i]
            t1 = tokens[i + 1]
            if re.fullmatch(r"[A-Z]{1,3}", t0) and re.match(r"^\d", t1):
                start_idx = i
                break

        if start_idx is None:
            self.parse_errors.append("Could not find LC class + number pattern")
            return

        # Rebuild text starting from LC portion only (drops prefixes like MATH/PHYS/ENGR)
        text = " ".join(tokens[start_idx:])
        
        # Extract class letters (1-3 letters at start)
        match = re.match(r'^([A-Z]{1,3})', text)
        if match:
            self.class_letters = match.group(1)
            text = text[len(self.class_letters):].strip()
        else:
            self.parse_errors.append("No class letters found")
            return
        
        # Extract class number (integer and optional decimal)
        match = re.match(r'^(\d+)\.?(\d*)', text)
        if match:
            integer_part = match.group(1)
            decimal_part = match.group(2)
            
            self.class_number = float(integer_part)
            if decimal_part:
                self.class_decimal = decimal_part
                self.class_number += float("0." + decimal_part)
            
            text = text[match.end():].strip()
        else:
            self.parse_errors.append("No class number found")
        
        # Extract cutter number (typically .L123 or .A12b3)
        match = re.search(r'\.([A-Z]\d+[A-Z]?\d*)', text)
        if match:
            self.cutter = match.group(1)
            text = text.replace(match.group(0), '').strip()
        
        # Extract year (4 digits)
        match = re.search(r'\b(19\d{2}|20\d{2})\b', text)
        if match:
            self.year = int(match.group(1))
            text = text.replace(match.group(0), '').strip()
        
        # Store remaining components
        if text:
            self.additional = text.split()
    
    def __lt__(self, other):
        """Compare two call numbers for sorting"""
        if not isinstance(other, LCCallNumber):
            return NotImplemented
        
        # Compare class letters alphabetically
        if self.class_letters != other.class_letters:
            return self.class_letters < other.class_letters
        
        # Compare class numbers numerically
        if abs(self.class_number - other.class_number) > 0.001:
            return self.class_number < other.class_number
        
        # Compare cutter numbers
        if self.cutter != other.cutter:
            return self.cutter < other.cutter
        
        # Compare years
        if self.year != other.year:
            if self.year is None:
                return True
            if other.year is None:
                return False
            return self.year < other.year
        
        # If all else equal, maintain original order
        return False
    
    def __repr__(self):
        return f"LCCallNumber('{self.original}')"
    
    def __str__(self):
        return self.original
    
    def is_valid(self):
        """Check if the call number was parsed successfully"""
        return len(self.parse_errors) == 0 and self.class_letters != ""

def load_shelf_order(json_file: str) -> Dict:
    """Load shelf order from JSON file"""
    with open(json_file, 'r') as f:
        return json.load(f)

def validate_order(shelf_data: Dict) -> Tuple[bool, List[Dict]]:
    """
    Validate if books are in correct LC order.
    Returns (is_correct, issues_list)
    """
    books = shelf_data["books"]
    
    if len(books) < 2:
        return True, []
    
    # Parse all call numbers
    parsed_books = []
    for book in books:
        lc_num = LCCallNumber(book["call_number"])
        parsed_books.append({
            "position": book["position"],
            "call_number": book["call_number"],
            "lc_object": lc_num,
            "valid": lc_num.is_valid(),
            "errors": lc_num.parse_errors
        })
    
    # Check for parsing errors
    invalid_books = [b for b in parsed_books if not b["valid"]]
    if invalid_books:
        print("\n⚠ WARNING: Some call numbers could not be parsed:")
        for book in invalid_books:
            print(f"  Position {book['position']}: {book['call_number']}")
            print(f"    Errors: {', '.join(book['errors'])}")
        print()
    
    # Check order
    issues = []
    for i in range(len(parsed_books) - 1):
        current = parsed_books[i]
        next_book = parsed_books[i + 1]
        
        if not current["valid"] or not next_book["valid"]:
            continue
        
        # Check if current should come after next
        if current["lc_object"] > next_book["lc_object"]:
            issues.append({
                "type": "out_of_order",
                "position_1": current["position"],
                "call_number_1": current["call_number"],
                "position_2": next_book["position"],
                "call_number_2": next_book["call_number"],
                "description": f"Book at position {current['position']} ({current['call_number']}) should come AFTER position {next_book['position']} ({next_book['call_number']})"
            })
    
    return len(issues) == 0, issues

def find_swaps_to_fix(shelf_data: Dict) -> List[Tuple[int, int]]:
    """
    Find the minimum swaps needed to fix the order.
    Returns list of (position1, position2) tuples indicating books to swap.
    """
    books = shelf_data["books"]
    
    # Parse and sort
    parsed_books = []
    for book in books:
        lc_num = LCCallNumber(book["call_number"])
        if lc_num.is_valid():
            parsed_books.append({
                "original_position": book["position"],
                "call_number": book["call_number"],
                "lc_object": lc_num
            })
    
    # Get correct order
    sorted_books = sorted(parsed_books, key=lambda x: x["lc_object"])
    
    # Find swaps needed
    swaps = []
    current_positions = {b["original_position"]: i for i, b in enumerate(parsed_books)}
    
    for target_idx, target_book in enumerate(sorted_books):
        current_idx = current_positions[target_book["original_position"]]
        
        if current_idx != target_idx:
            # Find what's currently in target position
            current_at_target = None
            for pos, idx in current_positions.items():
                if idx == target_idx:
                    current_at_target = pos
                    break
            
            if current_at_target:
                swaps.append((
                    target_book["original_position"],
                    current_at_target,
                    target_book["call_number"],
                    next(b["call_number"] for b in parsed_books if b["original_position"] == current_at_target)
                ))
                
                # Update positions
                current_positions[target_book["original_position"]] = target_idx
                current_positions[current_at_target] = current_idx
    
    return swaps

def print_report(shelf_data: Dict, is_correct: bool, issues: List[Dict], swaps: List[Tuple]):
    """Print a detailed validation report"""
    print("\n" + "="*70)
    print("LIBRARY OF CONGRESS CALL NUMBER ORDER VALIDATION REPORT")
    print("="*70)
    
    print(f"\nImage: {shelf_data['image_file']}")
    print(f"Total Books: {shelf_data['num_books']}")
    
    print("\n" + "-"*70)
    print("CURRENT SHELF ORDER (Left to Right):")
    print("-"*70)
    for book in shelf_data["books"]:
        print(f"{book['position']}. {book['call_number']}")
    
    print("\n" + "-"*70)
    if is_correct:
        print("✓ RESULT: Books are in CORRECT Library of Congress order!")
    else:
        print("✗ RESULT: Books are NOT in correct order")
        print(f"   Found {len(issues)} ordering issue(s)")
    print("-"*70)
    
    if not is_correct and issues:
        print("\nISSUES DETECTED:")
        for i, issue in enumerate(issues, 1):
            print(f"\n  Issue {i}:")
            print(f"    {issue['description']}")
            print(f"    Position {issue['position_1']}: {issue['call_number_1']}")
            print(f"    Position {issue['position_2']}: {issue['call_number_2']}")
    
    if swaps:
        print("\n" + "-"*70)
        print("RECOMMENDED SWAPS TO FIX ORDER:")
        print("-"*70)
        for i, (pos1, pos2, call1, call2) in enumerate(swaps, 1):
            print(f"\nSwap {i}:")
            print(f"  Swap position {pos1} ({call1})")
            print(f"  with position {pos2} ({call2})")
    
    # Show correct order
    books_with_lc = []
    for book in shelf_data["books"]:
        lc_num = LCCallNumber(book["call_number"])
        if lc_num.is_valid():
            books_with_lc.append((book["position"], book["call_number"], lc_num))
    
    books_with_lc.sort(key=lambda x: x[2])
    
    print("\n" + "-"*70)
    print("CORRECT ORDER (Library of Congress):")
    print("-"*70)
    for i, (orig_pos, call_num, lc_obj) in enumerate(books_with_lc, 1):
        status = "✓" if orig_pos == i else f"✗ (currently at position {orig_pos})"
        print(f"{i}. {call_num} {status}")
    
    print("\n" + "="*70)

def main():
    import sys
    
    # Get JSON file from command line or use default
    json_file = sys.argv[1] if len(sys.argv) > 1 else "shelf_order.json"
    
    try:
        # Load shelf order
        print(f"Loading shelf order from: {json_file}")
        shelf_data = load_shelf_order(json_file)
        
        # Validate order
        is_correct, issues = validate_order(shelf_data)
        
        # Find swaps needed
        swaps = find_swaps_to_fix(shelf_data) if not is_correct else []
        
        # Print report
        print_report(shelf_data, is_correct, issues, swaps)
        
        # Save report to file
        report_file = json_file.replace(".json", "_validation_report.txt")
        with open(report_file, 'w') as f:
            import sys
            old_stdout = sys.stdout
            sys.stdout = f
            print_report(shelf_data, is_correct, issues, swaps)
            sys.stdout = old_stdout
        
        print(f"\nReport saved to: {report_file}")
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{json_file}'")
        print("Usage: python lc_validator.py [shelf_order.json]")
    except json.JSONDecodeError:
        print(f"Error: '{json_file}' is not a valid JSON file")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
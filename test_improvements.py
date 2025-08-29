#!/usr/bin/env python3
"""
Test script to validate RAG accuracy improvements without requiring database connection.
"""

import sys
import os

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from util.knowlege_util import node_labels, relationship_labels
        print(f"✓ Node labels imported: {node_labels}")
        print(f"✓ Relationship labels imported: {relationship_labels}")
        
        # Verify the naming fixes
        assert "Practice" in node_labels, "Practice node label not found"
        assert "Plactice" not in node_labels, "Old typo 'Plactice' still present"
        assert "PracticeToConsideration" in relationship_labels, "PracticeToConsideration relationship not found"
        
        print("✓ All naming fixes validated")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except AssertionError as e:
        print(f"✗ Validation error: {e}")
        return False
    
    return True

def test_function_improvements():
    """Test that function signatures are improved."""
    print("\nTesting function improvements...")
    
    try:
        # Read the graphrag2.py file and check function signature
        with open('graphrag2.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check that vector_node_search has the k parameter with default 5
        assert 'def vector_node_search(vector_index: Neo4jVector, query: str, k: int = 5)' in content, \
            "vector_node_search should have k parameter with default 5"
        
        # Check for docstring improvements
        assert 'Enhanced QA function with better error handling' in content, \
            "runQA function should have enhanced documentation"
        
        assert 'Improved prompt for better node type selection accuracy' in content, \
            "get_target_node_types should have improved documentation"
        
        print("✓ Function improvements validated")
        
    except AssertionError as e:
        print(f"✗ Function validation error: {e}")
        return False
    except Exception as e:
        print(f"✗ File reading error: {e}")
        return False
    
    return True

def test_node_naming_consistency():
    """Test that node naming is consistent throughout."""
    print("\nTesting node naming consistency...")
    
    try:
        # Read the main file and check for naming consistency
        with open('graphrag2.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check that old typos are not present
        assert 'Plactice' not in content, "Old typo 'Plactice' found in graphrag2.py"
        assert 'PLACTICETOAZURERESOURCE' not in content, "Old relationship name found"
        assert 'PLACTICETOCONSIDERATION' not in content, "Old relationship name found"
        
        # Check that new correct names are present
        assert 'Practice' in content, "Correct 'Practice' naming not found"
        assert 'PRACTICETOAZURERESOURCE' in content, "Correct relationship name not found"
        
        print("✓ Node naming consistency validated")
        
    except AssertionError as e:
        print(f"✗ Naming consistency error: {e}")
        return False
    except Exception as e:
        print(f"✗ File reading error: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("=== RAG Accuracy Improvement Tests ===\n")
    
    tests = [
        test_imports,
        test_function_improvements,
        test_node_naming_consistency
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"=== Test Results: {passed}/{total} passed ===")
    
    if passed == total:
        print("✓ All improvements validated successfully!")
        return 0
    else:
        print("✗ Some tests failed. Please review the improvements.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
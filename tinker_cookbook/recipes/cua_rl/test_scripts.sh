#!/bin/bash
# Quick test script to verify train.sh and benchmark.sh work correctly
# This doesn't actually run training/benchmark, just validates the scripts

set -e

echo "============================================"
echo "Script Validation Test"
echo "============================================"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Testing train.sh..."
echo "--------------------"

# Test train.sh help
if ! "$SCRIPT_DIR/train.sh" --help > /dev/null 2>&1; then
    echo "❌ train.sh --help failed"
    exit 1
fi
echo "✓ train.sh --help works"

# Test train.sh with invalid option (should fail gracefully)
if "$SCRIPT_DIR/train.sh" --invalid-option > /dev/null 2>&1; then
    echo "❌ train.sh should reject invalid options"
    exit 1
fi
echo "✓ train.sh rejects invalid options"

echo ""
echo "Testing benchmark.sh..."
echo "--------------------"

# Test benchmark.sh help
if ! "$SCRIPT_DIR/benchmark.sh" --help > /dev/null 2>&1; then
    echo "❌ benchmark.sh --help failed"
    exit 1
fi
echo "✓ benchmark.sh --help works"

# Test benchmark.sh with invalid option (should fail gracefully)
if "$SCRIPT_DIR/benchmark.sh" --invalid-option > /dev/null 2>&1; then
    echo "❌ benchmark.sh should reject invalid options"
    exit 1
fi
echo "✓ benchmark.sh rejects invalid options"

echo ""
echo "Testing Python modules..."
echo "--------------------"

# Test if train.py exists
if [ ! -f "$SCRIPT_DIR/train.py" ]; then
    echo "❌ train.py not found"
    exit 1
fi
echo "✓ train.py exists"

# Test if benchmark.py exists
if [ ! -f "$SCRIPT_DIR/benchmark.py" ]; then
    echo "❌ benchmark.py not found"
    exit 1
fi
echo "✓ benchmark.py exists"

# Test if Python modules can be imported (syntax check)
if ! python3 -c "import ast; ast.parse(open('$SCRIPT_DIR/benchmark.py').read())" 2>&1; then
    echo "❌ benchmark.py has syntax errors"
    exit 1
fi
echo "✓ benchmark.py syntax is valid"

echo ""
echo "============================================"
echo "All tests passed! ✓"
echo "============================================"
echo ""
echo "The scripts are ready to use. Examples:"
echo ""
echo "  # Show help for training"
echo "  ./train.sh --help"
echo ""
echo "  # Show help for benchmark"
echo "  ./benchmark.sh --help"
echo ""
echo "  # Run training (requires API keys)"
echo "  export GBOX_API_KEY=your_api_key"
echo "  ./train.sh"
echo ""
echo "  # Run benchmark (requires API keys)"
echo "  ./benchmark.sh"
echo ""


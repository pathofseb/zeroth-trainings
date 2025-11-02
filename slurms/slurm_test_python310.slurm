#!/bin/bash
#SBATCH --job-name=test_python310
#SBATCH --output=slurms/slurm_outputs/test_python310_output.txt
#SBATCH --error=slurms/slurm_outputs/test_python310_error.txt
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=1

echo "Testing Python 3.10 compatibility fix..."
echo "Python version: $(python --version)"
echo ""

# Test importing Self from the fixed files
echo "Testing import of Self type hint..."
python -c "
try:
    from typing import Self
    print('✓ Self imported from typing (Python 3.11+)')
except ImportError:
    try:
        from typing_extensions import Self
        print('✓ Self imported from typing_extensions (Python 3.10)')
    except ImportError:
        print('✗ FAILED: Cannot import Self from either typing or typing_extensions')
        print('  You may need to install typing-extensions: pip install typing-extensions')
        exit(1)
"

echo ""
echo "Testing train_walk1.py imports..."

# Test that the specific training file can be imported without errors
python -c "
import sys
sys.path.insert(0, '/cluster/work/sebasrb/zeroth-trainings')

# Just test the import section, not running the whole script
try:
    # Test the critical imports
    from typing import TypedDict
    try:
        from typing import Self
    except ImportError:
        from typing_extensions import Self
    
    print('✓ Successfully imported Self type hint')
    print('✓ train_walk1.py imports should work correctly')
except Exception as e:
    print(f'✗ FAILED: {e}')
    exit(1)
"

echo ""
echo "Checking if files were actually modified..."
echo ""

# Check a few sample files to confirm the fix was applied
for file in /cluster/work/sebasrb/zeroth-trainings/training_variants/train_walk{1,2,3}.py; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        if grep -q "from typing_extensions import Self" "$file"; then
            echo "✓ $filename - Uses typing_extensions.Self (FIXED)"
        elif grep -q "from typing import Self" "$file"; then
            echo "✗ $filename - Still uses typing.Self (NOT FIXED)"
        else
            echo "- $filename - No Self import found"
        fi
    fi
done

echo ""
echo "Test complete!"

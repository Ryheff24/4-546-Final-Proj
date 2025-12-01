#!/bin/bash
# Set library path for libfreenect2
export DYLD_LIBRARY_PATH=/Users/ryanheffernan/Documents/Buffalo/CSE446/CSE-4-546-Final-Project-Team-49/libfreenect2/build/lib:$DYLD_LIBRARY_PATH

# Run the visual.py script
/Users/ryanheffernan/Documents/Buffalo/CSE446/CSE-4-546-Final-Project-Team-49/.venv/bin/python /Users/ryanheffernan/Documents/Buffalo/CSE446/CSE-4-546-Final-Project-Team-49/visual.py

echo ""
echo "Done! Check for output.pcd and output_big.pcd files"
cd /Users/ryanheffernan/Documents/Buffalo/CSE446/CSE-4-546-Final-Project-Team-49
ls -lh output*.pcd 2>/dev/null || echo "No PCD files created"

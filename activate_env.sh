#!/bin/bash
# Activate the research environment
source research_env/bin/activate
echo "✅ Research environment is now active. Using:"
python --version
pip --version
#!/bin/bash
# ARIA Startup Script — run with: ./start_aria.sh
cd ~/Desktop/Projects/aria-career-assistant

export ARIA_DB_TYPE=mongodb
export ARIA_MONGO_URI=mongodb://localhost:27017
export ARIA_MONGO_DB=aria_career
export ARIA_EMAIL_USER=${ARIA_EMAIL_USER:-your@gmail.com}
export ARIA_EMAIL_TO=${ARIA_EMAIL_TO:-your@gmail.com}

# Load app password from secure file
if [ -f ~/.aria_secrets ]; then
    source ~/.aria_secrets
fi

echo ""
echo "  ARIA starting with:"
echo "  DB:    $ARIA_DB_TYPE ($ARIA_MONGO_DB)"
echo "  Email: $ARIA_EMAIL_USER"
echo "  Pass:  $(echo -n $ARIA_EMAIL_PASS | wc -c | tr -d ' ') chars loaded"
echo ""

python aria_refactored.py

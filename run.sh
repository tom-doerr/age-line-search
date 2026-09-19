#!/bin/sh
# System python on purpose: it has the GB10 (cu130) torch build.
cd "$(dirname "$0")" && exec python3 -m streamlit run app.py --server.port "${PORT:-8540}" --server.headless true

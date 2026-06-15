#!/usr/bin/env bash
set -e
# Run the project's Streamlit app using the virtualenv binary if present
if [ -x ".venv/bin/streamlit" ]; then
  .venv/bin/streamlit run app.py
else
  # fallback to system streamlit
  streamlit run app.py
fi

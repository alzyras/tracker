#!/bin/bash
# Script to stop the Streamlit app

echo "Stopping Streamlit app..."
pkill -f "streamlit run streamlit_app/streamlit_configurator.py"
echo "Streamlit app stopped."
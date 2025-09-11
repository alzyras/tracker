# UV App Streamlit Configurator

A beautiful and dynamic configuration interface for the UV App tracking system.

## Features

- **Plugin Management**: Enable/disable plugins with a simple toggle
- **Dynamic Settings**: Plugin-specific configuration options that adapt to each plugin's requirements
- **UI Customization**: Full control over colors, fonts, sizes, and other visual elements
- **Display Options**: Choose between Streamlit, OpenCV window, or no display
- **Configuration Persistence**: Save and load your settings

## How It Works

The Streamlit configurator saves your settings to a `uv_app_config.json` file in the project root directory. The main UV App automatically loads these settings when it starts, applying your configuration to the tracking system.

## Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure you have the main UV App installed in development mode:
   ```bash
   uv pip install -e .
   ```

## Usage

Run the Streamlit app:
```bash
streamlit run streamlit_configurator.py
```

The configurator will be available at `http://localhost:8501` by default.

Configure your plugins and UI settings, then click "Save Configuration". The settings will be automatically applied when you run the main UV App.

## Configuration

The app automatically saves your configuration to `uv_app_config.json` in the project root directory. You can also manually edit this file if needed.
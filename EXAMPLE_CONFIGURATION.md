# Example: Using the Streamlit Configurator

This example shows how to use the Streamlit configurator to set up your tracking system.

## Step 1: Configure Plugins with Streamlit

1. Run the Streamlit configurator:
   ```bash
   cd streamlit_app
   streamlit run streamlit_configurator.py
   ```

2. In your browser, go to `http://localhost:8501`

3. Enable the plugins you want to use:
   - Enable "Api Emotion Plugin" for emotion detection
   - Enable "Smolvlm Activity Plugin" for activity recognition
   - Adjust update intervals as needed
   - Configure API URLs for external services

4. Customize the UI:
   - Choose "OpenCV Window" for display mode
   - Set box colors and font sizes
   - Adjust border widths

5. Click "Save Configuration"

## Step 2: Run the Tracking Application

The configuration you saved will automatically be applied when you run the main application:

```bash
uv run python uv_app/app.py
```

The app will:
- Load your plugin configuration
- Register only the enabled plugins
- Pass your configured parameters to each plugin
- Display results according to your UI settings

## Example Configuration

Here's what a typical configuration might look like in `uv_app_config.json`:

```json
{
  "plugin_settings": {
    "api_emotion": {
      "enabled": true,
      "update_interval_ms": 300,
      "api_url": "http://localhost:8500"
    },
    "smolvlm_activity": {
      "enabled": true,
      "update_interval_ms": 5000,
      "api_url": "http://localhost:9000/describe"
    }
  },
  "ui_settings": {
    "display_mode": "opencv",
    "box_color": "#00FF00",
    "font_size": 14,
    "border_width": 2
  }
}
```

This configuration would:
- Enable emotion detection using an external API service
- Enable activity recognition using the SmolVLM service
- Display results in an OpenCV window
- Use green boxes with a border width of 2 pixels
- Use font size 14 for labels
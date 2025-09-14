import uv_app
import streamlit as st
import sys
import os
import inspect
import json
from pathlib import Path
from typing import Dict, Any

# Add the parent directory to sys.path to import uv_app modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from uv_app.plugins import PLUGIN_REGISTRY, PluginManager
from uv_app.config import PLUGIN_CONFIG
from uv_app.plugins.base import BasePlugin
from config_manager import config_manager

# Initialize session state
if 'plugin_manager' not in st.session_state:
    st.session_state.plugin_manager = PluginManager()
    
# Load configuration
config = config_manager.load_config()

if 'plugin_settings' not in st.session_state:
    st.session_state.plugin_settings = config.get("plugin_settings", {})

if 'ui_settings' not in st.session_state:
    st.session_state.ui_settings = config.get("ui_settings", {
        'display_mode': 'streamlit',  # 'streamlit', 'opencv', 'none'
        'box_color': '#00FF00',
        'font_size': 12,
        'font_color': '#FFFFFF',
        'border_width': 2
    })

# Custom CSS for styling
st.markdown(f"""
<style>
    .stApp {{
        background-color: #0E1117;
        color: #FAFAFA;
    }}
    .stSlider [data-baseweb=slider] {{
        color: #FF4B4B;
    }}
    .stSelectbox [data-baseweb=select] {{
        background-color: #262730;
        border-color: #4A4A4A;
    }}
    .stButton>button {{
        background-color: #FF4B4B;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
    }}
    .stButton>button:hover {{
        background-color: #FF6B6B;
    }}
    .plugin-card {{
        background-color: #1E1E1E;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
        border: 1px solid #333;
    }}
    .plugin-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }}
    .plugin-title {{
        font-size: 18px;
        font-weight: bold;
        color: #FAFAFA;
    }}
    .plugin-description {{
        font-size: 14px;
        color: #CCCCCC;
        margin-bottom: 12px;
    }}
    .settings-section {{
        background-color: #262730;
        border-radius: 8px;
        padding: 16px;
        margin-top: 24px;
    }}
    .section-title {{
        font-size: 20px;
        font-weight: bold;
        color: #FAFAFA;
        margin-bottom: 16px;
    }}
</style>
""", unsafe_allow_html=True)

# App title
st.title("🎯 UV App Plugin Configurator")

# Function to get plugin parameters
def get_plugin_parameters(plugin_class) -> Dict[str, Any]:
    """Extract plugin parameters from the __init__ method."""
    try:
        # Get the __init__ method signature
        sig = inspect.signature(plugin_class.__init__)
        params = {}
        
        # Skip 'self' parameter
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
                
            # Get default value or None
            default = param.default if param.default != inspect.Parameter.empty else None
            
            # Get type annotation if available
            annotation = param.annotation if param.annotation != inspect.Parameter.empty else None
            
            params[name] = {
                'default': default,
                'annotation': annotation,
                'kind': param.kind
            }
            
        return params
    except Exception as e:
        st.warning(f"Could not extract parameters for {plugin_class.__name__}: {e}")
        return {}

# Function to render parameter controls
def render_parameter_control(param_name: str, param_info: Dict[str, Any], plugin_name: str):
    """Render appropriate UI control for a parameter based on its type and default value."""
    
    # Create unique key for the widget
    widget_key = f"{plugin_name}_{param_name}"
    
    # Get current value from session state or use default
    current_value = st.session_state.plugin_settings.get(plugin_name, {}).get(param_name, param_info['default'])
    
    # Render appropriate control based on parameter type and name
    if param_name == 'update_interval_ms':
        return st.slider(
            "Update Interval (ms)",
            min_value=100,
            max_value=10000,
            value=current_value or 1000,
            key=widget_key
        )
    elif param_name == 'api_url':
        return st.text_input(
            "API URL",
            value=current_value or "",
            key=widget_key
        )
    elif param_name == 'api_key':
        return st.text_input(
            "API Key",
            value=current_value or "",
            type="password",
            key=widget_key
        )
    elif param_name.endswith('_url'):
        return st.text_input(
            param_name.replace('_', ' ').title(),
            value=current_value or "",
            key=widget_key
        )
    elif isinstance(current_value, bool):
        return st.checkbox(
            param_name.replace('_', ' ').title(),
            value=current_value,
            key=widget_key
        )
    elif isinstance(current_value, int):
        # Check if it's likely a small integer (like a threshold)
        if current_value < 100:
            return st.number_input(
                param_name.replace('_', ' ').title(),
                value=current_value,
                step=1,
                key=widget_key
            )
        else:
            # Likely a larger value like milliseconds
            return st.number_input(
                param_name.replace('_', ' ').title(),
                value=current_value,
                step=100,
                key=widget_key
            )
    elif isinstance(current_value, float):
        return st.number_input(
            param_name.replace('_', ' ').title(),
            value=current_value,
            step=0.1,
            key=widget_key
        )
    else:
        # Default to text input for strings or unknown types
        return st.text_input(
            param_name.replace('_', ' ').title(),
            value=str(current_value) if current_value is not None else "",
            key=widget_key
        )

# Sidebar for display settings
with st.sidebar:
    st.header("🖥️ Display Settings")
    
    # Display mode with better descriptions
    display_mode = st.selectbox(
        "Output Display Mode",
        options=[
            ("Streamlit", "Display tracking results in Streamlit interface"),
            ("OpenCV Window", "Display tracking results in OpenCV window"),
            ("None", "Run tracking without display")
        ],
        format_func=lambda x: x[0],
        index=[("Streamlit", ""), ("OpenCV Window", ""), ("None", "")].index(
            ({"streamlit": "Streamlit", "opencv": "OpenCV Window", "none": "None"}.get(
                st.session_state.ui_settings['display_mode'], "Streamlit"
            ), "")
        ),
        help="Choose how you want to view the tracking results"
    )
    st.session_state.ui_settings['display_mode'] = display_mode[0].lower().replace(" ", "_")
    
    # Show description of selected mode
    st.info(display_mode[1])
    
    st.markdown("---")
    
    st.subheader("🎥 Video Settings")
    frame_width = st.number_input(
        "Frame Width", 
        min_value=320, 
        max_value=1920, 
        value=st.session_state.ui_settings.get('frame_width', 640),
        key="frame_width_input"
    )
    frame_height = st.number_input(
        "Frame Height", 
        min_value=240, 
        max_value=1080, 
        value=st.session_state.ui_settings.get('frame_height', 480),
        key="frame_height_input"
    )
    
    # Update session state for video settings
    st.session_state.ui_settings['frame_width'] = frame_width
    st.session_state.ui_settings['frame_height'] = frame_height
    
    st.markdown("---")
    
    st.subheader("📊 Preview Settings")
    show_preview = st.checkbox(
        "Show Live Preview",
        value=st.session_state.ui_settings.get('show_preview', True),
        help="Display a live preview of the tracking in the Streamlit app"
    )
    preview_interval = st.slider(
        "Preview Update Interval (ms)",
        min_value=50,
        max_value=1000,
        value=st.session_state.ui_settings.get('preview_interval', 100),
        help="How often to update the preview (lower = more frequent updates)"
    )
    
    # Update session state for preview settings
    st.session_state.ui_settings['show_preview'] = show_preview
    st.session_state.ui_settings['preview_interval'] = preview_interval

# Main content area
st.header("🔌 Plugin Configuration")

# Plugin selection and configuration
for plugin_name, plugin_class in PLUGIN_REGISTRY.items():
    # Get current plugin status from config
    config_key = f"{plugin_name}_enabled"
    is_enabled = PLUGIN_CONFIG.get(config_key, False)
    
    # Plugin card
    with st.container():
        st.markdown('<div class="plugin-card">', unsafe_allow_html=True)
        
        # Plugin header with name and toggle
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f'<div class="plugin-title">{plugin_name.title()} Plugin</div>', unsafe_allow_html=True)
        with col2:
            enabled = st.checkbox("Enable", value=is_enabled, key=f"enable_{plugin_name}")
        
        # Plugin description
        st.markdown(f'<div class="plugin-description">{plugin_class.__doc__ or "No description available."}</div>', unsafe_allow_html=True)
        
        # Plugin-specific settings
        if enabled:
            with st.expander("⚙️ Settings"):
                # Get plugin parameters
                plugin_params = get_plugin_parameters(plugin_class)
                
                # Filter out parameters that are not typically configurable by users
                # (like 'name' which is usually set automatically)
                filtered_params = {k: v for k, v in plugin_params.items() if k not in ['name']}
                
                # Render controls for each parameter
                for param_name, param_info in filtered_params.items():
                    # Skip 'self' parameter if it somehow got through
                    if param_name == 'self':
                        continue
                    
                    # Get or create plugin settings dict
                    if plugin_name not in st.session_state.plugin_settings:
                        st.session_state.plugin_settings[plugin_name] = {}
                    
                    # Render the control and store the value
                    value = render_parameter_control(param_name, param_info, plugin_name)
                    st.session_state.plugin_settings[plugin_name][param_name] = value
                
                # If no parameters found, show a message
                if not filtered_params:
                    st.info("This plugin has no configurable parameters.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Store enabled status
        st.session_state.plugin_settings[plugin_name] = st.session_state.plugin_settings.get(plugin_name, {})
        st.session_state.plugin_settings[plugin_name]['enabled'] = enabled

# UI Settings Section
st.markdown('<div class="settings-section">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🎨 UI Settings</div>', unsafe_allow_html=True)

# Create tabs for different UI elements
ui_tabs = st.tabs(["Bounding Box", "Text", "Advanced"])

with ui_tabs[0]:
    st.subheader("Bounding Box Settings")
    box_color = st.color_picker("Box Color", st.session_state.ui_settings['box_color'], key="box_color_picker")
    border_width = st.slider("Border Width", 1, 10, st.session_state.ui_settings['border_width'], key="border_width_slider")
    box_style = st.selectbox(
        "Box Style",
        ["Solid", "Dashed", "Dotted"],
        index=["Solid", "Dashed", "Dotted"].index(st.session_state.ui_settings.get('box_style', 'Solid'))
    )
    
with ui_tabs[1]:
    st.subheader("Text Settings")
    font_size = st.slider("Font Size", 8, 24, st.session_state.ui_settings['font_size'], key="font_size_slider")
    font_color = st.color_picker("Font Color", st.session_state.ui_settings['font_color'], key="font_color_picker")
    font_family = st.selectbox(
        "Font Family",
        ["Arial", "Helvetica", "Times New Roman", "Courier New", "Verdana"],
        index=["Arial", "Helvetica", "Times New Roman", "Courier New", "Verdana"].index(
            st.session_state.ui_settings.get('font_family', 'Arial')
        )
    )
    
with ui_tabs[2]:
    st.subheader("Advanced UI Settings")
    bg_color = st.color_picker(
        "Background Color", 
        st.session_state.ui_settings.get('bg_color', '#0E1117'),
        key="bg_color_picker"
    )
    opacity = st.slider(
        "UI Opacity", 
        0.0, 1.0, 
        st.session_state.ui_settings.get('opacity', 0.8),
        key="opacity_slider"
    )
    show_fps = st.checkbox(
        "Show FPS Counter",
        value=st.session_state.ui_settings.get('show_fps', True),
        key="show_fps_checkbox"
    )

# Update session state
st.session_state.ui_settings.update({
    'box_color': box_color,
    'font_size': font_size,
    'font_color': font_color,
    'border_width': border_width,
    'box_style': box_style,
    'font_family': font_family,
    'bg_color': bg_color,
    'opacity': opacity,
    'show_fps': show_fps
})

st.markdown('</div>', unsafe_allow_html=True)

# Save configuration button
if st.button("💾 Save Configuration"):
    # Save configuration using the config manager
    config_to_save = {
        'plugin_settings': st.session_state.plugin_settings,
        'ui_settings': st.session_state.ui_settings
    }
    
    if config_manager.save_config(config_to_save):
        st.success("Configuration saved successfully!")
        st.json(config_to_save)
    else:
        st.error("Failed to save configuration. Check the logs for details.")

# Preview section
st.markdown('<div class="settings-section">', unsafe_allow_html=True)
st.markdown('<div class="section-title">👁️ Preview</div>', unsafe_allow_html=True)

# Preview of UI settings
st.write("Current UI Settings:")
st.json(st.session_state.ui_settings)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("UV App Plugin Configurator | Configure your tracking plugins and UI settings")
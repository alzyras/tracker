import sys
from pathlib import Path

# Add the parent directory to sys.path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

print("Parent directory:", parent_dir)
print("Sys path:", sys.path[:3])  # Show first 3 paths

try:
    from uv_app.plugins import PLUGIN_REGISTRY, PluginManager
    print("Successfully imported PLUGIN_REGISTRY and PluginManager")
    print("Available plugins:", list(PLUGIN_REGISTRY.keys()))
except Exception as e:
    print("Error importing plugins:", e)
    
try:
    from uv_app.config import PLUGIN_CONFIG
    print("Successfully imported PLUGIN_CONFIG")
    print("Plugin config keys:", list(PLUGIN_CONFIG.keys())[:5])  # Show first 5 keys
except Exception as e:
    print("Error importing config:", e)
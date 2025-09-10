# uv_app/plugins/__init__.py

from .base import BasePlugin, FacePlugin, BodyPlugin, PosePlugin
from .manager import PluginManager
from .emotion_plugin import EmotionPlugin, SimpleEmotionPlugin
from .api_emotion_plugin import APIEmotionPlugin
from .activity_plugin import ActivityPlugin
from .face_image_plugin import FaceImagePlugin
from .body_analysis_plugin import BodyAnalysisPlugin
from .pose_analysis_plugin import PoseAnalysisPlugin
from .emotion_logger_plugin import EmotionLoggerPlugin
from .person_event_logger import PersonEventLogger

# Plugin registry - all available plugins
PLUGIN_REGISTRY = {
    "emotion": EmotionPlugin,
    "simple_emotion": SimpleEmotionPlugin,
    "api_emotion": APIEmotionPlugin,
    "activity": ActivityPlugin,
    "face_image": FaceImagePlugin,
    "body_analysis": BodyAnalysisPlugin,
    "pose_analysis": PoseAnalysisPlugin,
    "emotion_logger": EmotionLoggerPlugin,
    "person_event_logger": PersonEventLogger
}

__all__ = [
    "BasePlugin", "FacePlugin", "BodyPlugin", "PosePlugin",
    "PluginManager", "PLUGIN_REGISTRY",
    "EmotionPlugin", "SimpleEmotionPlugin", "APIEmotionPlugin",
    "ActivityPlugin", "FaceImagePlugin",
    "BodyAnalysisPlugin", "PoseAnalysisPlugin", 
    "EmotionLoggerPlugin", "PersonEventLogger"
]
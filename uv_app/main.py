"""Main entry point for the tracking application."""

from .tracker import run_tracker
from .logging_config import setup_logging, get_logger

logger = get_logger(__name__)


def main() -> None:
    """Main function to run the tracker."""
    # Setup logging first
    setup_logging()
    logger.info("Starting tracking application")
    
    try:
        # ---------------- Example: Webcam ----------------
        logger.info("Starting tracker on webcam (face + body + pose)")
        run_tracker(
            video_source=0,       # webcam index
            enable_face=True,     # enable face detection
            enable_body=True,     # enable full-body bounding box
            enable_pose=True      # enable pose detection
        )

        # ---------------- Example: CCTV Stream ----------------
        # Uncomment and replace URL with your CCTV stream
        # logger.info("Starting tracker on CCTV stream")
        # run_tracker(
        #     video_source="http://192.168.1.31:8080/video",
        #     enable_face=True,
        #     enable_body=True,
        #     enable_pose=True
        # )
        
    except Exception as e:
        logger.error("Application failed: %s", e)
        raise


if __name__ == "__main__":
    main()

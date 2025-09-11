# uv_app/main.py

"""
Legacy main entry point for backward compatibility.
For new usage, use: python -m uv_app.app
"""

from app import TrackerApp

if __name__ == "__main__":
    # ---------------- Example: Webcam ----------------
    print("Starting tracker on webcam (face detection only)...")
    print("For more options, use: python -m uv_app.app --help")
    
    app = TrackerApp(
        enable_face=True,     # enable face detection
        enable_body=True,    # disable full-body bounding box
        enable_pose=False     # disable pose detection
    )
    
    try:
        app.run(video_source=0)  # webcam index
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")

    # ---------------- Example: CCTV Stream ----------------
    # Uncomment and replace URL with your CCTV stream
    # print("Starting tracker on CCTV stream...")
    # app = TrackerApp(enable_face=True, enable_body=True, enable_pose=True)
    # app.run(video_source="http://192.168.1.31:8080/video")

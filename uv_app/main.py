# uv_app/main.py

from uv_app.tracker import run_tracker

if __name__ == "__main__":
    # ---------------- Example: Webcam ----------------
    print("Starting tracker on webcam (face + body + pose)...")
    run_tracker(
        video_source=0,       # webcam index
        enable_face=True,     # enable face detection
        enable_body=False,     # enable full-body bounding box
        enable_pose=False      # enable pose detection
    )

    # ---------------- Example: CCTV Stream ----------------
    # Uncomment and replace URL with your CCTV stream
    # print("Starting tracker on CCTV stream...")
    # run_tracker(
    #     video_source="http://192.168.1.31:8080/video",
    #     enable_face=True,
    #     enable_body=True,
    #     enable_pose=True
    # )

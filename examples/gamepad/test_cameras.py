#!/usr/bin/env python
"""
Test available cameras by displaying their feeds in OpenCV windows.

Usage:
  python examples/gamepad/test_cameras.py
"""

import cv2


def main():
    # Try camera indices 0-5
    caps = {}
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                print(f"Camera {i}: {w}x{h} — OK")
                caps[i] = cap
            else:
                print(f"Camera {i}: opened but no frame")
                cap.release()
        else:
            print(f"Camera {i}: not found")

    if not caps:
        print("No cameras found!")
        return

    print(f"\nShowing {len(caps)} camera(s). Press 'q' to quit.")

    while True:
        for i, cap in caps.items():
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f"Camera {i}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

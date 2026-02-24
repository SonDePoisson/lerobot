#!/usr/bin/env python
"""Test script to display Stadia controller inputs in real-time."""

import hid
import time
import sys


def main():
    print("Searching for Stadia controller...")
    devices = hid.enumerate()
    device_info = None
    for d in devices:
        ps = d.get("product_string", "")
        if "Stadia" in ps:
            device_info = d
            break

    if not device_info:
        print("No Stadia controller found!")
        sys.exit(1)

    print(f"Found: {device_info['manufacturer_string']} {device_info['product_string']}")

    dev = hid.device()
    dev.open_path(device_info["path"])
    dev.set_nonblocking(1)

    print("\nReading inputs... Press Ctrl+C to quit.\n")

    DPAD_NAMES = {
        0: "Up", 1: "Up-Right", 2: "Right", 3: "Down-Right",
        4: "Down", 5: "Down-Left", 6: "Left", 7: "Up-Left", 8: "Neutral",
    }

    try:
        while True:
            data = dev.read(64)
            if not data or len(data) < 10:
                time.sleep(0.01)
                continue

            # Sticks (0-255, 128=center)
            lx = data[4]
            ly = data[5]
            rx = data[6]
            ry = data[7]

            # Triggers analog (0-255)
            l2_val = data[8]
            r2_val = data[9]

            # D-pad
            dpad = DPAD_NAMES.get(data[1], "?")

            # Face buttons (byte 3)
            btns = data[3]
            a = bool(btns & 64)
            b = bool(btns & 32)
            x = bool(btns & 16)
            y = bool(btns & 8)
            l1 = bool(btns & 4)
            r1 = bool(btns & 2)

            line = (
                f"LStick({lx:3d},{ly:3d}) "
                f"RStick({rx:3d},{ry:3d}) "
                f"L2={l2_val:3d} R2={r2_val:3d} "
                f"| A={int(a)} B={int(b)} X={int(x)} Y={int(y)} "
                f"L1={int(l1)} R1={int(r1)} "
                f"DPad={dpad:<10}"
            )
            print(f"\r{line}", end="", flush=True)

    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        dev.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
AMPBANK M256 Thermal Camera Reader
----------------------------------
Copyright 2025 Ivan Gordeyev

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Based on Les Wright's work, adapted for AMPBANK M256 + Raspberry Pi + Thonny by Ivan Gordeyev October 2025
- Apache License 2.0
- Original License for Les Wright script and copyright notice at https://github.com/leswright1977/PyThermalCamera/blob/main/LICENSE
- Auto-detects the correct /dev/video* device - thanks LLMs
- Fixes uint8 overflow in temperature maths - thanks LLMs
- Sets QT_QPA_PLATFORM at runtime (no terminal export needed)
- Runs safely inside Thonny

"""

print('Les Wright 21 June 2023 - original author,')
print('https://youtube.com/leslaboratory')
print('A Python program to read, parse and display thermal data from the Topdon TC001 Thermal camera!')
print('')
print('Tested on Debian/Raspberry Pi OS all features are working correctly')
print('This will work on the Pi 4b However a number of workarounds are implemented!')
print('Seemingly there are bugs in the compiled version of cv2 that ships with the Pi!')
print('')
print('Key Bindings:')
print('')
print('a z: Increase/Decrease Blur')
print('s x: Floating High and Low Temp Label Threshold')
print('d c: Change Interpolated scale Note: This will not change the window size on the Pi')
print('f v: Contrast')
print('q w: Fullscreen Windowed (note going back to windowed does not seem to work on the Pi!)')
print('r t: Record and Stop')
print('p : Snapshot')
print('m : Cycle through ColorMaps')
print('h : Toggle HUD')

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"  # Prevent Wayland plugin error

import cv2
import numpy as np
import time
import io
import subprocess

def is_raspberrypi():
    try:
        with io.open('/sys/firmware/devicetree/base/model', 'r') as m:
            return 'raspberry pi' in m.read().lower()
    except Exception:
        return False

isPi = is_raspberrypi()

# --- Auto-detect the correct camera device ---
def find_tc001_device():
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True, text=True, check=True
        )
        lines = result.stdout.splitlines()
        for i, line in enumerate(lines):
            if "Camera" in line and "usb" in line.lower():
                next_lines = lines[i + 1 : i + 4]
                for dev in next_lines:
                    if "/dev/video" in dev:
                        return dev.strip()
    except Exception as e:
        print("⚠️ Could not auto-detect camera:", e)
    return "/dev/video0"

device_path = find_tc001_device()
print(f"🎥 Using device: {device_path}")

# --- Video setup ---
cap = cv2.VideoCapture(device_path, cv2.CAP_V4L)
if isPi:
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
else:
    cap.set(cv2.CAP_PROP_CONVERT_RGB, False)

if not cap.isOpened():
    raise RuntimeError(f"❌ Could not open {device_path}")

# --- Settings ---
width, height = 256, 192
scale = 3
newWidth, newHeight = width * scale, height * scale
alpha = 1.0
colormap = 0
rad = 0
threshold = 2
hud = True
recording = False
elapsed = "00:00:00"
snaptime = "None"

cv2.namedWindow('Thermal', cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow('Thermal', newWidth, newHeight)

def rec():
    now = time.strftime("%Y%m%d--%H%M%S")
    print(f"File being saved to {now}_output.avi")
    return cv2.VideoWriter(f"{now}_output.avi", cv2.VideoWriter_fourcc(*'XVID'), 25, (newWidth, newHeight))
    
def snapshot(heatmap):
    now = time.strftime("%Y%m%d-%H%M%S")
    snaptime = time.strftime("%H:%M:%S")
    filename = f"M256_001_{now}.png"
    cv2.imwrite(filename, heatmap)
    print(f"📸 Snapshot saved: {filename}")
    return snaptime

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No frame received.")
        break

    imdata, thdata = np.array_split(frame, 2)
    thdata = thdata.astype(np.uint16)

    # Center pixel
    hi = int(thdata[96, 128, 0])
    lo = int(thdata[96, 128, 1])
    temp = round(((hi + (lo << 8)) / 64) - 273.15, 2)

    # Max temperature
    posmax = thdata[..., 1].argmax()
    mcol, mrow = divmod(posmax, width)
    hi_max = int(thdata[mcol, mrow, 0])
    lo_max = int(thdata[mcol, mrow, 1])
    maxtemp = round(((hi_max + (lo_max << 8)) / 64) - 273.15, 2)

    # Min temperature
    posmin = thdata[..., 1].argmin()
    lcol, lrow = divmod(posmin, width)
    hi_min = int(thdata[lcol, lrow, 0])
    lo_min = int(thdata[lcol, lrow, 1])
    mintemp = round(((hi_min + (lo_min << 8)) / 64) - 273.15, 2)

    # Average
    hi_avg = int(thdata[..., 0].mean())
    lo_avg = int(thdata[..., 1].mean())
    avgtemp = round(((hi_avg + (lo_avg << 8)) / 64) - 273.15, 2)

    # Process display image
    bgr = cv2.cvtColor(imdata, cv2.COLOR_YUV2BGR_YUYV)
    bgr = cv2.convertScaleAbs(bgr, alpha=alpha)
    bgr = cv2.resize(bgr, (newWidth, newHeight), interpolation=cv2.INTER_CUBIC)
    if rad > 0:
        bgr = cv2.blur(bgr, (rad, rad))

    colormaps = [
        (cv2.COLORMAP_JET, 'Jet'), (cv2.COLORMAP_HOT, 'Hot'),
        (cv2.COLORMAP_MAGMA, 'Magma'), (cv2.COLORMAP_INFERNO, 'Inferno'),
        (cv2.COLORMAP_PLASMA, 'Plasma'), (cv2.COLORMAP_BONE, 'Bone'),
        (cv2.COLORMAP_SPRING, 'Spring'), (cv2.COLORMAP_AUTUMN, 'Autumn'),
        (cv2.COLORMAP_VIRIDIS, 'Viridis'), (cv2.COLORMAP_RAINBOW, 'Rainbow'),
    ]
    cmap, cmapText = colormaps[colormap % len(colormaps)]
    heatmap = cv2.applyColorMap(bgr, cmap)
    if cmapText == 'Rainbow':
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Crosshair
    cx, cy = newWidth // 2, newHeight // 2
    cv2.drawMarker(heatmap, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 30, 2)
    cv2.putText(heatmap, f"{temp} C", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    # HUD
    if hud:
        cv2.rectangle(heatmap, (0, 0), (180, 180), (0, 0, 0), -1)
        info = [
            f"Avg: {avgtemp} C",
            f"Max: {maxtemp} C",
            f"Min: {mintemp} C",
            f"Colormap: {cmapText}",
            f"Contrast: {alpha}",
            f"Blur: {rad}",
            f"Scale: {scale}",
            " ",
            "m Cycle through ColorMaps",
            "r t Record and Stop",
            "p to Take Snapshot",
            "h Toggle HUD"
        ]
        for i, line in enumerate(info):
            cv2.putText(heatmap, line, (10, 16 + i * 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Thermal", heatmap)

    if recording:
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
        videoOut.write(heatmap)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('a'):
        rad += 1
    elif key == ord('z'):
        rad = max(0, rad - 1)
    elif key == ord('s'):
        threshold += 1
    elif key == ord('x'):
        threshold = max(0, threshold - 1)
    elif key == ord('m'):
        colormap += 1
    elif key == ord('f'):
        alpha = min(3.0, round(alpha + 0.1, 1))
    elif key == ord('v'):
        alpha = max(0.1, round(alpha - 0.1, 1))
    elif key == ord('h'):
        hud = not hud
    elif key == ord('r') and not recording:
        videoOut = rec()
        recording = True
        start = time.time()
        print("Recording started")
    elif key == ord('t') and recording:
        recording = False
        elapsed = "00:00:00"
        videoOut.release()
        print("Recording stopped")
    elif key == ord('p'):
        snaptime = snapshot(heatmap)

cap.release()
cv2.destroyAllWindows()
print("✅ Thermal Camera session ended.")

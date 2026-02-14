import cv2
import logging

from cv2_enumerate_cameras import enumerate_cameras
from cv2_enumerate_cameras.camera_info import CameraInfo
from logging.handlers import RotatingFileHandler


KB = 1024
NAME_KEY = "name"
BACKEND_KEY = "backend"
FRAME_KEY = "frame"
CAMERA_NOT_FOUND_MESSAGE = "Camera not found!"
M256_UTIL_PY_LOGGER = "m256-util-py-logger"


if __name__ == "__main__":
    # Logging config below ONLY applies to util.py. This will NOT
    # take effect if instead M256.py is the entrypoint
    logger = logging.getLogger(M256_UTIL_PY_LOGGER)
    logger.setLevel(logging.DEBUG)
    log_handler = RotatingFileHandler(
        M256_UTIL_PY_LOGGER, maxBytes=50 * KB, backupCount=5
    )
    log_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - [%(levelname)s] - %(message)s")
    )
    logger.addHandler(log_handler)
else:
    logger = logging.getLogger(__name__)


# Helper: function to display index and name of each connected USB camera
def get_connected_cameras(use_windows: bool = True, apiPreference=cv2.CAP_ANY):
    cameras = {}
    for camera_info in enumerate_cameras(apiPreference):
        cameras.update(
            {
                camera_info.index: {
                    NAME_KEY: camera_info.name,
                    BACKEND_KEY: camera_info.backend,
                }
            }
        )

    print(f"Retrieved {len(cameras)} connected cameras")
    logger.info(f"Retrieved {len(cameras)} connected cameras")
    return cameras


# Helper: find USB camera by VID (Vendor ID) and PID (Product ID)
def find_camera(vid, pid, apiPreference=cv2.CAP_ANY):

    def is_vid_pid_match(info: CameraInfo, vid, pid):
        return info.vid == vid and info.pid == pid

    for cam_info in enumerate_cameras(apiPreference=apiPreference):
        if is_vid_pid_match(cam_info, vid, pid):
            return cv2.VideoCapture(cam_info.index, cam_info.backend)

    logger.error(CAMERA_NOT_FOUND_MESSAGE)
    raise RuntimeError(CAMERA_NOT_FOUND_MESSAGE)


# Helper: sanity check camera capture
def test_example_capture(cap: cv2.VideoCapture, n_frames_to_collect: int = 5):
    try:
        collected = []
        for i in range(n_frames_to_collect):
            ok, frame = cap.read()
            if not ok:
                continue
            collected.append(frame)

        if len(collected) > 0:
            cv2.imshow(FRAME_KEY, frame)

        print(
            f"Sanity test complete ({collected}/{n_frames_to_collect} frames captured)"
        )
        logger.info(
            f"Sanity test complete ({collected}/{n_frames_to_collect} frames captured)"
        )

    except Exception:
        logger.exception("Exception while sanity testing capture on given camera!")


# Sanity testing (only if this file is run explicitly)
if __name__ == "__main__":
    cams = get_connected_cameras()
    for index in list(cams.keys()):
        info = cams[index]

        print(f"\t[{index}]: {info.get('name')} ({info.get('backend')})")
        logger.debug(f"\t[{index}]: {info.get('name')} ({info.get('backend')})")

# Leave 1 space below this

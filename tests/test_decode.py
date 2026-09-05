import numpy as np

from ocr_layout_detector.decode import DecodeConfig, decode_maps
from ocr_layout_detector.model import TinyLayoutNet, parameter_count


def test_model_parameter_count() -> None:
    assert parameter_count(TinyLayoutNet()) == 7535


def test_decode_single_horizontal_line() -> None:
    center = np.full((16, 24), -10.0, dtype=np.float32)
    height = np.zeros((16, 24), dtype=np.float32)
    region = np.full((16, 24), -10.0, dtype=np.float32)

    center[7:9, 4:20] = 10.0
    height[7:9, 4:20] = np.log(3.0)
    region[5:11, 2:22] = 10.0

    detections = decode_maps(
        center,
        height,
        region,
        image_width=96,
        image_height=64,
        config=DecodeConfig(center_threshold=0.5, height_factor=0.5, vertical_padding_px=0.0),
    )

    assert len(detections) == 1
    x0, y0, x1, y1 = detections[0].box
    assert x0 < 20
    assert x1 > 75
    assert 24 < y0 < 32
    assert 32 < y1 < 40

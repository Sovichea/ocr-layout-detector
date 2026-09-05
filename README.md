# OCR Layout Detector

Lightweight document-region and text-line detection for OCR, designed for direct native and WebAssembly inference.

The project explores small fully convolutional models that locate text lines without depending on YOLO, OpenCV, or a large document-analysis runtime. The deployment target is a compact Rust/WASM implementation with simple tensor operations and lightweight post-processing.

## Initial model

The first proof of concept is a **7,535-parameter** fully convolutional detector:

```text
grayscale page
    -> Conv 1->8, stride 2 + ReLU
    -> Conv 8->12, stride 2 + ReLU
    -> 5 x Conv 12->12 + ReLU
       dilations: 1, 2, 4, 8, 1
    -> Conv 12->3, 1x1
         |-> line-center logit
         |-> log line height
         `-> text-region logit
```

The output stride is 4. Inference uses only:

- Conv2D
- ReLU
- sigmoid
- thresholding
- connected components
- simple box merging

The neural model intentionally avoids anchors, NMS-heavy object-detection heads, transformers, and framework-specific operators that would make a small Rust/WASM port harder.

## Initial experiment

The first local proof of concept used a 104-page Khmer digital-terminology PDF with PDF text geometry as training/evaluation supervision.

- 82 training pages
- 8 development pages
- 10 held-out quantitative test pages
- 2 qualitative holdout pages
- 500 CPU training steps

The initial test result before decoder refinement was:

| Metric | Result |
|---|---:|
| parameters | 7,535 |
| detection F1 50/50 | 81.02% |
| text-ink recall >=95% | 100.00% |
| clean text-ink >=95/5 | 46.52% |
| mean neighbor-ink contamination | 17.00% |

A separate local decoder-height sweep showed that the network's center predictions were substantially better than the initial box expansion. This is the main reason the next iteration focuses on decoder calibration, hard negatives, and better region use rather than immediately increasing model size.

See [`experiments/2026-09-05/REPORT.md`](experiments/2026-09-05/REPORT.md).

## Direction

Near-term priorities:

1. calibrate line-height decoding without clipping Khmer marks
2. add hard negatives for seals, logos, rules, figures, signatures, and decorative strokes
3. make the region head useful for grouping and false-positive suppression
4. benchmark against the deterministic `ocr-line-cropper` baseline on the same pages
5. test 15k to 30k parameter variants only if the 7.5k model saturates
6. export weights for a small Rust/WASM runtime and later quantize Conv weights to Q8

The detector is deliberately language-agnostic. Khmer-specific recognition remains the responsibility of the downstream OCR recognizer.

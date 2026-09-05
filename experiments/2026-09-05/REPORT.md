# Tiny Layout Detector v0

Date: 2026-09-05

## Goal

Test whether a very small learned detector can approach the existing deterministic OCR line cropper while remaining simple enough to port directly to Rust/WebAssembly.

This experiment intentionally avoids a YOLO-style detection stack. The model predicts dense line-center, line-height, and text-region maps and uses connected components for decoding.

## Architecture

- grayscale variable-size input
- output stride 4
- 7,535 parameters
- Conv2D + ReLU backbone only
- three 1x1 output channels:
  - line-center logit
  - log line height
  - text-region logit

The convolution widths are 1 -> 8 -> 12 and then five 12-channel convolutions with dilation sequence 1, 2, 4, 8, 1.

## Data split

The first proof of concept used a 104-page MPTC Khmer digital-terminology PDF with vector PDF geometry as detector supervision.

- train: 82 pages
- dev: pages 5, 15, 25, 45, 55, 65, 85, 95
- test: pages 11, 14, 35, 41, 61, 66, 75, 76, 92, 101
- qualitative holdout: pages 12, 19

PDF text extraction is used only to obtain visual line/block geometry for this detector experiment. OCR Unicode correctness is not evaluated here.

## Training

- 120 DPI page rasterization
- 256 x 256 random patches
- 500 steps
- Adam, learning rate 2e-3
- simple contrast/noise/speckle augmentation
- CPU training

## Initial result

The raw v0 decoder selected a development center threshold of 0.25.

Held-out 10-page result:

| Metric | Result |
|---|---:|
| GT lines | 273 |
| predicted lines | 238 |
| detection precision 50/50 | 86.97% |
| detection recall 50/50 | 75.82% |
| detection F1 50/50 | 81.02% |
| strict precision 95/80 | 75.63% |
| strict recall 95/80 | 65.93% |
| text-ink recall >=95% | 100.00% |
| clean text-ink >=95/5 | 46.52% |
| mean neighbor-ink contamination | 17.00% |

The 100% text-ink recall together with poor clean-crop rate showed that the first decoder expanded boxes too generously in the vertical direction.

A subsequent local line-height sweep improved the tradeoff substantially without retraining the neural model. This supports treating decoder calibration as a first-class part of the next iteration.

## Qualitative observations

The held-out scanned/formal-letter pages show that the detector learns real text-line structure, including lines that are hard to obtain with global projection methods. The main false positives come from text-like non-text structures such as:

- seals and logos
- decorative horizontal/vertical rules
- signatures
- repeated graphic strokes
- figure/chart elements

These are suitable hard negatives for the next training iteration.

## Next experiment

1. replace broad vertical expansion with calibrated line-height decoding
2. train with hard-negative graphics and document decorations
3. use the region head to suppress isolated false positives and preserve reading regions
4. compare on exactly the same holdout pages as `ocr-line-cropper`
5. only increase model size if the 7.5k model saturates
6. implement the decoder in Rust without SciPy/OpenCV and later export Q8 convolution weights

The raw experiment output is saved in [`results.json`](results.json).

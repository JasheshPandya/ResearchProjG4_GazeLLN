# Gaze-Guided vs. Uniform Compression -- Evaluation Report

- Generated: 2026-10-15T00:48:25
- Manifest: `/run/media/tejas-kumar-s/Windows/Users/Tejas/Smd/Data/KDAG/GazeLNN Video Encoder/ResearchProjG4_GazeLLN/Encoder/ablation_out/manifest.json`
- ROI definition: top 25% of macroblocks by predicted saliency (per frame), derived from the QP-offset maps of the canonical `roi_source` for each clip. Same mask used to score every variant of a clip.
- Pixel metrics use 8-bit OpenCV luma (Y): MSE, PSNR, SSIM and MS-SSIM. PSNR uses pooled pixel MSE (not mean frame dB); SSIM uses 11x11 Gaussian windows, sigma=1.5. MS-SSIM uses five scales and Wang weights; unavailable for undersized frames. These pixel metrics do not measure chroma fidelity.
- ROI selects an exact ranked macroblock count (rounded up, ties deterministic); reported ROI pixel fraction accounts for partial edge macroblocks. ROI masks are predicted gaze, not independent human eye-tracking measurements.
- VMAF: available.

## 1. Per-variant results

| Clip | Method | Label | Bitrate (kbps) | Size (MiB) | PSNR global/ROI/periph (dB) | SSIM global/ROI/periph | VMAF | Encode time (s) | Notes |
|---|---|---|---|---|---|---|---|---|---|
| sample_video | gaze | crf33 | 608.0 | 1.17 | 29.90 / 31.42 / 29.49 | 0.7722 / 0.8335 / 0.7518 | 46.36 | 4.59 |  |
| sample_video | gaze | crf30 | 811.8 | 1.56 | 31.44 / 33.05 / 31.01 | 0.8254 / 0.8746 / 0.8089 | 59.26 | 5.10 |  |
| sample_video | gaze | crf27 | 1085.9 | 2.08 | 33.09 / 34.77 / 32.65 | 0.8717 / 0.9093 / 0.8592 | 70.36 | 4.70 |  |
| sample_video | gaze | crf24 | 1485.6 | 2.84 | 34.89 / 36.63 / 34.43 | 0.9103 / 0.9374 / 0.9013 | 79.09 | 4.65 |  |
| sample_video | gaze | crf21 | 2076.8 | 3.97 | 36.79 / 38.58 / 36.33 | 0.9395 / 0.9579 / 0.9334 | 85.77 | 5.48 |  |
| sample_video | gaze | crf18 | 2989.9 | 5.71 | 38.73 / 40.56 / 38.26 | 0.9598 / 0.9720 / 0.9557 | 90.57 | 5.61 |  |
| sample_video | uniform | crf33 | 936.3 | 1.79 | 33.32 / 33.56 / 33.24 | 0.8790 / 0.8876 / 0.8761 | 69.57 | 4.72 |  |
| sample_video | uniform | crf30 | 1286.7 | 2.46 | 35.04 / 35.30 / 34.96 | 0.9138 / 0.9188 / 0.9122 | 78.27 | 4.71 |  |
| sample_video | uniform | crf27 | 1820.6 | 3.48 | 36.87 / 37.12 / 36.80 | 0.9408 / 0.9434 / 0.9399 | 85.07 | 4.99 |  |
| sample_video | uniform | crf24 | 2658.8 | 5.08 | 38.79 / 39.07 / 38.70 | 0.9603 / 0.9619 / 0.9598 | 90.06 | 5.49 |  |
| sample_video | uniform | crf21 | 3888.7 | 7.43 | 40.72 / 41.04 / 40.62 | 0.9735 / 0.9744 / 0.9732 | 93.55 | 5.92 |  |
| sample_video | uniform | crf18 | 5598.6 | 10.70 | 42.60 / 42.88 / 42.50 | 0.9822 / 0.9824 / 0.9821 | 95.85 | 6.61 |  |

## Additional quality measurements

| Clip | Method | Label | MSE global/ROI/periph | ROI pixels (%) | MS-SSIM | PSNR p5 (dB) | SSIM p5 | MS-SSIM p5 | VMAF p5/min | VMAF model |
|---|---|---|---|---|---|---|---|---|---|---|
| sample_video | uniform | crf18 | 3.576 / 3.347 / 3.653 | 25.03 | 0.9962 | 41.60 | 0.9783 | 0.9954 | 94.33 / 93.81 | version=vmaf_v0.6.1 |
| sample_video | gaze | crf18 | 8.703 / 5.713 / 9.701 | 25.03 | 0.9905 | 37.86 | 0.9514 | 0.9884 | 88.44 / 87.42 | version=vmaf_v0.6.1 |
| sample_video | uniform | crf21 | 5.508 / 5.121 / 5.637 | 25.03 | 0.9941 | 39.79 | 0.9677 | 0.9929 | 91.88 / 90.97 | version=vmaf_v0.6.1 |
| sample_video | gaze | crf21 | 13.606 / 9.025 / 15.136 | 25.03 | 0.9846 | 35.89 | 0.9255 | 0.9807 | 82.95 / 81.74 | version=vmaf_v0.6.1 |
| sample_video | uniform | crf24 | 8.588 / 8.051 / 8.768 | 25.03 | 0.9906 | 37.84 | 0.9508 | 0.9883 | 87.97 / 86.93 | version=vmaf_v0.6.1 |
| sample_video | gaze | crf24 | 21.099 / 14.140 / 23.423 | 25.03 | 0.9749 | 34.06 | 0.8897 | 0.9683 | 75.42 / 74.79 | version=vmaf_v0.6.1 |
| sample_video | uniform | crf27 | 13.356 / 12.621 / 13.601 | 25.03 | 0.9848 | 35.94 | 0.9252 | 0.9806 | 82.19 / 81.32 | version=vmaf_v0.6.1 |
| sample_video | gaze | crf27 | 31.928 / 21.680 / 35.351 | 25.03 | 0.9597 | 32.35 | 0.8466 | 0.9493 | 65.70 / 64.57 | version=vmaf_v0.6.1 |
| sample_video | uniform | crf30 | 20.354 / 19.206 / 20.737 | 25.03 | 0.9756 | 34.08 | 0.8888 | 0.9677 | 74.23 / 73.55 | version=vmaf_v0.6.1 |
| sample_video | gaze | crf30 | 46.727 / 32.249 / 51.562 | 25.03 | 0.9371 | 30.84 | 0.8006 | 0.9232 | 53.63 / 52.35 | version=vmaf_v0.6.1 |
| sample_video | uniform | crf33 | 30.292 / 28.624 / 30.849 | 25.03 | 0.9617 | 32.34 | 0.8448 | 0.9481 | 63.76 / 62.19 | version=vmaf_v0.6.1 |
| sample_video | gaze | crf33 | 66.575 / 46.906 / 73.144 | 25.03 | 0.9034 | 29.38 | 0.7507 | 0.8869 | 39.95 / 38.52 | version=vmaf_v0.6.1 |

p5 is the fifth percentile of per-frame scores, not the percentile of 20-frame averages. Higher is better. MSE is lower-is-better. Infinite PSNR means identical luma samples.

## Processing cost

| Clip | Method | Label | Encoder fps | QP generation (s) | Sequential pipeline (s) | Pipeline fps | Processing ms/frame | Bitrate source |
|---|---|---|---|---|---|---|---|---|
| sample_video | uniform | crf18 | 72.66 | 0.000 | 6.606 | 72.66 | 13.763 | video_stream |
| sample_video | gaze | crf18 | 85.61 | 11.199 | 16.805 | 28.56 | 35.011 | video_stream |
| sample_video | uniform | crf21 | 81.09 | 0.000 | 5.919 | 81.09 | 12.331 | video_stream |
| sample_video | gaze | crf21 | 87.67 | 11.199 | 16.674 | 28.79 | 34.737 | video_stream |
| sample_video | uniform | crf24 | 87.43 | 0.000 | 5.490 | 87.43 | 11.437 | video_stream |
| sample_video | gaze | crf24 | 103.15 | 11.199 | 15.852 | 30.28 | 33.025 | video_stream |
| sample_video | uniform | crf27 | 96.14 | 0.000 | 4.993 | 96.14 | 10.402 | video_stream |
| sample_video | gaze | crf27 | 102.21 | 11.199 | 15.895 | 30.20 | 33.114 | video_stream |
| sample_video | uniform | crf30 | 101.96 | 0.000 | 4.708 | 101.96 | 9.808 | video_stream |
| sample_video | gaze | crf30 | 94.06 | 11.199 | 16.302 | 29.44 | 33.962 | video_stream |
| sample_video | uniform | crf33 | 101.72 | 0.000 | 4.719 | 101.72 | 9.831 | video_stream |
| sample_video | gaze | crf33 | 104.51 | 11.199 | 15.791 | 30.40 | 32.899 | video_stream |

Pipeline time is QP generation plus encoding when both are measured, or an explicit pipeline_time_s. Each standalone gaze encode is charged the full QP-generation cost, even when the sweep caches those maps. Missing stage timings remain N/A. Throughput and average processing cost are not capture-to-display latency.

## 2. BD-Rate / BD-quality summary (gaze vs. uniform, per clip)

Negative BD-Rate % = gaze-guided encoding needs fewer bits for equal quality (bitrate savings). Positive BD-quality = gaze-guided has higher quality at equal bitrate. Both are computed over the bitrate/quality range common to both methods.

| Clip | Metric | #Uniform pts | #Gaze pts | BD-Rate (%) | BD-quality | Notes |
|---|---|---|---|---|---|---|
| sample_video | psnr | 6 | 6 | 17.08 | -0.8600 |  |
| sample_video | ssim | 6 | 6 | 18.29 | -0.0146 |  |
| sample_video | vmaf | 6 | 6 | 10.40 | -2.1167 |  |
| sample_video | ms_ssim | 6 | 6 | 16.25 | -0.0049 |  |
| sample_video | psnr_roi | 6 | 6 | -11.02 | 0.6321 |  |
| sample_video | ssim_roi | 6 | 6 | -10.06 | 0.0067 |  |
| sample_video | psnr_periphery | 6 | 6 | 25.58 | -1.2334 |  |
| sample_video | ssim_periphery | 6 | 6 | 26.97 | -0.0217 |  |
| sample_video | vmaf_p5 | 6 | 6 | 9.41 | -2.2133 |  |

## 3. How to read this

- **Global** columns show whole-frame quality -- the classic RD comparison.
- **ROI** columns isolate the macroblocks the saliency model flagged as important; this is where the gaze-guided method should show a clear advantage.
- **Periphery** columns show the rest of the frame; some quality loss here relative to uniform encoding at the same overall bitrate is expected and is the mechanism by which bits were saved/reallocated.
- BD-Rate/BD-quality require >=2 RD points per method to compute at all, and the cubic fit uses up to degree 3; >=4 points are recommended but do not guarantee a reliable fit. Invalid/nonmonotonic curves are reported as N/A. Sweep more CRF/QP-scale values if you see that flag.
- Compare methods at equal bitrate or equal quality, not equal CRF alone.
- PSNR/SSIM implementation, pooling, decoding path and VMAF model must stay fixed across comparisons. Old mean-frame-dB results are not directly comparable to pooled-MSE PSNR.


## Generated figures

See [figure index](plots/README.md) for PNG/PDF figures, bitrate-pair details and unavailable inputs.

# All settings in one place

These values are the defaults of the scripts and match the technical report. The file that sets each value is given in brackets.

## CSI → Doppler heatmap (`csi_tools/`)
| setting | value |
|---|---|
| traffic | ping to the router every 2 ms (`record.sh`) |
| packets kept | about 150 per second, unused tones dropped (`csi_to_npz.py --rate 150`) |
| ratio | antenna A / antenna B per tone, then neighbouring tones (`preprocess_csi.py`) |
| outlier rule | values above 5 × the typical level are errors (`CLIP_MULT = 5`) |
| uniform grid | 100 Hz (`--fs`) |
| Doppler window | 0.5 s Hann, step 1 sample, FFT size 100 (`--win`, `--sliding`, `--n-fft`) |
| carrier | 5.775 GHz (`--fc`), 1 Hz ≈ 0.10 m/s |
| display | log scale, ±30 Hz in the picture (`--fmax`) |
| movement | above 3 % of the recording's peak (`make_dataset.py --min-motion 0.03`) |

## Doppler SNN (`doppler_snn/train_snn.py`)
| setting | value |
|---|---|
| frame | 1 s of heatmap, 100 steps of 10 ms, |f| ≤ 40 Hz (`--fmax`), dB −35…0 scaled to 0…1 |
| layers | 256 → 128 → 3 spiking neurons (`--h1`, `--h2`), dropout 0.3 |
| neuron | decay 0.9 (learnable per neuron), threshold 1.0, soft reset |
| surrogate gradient | fast sigmoid, slope 25 |
| optimiser | AdamW, lr 0.002, weight decay 0.001, cosine decay, 40 epochs, batch 64 |
| also in the code | label smoothing 0.05, gradient clipping 1.0, best epoch chosen on the validation person, small gain and noise augmentation (`--no-augment` switches it off) |
| split | by person: two train, one validation, one test; repeated for every person (`--mode loso`) |
| people | Divij and Nikita excluded (`--exclude`) |

## Pose labels (`siwis_pose/prepare_percom.py`)
| setting | value |
|---|---|
| frame rate | 2 per second (`--label_fps`), height 480 px (`--max_h`) |
| networks | torchvision Mask R-CNN and Keypoint R-CNN (COCO, no training) |
| thresholds | person detection ≥ 0.7, mask ≥ 0.5, joint visible if score > 1.0 (`--kp_thresh`) |
| labels | mask 48 × 64, 17 joint heatmaps (σ = 2) |
| CSI window | 1.7 s = 255 samples at 150 Hz, ends at the frame time |

## Pose model (SiWiS)
| | earlier model (`train_percom.py`) | final cross-person (`train_percom_v2.py`) | within-person (`within_pipeline.py`, `segment_pipeline.py`) |
|---|---|---|---|
| channels | 2 (amplitude spread) | 10 | 10 |
| loss | repository loss | mask loss / pixels + 500 × joint loss | same |
| optimiser | Adam, repository lr (1e-3) | AdamW, lr 3e-4, cosine, clip 1.0 | same |
| epochs | 50 | 80 | 60 |
| batch size | 16 | 16 | 16 |

## SNN on the pose
| | cross-person windows (`pose_snn.py`) | within-person windows (`within_pipeline.py`) | complete gestures (`segment_pipeline.py`) |
|---|---|---|---|
| unit | 10 frames (5 s), stride 5 | 10 frames, train stride 2, test stride 5 | one gesture, 1 step per 0.5 s frame |
| input | outline 192 / joints 36 / both 228, optionally + speed | same, speed always on | joints 36 / outline 192 / both 228, speed (72 / 384 / 456) |
| preparation | subtract recording mean, scale with training statistics | same | subtract the person's mean pose from training gestures, scale by training std |
| layers | 256 → 128 → 3 | same | Linear → 256 LIF → Linear → 128 LIF → Linear → 3 |
| dropout | none | none | 0.3 on the input |
| read-out | leaky, last step | leaky, last step | leaky, averaged over all real frames |
| neuron | decay 0.9, threshold 1.0, soft reset | same | same |
| surrogate | fast sigmoid, slope 10 | same | same |
| loss | class-weighted cross-entropy | same | same |
| optimiser | Adam, lr 0.002, no weight decay, no clipping | same | Adam, lr 0.001, weight decay 1e-4, clip 1.0 |
| batch / epochs | 64 / 40 | 64 / 40 | 16 / 60 |
| augmentation | none | none | trim ≤ 2 frames per end (if ≥ 6 remain), Gaussian noise σ = 0.05 |
| validation | none | none | none (settings fixed in advance) |
| seeds | 1 | 3 | 3 (mean ± std, Wilson 95 % interval) |
| test | leave one person out | last 20 % of each recording | last ~20 % of the gestures of each recording |

## Gesture cutting (`siwis_pose/video_segments.py`)
Keypoint R-CNN at 10 fps and 480 px, hand height = (elbow − wrist) / torso height per hand, smoothing 0.2 s, peaks at least 1.5 s apart,
prominence 35 % of the hand range, rest below 30 % of the range, cut at the valleys on both sides of a peak.

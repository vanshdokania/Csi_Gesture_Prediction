# Results in detail

All numbers are from the technical report. "Window" = one classified window or frame, "recording" = majority vote over all windows of a recording.
Chance level is 33 % for the three gestures.

## 1. Doppler heatmap → SNN (new person)
Train on three people, test on the fourth, averaged over the four test people: **35 %** per frame, **44 %** per recording.
Training accuracy was 95–100 %, so the network learned the people instead of the gesture.

## 2. Pose model for a new person
| model | outline score | wrists found |
|---|---|---|
| earlier model (2 channels) | 0.35 | 9 % |
| final model (10 channels) | 0.42 | 38 % |
| mean-pose baseline | 0.40 | 32 % |

## 3. Cross-person SNN
| input | window | recording |
|---|---|---|
| outline, no speed | 34 % | 34 % |
| joints + speed | 43 % | 54 % |
| outline + speed | 45 % | 43 % |
| outline + joints + speed | 43 % | 46 % |

Per-hand accuracy with joints + speed: both hands 30 %, left hand 31 %, right hand 67 %. Window accuracy across the test people: 37–49 %.

Control (poses from the video instead of the Wi-Fi prediction): 67 % / 71 % without speed, 75 % / 80 % with speed.
So the SNN works, and the limit is the pose prediction for a new person.

## 4. Within person, fixed 5 s windows
First 80 % of each recording train, last 20 % test, mean ± std over 3 seeds, pooled over the four people.

| input (all with speed except the outline row) | window | recording |
|---|---|---|
| arm joints + speed | 77 % | 81 % |
| outline | 51 % | 63 % |
| outline + joints | 56 % | 63 % |

Per person (window): P1 81 %, P2 90 %, P3 49 %, P4 89 %.
Pose quality on the test part: outline 0.72 (baseline 0.65), body points 0.82 (baseline 0.65), wrists found 0.60 (baseline 0.19).

Limitations of this experiment: windows do not represent complete gestures, and the overlapping test windows are not independent.

## 5. Within person, complete gestures
269 test gestures, three seeds. The best input (arm joints + speed) reaches about **89 %** (89.2 ± 1.4 %). The other two input types gave 88.6 ± 1.7 % and 86.9 ± 1.7 %.
`segment_pipeline.py` writes the exact table (with Wilson 95 % intervals), a per-person breakdown and every single prediction to `seg_results.json` and `seg_predictions.csv`.
Control with video poses: 99–100 %.

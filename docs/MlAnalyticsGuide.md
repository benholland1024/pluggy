# Ultralytics results guide

## How to generate the ultralytics results

After generating training data, the outlet recognition system can be trained with this command:
```bash
uv run yolo detect train data=datasets/outlets/dataset.yaml model=yolo11n.pt epochs=50 imgsz=640
```

This generates `runs/detect/train/` — curves, metrics, and sample images, all written
by training itself (the sample images come from a final validation pass at the end).

Running predictions afterwards writes to a **separate** directory, `runs/detect/predict/`
— one annotated copy of each source image. It does not add anything to `train/`:
```bash
uv run yolo predict model=runs/detect/train/weights/best.pt source=datasets/outlets/images/val
```

## How to interpret the ultralytics results

`runs/detect/train/results.png` shows 2 rows of graphs, with 5 graphs each. Columns 1 to 3
are losses, columns 4 and 5 are metrics. The top row = training set, the bottom row =
validation set. The x-axis is epochs of training.

### The three losses

Lower is better; the absolute values are meaningless, only the trend matters.

 - `box_loss` — how badly the predicted rectangle's coordinates miss the true one.
 - `cls_loss` — how badly it misjudges what the object is. With one class, this is
   effectively "is this an outlet or background?"
 - `dfl_loss` — distribution focal loss, a refinement on box edges. YOLO predicts each
   edge as a probability distribution over positions rather than a single number; this
   sharpens it. Ignore it unless it diverges.

### The four metrics

Higher is better, all 0→1.

| Metric | Meaning |
|---|---|
| `precision` | Of the boxes it drew, what fraction were real outlets. Low = hallucinating outlets. |
| `recall` | Of the real outlets, what fraction it found. Low = missing outlets. |
| `mAP50` | Overall accuracy, counting a box "correct" if it overlaps truth by ≥50%. Lenient. |
| `mAP50-95` | Same, averaged over overlap thresholds 50%→95%. Strict — grades how *tightly* the box fits. |

Precision and recall trade against each other and depend on the confidence threshold you
pick. mAP integrates over all thresholds, which is why it's the headline number.
`mAP50-95` is the one to watch for docking, where box tightness becomes a bearing error.

### Two things that look wrong but aren't

- **Validation loss below training loss.** Expected here: training images get heavy
  augmentation (mosaic, flips, colour jitter) and validation images don't, so the model
  is graded on easier pictures than it practises on. The real overfitting signal is val
  loss *rising* while train loss falls.
- **A sudden step down in the train losses ~10 epochs from the end.** That's
  `close_mosaic` (default 10) switching mosaic augmentation off for the final epochs.
  Deliberate — it's what sharpens box quality at the end.

## Always eyeball the images, not just the curves

Training writes paired images: `val_batch0_labels.jpg` is the truth, `val_batch0_pred.jpg`
is what the model guessed (with confidences), same 16 images in the same layout. Flick
between them and look for tiles that disagree. Also check
`datasets/outlets/contact_sheet.png`, which the dataset generator writes with the
ground-truth boxes drawn — that is what you *taught* the model.

Metrics tell you how much is wrong; these tell you what. **A disagreement can mean the
label is wrong, not the model.** That is exactly how the MSAA segmentation bug was caught
(see SimNotes.md): the model drew two tight, correct boxes where the ground truth was one
absurd wide bar. The metrics alone just looked like a slightly imperfect model.

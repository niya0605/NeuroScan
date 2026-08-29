# Four-class classification audit

The trained Keras artifact has input shape `(224, 224, 3)`, a four-unit softmax output, and the declared class order `glioma`, `meningioma`, `notumor`, `pituitary`. Its EfficientNet backbone contains an internal Rescaling layer, so the deployed worker now matches app.py by sending raw 0–255 pixels.

The exact worker output on the user-supplied labeled files is:

| File           | User-provided label | Model top class | Confidence | Distribution                                                     |
| -------------- | ------------------- | --------------: | ---------: | ---------------------------------------------------------------- |
| `testg.jpeg`   | glioma              |         notumor |     81.33% | glioma 1.14%, meningioma 17.53%, notumor 81.33%, pituitary 0.00% |
| `test2-m.jpg`  | meningioma          |         notumor |     65.77% | glioma 2.73%, meningioma 26.32%, notumor 65.77%, pituitary 5.18% |
| `test3-n.jpeg` | no tumor            |         notumor |     90.61% | glioma 0.27%, meningioma 1.30%, notumor 90.61%, pituitary 7.81%  |
| `test4-p.jpg`  | pituitary           |         notumor |     80.39% | glioma 0.60%, meningioma 16.96%, notumor 80.39%, pituitary 2.05% |

The earlier normalized 0–1 input made every supplied file produce approximately 99.2% glioma, which is a clear preprocessing mismatch. Raw input fixes the no-tumor case but does not make the artifact accurate on the supplied glioma, meningioma, or pituitary images. This is a model-artifact/training-quality problem, not a frontend class-mapping bug. The app must not relabel these outputs to match filenames or force probabilities to 100%. Reliable correction requires the original labeled training/validation data or a retrained model with documented evaluation metrics. A true correction cannot be performed from inference outputs alone; future retraining or recalibration requires the exact dataset split, preprocessing configuration, class-index mapping, training code, and evaluation metrics.

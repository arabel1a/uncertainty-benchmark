# Uncertainty Evaluation + wilds
This repository contain a fork of [WILDS](https://github.com/p-lambda/wilds) with minor compatibility fixes and a monkey-patch for ECE and MCE evaluation.

# Kaggle usage
Since I do not own a GPU cluster, I used a Kaggle's free GPU pool to run experimetnts. The example notebook can be found [here](https://www.kaggle.com/code/arabel1a/t4-ub-civilcoments), it will pull the latest version of this repository, perform a run and save WILDS logs and checkpoints. I do this by "Save notebook version" functionality. Feel free to look at different versoins, there you may find all the checkpoints and logs described below.

# Changes

## `wilds_git` commits 

1. `9efbb70` - updated deprecated imports form `transformers` and disabled SSL verification for datasets downloading since some sources have outdated certificate.
2. `457b77f` - Noticed that at least one WIDLS dataset (civilcomments) supports `prediction_fn` argument (a function that maps logits into predictions, e.g. softmax), but this functionality is not supported by dataset splits and train code. Fixed. Now dataset evaluators have an access to raw probabilities instead of point predictions.
3. `7c8c143` - Wilds uses custom loggers, which close stdout on exit. This results in a kernel freeze on Jupyter-based platforms (e.g. kaggle) and breaks outputs when using self-hosted jupyter notebooks.
4. `1896393` - added mixed precision training support. This allow to leverage TensorCores on some hardware, dramatically improving throughput. For example, kaggle's free-tier 2xT4 machine would have ~16 TFLOPS in total for FP32 and ~130 TFLOPS for FP16.

> Additionally, there is a `windows` branch which have fixes to run on windows machine (but it needs to be merged with main). Windows uses different process spawning strategy, and because of this all code executed on workers should be serializable. This forbids function definition within a function, which was used in wilds original code.

## `evaluate_uncertainty.py`

This is just a minimal monkey-patch that allows to evaluate uncertainty measures. 

TODOs:
* Edit wilds directly instead of patching
* Edit metrics to make them work with wilds `groups` (their abstraction above ID / OOD data)
* Add other metrics 

## Results
**TL;DR** using torch.amp + data parallelism allow to reduce experiment running time up to 2 times, but it is CPU-bottlenecked (most of the time spent in dataloader). This can be overcomed via increasing batch size. Larger batches require more epochs to converge, but (due to higher parallelism) it is still worth it. It seems that bs=256 and half-precision baseline outperforms default wilds baseline both in terms of callibration and OOD performance while being marginally faster to train (>15 experiments / week fits within kaggle free tier, compared to 3.5 if using original wilds code)

| time per epoch | format | ece (global) | worst-group accuracy | batch size | n epochs | comments |
| --- | --- | --- | --- | --- | --- | --- |
| 93m | fp32 | 0.0671 | 0.544 | 16 | 5 | this value matches the WILDS leaderbord exactly |
| 93m | fp32 | 0.0687 | 0.555 | 16 | 5 | same parameters, but different seed to quickly estimate metric stability |
| 45m | fp16 | 0.0781 | 0.541 | 16 | 7 | |
| 24m | fp16 | 0.0326 | 0.546 | 64 | 5 | |
| 18m | fp16 | 0.0019 | 0.573 | 256 | 5 | |


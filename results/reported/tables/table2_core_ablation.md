# Table 2. Core 2x2 ablation of metadata-guided fusion and evidential learning

| Model | Image | Metadata/MGGF | CE | EDL | Accuracy | Precision | Recall | F1 | Specificity | AUC | Error-detection AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Image + CE | Yes | No | Yes | No | 0.7751 | 0.4496 | 0.8056 | 0.5771 | 0.7680 | 0.8534 | 0.7040 |
| Image + Metadata + CE | Yes | Yes | Yes | No | 0.8426 | 0.5772 | 0.6493 | 0.6111 | 0.8881 | 0.8068 | 0.6815 |
| Image + CE + EDL | Yes | No | Yes | Yes | 0.7963 | 0.4762 | 0.6944 | 0.5650 | 0.8203 | 0.8543 | 0.7112 |
| Image + Metadata + CE + EDL (Ours) | Yes | Yes | Yes | Yes | 0.8472 | 0.5953 | 0.6181 | 0.6065 | 0.9011 | 0.8575 | 0.7381 |

Note: CE-only runs use normalized predictive entropy; EDL runs use u = K / S. Error-detection AUC is scale-independent and is therefore the preferred reliability comparison.
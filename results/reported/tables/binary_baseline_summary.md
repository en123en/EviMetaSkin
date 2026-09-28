# Binary Baseline Summary

```text
          model          run_name accuracy balanced_accuracy precision recall     f1 specificity    auc threshold       fps  tp  fp   tn  fn best_val_f1
efficientnet_b0 recall085_sampler   0.7963            0.8025    0.4795 0.8125 0.6031      0.7925 0.8790    0.5800 3159.8800 234 254  970  54      0.7429
efficientnet_b3 recall085_sampler   0.7870            0.7941    0.4659 0.8056 0.5903      0.7827 0.8703    0.5150  248.2284 232 266  958  56      0.7619
       resnet50 recall085_sampler   0.7884            0.7883    0.4671 0.7882 0.5866      0.7884 0.8741    0.6300 3206.6366 227 259  965  61      0.7321
efficientnet_b0           default   0.7745            0.7890    0.4491 0.8125 0.5785      0.7655 0.8618    0.5700  542.5599 234 287  937  54      0.7241
       resnet50           default   0.8089            0.7585    0.4987 0.6771 0.5744      0.8399 0.8639    0.6200 3313.0573 195 196 1028  93      0.7257
efficientnet_b3           default   0.8188            0.7460    0.5201 0.6285 0.5692      0.8636 0.8627    0.7250 1528.2462 181 167 1057 107      0.7434
```

Source root: `results/classification/baselines/binary`
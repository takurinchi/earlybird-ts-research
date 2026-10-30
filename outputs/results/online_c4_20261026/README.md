# Online and posthoc C1-C4 result bundle

CIFAR-10 / VGG16-BN, seed 1. The dense baseline detected t_m=28. Online and posthoc each compare C1=t_m-10 (18), C2=t_m (28), C3=t_m+10 (38), and C4=fixed t_s=40.

The bundle contains configuration, environment metadata, logs, per-epoch metrics, summaries, mask traces, selected masks, aggregate CSVs, and an accuracy plot comparing information modes. Each per-run metrics.csv also contains final test and compact-model metrics in final_* columns. Large model checkpoints are excluded from Git.

All four online and all four posthoc conditions completed 160 epochs and final test evaluation.

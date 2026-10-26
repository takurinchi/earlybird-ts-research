# Online C1-C4 result bundle

CIFAR-10 / VGG16-BN, seed 1. The dense baseline detected t_m=28; online pruning conditions are C1=t_m-10 (18), C2=t_m (28), C3=t_m+10 (38), and C4=fixed t_s=40.

The bundle contains run-level configuration, environment metadata, logs, per-epoch metrics, summaries, the baseline mask-distance trace, selected channel masks, an aggregate CSV, and the online test-accuracy plot. Large initial/final model checkpoints are intentionally excluded; the original run directories remain on this machine under outputs/ but are ignored by Git.

C4 completed 160 epochs and final test evaluation. Posthoc conditions were not run.

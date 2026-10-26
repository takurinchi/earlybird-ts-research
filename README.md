# Early-Bird t_m / t_s 比較実験

CIFAR-10 / VGG16-BNで実行できるEarly-Bird検出・構造化チャネル枝刈り・t_s比較の実験コードです。Early-Bird検出は公式実装のBNスケールによるチャネル選択、正規化Hamming距離、5マスク窓を反映しています。

Rewinding (`t_r`) は追加していません。

## 実装済みの基盤

- 設定読込・共通設定マージ・未設定必須値の検出 (`utils/config.py`)
- 外部モデルとDataLoader factoryの取り込み (`models/model_factory.py`, `data/datasets.py`)
- Dense/Sparse共通の分類学習ループ、マスク適用、チェックポイント、seed/device/logging
- BNスケールによるチャネルマスク候補、BN出力ゲート、マスク保存・復元
- 正規化Hamming距離、公式Early-Birdの5マスク安定判定
- CIFAR向けSGD/MultiStepLRの参考構築関数
- Accuracy/Macro-F1、経過時間、推論レイテンシ、ゼロ率、FLOPs adapter、ファイルサイズ
- 出力ディレクトリと環境メタデータの準備、実験ジョブ計画の生成
- CSV集計、比較表、マスク距離・t_sのプロット関数
- 仕様テストの初期版 (`tests/`)

## 実験設定

`configs/default.yaml` は今回の実行条件を定義します。

- `project.seed`: 公式コード既定の1
- `experiment.dataset`: CIFAR-10
- `experiment.epochs`: 160
- `pruning.selected_ratio`: 0.5
- モデル: CIFAR VGG16-BN
- 学習: batch size 256、SGD (lr 0.1, momentum 0.9, weight decay 1e-4)、80/120 epochでlrを0.1倍

公式リポジトリREADMEはbatch size 256を論文の標準値としていますが、配布コードのCLI既定値は64です。ここでは添付資料に合わせて256を明示しました。枝刈り率候補0.1〜0.8から0.5を固定値として選んでいます。

Early-Bird閾値は `epsilon=0.1`、窓は5個のマスクに設定しました。公式コードでは「最新マスクと直前4マスク」の距離を計算し、すべてが `0.1` 以下なら検出します（公式コードは距離が `0.1` より大きい場合にのみ不安定と判定）。添付文の「直近5回の距離」とは比較回数に1つ差があります。実装は公式コード（5マスク、4距離、閾値同値を許容）に合わせています。

CIFAR-10は元アーカイブと同一内容の[固定コミットミラー](https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python/commit/c6e562a)から取得し、torchvisionのMD5 (`c58f30108f718f92721af3b95e74349a`) で検証します。訓練45,000枚、seed固定の検証5,000枚に分けており、公式設定からこのholdout分だけ変更しています。

## 選択した再現条件と実行

既存の研究条件が共有されていないため、実行用の初期設定として CIFAR-10 / VGG16-BN / 50%チャネル枝刈り / seed 1 を選びました。t_sは `t_m-10`, `t_m`, `t_m+10`, 固定epoch 40とし、オンライン方式・事後比較方式を別々に実行します。前時点が1未満になる場合はepoch 1に丸めます。固定epoch 40と±10はこの比較を実行するための研究設計上の選択で、原論文の推奨値ではありません。

Windowsでは `setup_windows.ps1` がプロジェクト内 `.venv` と CUDA対応PyTorch 2.14.1を用意します。プロジェクト直下で以下を実行します。

```bash
.\.venv\Scripts\python.exe run_experiments.py --config configs/ts_sweep.yaml
.\.venv\Scripts\python.exe run_experiments.py --config configs/ts_sweep.yaml --execute
```

初回はCIFAR-10をダウンロードします。`--execute` はDense baselineとEarly-Bird検出を一つのdense runで行い、オンライン4条件、事後比較4条件を順に実行します。t_mは各epochの学習開始前に検出する公式ループの順序に合わせ、値は完了済みepoch数として記録します。各t_s条件は160 epochで同じseedから学習し直すため、実行時間が長くなります。CUDA対応PyTorchがない場合は実行を停止します。

Dense run完了後に処理が中断した場合は、完了済みbaselineから続けられます。

```powershell
.\.venv\Scripts\python.exe run_experiments.py --config configs/ts_sweep.yaml --execute --resume-baseline outputs/<baseline_run_id>
```

FLOPsはConv2dとLinear演算から1画像あたりの積和FLOPsを計算します。推論時間はbatch size 1でウォームアップ10回、測定100回です。学習データの45,000枚を訓練、5,000枚をseed固定の検証用とし、公式CIFAR-10設定からこのholdout分だけ変更しています。テストデータは最終評価にのみ使用します。

## 出力仕様

`outputs/<experiment_id>/` に `config.yaml`, `environment.json`, `run.log`, `metrics.csv`, `mask_distances.csv`, `summary.json`, checkpoints, masks, plotsを保存します。未検出の `t_m` はsummaryに検出失敗として残し、t_s sweepは開始しません。

## TODOや仕様書の重複・注意点

1. **重複する成果物要件**：5章の評価指標表、7章の `metrics.csv` 列、7章の `summary.json` 項目がAccuracy/F1、時間、サイズ、疎性などで重なっています。正本をCSVの機械可読列とし、summaryは実験単位の設定・完了状態・集約値だけに絞ると二重管理を避けられます。
2. **重複するマスク検証要件**：2章・3.2・8章で距離定義、検出失敗、安定判定が繰り返されています。アルゴリズム定義を一か所に固定し、他の節は参照する形が保守しやすいです。実装は公式コードの閾値境界・窓長に合わせています。
3. **構造化枝刈りの実測**：学習中は `BNChannelMaskManager` がBN出力をゲートし、最終評価前にVGG16のConv/BN/Linear重みを選択チャネルへ物理的に縮約します。FLOPsとサイズは縮約モデルから測ります。推論時間は環境依存の測定値として別途解釈してください。
4. **時間指標の境界**：「学習時間」と「マスク生成・検出時間」の合算・内訳が曖昧です。実行コードでは通常学習、マスク生成/検出、スパース化後学習、評価を別々に計時してください。
5. **t_s設計の採用値**：現設定はC1=`t_m-10`, C2=`t_m`, C3=`t_m+10`, C4=40としています。±10と固定40は本実験の設計選択であり、先行研究で最適化された値ではありません。C1は最低epoch 1に丸めます。
6. **比較の公平性**：オンライン方式は各t_s時点のBN重要度からマスクを作り、事後比較方式は独立した検出runの安定マスクを用います。両方式を別IDで保存します。t_m未検出時はrunを失敗状態として残し、t_s比較を中断します。
7. **評価指標の詳細**：クラスラベル集合、Macro-F1の未観測クラス扱い、推論batch size、入力形状、ウォームアップと繰り返し回数、チェックポイント圧縮形式を設定に追加すると再現性が上がります。
8. **「全グラフ生成」と不足データ**：t_m未検出・FLOPs非対応・欠損metricsがあり得るため、グラフを常に生成するのか、生成不能として状態を記録するのかを定義してください。

## 検証

`pytest -q`: 15件成功。GPUスモーク実行（1 epoch）成功。ベースラインとonline方式C1〜C4を各160 epoch実行し、最終テスト評価まで完了しました。posthoc方式は未実行です。

## 保存済み結果（2026-10-26）

Gitに含める再現・分析用成果物を `outputs/results/online_c4_20261026/` にまとめています。`summary.csv` はbaselineとC1〜C4の最終評価、`metrics_all_epochs.csv` は全800 epochの訓練・検証指標です。`accuracy_by_ts.png` はonline条件のテスト精度比較、baselineのマスク安定距離図は `runs/baseline/plots/mask_distance.png` にあります。各runには設定、環境情報、ログ、epoch別CSV、JSON要約を含め、baseline安定マスクと各online遷移マスクも保存しています。

大きな学習済み重みチェックポイントはGitには含めていません。C4最終結果は `runs/C4_fixed_ts_40/summary.json` に記録されています（CIFAR-10テスト精度92.77%、Macro-F1 92.75%）。C4完了後、posthoc条件は開始していません。

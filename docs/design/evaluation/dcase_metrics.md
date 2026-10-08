# DCASE 指標

公式 evaluator（https://github.com/nttcslab/dcase2026_task2_evaluator）の定義を移植。

| 指標 | 使うクリップ |
|---|---|
| AUC (source) | source の正常 + **全ドメインの異常** |
| AUC (target) | target の正常 + **全ドメインの異常** |
| pAUC | 全クリップ、`roc_auc_score(max_fpr=0.1)`（McClish 補正） |

- `hmean(values)`: `sys.float_info.epsilon` でクリップしてから調和平均
- 公式スコア（`official_score`）= 全機種の AUC(source), AUC(target), pAUC をまとめた調和平均

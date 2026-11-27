# Row-local配置の実験

このパッチは `current/cpu/` の簡素化したキャッシュ実装に適用できます。別の作業ディレクトリで適用すると、2026年11月12日の `iq3-cache.cpp` と `placement.inc` をバイト単位で復元します。実験用ビルドは通常のCPUライブラリと異なるディレクトリに作ります。

```bash
cp -r current/cpu /tmp/xeon-iq-row-local
git -C /tmp/xeon-iq-row-local apply --check "$PWD/current/experiments/row-local/enable.patch"
git -C /tmp/xeon-iq-row-local apply "$PWD/current/experiments/row-local/enable.patch"
```

使用する場合は `GGML_CPU_IQ_ROW_LOCAL` で配置を、`GGML_CPU_IQ_AUDIT_DIR` で監査記録を指定します。元実装は4KiBページ・特定の12物理コアとNUMA番号・THP無効・指定した行列形状を前提とします。CPU交換後は新しい番号と負荷で再評価してください。

この実機で配置実験に使える再構築済みCPUは `/home/codex/projects/llm-lab/ecc9-simplification-20261119/build/control/libggml-cpu.so.0.28.0` です。公開手順の `build_cpu.py --output DIR` は `DIR/control/` にこの同一バイナリ、`DIR/ordinary/` に簡素化版を作ります。上のパッチは両ソースの関係を確認するためにも使えます。

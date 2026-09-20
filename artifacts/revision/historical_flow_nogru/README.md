# Historical Flow no-GRU reference

Preserved historical artifacts only. **The 94.5% wooden-pickaxe result is not claimed reproduced under the corrected genuine-prefix protocol.**

- `model.pkl`: exact retained final policy; local and remote SHA256 verified.
- `config.json`, `wmgym_5m_nogru.log`: matching configuration and full training log.
- `ledger_wmgym.jsonl`: original complete native evaluation ledger.
- `historical_native256_row.json`: the exact final-policy row extracted from that ledger.
- `manifest.json`: identities, native rates, current source hashes, and provenance limits.
- `live_remote_provenance.json`, `current_reset_data.sha256`: read-only remote capture and bounded current reset-data hash.

Historical native evaluation: seed 123, 256 episodes, return 6.220 ± 0.103 SEM; wooden pickaxe 94.5%, wooden sword 3.1%, stone pickaxe 0.4%, stone sword 0%. Rates were originally rounded. No new evaluation was run.

The reset-data path currently resolves to `train_disco_clean200k.npz` (SHA256 536042593dffb63f2b1e4331de5fc84acbdbf87a954787094e35a8de1d85782e). This is separate from the WM metadata's 500k training archive. The historical symlink target and historical policy-training source hashes were not archived, so present-day identities must not be projected backward.

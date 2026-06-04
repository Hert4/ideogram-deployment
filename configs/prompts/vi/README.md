# Vietnamese Smoke Prompts

These prompts are structured JSON captions for `--no-magic-prompt` runs.

Samples:

- `ca-phe-misa-product-ad.json`: product ad with Vietnamese brand text.
- `poster-tet-an-lanh.json`: modern Tet poster with Vietnamese typography.
- `bien-hieu-pho-dem-ha-noi.json`: Hanoi night street scene with signage.

Example:

```bash
IDEOGRAM_CAPTION="$(python scripts/read_prompt.py configs/prompts/vi/ca-phe-misa-product-ad.json)" \
scripts/smoke_mac_mps_int8.sh
```

Run all three on Mac MPS:

```bash
scripts/run_vi_prompts_mac.sh
```

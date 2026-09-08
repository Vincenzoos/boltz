# Boltz UI add-ons

Helpers that sit **on top of** native `boltz predict`. They are not Boltz features.

The notebook UI lives here too (`predict_ui.py`). Each Input mode that needs custom
batching calls the matching module directly — e.g. **Binder remodel** always uses
`binder_remodel.py`. JSON configs do **not** include an `addon` field.

## Layout

```text
add_ons/
  predict_ui.py           # ipywidgets UI (launch_ui)
  binder_remodel.py       # target + many binders → per-pair YAMLs
  common.py
  configs/
    binder_remodel.example.json
    IFIT5_FL_binder_remodel.json   # example job only
```

## binder_remodel schema

```json
{
  "job_name": "my_job",
  "target": { "id": "A", "name": "Target", "sequence": "M..." },
  "binder_id": "B",
  "binders": [
    { "name": "binder_1", "sequence": "S..." }
  ]
}
```

`binders` may also be `{ "binder_1": "S...", ... }`.

Put your own configs in `configs/` (or any path) and load them from the UI.

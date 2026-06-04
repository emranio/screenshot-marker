# Test Fixtures

This directory keeps local screenshot-marker fixtures in three separate
folders:

```text
tests/
├── screens/       # source screenshots
├── annotations/   # annotation result JSON sidecars
└── rendered/      # annotated PNG outputs
```

Run all fixtures from the repository root:

```bash
./run_tests.sh --allow-unresolved
```

The runner reads request text from each matching
`tests/annotations/<stem>.json`, annotates `tests/screens/<stem>.*`, writes
`tests/rendered/<stem>.png`, and replaces the annotation JSON with the fresh
CLI result.

Screenshots without a matching annotation JSON are skipped.

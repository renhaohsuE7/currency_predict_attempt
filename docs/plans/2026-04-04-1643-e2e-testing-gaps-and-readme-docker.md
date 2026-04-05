# E2E Testing Gaps + README Docker Commands

- **Date**: 2026-04-04 16:43
- **Status**: completed
- **Module**: tests, README

## Goal

Fix two independent issues:

1. **E2E Testing Gaps**: `--backtest` CLI routing untested, no backtest offline E2E with real models, no pipeline output verification
2. **README Docker Commands**: Make Docker the primary command style throughout README

## Impact

### Issue 1: Testing
- `tests/test_e2e_cli.py` — add backtest arg parsing, routing, validation tests
- `tests/test_e2e_offline.py` — add `TestOfflineE2EBacktest` with real PatchTSTSklearn
- `tests/test_e2e_pipeline_output.py` — new file for pipeline output verification

### Issue 2: README
- `README.md` — restructure all command sections: Docker primary, Local alternative

## Implementation

See full plan details in `.claude/plans/squishy-mixing-pnueli.md`

## Completion Criteria

- 15 new tests pass (12 fast + 3 slow)
- Existing 522 fast tests unaffected
- README Docker-first throughout

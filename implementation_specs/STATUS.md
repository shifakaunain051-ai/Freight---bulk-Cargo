# NaviFreight Implementation Status

## Source of truth
The files in this directory are the implementation specification.

Read all specification files before changing the repository.

## Completed
- Genuine freight dataset established
- Model V3 frozen
- Commodity procurement service
- Vessel suitability service
- Decision orchestration
- Backend regression/integration tests
- Landed-cost specification
- Frontend specification
- Audit/QA specification

## Immediate next steps
1. Inspect repository against all specs.
2. Audit decision semantics and weather override behavior.
3. Implement/finalize landed-cost backend logic.
4. Implement frontend in bounded stages.
5. Run full QA and demo scenarios.

## Operating rule
Do not rewrite or replace completed backend/model work without a concrete specification reason.

Do not modify Model V3 or the genuine training dataset.

Use small, bounded implementation tasks.

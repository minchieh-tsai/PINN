# Generated POC Spec Pointer

The source specification for this generated repository is:

`PINN_EPI_CODEX_SPEC_H350_W200_POC.md`

The implementation follows the POC scope in that file:

- XLSX state loading with strict `(350, 200)` validation.
- Signed-distance level-set preprocessing.
- 20-point contour extraction.
- Separate `DepositionPINN` and `EtchPINN` models.
- Rollout from `2E` through `5E`.
- Holdout evaluation against `5M` and `5E`.
- Known-average-rate baseline.

The source spec explicitly excludes pytest, unittest, synthetic test code,
`tests/`, and CI/CD artifacts, so those are not generated here.

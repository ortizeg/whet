---
name: library-review
description: >
  Use this skill when deciding whether to adopt a third-party Python library — evaluating
  maintenance, community, docs, type-hint support, license compatibility, security
  history, and performance, then classifying it Adopt/Trial/Assess/Hold. Reach for it any
  time you'd otherwise add a dependency on gut feel or wonder "is this package safe and
  maintained enough to depend on?", even if the user doesn't say "library review". For
  designing wrappers around a library once adopted, see abstraction-patterns.
---

# Library Review and Evaluation Framework

Every dependency is a long-term commitment carrying security, maintenance, breaking-change,
license, and transitive-dependency risk. Evaluate new libraries before adoption using a
technology radar (Adopt / Trial / Assess / Hold) plus a wrapping strategy that isolates
third-party APIs behind project-owned interfaces. This page is the review loop; the deep
dives hold the scoring tables, radar criteria, and wrapper patterns.

## The review loop

Score the candidate on eight criteria, each rated green / yellow / red:

1. **Maintenance status** — last commit, last release, issue triage, CI health, Python support.
2. **Community size** — stars, contributors, PyPI downloads, Stack Overflow presence.
3. **Documentation quality** — API reference, tutorials, runnable examples, changelog, migration guides.
4. **Type hint support** — inline hints, `py.typed` marker, passes `mypy --strict`, available stubs.
5. **License compatibility** — permissive (MIT/Apache/BSD) is fine; GPL/AGPL is disqualifying for MIT or Apache projects.
6. **Security history** — CVE record and patch speed, clean `pip-audit`, published SECURITY.md.
7. **Performance** — benchmark candidates head-to-head when the library sits in a hot path.
8. **Stack integration** — Python version, `pip check` conflicts, platform wheels, PyTorch/Lightning and uv/conda availability.

The red thresholds that most often sink a candidate:

| Signal | Red |
|--------|-----|
| Last commit | > 12 months ago |
| Last release | > 18 months ago |
| GitHub stars / contributors | < 100 stars, < 5 contributors |
| PyPI downloads | < 10K/month |
| Type hints | No inline hints, no `py.typed`, no stubs |
| License | GPL or AGPL (copyleft, viral) |
| Security | Multiple CVEs with slow patches, or critical audit findings |
| Docs | No API reference, no changelog |

A fast first pass on the two cheapest signals:

```bash
gh api repos/OWNER/REPO --jq '{pushed: .pushed_at, stars: .stargazers_count, license: .license.spdx_id}'
pip-audit --requirement requirements.txt
```

Then classify:

| Verdict | Meaning |
|---------|---------|
| **Adopt** | All criteria green/yellow, production-proven, active maintenance. Use freely. |
| **Trial** | Mostly green/yellow, gaining traction, has a champion. Use in a non-critical path, wrapped. |
| **Assess** | Mixed signals or no hands-on experience. Proof-of-concept in a branch first. |
| **Hold** | Multiple reds, abandoned, incompatible license, or better alternatives exist. Do not adopt. |

Record the outcome with the review template (see the technology-radar deep dive) so the
decision is auditable rather than folklore.

## Wrap what you adopt

Never call a third-party API directly from across the codebase. Define a project-owned
Protocol or ABC and put the library behind it — then a breaking change or a swap touches
one file:

```python
from typing import Protocol

class ImageProcessor(Protocol):
    """Interface for image processing."""
    def process(self, image: np.ndarray, threshold: float = 0.5) -> np.ndarray: ...

class SomeLibraryProcessor:
    """Image processor using some_library."""

    def process(self, image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return some_library.process(image, mode="fast", threshold=threshold)
```

For optional dependencies, ship a no-op implementation of the same interface (Null Object
pattern) so calling code never branches on whether the library is installed.

## Conventions

1. **Evaluate before installing**: Run through the checklist before adding any dependency.
2. **Wrap all third-party APIs**: Isolate external code behind project interfaces.
3. **Minimize transitive dependencies**: Prefer libraries with few dependencies of their own.
4. **Pin versions**: Use exact versions in lock files, compatible ranges in requirements.
5. **Audit regularly**: Run `pip-audit` in CI and review outdated packages monthly.
6. **Document decisions**: Use the review template for every new dependency.
7. **Use the Null Object pattern**: Provide no-op implementations for optional dependencies.
8. **Prefer standard library**: Use built-in modules when they are sufficient.
9. **Track the Technology Radar**: Maintain a team document classifying all dependencies.
10. **Plan for migration**: Assume every library will eventually need to be replaced.

## Anti-patterns

- **Adding a dependency on gut feel** — a single convenient function is not worth an unscored, unaudited package.
- **Scattering library calls across the codebase** — direct `some_library.foo()` calls in twenty files make every upgrade a twenty-file diff.
- **Ignoring the license** — a GPL or AGPL dependency in an MIT or Apache project is a legal problem, not a style preference.
- **Reviewing only at adoption** — a green library goes red silently; re-audit monthly and re-evaluate annually.
- **Unpinned versions** — without a lock file, "it worked yesterday" is not reproducible.
- **Adopting on popularity alone** — high star counts do not imply type support, a compatible license, or a clean CVE record.
- **Skipping the wrapper for a Trial library** — Trial exists precisely because you may have to rip it back out.

## Deep dives

- `references/dependency-risk-assessment.md` — read when scoring a candidate in detail: the full green/yellow/red tables and the `gh`/`pip`/benchmark commands for maintenance, community, docs, types, performance, and stack integration.
- `references/security-and-license-review.md` — read when checking license compatibility or CVE history, wiring `pip-audit` and dependency review into CI, or setting up the recurring audit schedule.
- `references/technology-radar.md` — read when assigning a final Adopt/Trial/Assess/Hold verdict or writing up the dependency proposal with the review template.
- `references/api-wrapping.md` — read when designing the wrapper interface for a newly adopted library, adding a Null Object fallback, or planning a migration off an existing dependency.

# ADR 0004: Publish architecture only by explicit action

- **Status:** Accepted
- **Date:** 2026-07-30

## Context

GitHub Pages content can be public even when its source repository is private.
Architecture diagrams are suitable for publication only when the output excludes
private plans, datasets, model files, secrets and runtime application assets.

## Decision

The architecture workflow runs only by manual dispatch. It installs the locked
LikeC4 dependency, validates and builds with the repository base path, scans links
and publication boundaries, then uses GitHub's official Pages artifact and deploy
actions.

## Consequences

- Publication is a conscious human action.
- Repository settings and account eligibility must still be enabled by an owner.
- The generated site contains documentation only; it is not the screening app.

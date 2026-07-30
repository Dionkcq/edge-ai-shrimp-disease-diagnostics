# API contract

**Current schema version: 1.0.0**

The files in this directory are **generated**. The Pydantic models under
`backend/src/shrimp_screening/contracts/` are the single source of truth; these
JSON Schema documents are what every other language binds to.

```bash
make schema        # regenerate
make test          # a contract test byte-compares committed vs freshly generated
```

A pull request that changes a contract model without rerunning `make schema` fails
CI. That is deliberate: a generated artifact that can drift is not a contract.

| File | Produced by | Describes |
|---|---|---|
| `screening_result.schema.json` | `contracts/screening.py` | `200` body of `POST /api/v1/screenings` |
| `problem_detail.schema.json` | `contracts/problem.py` | every `4xx`/`5xx` body, `application/problem+json` |
| `guidance_document.schema.json` | `contracts/guidance.py` | `200` body of `GET /api/v1/guidance/{decision}` |

## Versioning rules

`schema_version` follows semver over the **wire format**, not the codebase.

**Patch** — documentation, descriptions, examples. No client can observe it.

**Minor** — a new optional field, or a new member of an *open* vocabulary
(`NoticeCode`, `QualityReason`, `ProblemCode`). Clients must ignore unknown members
of these rather than crash, and every one of them is additive by design.

**Major** — anything else: removing or renaming a field, tightening a type,
changing a unit, or **adding a member to a closed vocabulary**.

## Closed vocabularies

Two enumerations are closed and a change to either is a major version bump *and* a
safety review, not a schema edit:

- **`Decision`** — exactly five members. A sixth would be the most plausible route
  by which this project starts making a claim it cannot support. `UNABLE_TO_ASSESS`
  absorbs quality failure, model unavailability, low confidence and inference
  failure; the *reason* lives in `abstention_reason`, never in the decision.
- **`MarkerRole`** — the screening meaning of a detected class. Adding a role means
  the product screens for something new, which is a scope change.

## Invariants a client may rely on

1. `decision` is always exactly one of the five members, in every response, in
   every failure mode short of an HTTP error.
2. `abstention_reason` is non-null **if and only if** `decision` is
   `UNABLE_TO_ASSESS`.
3. Boxes are normalized `xyxy` in the EXIF-corrected **original** image frame, and
   `image.width` / `image.height` are post-transpose. A client never has to reason
   about orientation, and an overlay positioned by percentage cannot drift from the
   photograph.
4. Labels come from `model.class_names`. **No index-to-label table exists anywhere
   in the backend.** If an exported model's class order flips, the labels in the
   response change with it instead of silently mislabelling.
5. `model.is_demonstration_data` is `true` for every synthetic response. A user
   interface is required to render that permanently and unmissably.
6. `quality.policy_hash` and the policy identifiers make a result reproducible:
   same image plus same hashes yields the same decision.
7. `limitations[]` contains identifiers defined in `docs/LIMITATIONS.md`, and a
   test asserts every emitted identifier exists there.

## Errors

`application/problem+json` per RFC 9457, with a stable `code` extension member.
`type`, `title` and `detail` are human-facing and may be reworded without a
contract break; `code` and `status` may not.

| `code` | HTTP | When |
|---|---|---|
| `MALFORMED_REQUEST` | 400 | no `image` part, bad multipart, illegal parameter |
| `UNSUPPORTED_MEDIA_TYPE` | 415 | not `multipart/form-data`, or not a JPEG/PNG by magic bytes |
| `PAYLOAD_TOO_LARGE` | 413 | body exceeds the cap, or decoded pixels exceed the cap |
| `UNDECODABLE_IMAGE` | 422 | truncated, animated, or otherwise undecodable |
| `SERVICE_BUSY` | 503 | inference queue timed out; carries `Retry-After` |
| `NOT_FOUND` | 404 | unknown decision or route |
| `INTERNAL_ERROR` | 500 | anything unhandled; the cause is never echoed |

`detail` is a fixed string chosen at the raise site. It never contains anything
derived from the request body, the filename, the headers or EXIF.

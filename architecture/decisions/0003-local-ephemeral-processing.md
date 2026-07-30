# ADR 0003: Local, ephemeral image processing

- **Status:** Accepted
- **Date:** 2026-07-30

## Context

Farm images and their metadata can be sensitive. The target deployment must work
on an isolated local network and does not need analytics, cloud inference or
research logging.

## Decision

The browser sends one original image over the local LAN to the laptop service.
The service performs bounded parsing, strips metadata from the processing path,
keeps encoded bytes and decoded pixels in memory only, and logs only a generated
request identifier. The runtime does not call internet services.

## Consequences

- No image archive, EXIF/GPS store, telemetry stream or cloud inference endpoint is
  part of the runtime architecture.
- Input remains untrusted until magic-byte, dimension, decode and quality checks
  pass.
- Operators must deliberately preserve the WAN-disabled deployment posture.

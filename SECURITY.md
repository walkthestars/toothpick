# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

Please **do not** report security vulnerabilities through public GitHub issues.

Use [GitHub Private Vulnerability Reporting](https://github.com/walkthestars/toothpick/security/advisories/new)
to submit a report privately to the maintainer. Include:

- The affected command, parser, or module (for example, `toothpick unpack`, `toothpick jvm ingest`, or `toothpick mcp`).
- A minimal reproducer input or command sequence.
- Impact assessment (for example, path traversal on archive extraction, out-of-bounds memory read, or command injection).

You will receive an acknowledgment within 72 hours and a coordinated remediation timeline once the issue is reproduced.

## Security Model & Operating Boundaries

Toothpick analyzes untrusted binaries, archives, and memory dumps. Keep the following boundaries in mind when operating it:

1. **Static analysis vs. dynamic execution**: `toothpick triage`, `toothpick static`, `toothpick query`, `toothpick solve`, `toothpick rtti`, and `toothpick jvm ingest` do not execute target binaries on the host OS. `toothpick unpack` runs packer stubs inside an isolated Unicorn CPU micro-emulator, not natively on the host CPU.
2. **Dynamic commands attach to live processes**: `toothpick ground` (without `--static-only`), `toothpick scan --pid`, `toothpick cdb`, and `toothpick dbi` interact with live OS processes or debuggers. Only run dynamic instrumentation against untrusted binaries inside an isolated, snapshotted virtual machine.
3. **Archive and parser bounds**: PCK (`toothpick pck extract`), JAR (`toothpick jvm ingest`), and Lua Workshop (`toothpick lua`) parsers enforce strict zip-slip path traversal rejection and decompression size ceilings.
4. **Offline operation**: Toothpick makes zero outbound network requests and never transmits binary bytes, symbols, or decompiled code off the host machine.

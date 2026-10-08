# Security and configuration review

**Review date:** 2026-10-08  
**Scope:** InsightAI source, configuration examples, upload handling, database access, and LLM error paths.

## Findings

- **Secrets:** `.env` is excluded by `.gitignore`; `.env.example` contains an empty Groq key and a local SQLite URL. Git does not track `.env`. Keep real provider and PostgreSQL credentials in environment variables or the private `.env` file.
- **Uploads:** Sales uploads are limited to 10 MiB and accepted by extension plus the existing data parser and schema validator. PDFs are limited to 15 MiB and 500 pages; encrypted or textless PDFs are rejected. Filenames are reduced to basenames before storage.
- **Database queries:** Sales persistence uses SQLAlchemy ORM expressions. Document-store SQL uses bound parameters for values; the optional document filter changes only a fixed SQL clause. No user-provided query text is executed as SQL.
- **LLM and code execution:** Groq credentials come from configuration. Provider and database failures use sanitized application messages. User questions and PDF contents are input data; the model does not generate executable Python or SQL.
- **Dependency configuration:** Runtime dependencies are listed in `requirements.txt`. `pip check` passed during this review; no dependency vulnerability scanner was run.

## Data handling

When document Q&A is requested, retrieved PDF excerpts and the question are sent to Groq for answer generation. Do not upload confidential or regulated documents unless this transfer is approved for the configured Groq account and environment. Sales records are stored in the configured SQL database. Indexed PDF text and vectors are stored in the local SQLite document store.

## Known limits

- The API has no authentication, authorization, rate limiting, or per-user data separation. Run it only on a trusted local network; do not expose it publicly as-is.
- Upload size and PDF page limits reduce resource use but do not provide a full malware scan, OCR, or a strict CPU/memory budget for hostile files.
- The app does not define document or sales-data retention policies. Deleting a document through the app removes it from the local document store; backups and filesystem snapshots are outside the app's control.
- HTTPS, private database networking, production secret management, and deployment hardening remain deployment responsibilities and have not been validated here.

This is a focused portfolio-project review, not a penetration test or production security certification.

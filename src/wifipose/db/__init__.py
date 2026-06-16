"""Persistence: async SQLAlchemy engine, schema, and the data-access repository.

Designed to run on TimescaleDB locally and on plain Postgres (Supabase/Neon) in
free-tier cloud. TimescaleDB hypertables and pgvector similarity search are used
when the extensions are present, and the code degrades gracefully when they are
not — detected once at startup (see `session.Capabilities`).
"""

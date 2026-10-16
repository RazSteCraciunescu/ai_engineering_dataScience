# Architecture

The repository follows a layered flow: immutable synthetic exports, format-specific collectors, canonical preparation, validation and SQLite loading, then analytics, modeling, retrieval, image quality and serving.

The core uses local files, SQLite and compact scikit-learn artifacts. Hosted tracking tools, external vector databases and large neural models are adapters rather than hidden dependencies.

## Boundaries

- The API is read-oriented and binds to localhost by default.
- Model features use information available at initial triage; final status and post-resolution outcomes are excluded.
- Raw resolution values remain available even when a capped training target is used.
- SQLite suits portfolio scale; production would use controlled migrations and a managed transactional store.

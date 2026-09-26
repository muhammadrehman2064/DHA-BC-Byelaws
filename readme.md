# Data directory

Place the source building byelaw PDF here.

Expected filename:

building_bylaws.pdf

Then run:

python ingest.py

This will create:

index.json

The `index.json` file contains:

- extracted document chunks
- PDF page numbers
- vector embeddings
- metadata required for retrieval

Do not manually edit index.json.
Regenerate it whenever the source PDF changes.

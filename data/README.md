# Document Data

Place the source building byelaw PDF in this folder.

The expected filename is:

building_bylaws.pdf

After placing the PDF here, run:

python ingest.py

This will create:

index.json

The index contains:

- PDF text chunks
- PDF page numbers
- local embeddings
- document metadata

Do not manually edit index.json.

If the PDF changes, run:

python ingest.py

again.

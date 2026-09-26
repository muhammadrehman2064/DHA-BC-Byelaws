# 🏗️ Building Byelaw RAG Assistant

A beginner-friendly Retrieval-Augmented Generation (RAG)
application for asking questions about a building byelaw PDF.

The application uses:

- Streamlit
- Groq API
- Sentence Transformers
- PyPDF
- NumPy

---

# Architecture

```text
                 BUILDING BYELAW PDF
                         |
                         v
                    ingest.py
                         |
                         v
                 Extract PDF text
                         |
                         v
                      Chunks
                         |
                         v
              Sentence Transformer
                         |
                         v
                  Embeddings
                         |
                         v
                    index.json
                         |
                         |
                         v
User Question ---> Streamlit
                         |
                         v
                Question Embedding
                         |
                         v
                 Similarity Search
                         |
                  +------+------+
                  |             |
              Weak Match     Good Match
                  |             |
                  v             v
             Refuse          Groq LLM
                                |
                                v
                         Grounded Answer
                                |
                                v
                          Page Citations

1. Requirements

Python 3.10 or newer is recommended.

Install dependencies:

pip install -r requirements.txt

2. Add your PDF

Place your building byelaw PDF here:

data/building_bylaws.pdf

3. Configure Groq

Create a local file:

.env


Add:

GROQ_API_KEY=your-groq-api-key


Never commit .env to GitHub.

4. Build the document index

Run:

python ingest.py


This will:

Read the PDF.

Extract text page by page.

Split the text into chunks.

Generate local embeddings.

Save the embeddings to:

data/index.json

5. Start the application

Run:

streamlit run app.py

6. Ask questions

Example:

What is the minimum required width of a staircase?


The application will:

Convert the question into an embedding.

Search the document.

Retrieve the most relevant passages.

Check similarity.

Send relevant passages to Groq.

Generate a grounded answer.

Display source PDF pages.

7. Hallucination protection

The application uses several controls.

Retrieval threshold

If the document does not appear sufficiently related
to the question, the LLM is not called.

The application responds:

I could not find sufficient information in the
provided building byelaw document to answer this.

Grounded prompt

The Groq model is instructed to:

use only retrieved passages

not use outside knowledge

not invent measurements

not invent clauses

not invent regulations

not invent exceptions

cite PDF pages

refuse when information is insufficient

8. Updating the PDF

If you replace the PDF:

data/building_bylaws.pdf


run:

python ingest.py


again.

This regenerates:

data/index.json


Then commit both the changed PDF/index if appropriate.

9. GitHub

Initialize Git:

git init


Add files:

git add .


Commit:

git commit -m "Initial building byelaw RAG app"


Push:

git push

10. Streamlit Community Cloud

Deploy the repository using Streamlit Community Cloud.

Set the main file to:

app.py


Then configure the secret:

GROQ_API_KEY = "your-groq-api-key"


Do NOT put the Groq API key inside Python files.

11. Important PDF limitation

If the PDF is a scanned/image-only PDF, PyPDF may not
extract its text.

If ingestion reports:

Warning: Page X has no extractable text.


the PDF may require OCR before it can be used effectively.

12. RAG settings

The application currently uses:

TOP_K = 5

SIMILARITY_THRESHOLD = 0.35

CHUNK_SIZE = 1200

CHUNK_OVERLAP = 200


These are starting values, not universal values.

For a better application, test the system with questions
whose answers are both inside and outside the document.

13. Important security rule

Never commit:

.env


or:

.streamlit/secrets.toml


to GitHub.

Your API key should remain in Streamlit Secrets when deployed.

:::

---

# 11. Ab deployment ka exact process

Ab **old project ko patch nahi karna**. Ye sequence follow karo.

### Step A — GitHub

Repository mein ye files rakho:

```text
app.py
config.py
ingest.py
rag.py
requirements.txt
.env.example
.gitignore
README.md

data/
    building_bylaws.pdf
    README.md

.streamlit/
    config.toml


Old index.json delete karo.

Step B — Local installation

Terminal mein:

pip install -r requirements.txt


Phir:

python ingest.py


Expected output kuch is tarah hoga:

BUILDING BYELAW RAG - PDF INGESTION

Reading PDF: data/building_bylaws.pdf
Total PDF pages: 150

Loading embedding model:
sentence-transformers/all-MiniLM-L6-v2

Creating embeddings for 300 chunks...

INDEX CREATED SUCCESSFULLY
Index: .../data/index.json
Chunks: 300

Ingestion completed.


Phir check karo:

data/index.json


exist karta hai.

Step C — Local Streamlit test

.env:

GROQ_API_KEY=gsk_xxxxxxxxx


Phir:

streamlit run app.py


Browser mein app khulega.

Ek question test karo:

What is the minimum required staircase width?

12. Phir GitHub par push

index.json banne ke baad:

git add .
git commit -m "Rebuild RAG application with Groq"
git push


Streamlit ko redeploy karne do.

13. Streamlit Cloud Secrets

Streamlit Cloud mein:

Manage app → Settings → Secrets

Sirf ye add karo:

GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxx"


Streamlit ki documentation bhi deployed applications ke liye secrets ko application settings mein store karne ka recommended workflow batati hai. {"fallbackMarkdown":"(Streamlit Docs
)","reference":{"matched_text":"","prefix":null,"start_idx":39123,"end_idx":39156,"safe_urls":["https://docs.streamlit.io/deploy/concepts/secrets","https://docs.streamlit.io/deploy/concepts/secrets?utm_source=chatgpt.com","https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management","https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management?utm_source=chatgpt.com"],"refs":[],"alt":"(Streamlit Docs
)","prompt_text":null,"type":"grouped_webpages","items":[{"title":"Secrets management for your Community Cloud app - Streamlit Docs","url":"https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management?utm_source=chatgpt.com","attribution":"Streamlit Docs","pub_date":null,"snippet":"","attribution_segments":null,"supporting_websites":[{"title":"Managing secrets when deploying your app - Streamlit Docs","url":"https://docs.streamlit.io/deploy/concepts/secrets?utm_source=chatgpt.com","pub_date":null,"snippet":"","attribution":"Streamlit Docs"}],"refs":[{"turn_index":0,"ref_type":"search","ref_index":1},{"turn_index":0,"ref_type":"search","ref_index":14}],"hue":null,"attributions":null}],"error":null,"style":null,"status":"done","fallback_items":null},"showLoginRequiredCard":false}

Ek bahut important correction

Is naye project mein OpenAI package/API kahin use nahi ho raha:

❌ from openai import OpenAI
❌ import openai
❌ OPENAI_API_KEY


LLM:

Groq
    ↓
openai/gpt-oss-20b


Embedding:

Sentence Transformers
    ↓
all-MiniLM-L6-v2


Groq ka official Python example bhi from groq import Groq aur client.chat.completions.create(...) use karta hai. {"fallbackMarkdown":"(Groq Console
)","reference":{"matched_text":"","prefix":null,"start_idx":39576,"end_idx":39595,"safe_urls":["https://console.groq.com/docs/text-chat","https://console.groq.com/docs/text-chat?utm_source=chatgpt.com"],"refs":[],"alt":"(Groq Console
)","prompt_text":null,"type":"grouped_webpages","items":[{"title":"Text Generation - GroqDocs","url":"https://console.groq.com/docs/text-chat?utm_source=chatgpt.com","attribution":"Groq Console","pub_date":null,"snippet":"","attribution_segments":null,"supporting_websites":[],"refs":[{"turn_index":0,"ref_type":"search","ref_index":3}],"hue":null,"attributions":null}],"error":null,"style":null,"status":"done","fallback_items":null},"showLoginRequiredCard":false}

Aur ek critical baat: ingest.py ko naye embedding model ke saath dobara run karna hai. Purana index.json agar OpenAI embeddings se bana hua hai to use naye rag.py ke saath use mat karna.

Abhi aap sirf ye 4 kaam karein:
1. Purani Python files + index.json delete
2. Upar wali fresh files copy
3. python ingest.py
4. streamlit run app.py


Uske baad deployment karein. Is tarah agar error aata hai to hum ek clean baseline se debug karenge, purane OpenAI/Groq code ke mix ko nahi.{"fallbackMarkdown":"","reference":{"matched_text":" ","prefix":null,"start_idx":40110,"end_idx":40110,"safe_urls":[],"refs":[],"alt":"","prompt_text":null,"type":"sources_footnote","sources":[{"title":"Supported Models - GroqDocs","url":"https://console.groq.com/docs/models?utm_source=chatgpt.com","attribution":"Groq Console"},{"title":"Secrets management for your Community Cloud app - Streamlit Docs","url":"https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management?utm_source=chatgpt.com","attribution":"Streamlit Docs"}],"has_images":false},"showLoginRequiredCard":false}

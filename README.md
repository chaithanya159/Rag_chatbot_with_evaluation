# Marginalia: a RAG document Q&A chatbot with an evaluation harness

Upload a PDF, DOCX, TXT or MD file and ask questions about it. Each answer comes with the passages it was based on. The project also includes a small evaluation setup that I used to measure the system and tune how documents are split into passages.

![Same question asked at 300-word and 200-word passages](docs/live-demo-chunk-size.png)

*The same question ("explain map") asked at two chunk sizes. At 300-word passages the chatbot says the document does not contain the answer. After re-indexing at 200 words, it answers correctly.*

<!-- Demo video: add your link here, e.g. [Watch the 1-minute demo](https://...) -->

## What it does

- Loads PDF, DOCX, TXT and MD files and answers questions using only the document's content.
- Shows the source passages (with page numbers for PDFs) behind every answer.
- Lets you change chunk size and overlap in the UI and re-index the same file, keeping the chat, so you can compare settings side by side. Each answer is labeled with the settings it was produced under.
- Says so when the document does not contain the answer, instead of inventing one.
- Remembers the last few turns for follow-up questions.

## How it works

1. **Load:** `pypdf` and `python-docx` extract text, keeping page numbers for PDFs.
2. **Chunk:** text is split by paragraph, then packed sentence by sentence into chunks of about N words, with M words of overlap between chunks (own implementation, no LangChain).
3. **Embed:** each chunk becomes a 384-dimensional vector with `all-MiniLM-L6-v2` (runs locally, normalized).
4. **Store:** vectors go into a FAISS `IndexFlatIP` index (exact cosine similarity); chunk text and metadata are kept alongside and saved to disk.
5. **Retrieve:** the question is embedded the same way and the top passages are returned, with a minimum similarity score.
6. **Generate:** the passages, the last 5 turns and a system prompt are sent to a model on Groq (Qwen3.8-27B). The prompt says to answer only from the passages and to admit when the answer is missing.
7. **Post-process:** markdown is stripped, sources are attached, and the turn is saved to a history file.

## Results

The test document is a 22-page excerpt of NIST AI 100-1 (AI RMF 1.0): printed pages 1 to 9 and 12 to 24. I wrote 15 questions with reference answers (13 answerable, 2 that the document cannot answer), checked each against the PDF text, and ran them through the chatbot.

### Tuning chunk size and overlap

The first evaluation showed one clear failure: the question "What is the purpose of the MAP function?" came back as "not in the document" even though the answer is on page 22. The retrieval-only experiment (`experiment.py`, no LLM calls) then compared chunk settings:

| Chunk size / overlap | Chunks | hit@5 | MRR | Rank of the MAP passage |
|---|---|---|---|---|
| 100 / 25 | 136 | 0.92 | 0.77 | 2 |
| 100 / 50 | 196 | 0.92 | 0.85 | 2 |
| 150 / 25 | 80 | 0.92 | 0.86 | 1 |
| 150 / 50 | 96 | 0.92 | 0.89 | 2 |
| 200 / 25 | 59 | 0.92 | 0.85 | 2 |
| **200 / 50** | **65** | **0.92** | **0.89** | **1** |
| 300 / 25 | 39 | 0.85 | 0.86 | below 10 |
| 300 / 50 (original) | 39 | 0.85 | 0.85 | below 10 |

*hit@5: share of the 13 answerable questions where one of the top 5 passages comes from an expected page. MRR: how high the first correct-page passage ranks (1.0 is best).*

The cause: at 300 words, the sentence defining MAP made up about 5% of a chunk that was otherwise a table about a different function, so the chunk did not match a question about MAP. Smaller chunks fixed it. Full evaluation before and after the change (same 15 questions, same models):

| Metric | 300 / 50 | 200 / 50 |
|---|---|---|
| Correctness (against reference answers) | 0.93 | **1.00** |
| Page hit rate (right page retrieved) | 0.85 | **0.92** |
| Answer relevancy | 0.95 | **0.99** |
| Faithfulness (answer stays within the retrieved passages) | **1.00** | 0.96 |
| Correctly declined the 2 unanswerable questions | 2 of 2 | 2 of 2 |

Faithfulness dipped slightly after the change. The MAP question moved from correctness 0.0 to 1.0.

### Hold-out check

I then wrote 10 further questions (8 answerable, 2 unanswerable) and ran them once, without tuning on them:

| Metric | Hold-out (200 / 50) |
|---|---|
| Correctness | 0.90 (9 of 10) |
| Page hit rate | 0.88 (7 of 8) |
| Faithfulness / relevancy | 1.00 / 1.00 |
| Correctly declined the 2 unanswerable questions | 2 of 2 |

The one miss: "What does the AI RMF say makes an AI system 'safe'?" The passage with the definition was not retrieved, and the answer described other safety practices instead. Faithfulness and relevancy both scored 1.0 on it, and only the reference-based correctness metric caught it. Two of the 10 questions reuse passages from the tuning set (one of them is the MAP question again, as a regression check), so only 7 are fully unseen.

### How the evaluation works (`evaluate.py`)

- **Page hit:** whether a retrieved passage comes from a page that holds the answer.
- **Keyword coverage:** whether key terms from the reference appear in the retrieved text.
- **Faithfulness and answer relevancy:** judged by an LLM that sees the question, the retrieved passages and the answer.
- **Correctness:** judged by an LLM against the reference answer. It does not see the retrieved passages, so it can catch an answer that is faithful to the wrong passages.
- **Abstention:** whether the answer says the information is not in the document, for the unanswerable questions.
- **Separate judge:** the answers come from Qwen3.8-27B and are graded by a different model, `gpt-oss-120b`, to reduce a model favoring its own writing. Both run on Groq.

## Limitations

- **Small test sets.** With 15 questions, one question moves a score by about 0.07, so treat the gains as a good sign and not proof.
- **One document.** The settings were tuned on a single NIST document and may not be best for other documents.
- **Tuned on the main set.** The hold-out set is the fairer number, and it is lower (0.90 correctness).
- **LLM judges are imperfect.** They can be strict or lenient, and scores vary slightly between runs. I checked a few judgments by hand but did not measure agreement with human grading.
- **Page numbers inside answers.** The model writes page numbers itself and they are not always exact. The source tabs show the real pages.
- **Preview model.** The generator is a preview model on Groq. Model names and availability change, so you may need to pick a different one (see below).

## Not production-ready

This is a single-user local app. There is one shared index and one chat history file, no login, and it uses Flask's development server. The server listens only on your own computer. Do not put it on the public internet as it is.

## Run it

You need Python 3.10 or newer and a free Groq API key from console.groq.com.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and put your key in it:

```
GROQ_API_KEY=your_key_here
```

Start the app and open http://localhost:5000:

```bash
python app.py
```

The first start downloads the embedding model (about 90 MB).

**Choosing the model.** The answer model is set by `GROQ_MODEL` in `rag/pipeline.py`. Groq's available models change, so list the ones your key can use and edit that line if needed:

```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); from groq import Groq; c=Groq(api_key=os.getenv('GROQ_API_KEY')); print('\n'.join(sorted(m.id for m in c.models.list().data)))"
```

If your model does not support the `reasoning_effort` option, remove the `extra_body` line next to it. The judge model is `JUDGE_MODEL` (default `openai/gpt-oss-120b`) and can be set in `.env`.

### Run the evaluation

1. Start the app, load the NIST excerpt, then stop the app (`Ctrl+C`).
2. Run:

```bash
python evaluate.py                 # main set (15 questions)
python evaluate.py --set holdout   # hold-out set (10 questions)
python experiment.py               # retrieval-only chunk-size comparison (no API calls)
python experiment.py --set holdout
```

The questions and page numbers in `evaluate.py` are written for the specific 22-page excerpt described above, and the page numbers are the PDF's physical page numbers. If you use a different file, edit `TEST_CASES` and `HOLDOUT_CASES` first. Results are written to `evaluation_results_*.json` and `experiment_results*.csv`.

The source document is NIST AI 100-1, *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*, published by NIST. It is not included here; download it from NIST.

## Project layout

```
app.py                     Flask server and API routes
evaluate.py                Evaluation harness and test questions
experiment.py              Retrieval-only chunking experiments
loaders/document_loader.py File loading and chunking
rag/embeddings.py          Embedding model wrapper
rag/vector_store.py        FAISS index and saving to disk
rag/retriever.py           Retrieval with a minimum score
rag/pipeline.py            Prompt building and the Groq call
rag/history.py             Chat history
frontend/index.html        The whole web interface
results/                   Saved evaluation outputs
docs/                      Screenshots
```

## Built with

Python, Flask, sentence-transformers (`all-MiniLM-L6-v2`), FAISS, pypdf, python-docx, the Groq API.

## Ideas for next steps

- Hybrid search (keyword plus embeddings) and a reranker, measured on a larger and harder question set.
- A second, independent judge model, to check how much the scores depend on the judge.
- Test on documents of other kinds (tables, multi-column papers) and write questions for them.

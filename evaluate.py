"""
evaluate.py
-----------
A lightweight RAG evaluation harness — no RAGAS dependency, no extra API
costs beyond what you already have. Uses the same Groq LLM as an
"LLM-as-judge" to score faithfulness and answer relevancy, plus retrieval
metrics computed directly from FAISS similarity scores.

Metrics produced:
  - top_retrieval_score   — how confident retrieval was (highest cosine sim)
  - keyword_coverage       — did the retrieved context actually contain the
                              facts we expect it to? (a cheap proxy for
                              context recall, no LLM call needed)
  - faithfulness           — does the generated answer only state things
                              actually supported by the retrieved context?
                              (LLM-as-judge, catches hallucination)
  - answer_relevancy       — does the answer actually address the question
                              asked? (LLM-as-judge)

Usage:
    1. Make sure a document is already loaded (upload it via the app once).
    2. Edit TEST_CASES below to match questions relevant to that document.
    3. Run: python evaluate.py
    4. Results print to the terminal and save to evaluation_results.json
"""

import os
import re
import json
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from rag.embeddings import EmbeddingManager
from rag.vector_store import VectorStore
from rag.retriever import RAGRetriever
from rag.pipeline import get_llm, rag_advanced, GROQ_MODEL

VECTOR_STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "vector_store")


def read_index_settings():
    """Chunk settings the loaded index was built with (written by app.py), or None."""
    try:
        with open(os.path.join(VECTOR_STORE_DIR, "index_settings.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Judge model — deliberately DIFFERENT from the generator (GROQ_MODEL, a Llama
# model) to reduce self-preference bias. Hosted on Groq, so the same
# GROQ_API_KEY works. Override with JUDGE_MODEL in .env if needed.
# ---------------------------------------------------------------------------
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")
JUDGE_MAX_TOKENS = 2000          # reasoning models spend tokens "thinking" before the JSON
JUDGE_REASONING_EFFORT = "low"   # low | medium | high  (gpt-oss only)
SECONDS_BETWEEN_QUESTIONS = 3    # gentle pacing for Groq free-tier rate limits

# One results file per judge, so runs with different judges can be compared later
# e.g. evaluation_results_openai_gpt-oss-120b.json
_judge_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", JUDGE_MODEL)


def results_path(set_name: str = "main") -> str:
    """main -> evaluation_results_<judge>.json ; holdout -> evaluation_results_holdout_<judge>.json"""
    prefix = "evaluation_results" if set_name == "main" else f"evaluation_results_{set_name}"
    return os.path.join(BASE_DIR, f"{prefix}_{_judge_tag}.json")

# ---------------------------------------------------------------------------
# Test set — EDIT THIS to match whatever document is currently indexed.
# expected_keywords: words/phrases you know should appear in the correct
# source passage — used to sanity-check retrieval without needing an LLM call.
# ---------------------------------------------------------------------------
TEST_CASES = [
    {
        "id": "Q1",
        "type": "factual",
        "question": "What is the main purpose of the NIST AI Risk Management Framework?",
        "reference_answer": "The AI RMF helps organizations manage the risks of AI and promote the trustworthy and responsible development and use of AI systems.",
        "expected_pages": [2],  # physical PDF pages (what the chatbot reports)
        "expected_keywords": ["trustworthy", "responsible"],
    },
    {
        "id": "Q2",
        "type": "factual",
        "question": "What are the four core functions of the AI RMF?",
        "reference_answer": "The four functions are GOVERN, MAP, MEASURE, and MANAGE.",
        "expected_pages": [3, 18],  # physical PDF pages (what the chatbot reports)
        "expected_keywords": ["govern", "map", "measure", "manage"],
    },
    {
        "id": "Q3",
        "type": "factual",
        "question": "How does the framework define risk?",
        "reference_answer": "Risk is the composite measure of an event's probability of occurring and the magnitude or degree of its consequences.",
        "expected_pages": [4],  # physical PDF pages (what the chatbot reports)
        "expected_keywords": ["probability", "magnitude"],
    },
    {
        "id": "Q4",
        "type": "factual",
        "question": "What does risk tolerance mean in the AI RMF?",
        "reference_answer": "Risk tolerance refers to an organization's or AI actor's readiness to bear risk in order to achieve its objectives.",
        "expected_pages": [7],  # physical PDF pages (what the chatbot reports)
        "expected_keywords": ["readiness", "bear"],
    },
    {
        "id": "Q5",
        "type": "factual",
        "question": "What characteristics make an AI system trustworthy according to NIST?",
        "reference_answer": "Trustworthy AI characteristics include validity and reliability, safety, security and resilience, accountability and transparency, explainability and interpretability, privacy enhancement, and fairness with harmful bias managed.",
        "expected_pages": [3, 10],  # physical PDF pages (what the chatbot reports)
        "expected_keywords": ["valid", "reliable", "safe", "resilient", "accountable", "transparent", "explainable", "interpretable", "privacy", "fair"],
    },
    {
        "id": "Q6",
        "type": "factual",
        "question": "What is the difference between explainability and interpretability?",
        "reference_answer": "Explainability refers to a representation of the mechanisms underlying AI system operation, whereas interpretability refers to the meaning of AI system output in the context of its designed functional purposes.",
        "expected_pages": [14],  # physical PDF pages (what the chatbot reports)
        "expected_keywords": ["mechanisms", "meaning", "output"],
    },
    {
        "id": "Q7",
        "type": "factual",
        "question": "What are the three major categories of AI bias identified by NIST?",
        "reference_answer": "The three categories are systemic bias, computational and statistical bias, and human-cognitive bias.",
        "expected_pages": [16],  # physical PDF pages (what the chatbot reports)
        "expected_keywords": ["systemic", "computational", "human-cognitive"],
    },
    {
        "id": "Q8",
        "type": "factual",
        "question": "What is the purpose of the MAP function?",
        "reference_answer": "The MAP function establishes the context to frame risks related to an AI system.",
        "expected_pages": [22],  # physical PDF pages (what the chatbot reports)
        "expected_keywords": ["establishes the context", "frame risks"],
    },
    {
        "id": "Q9",
        "type": "multi-part",
        "question": "Why can AI risks be difficult to measure, and what should organizations do to address emerging risks?",
        "reference_answer": "AI risks may be difficult to measure when failures are not well-defined or adequately understood. Organizations should identify and track emerging risks and consider techniques for measuring them.",
        "expected_pages": [5, 6],  # physical PDF pages (what the chatbot reports)
        "expected_keywords": ["not well-defined", "emergent", "measur"],
    },
    {
        "id": "Q10",
        "type": "multi-part",
        "question": "How should organizations prioritize AI risks, and what should they do when the risks become unacceptable?",
        "reference_answer": "Organizations should prioritize policies and resources according to the assessed risk level and potential impact of an AI system. If risks are unacceptable, development and deployment should cease safely until the risks can be sufficiently managed.",
        "expected_pages": [7, 8],  # physical PDF pages (what the chatbot reports)
        "expected_keywords": ["prioritized", "cease", "unacceptable"],
    },
    {
        "id": "Q11",
        "type": "multi-part",
        "question": "How do governance and diverse team participation contribute to AI risk management?",
        "reference_answer": "Governance establishes organizational policies, responsibilities, and practices for managing AI risks. Diverse, multidisciplinary teams bring different experiences and expertise, helping identify problems and emerging risks throughout the AI lifecycle.",
        "expected_pages": [9, 18, 19, 20, 21],  # physical PDF pages (what the chatbot reports)
        "expected_keywords": ["diverse", "governance", "emergent"],
    },
    {
        "id": "Q12",
        "type": "paraphrased",
        "question": "Can an AI system be considered trustworthy just because it is accurate? Why or why not?",
        "reference_answer": "No. Trustworthiness requires balancing multiple characteristics, and addressing these characteristics individually does not guarantee that an AI system is trustworthy.",
        "expected_pages": [10, 11],  # physical PDF pages (what the chatbot reports)
        "expected_keywords": ["individually", "tradeoffs"],
    },
    {
        "id": "Q13",
        "type": "paraphrased",
        "question": "Why might an AI system perform well during laboratory testing but create problems after deployment?",
        "reference_answer": "Measurements in a laboratory or controlled environment may differ from risks that emerge in operational, real-world settings. Some risks may also be latent early on and increase as AI systems adapt and evolve.",
        "expected_pages": [6],  # physical PDF pages (what the chatbot reports)
        "expected_keywords": ["laboratory", "controlled environment", "real-world"],
    },
    {
        "id": "Q14",
        "type": "unanswerable",
        "question": "What exact monetary fine does NIST prescribe for organizations that fail to follow the AI RMF?",
        "reference_answer": "The document does not specify a monetary fine for failing to follow the AI RMF. The framework is described as voluntary.",
        "expected_pages": [],  # physical PDF pages (what the chatbot reports)
        "expected_keywords": [],
    },
    {
        "id": "Q15",
        "type": "unanswerable",
        "question": "What is the mandatory deadline for every organization to complete implementation of the AI RMF?",
        "reference_answer": "I couldn't find a mandatory implementation deadline in the document. The AI RMF is voluntary, and organizations can apply its approaches according to their needs and capabilities.",
        "expected_pages": [],  # physical PDF pages (what the chatbot reports)
        "expected_keywords": [],
    },
]

# ---------------------------------------------------------------------------
# HOLD-OUT SET. Written separately from the main set. Run it ONCE and report the
# numbers; do NOT tune settings on it (that would make it a tuning set again).
# 'overlap' says how much it resembles questions used while tuning:
#   new = unseen passage | partial = same passage/page as a main-set question
#   regression = essentially the same question as a main-set one (checks a fix held)
# expected_pages are PHYSICAL PDF pages (printed page - 2 from printed page 12 on).
# ---------------------------------------------------------------------------
HOLDOUT_CASES = [
    {
        "id": "H1",
        "type": "factual",
        "overlap": "new",
        "question": "What does NIST mean by \"validation\" for an AI system?",
        "reference_answer": "Validation is confirmation, through objective evidence, that the requirements for a specific intended use or application have been fulfilled.",
        "expected_pages": [11],
        "expected_keywords": ["objective evidence", "intended use"],
    },
    {
        "id": "H2",
        "type": "factual",
        "overlap": "new",
        "question": "How does the document define accuracy, and what should accuracy measurements be paired with?",
        "reference_answer": "Accuracy is the closeness of observations, computations, or estimates to true or accepted values. Measurements should be paired with clearly defined, realistic test sets representative of expected-use conditions and with details of the test methodology.",
        "expected_pages": [12],
        "expected_keywords": ["closeness", "true values", "realistic test sets", "test methodology"],
    },
    {
        "id": "H3",
        "type": "factual",
        "overlap": "new",
        "question": "What does the AI RMF say makes an AI system \"safe\"?",
        "reference_answer": "An AI system should not, under defined conditions, lead to a state in which human life, health, property, or the environment is endangered.",
        "expected_pages": [12],
        "expected_keywords": ["human life", "endangered"],
    },
    {
        "id": "H4",
        "type": "factual",
        "overlap": "new",
        "question": "What is meant by \"inscrutability\" as a challenge to measuring AI risk?",
        "reference_answer": "Inscrutability can complicate risk measurement and may result from limited explainability or interpretability, lack of transparency or documentation, or inherent uncertainty in AI systems.",
        "expected_pages": [6],
        "expected_keywords": ["inscrutab", "opaque", "transparency or documentation", "uncertainties"],
    },
    {
        "id": "H5",
        "type": "factual",
        "overlap": "new",
        "question": "What does the framework mean by external validity when discussing AI accuracy?",
        "reference_answer": "The document says accuracy measures should demonstrate external validity, meaning generalizability beyond the training conditions.",
        "expected_pages": [12],
        "expected_keywords": ["external validity", "generalizable", "training conditions"],
    },
    {
        "id": "H6",
        "type": "factual",
        "overlap": "partial",
        "question": "Why might measuring risk at different points in an AI system's lifecycle produce different results?",
        "reference_answer": "Some risks may be latent at one point and increase as AI systems adapt and evolve. Different AI actors across the lifecycle may also have different perspectives on risk.",
        "expected_pages": [6],
        "expected_keywords": ["latent", "earlier stage", "risk perspectives"],
    },
    {
        "id": "H7",
        "type": "factual",
        "overlap": "partial",
        "question": "What kinds of bias can arise from non-representative samples, according to the document?",
        "reference_answer": "Computational and statistical biases can occur in AI datasets and algorithmic processes, and often stem from systematic errors due to non-representative samples.",
        "expected_pages": [16],
        "expected_keywords": ["non-representative samples", "computational and statistical"],
    },
    {
        "id": "H8",
        "type": "multi-part",
        "overlap": "regression",
        "question": "What does the MAP function help AI actors understand, and why can this be difficult?",
        "reference_answer": "MAP establishes the context for framing AI system risks. Anticipating impacts can be difficult because lifecycle activities are interdependent and actors may not have full visibility or control over other parts of the process.",
        "expected_pages": [22],
        "expected_keywords": ["establishes the context", "frame risks", "visibility"],
    },
    {
        "id": "H9",
        "type": "unanswerable",
        "overlap": "new",
        "question": "What is the exact minimum percentage of model accuracy required for an AI system to be considered trustworthy under the AI RMF?",
        "reference_answer": "The PDF does not specify a universal minimum accuracy percentage for trustworthiness. It says accuracy measures should use realistic test sets and that trustworthiness depends on balancing multiple characteristics.",
        "expected_pages": [],
        "expected_keywords": [],
    },
    {
        "id": "H10",
        "type": "unanswerable",
        "overlap": "new",
        "question": "How many independent external auditors must each organization hire annually to comply with the AI RMF?",
        "reference_answer": "The PDF does not specify a required number of independent external auditors or an annual external-audit requirement.",
        "expected_pages": [],
        "expected_keywords": [],
    },
]

TEST_SETS = {"main": TEST_CASES, "holdout": HOLDOUT_CASES}

JUDGE_SYSTEM_PROMPT = """You are an evaluator for a RAG (Retrieval-Augmented Generation) system.
You will be given a QUESTION, the CONTEXT that was retrieved, and the ANSWER that was generated.

Score two things, each from 0.0 to 1.0:
1. "faithfulness" - does the ANSWER only contain claims actually supported by the CONTEXT?
   1.0 = fully grounded in the context, 0.0 = contradicts the context or invents information not present in it.
2. "answer_relevancy" - does the ANSWER actually address the QUESTION asked?
   1.0 = directly and completely answers it, 0.0 = off-topic or non-responsive.

Respond with ONLY a JSON object in exactly this format, nothing else, no markdown fences:
{"faithfulness": 0.0, "answer_relevancy": 0.0, "reasoning": "one sentence explanation"}
"""


def _extract_json(raw: str):
    """Pull the first {...} object out of the judge's reply (handles fences / stray text)."""
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _clamp(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _call_judge(llm, system_prompt: str, user_message: str):
    """One judge call with retries. Returns (parsed_dict_or_None, error_text_or_None)."""
    kwargs = dict(
        model=JUDGE_MODEL,
        max_tokens=JUDGE_MAX_TOKENS,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    raw = ""
    last_error = None
    for attempt in range(3):
        try:
            try:
                response = llm.chat.completions.create(
                    **kwargs, extra_body={"reasoning_effort": JUDGE_REASONING_EFFORT}
                )
            except Exception as e:
                # If the reasoning_effort option is rejected, retry without it.
                if "reasoning" in str(e).lower():
                    response = llm.chat.completions.create(**kwargs)
                else:
                    raise
            raw = response.choices[0].message.content or ""
            break
        except Exception as e:  # rate limits (429), model not available, network...
            last_error = e
            time.sleep(5 * (attempt + 1))
    else:
        return None, f"Judge call failed ({JUDGE_MODEL}): {str(last_error)[:150]}"

    parsed = _extract_json(raw)
    if not parsed:
        return None, f"Could not parse judge output: {raw[:150]}"
    return parsed, None


def judge_answer(llm, question: str, context: str, answer: str) -> dict:
    """Faithfulness + answer relevancy (judged against the RETRIEVED CONTEXT)."""
    judge_message = f"""QUESTION: {question}

CONTEXT:
{context}

ANSWER:
{answer}"""
    parsed, error = _call_judge(llm, JUDGE_SYSTEM_PROMPT, judge_message)
    if error:
        return {"faithfulness": None, "answer_relevancy": None, "reasoning": error}
    return {
        "faithfulness": _clamp(parsed.get("faithfulness")),
        "answer_relevancy": _clamp(parsed.get("answer_relevancy")),
        "reasoning": parsed.get("reasoning"),
    }


CORRECTNESS_SYSTEM_PROMPT = """You are grading a question-answering system against a trusted REFERENCE ANSWER.
You will be given a QUESTION, the REFERENCE ANSWER (ground truth), and the system's ANSWER.
You do NOT see the source document. Judge only whether the ANSWER agrees with the REFERENCE.

Score "correctness" from 0.0 to 1.0:
  1.0 = the ANSWER conveys all key facts of the REFERENCE and contradicts nothing in it.
  0.5 = the ANSWER gets some key facts right but misses or garbles others.
  0.0 = the ANSWER is wrong, contradicts the REFERENCE, or misses the key facts entirely.

Rules:
- Wording can differ; judge meaning, not exact words.
- Extra detail in the ANSWER is fine as long as it does not contradict the REFERENCE.
- If the REFERENCE contains a real answer but the ANSWER says the information is not available or not in the document, score 0.0.
- If the REFERENCE says the information is NOT in the document (the question is unanswerable), the ANSWER is correct (1.0) only if it also says the information is not available; if it invents a specific answer, score 0.0.

Respond with ONLY a JSON object in exactly this format, nothing else, no markdown fences:
{"correctness": 0.0, "reasoning": "one sentence explanation"}
"""


def judge_correctness(llm, question: str, reference_answer: str, answer: str) -> dict:
    """Reference-based correctness. Deliberately does NOT see the retrieved context."""
    if not reference_answer:
        return {"correctness": None, "reasoning": "No reference_answer provided for this case."}
    judge_message = f"""QUESTION: {question}

REFERENCE ANSWER:
{reference_answer}

ANSWER:
{answer}"""
    parsed, error = _call_judge(llm, CORRECTNESS_SYSTEM_PROMPT, judge_message)
    if error:
        return {"correctness": None, "reasoning": error}
    return {
        "correctness": _clamp(parsed.get("correctness")),
        "reasoning": parsed.get("reasoning"),
    }


def keyword_coverage(expected_keywords: list, retrieved_text: str):
    if not expected_keywords:
        return None, []
    retrieved_lower = retrieved_text.lower()
    found = [kw for kw in expected_keywords if kw.lower() in retrieved_lower]
    return round(len(found) / len(expected_keywords), 2), found


ABSTAIN_PHRASES = [
    "couldn't find", "could not find", "not specified", "does not specify",
    "doesn't specify", "not mention", "no mention", "does not provide",
    "doesn't provide", "not contain", "does not contain", "not include",
    "does not include", "no information", "not stated",
]


def looks_like_abstention(answer: str) -> bool:
    """Rough heuristic: did the bot say the document doesn't contain the answer?
    Check a few results by eye, since phrasing varies."""
    a = answer.lower()
    return any(p in a for p in ABSTAIN_PHRASES)


def run_evaluation(set_name: str = "main"):
    cases = TEST_SETS[set_name]
    print("=" * 70)
    print(f"RAG EVALUATION  (test set: {set_name}, {len(cases)} questions)")
    print("=" * 70)
    if set_name == "holdout":
        print("HOLD-OUT RUN: report these numbers as they are. Do not tune on them.")

    embedding_manager = EmbeddingManager()
    vector_store = VectorStore(persist_directory=VECTOR_STORE_DIR, dim=embedding_manager.dim)

    if vector_store.get_count() == 0:
        print("\nNo document is currently indexed.")
        print("Upload a document through the app first, then run this script again.")
        return []

    retriever = RAGRetriever(vector_store, embedding_manager)
    llm = get_llm()

    print(f"Generator model : {GROQ_MODEL}")
    print(f"Judge model     : {JUDGE_MODEL}")
    index_settings = read_index_settings()
    if index_settings:
        print(f"Index settings  : {index_settings['chunk_size']}-word passages, "
              f"{index_settings['chunk_overlap']} overlap ({index_settings['chunks']} chunks)")
    else:
        print("Index settings  : unknown (index was built before settings were recorded; "
              "load the PDF again in the app, then re-run)")
    if JUDGE_MODEL == GROQ_MODEL:
        print("WARNING: judge and generator are the same model (self-preference bias).")

    results = []

    for i, case in enumerate(cases, start=1):
        question = case["question"]
        expected_keywords = case.get("expected_keywords", [])
        expected_pages = case.get("expected_pages", [])
        case_type = case.get("type")

        print(f"\n[{i}/{len(cases)}] {question}")

        retrieved = retriever.retrieve(question, top_k=5, min_score=0.0)
        context_text = "\n\n".join(doc.page_content for doc, _ in retrieved)
        top_score = round(retrieved[0][1], 3) if retrieved else 0.0

        coverage, found_keywords = keyword_coverage(expected_keywords, context_text)

        retrieved_pages = sorted({doc.metadata.get("page") for doc, _ in retrieved
                                  if doc.metadata.get("page") is not None})
        page_hit = (1.0 if set(retrieved_pages) & set(expected_pages) else 0.0) if expected_pages else None

        rag_result = rag_advanced(query=question, retriever=retriever, llm=llm, top_k=5, min_score=0.0)
        answer = rag_result["answer"]

        judge = judge_answer(llm, question, context_text, answer)
        correct = judge_correctness(llm, question, case.get("reference_answer"), answer)
        abstained = looks_like_abstention(answer) if case_type == "unanswerable" else None

        result = {
            "id": case.get("id"),
            "set": set_name,
            "overlap": case.get("overlap"),
            "type": case_type,
            "question": question,
            "reference_answer": case.get("reference_answer"),
            "answer": answer,
            "expected_pages": expected_pages,
            "retrieved_pages": retrieved_pages,
            "page_hit": page_hit,
            "abstained": abstained,
            "top_retrieval_score": top_score,
            "keyword_coverage": coverage,
            "keywords_found": found_keywords,
            "faithfulness": judge.get("faithfulness"),
            "answer_relevancy": judge.get("answer_relevancy"),
            "judge_reasoning": judge.get("reasoning"),
            "correctness": correct.get("correctness"),
            "correctness_reasoning": correct.get("reasoning"),
            "judge_model": JUDGE_MODEL,
            "index_settings": index_settings,
        }
        results.append(result)
        time.sleep(SECONDS_BETWEEN_QUESTIONS)

        print(f"  Top retrieval score : {result['top_retrieval_score']}")
        print(f"  Keyword coverage    : {result['keyword_coverage']}  (found: {found_keywords})")
        print(f"  Page hit            : {result['page_hit']}  (expected {expected_pages}, retrieved {retrieved_pages})")
        if abstained is not None:
            print(f"  Abstained (heuristic): {abstained}")
        print(f"  Faithfulness        : {result['faithfulness']}")
        print(f"  Answer relevancy    : {result['answer_relevancy']}")
        print(f"  Correctness (vs ref): {result['correctness']}")
        print(f"  Judge reasoning     : {result['judge_reasoning']}")

    def avg(key):
        vals = [r[key] for r in results if r[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if index_settings:
        print(f"Index settings          : {index_settings['chunk_size']} / {index_settings['chunk_overlap']} "
              f"({index_settings['chunks']} chunks)")
    print(f"Avg top retrieval score : {avg('top_retrieval_score')}")
    print(f"Avg keyword coverage    : {avg('keyword_coverage')}")
    print(f"Page hit rate           : {avg('page_hit')}")
    unans = [r for r in results if r["type"] == "unanswerable"]
    if unans:
        print(f"Abstained on unanswerable: {sum(1 for r in unans if r['abstained'])}/{len(unans)}")
    print(f"Avg faithfulness        : {avg('faithfulness')}")
    print(f"Avg answer relevancy    : {avg('answer_relevancy')}")
    print(f"Avg correctness         : {avg('correctness')}")

    # Per question-type correctness
    types = sorted({r["type"] for r in results if r.get("type")})
    for t in types:
        vals = [r["correctness"] for r in results if r.get("type") == t and r["correctness"] is not None]
        if vals:
            print(f"  correctness [{t}] : {round(sum(vals) / len(vals), 3)}  (n={len(vals)})")

    # Flag answers worth reading by hand
    weak = [r for r in results if r["correctness"] is not None and r["correctness"] < 0.7]
    if weak:
        print("\nLow correctness (review these by hand):")
        for r in weak:
            print(f"  {r.get('id')}: correctness={r['correctness']}  faithfulness={r['faithfulness']}  -> {r['correctness_reasoning']}")

    with open(results_path(set_name), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {results_path(set_name)}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate the RAG chatbot on a test set.")
    parser.add_argument("--set", choices=sorted(TEST_SETS), default="main",
                        help="which test set to run (default: main)")
    args = parser.parse_args()
    run_evaluation(args.set)

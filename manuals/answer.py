"""Answer generation: any ladder rung's top 5 chunks -> gpt-4o-mini -> an answer with [n] citations.

The ladder scored retrieval only (empty answers). This adds the last step, so the harness can also judge the
ANSWER: faithfulness / correctness with RAGAS, and abstention on the no-answer questions (Danfoss alarm 21 does
not exist). Each chunk is shown with its manual and page, so the model knows which machine a code belongs to and
can cite it. Refusals use the sprint's exact line: the harness only counts answers that START with "I don't know".

Run from hybrid_rag_in_depth/ (Qdrant running, OpenAI key loaded):
    uv run manuals/answer.py r6a                 # tune split, answers printed, no judge (cents)
    uv run manuals/answer.py r6a --judge         # tune split + RAGAS judge
    uv run manuals/answer.py r0 r6a --judge --test   # the final run: held-out test split
"""

import sys

from eval_harness import PipelineOutput, evaluate, settings, traced_openai

import hyde
from chunks import MANUAL_NAMES
from ladder import RUNGS, TOP_K, TUNE

TEST = "golden_manufacturing_test_v1.jsonl"
NO_ANSWER = "I don't know based on the provided documents."

PROMPT = """You help maintenance technicians troubleshoot equipment. Answer the question using ONLY the manual
excerpts below. Answer for the machine the question names; fault codes mean different things on different machines.
Give the code's meaning, then the causes and the fixes the excerpts list. Be concise. Cite excerpts like [2].
If the excerpts do not answer the question, reply exactly: "{no_answer}"

Excerpts:
{context}

Question: {question}"""


def make_pipeline(rung: str):
    """question -> PipelineOutput for this rung: its retrieval, then one generation call."""
    search = RUNGS[rung][1]
    client = traced_openai()

    def pipeline(question: str) -> PipelineOutput:
        hyde.USAGE.update(tokens_in=0, tokens_out=0, llm_ms=0.0)  # HyDE rungs add their call here
        hits = search(question, TOP_K)
        context = "\n\n".join(f"[{i}] {MANUAL_NAMES[h.doc_id]}, page {h.page}\n{h.text}"
                              for i, h in enumerate(hits, start=1))
        resp = client.chat.completions.create(
            model=settings.llm_model, temperature=0,
            messages=[{"role": "user", "content": PROMPT.format(no_answer=NO_ANSWER, context=context,
                                                                question=question)}])
        return PipelineOutput(answer=resp.choices[0].message.content.strip(),
                              contexts=[h.text for h in hits], context_ids=[h.doc_id for h in hits],
                              tokens_in=hyde.USAGE["tokens_in"] + resp.usage.prompt_tokens,
                              tokens_out=hyde.USAGE["tokens_out"] + resp.usage.completion_tokens)

    return pipeline


def main():
    args = sys.argv[1:]
    rungs = [a for a in args if not a.startswith("--")]
    split = TEST if "--test" in args else TUNE
    for rung in rungs:
        run = evaluate(make_pipeline(rung), arch=f"manuals_{rung}_answer", golden_set=split,
                       use_llm_judge="--judge" in args, compare_to=None)
        if "--judge" not in args:  # without the judge, show the answers so they can be read
            for q in run.questions:
                print(f"\n--- {q.golden_id}\n{q.output.answer if q.output else q.error}")


if __name__ == "__main__":
    main()

"""Answer a question from retrieved chunks with the LLM, citing sources as [S1], [S2], ...

Run: uv run python -m research_rag.ask "your question" (generate alone runs ask without saving)
"""

import sys
import time

from ollama import Client

SYSTEM_PROMPT = """You answer questions about research papers on image captioning.
Use ONLY the numbered sources given with the question. Do not use your own knowledge.
Cite every claim with the label of the source it comes from, for example [S2].
Answer in 1 to 4 short sentences.
If the sources do not contain the answer, reply exactly: {no_evidence}"""


def format_sources(chunks: list[dict]) -> str:
    return "\n\n".join(f"[S{i}] {c['title']}, page {c['page']}\n{c['text']}" for i, c in enumerate(chunks, start=1))


def build_messages(question: str, chunks: list[dict], no_evidence: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(no_evidence=no_evidence)},
        {"role": "user", "content": f"Sources:\n\n{format_sources(chunks)}\n\nQuestion: {question}"},
    ]


def generate(client: Client, config: dict, question: str, chunks: list[dict]) -> dict:
    gen = config["generation"]
    start = time.perf_counter()
    reply = client.chat(
        model=config["models"]["llm"],
        messages=build_messages(question, chunks, gen["no_evidence_reply"]),
        think=config["models"]["think"],
        options={"temperature": gen["temperature"], "num_ctx": gen["num_ctx"]},
    )
    return {
        "answer": reply.message.content.strip(),
        "prompt_tokens": reply.prompt_eval_count,
        "answer_tokens": reply.eval_count,
        "seconds": round(time.perf_counter() - start, 1),
    }


def main() -> None:
    from research_rag.ask import main as ask_main

    sys.argv.append("--no-save")
    ask_main()


if __name__ == "__main__":
    main()

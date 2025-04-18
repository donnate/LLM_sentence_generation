import torch
import asyncio, os, re, random
from typing import List, Dict
from openai import AsyncOpenAI, RateLimitError
from tqdm.asyncio import tqdm_asyncio as tqdm
import json

# Async client setup
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_sentence_re = re.compile(r"(?<=[.!?])\s+")

def strip_leading_number(s: str) -> str:
    """
    Remove '1. ', '23) ', '004. ', etc. from the start of a sentence.
    """
    return re.sub(r'^\s*\d+[\.)]\s*', '', s)

def split_sentences(text: str) -> List[str]:
    # crude but serviceable for plain English prose
    return [s.strip() for s in _sentence_re.split(text.strip()) if s.strip()]

# --- 2. core async call -------------------------------------------
async def alt_for_hidden_sentence_async(context: str,
                                        n_alts: int = 10,
                                        max_retries: int = 5,
                                        temperature=0.9,
                                        model="gpt-4.1-nano") -> List[str]:
    """
    Given a paragraph with ONE sentence omitted (context),
    ask for `n_alts` plausible replacements.
    """
    # Note: the model is not guaranteed to return exactly `n_alts` sentences.

    def build_prompt(context: str, n_alts: int = 10) -> str:
        return (
            "Below is a paragraph from an academic scientific abstract in which ONE sentence is missing and was replaced by the token '<MISSING_SENTENCE>'."
            "Write *exactly {n} different* replacement sentences to fill the missing value. "
            "Return them **only** as a JSON array called `sentences` — "
            "no commentary, no numbering, no extra keys.\n\n"
            "PARAGRAPH WITH MISSING SENTENCE:\n"
            "{context}\n\n"
            "###\n"
            "{{\"sentences\": [\"<alt1>\", \"<alt2>\", …]}}"
        ).format(n=n_alts, context=context)

    prompt = build_prompt(context, n_alts)

    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system",
                     "content": "You are a helpful and smart writing assistant for writing scientific abstracts."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,          # plenty for 10 short sentences
                temperature=temperature,
                response_format={"type": "json_object"},  # β feature, if enabled
            )
            data = json.loads(resp.choices[0].message.content)

            # ——— basic sanity checks ———
            if (
                isinstance(data, dict)
                and isinstance(data.get("sentences"), list)
                and len(data["sentences"]) == n_alts
                and all(isinstance(s, str) and s.strip() for s in data["sentences"])
            ):
                return data["sentences"]          # ✅ success

        except (json.JSONDecodeError, KeyError, AssertionError):
            pass                                  # bad format → retry
        except RateLimitError:
            await asyncio.sleep(2 ** attempt + random.random())
        except Exception as e:
            print("Unexpected error:", e); break
    return []  # give up after max_retries


    #         alts = [ln.strip("•- \t") for ln in raw.splitlines() if ln.strip()]
    #         # If the model returned fewer than requested, keep the non‑empty ones
    #         return alts[:n_alts]
    #     except RateLimitError:
    #         backoff = min(60, 2 ** attempt + random.random())
    #         await asyncio.sleep(backoff)
    #     except Exception as e:
    #         print(f"Unexpected error: {e}")
    #         break
    # return []          # give up after `max_retries`

# --- 3. process ONE document --------------------------------------


async def process_document_async(
    doc: str,
    n_alts_per_sentence: int,
    semaphore: asyncio.Semaphore
) -> dict[int, list[str]]:
    """
    Return {sentence_idx: [alt1, alt2, …, alt_n]} for ONE document.
    """
    async with semaphore:                       # limits concurrent API calls
        sentences = split_sentences(doc)

        async def handle_one(idx: int):
            ctx = " ".join(sentences[:idx] + ["<MISSING>"] + sentences[idx+1:])
            return idx, await alt_for_hidden_sentence_async(ctx, n_alts_per_sentence)

        idx_alt_pairs = await asyncio.gather(
            *(handle_one(i) for i in range(len(sentences)))
        )
        return dict(idx_alt_pairs)


# --- 4. top‑level: many documents ---------------------------------
async def hide_and_generate_async(
    docs: List[str],
    n_alts_per_sentence: int = 5,
    max_concurrent_docs: int = 5,
):
    """
    Returns
      • per_doc_results[i]  ==  {sent_idx: [alt1, …]}  for docs[i]
      • datasets[k][i]      ==  document i reconstructed from the k‑th alternatives
    """
    sem = asyncio.Semaphore(max_concurrent_docs)

    async def handle_doc(idx: int, doc: str):
        """Run one doc and remember its original position."""
        res = await process_document_async(doc, n_alts_per_sentence, sem)
        return idx, res

    # --------------------------------------------------------------
    coros = [handle_doc(i, d) for i, d in enumerate(docs)]

    per_doc_results: list[dict[int, list[str]]] = [None] * len(docs)
    datasets: dict[int, list[str]] = {
        k: [None] * len(docs) for k in range(n_alts_per_sentence + 1)
    }

    for fut in tqdm.as_completed(
        coros, total=len(coros), desc="Generating alt sentences"
    ):
        idx, result = await fut                # <- original position
        per_doc_results[idx] = result

        # rebuild kth‑alternative docs (same order as input!)
        for k in range( n_alts_per_sentence + 1):
            kth_sentences = [
                strip_leading_number(alts[alt_idx])          # alt_idx = which alternative to keep
                for sent_idx, alts in sorted(result.items()) # result: {sentence_idx: [alt0,…]}
                for alt_idx in range(len(alts))                   # alts: [alt0, alt1, …]
                if alt_idx < k
            ]
            datasets[k][idx] = docs[idx] +  " ".join(kth_sentences)

    return per_doc_results, datasets

# -------------- example usage -------------------------------------
# if __name__ == "__main__":
#     docs = ["First sentence. Second sentence? Third one!",
#             "Another short document with two sentences."]
#     results = asyncio.run(hide_and_generate_async(docs, n_alts_per_sentence=10))
#     print(results)


# How PromptLite Turns Big, Rough Prompts into Small, Accurate Prompts

PromptLite is a local, API-free prompt compressor. It does not call an LLM to
rewrite your text. It runs a constrained NLP pipeline: **understand → protect →
remove redundancy → validate meaning → count tokens**.

Flow: `frontend/src/App.tsx` → `POST /api/compress` or `POST /api/upload`
(`backend/app/main.py:run_compression`) → JSON with the optimized prompt plus
explainability data.

## 1. Input

- Typed/pasted text, or a `.txt` / `.md` / `.pdf` upload.
- Uploads go through `backend/app/services/document_processor.py`:
  `extract()` decodes text or pulls PDF pages with `pypdf`, `clean()` collapses
  extra whitespace. Files over 10 MB or other extensions are rejected.
- Request shape (`backend/app/schemas/api.py`): `{ text, mode, task? }` where
  `mode` is `safe | balanced | aggressive`.

## 2. Understand the prompt (`nlp_analyzer.py`)

`NLPAnalyzer` loads `en_core_web_sm` (falls back to a blank English pipeline
with a sentencizer) and produces an `Analysis`:

- **Sentences** via spaCy sentence segmentation.
- **Entities** (`doc.ents`), **noun phrases** (`doc.noun_chunks`).
- **Negations**: tokens with `dep_ == "neg"` plus sentences containing
  `not / never / without / no`.
- **Numbers**: `token.like_num` (quantities, dates, IDs like `42`).
- **Quotes**: regex for `"..."` / `'...'` content.
- **Imperative verbs**: `ROOT` verbs (the action of the prompt).
- **Constraints**: sentences matching
  `do not | don't | must | only | at least | no more than | return exactly |
  in json | include | exclude | required`.
- **Task detection**: first-matching regex over 10 classes —
  Summarization, Explanation, Analysis, Comparison, Classification, Extraction,
  Question answering, Code generation, Writing/rewrite, Translation —
  else `Other / general`.

This is why "tell me about major problems" is labeled `Analysis` instead of
being treated as generic text.

## 3. Protect what must survive (`protector.py`)

`SpanProtector.protect()` builds the no-touch list:

- Everything from step 2 that carries meaning: entities, numbers, quotes,
  negated sentences, constraint sentences.
- Plus regex extras: URLs (`https?://...`), emails, multi-word capitalized
  names (`Microsoft Azure`).
- Deduplicated, order-preserved.

Example: in `Do not remove 42 records from Microsoft Azure`, `42`,
`Microsoft Azure`, and the whole `Do not...` sentence are protected. The
compressor may delete filler around them but never the protected spans
(see `tests/test_pipeline.py:test_protected_items`).

## 4. Find redundancy (`redundancy_detector.py` + `semantic_validator.py`)

`RedundancyDetector.detect(sentences, mode)` reports three things:

1. **Filler matches** — a large `FILLER` regex: `could you please`,
   `i would like you to`, `if possible`, `kindly`, `in fact`, `basically`, etc.
2. **Duplicate sentences** — pairwise cosine similarity between sentences.
   Embeddings come from local `all-MiniLM-L6-v2`; if it cannot load, a local
   lexical TF-cosine fallback is used. Thresholds by mode:
   `safe: 0.94`, `balanced: 0.88`, `aggressive: 0.80`. The later duplicate
   index is dropped.
3. **Consolidation notes** — e.g. repeated `careful / detailed / thorough`
   instructions collapsed into one note.

## 5. Compress (`compressor.py`, plus `llmlingua_compressor.py`)

There are two engines. `backend/app/main.py` picks per mode:

### Safe mode — deterministic spaCy pipeline only

`PromptCompressor.compress()` is grammatical and conservative:

1. Drop duplicate sentences found in step 4.
2. Strip filler phrases found in step 4 (`Conversational filler: ...`).
3. Simplify verbose multiword constructions (`RedundancyDetector.MULTIWORD`):
   `a large number of → many`, `due to the fact that → because`,
   `in order to → to`, `make sure → ensure`, `prior to → before`, ... (1 pass,
   3 passes in aggressive).
4. Strip polite lead-ins (`please note that`, `i think that`, `can you`, ...).
5. Strip `please ensure/make sure/confirm that`.
6. Clean leftover punctuation/spacing.
7. Prefix `Task: ` (e.g. `Analysis: ...`) only when a real task was detected
   and there is more than one sentence (skipped in aggressive mode to save tokens).

Aggressive deterministic fallback (used on Render/slim installs where LLMLingua
is unavailable, and as the safety net locally) additionally runs:
`AGGRESSIVE_MULTIWORD` deletions first (e.g. `in great detail → ""`,
`each and every → each`, `as soon as possible → ASAP`), parenthetical removal,
then protected-masked thinning of articles / low-value adjectives /
relativizer `that` / `also`, plus dangling-conjunction cleanup. Duplicate
threshold is lower in aggressive mode (0.72 vs 0.94 safe / 0.88 balanced), so
near-duplicates are also dropped harder. This is what lets a 518-token prompt
reach ~230–270 tokens (≈45–55% reduction) even without LLMLingua, while numbers,
entities, URLs, emails, and constraint keywords survive via masking.

### Balanced / Aggressive — LLMLingua-2 token dropping with guardrails

`LLMLinguaCompressor` uses local Microsoft
`microsoft/llmlingua-2-xlm-roberta-large-meetingbank` (CPU, no API key):

- Keep-fraction: `balanced: 0.6` (~40% fewer tokens),
  `aggressive: 0.30` (~70% fewer, intentionally terse "caveman" style).
- Inputs under 60 chars skip LLMLingua and use the deterministic pipeline.
- **Layer 1 — pinning:** short protected spans only (≤80 chars, up to 100,
  longest first) are passed as `force_tokens`, plus `force_reserve_digit=True`
  pins all digits. Long constraint/negation sentences are excluded from pinning
  so token dropping is not frozen (their numbers/entities stay pinned
  individually).
- **Layer 2 — guarantee:** after compression, any protected span missing from
  the output is re-appended under `Preserved: ...` and reported.
- `_retokenize()` fixes gluing from extractive dropping
  (`Contactsupport@x.com`, `insights.Do`).
- Any load/inference failure returns `(None, reason)` and the caller falls back
  to the deterministic pipeline with a warning. Set
  `PROMPTLITE_DISABLE_LLMLINGUA=1` to force deterministic-only mode.

Every removal is logged into `removed_items_summary` (engine name + each rule
fired + consolidation notes), which the UI shows under "What changed?".

## 6. Validate meaning and fail safe (`main.py`, `semantic_validator.py`)

After compression:

- `SemanticValidator.similarity(original, optimized)` gives a 0–1 cosine score
  (MiniLM or lexical fallback), shown as **estimated semantic preservation %**.
- Floors: `safe: 0.90`, `balanced: 0.80`, `aggressive: 0.68`. Balanced and
  aggressive floors are lower because token-dropping output is ungrammatical by
  design and scores lower than a rewrite.
- If the score is below the floor, the pipeline recompresses one level safer
  (`balanced → safe`, `aggressive → balanced`), keeps whichever scored higher,
  and adds the warning `Compression was softened to ... to protect semantic
  preservation.` The returned `mode_used` reflects what actually ran.

## 7. Count tokens (`token_counter.py`)

`LocalTokenCounter` uses `tiktoken` `cl100k_base` when installed, else a
transparent `words × 1.3` approximation (same fallback the frontend uses for its
live estimate). Response includes `original_tokens`, `optimized_tokens`,
`token_reduction_percentage`, and `compression_ratio`.

## 8. What the UI shows

`frontend/src/App.tsx` + `frontend/src/services/api.ts`:

- Input panel with live token estimate, Paste / Upload / Clear.
- Mode switch (`Safe` = maximum preservation, `Balanced` = recommended,
  `Aggressive` = stronger reduction) + Compress button.
- Output panel with the optimized prompt + Copy.
- Metrics row: token reduction %, semantic preservation %, token count
  `before → after`.
- "What changed?" panel: detected task, mode used, protected items, removed /
  consolidated list, warnings.

## Worked example

Input:

> Could you please carefully analyze the customer reviews that I have provided
> and tell me about the major problems that customers are experiencing? Also,
> please make sure you identify any suggestions ... Do not remove reviews with
> fewer than 10 words.

Pipeline: task → `Analysis`; protected → `10`, `Do not remove reviews with
fewer than 10 words`; filler → `could you please`, `please make sure`;
duplicates → none; compress → drops politeness, `in order to → to`-style
simplifications, keeps the constraint verbatim; validate → similarity checked
against the floor, softened if needed; count → tokens before/after reported.

## Limitations

- Similarity is an estimate, not a proof of equal meaning.
- Deterministic rules cannot match a generative rewrite for style.
- Token counts are `cl100k`-based estimates; other model tokenizers differ.
- Very short inputs skip LLMLingua; very large models need ~2 GB cache on
  first Balanced/Aggressive run.

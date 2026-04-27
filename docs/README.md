# Philosophy Recommender

This project is a small recommendation system prototype built around a simple question:

> How can we recommend philosophy books from sparse, messy public metadata, while still keeping the system explainable and cheap enough to iterate on?

The current version shows a progression from a basic retrieval baseline to a more semantic recommender, while keeping the architecture small enough to reason about end to end.

![UI demo](./mygif.gif)

## Project Framing

The product idea is a reading companion for philosophy-oriented users. A user writes a prompt such as:

> "I liked Nietzsche but now want something more practical and less dark about meaning."

The system then tries to:

1. understand what the user is looking for,
2. search a local catalogue of philosophy books,
3. retrieve semantically related titles,
4. present them in a graph-oriented interface that invites exploration.

This repo is about building a recommendation pipeline under realistic constraints:

- public metadata is noisy,
- philosophy is a fuzzy domain,
- many books have weak or no descriptions,
- the catalogue has to be curated before retrieval quality can improve,
- and the recommendation logic should evolve in phases rather than jump immediately to a heavy system.

## Design Goals

The implementation is guided by four design decisions.

### 1. Keep the system inspectable

I deliberately avoided hiding the whole recommendation process behind a single opaque service. The pipeline is split into small stages:

- catalogue construction,
- structured semantic enrichment,
- retrieval,
- UI rendering.


### 2. Designing a usable product

The current design favors:

- a local catalogue over live API dependency,
- batch enrichment over online per-book calls,
- semantic retrieval over hand-written keyword ranking,
- a lightweight custom HTTP server over a larger web framework,
- explainable fallbacks when data is sparse.

### 3. Separate offline intelligence from online retrieval

One of the key choices in the current version is that expensive book understanding happens offline.

- **offline:** enrich books, embed books, build the semantic index
- **online:** extract one user preference profile, embed one query, retrieve nearest books

This keeps the online experience responsive while still allowing richer semantic matching than a purely lexical baseline.

### 4. Treat data quality as part of the recommender

For this project, the catalogue is not just an input. It is part of the recommendation logic.

That became especially clear after seeing noisy entries such as fiction, self-help, programming, or generally non-philosophy books enter the corpus through permissive Open Library subject metadata.

So a large part of the engineering work here is really:

- deciding what counts as "in-domain,"
- filtering,
- and estabilish the tradeoffs between catalogue size and catalogue quality.

## System Overview

Current high-level pipeline:

```text
User prompt
-> LLM preference profile
-> query embedding
-> similarity search over pre-embedded books
-> recommendation list + graph UI
```

This can be understood as three stages.

### Stage 1: Catalogue building

The project uses Open Library dumps rather than a live external books API.

That decision was made for three reasons:

1. **repeatability**: recommendations should not change because a third-party API is rate limiting or behaving differently;
2. **cost and control**: local catalogues are easier to inspect and rebuild;
3. **better system design**: retrieval quality starts with corpus quality.

The catalogue builder:

- scans Open Library works, authors, and ratings dumps,
- applies philosophy-oriented subject rules,
- resolves author names,
- ranks by number of ratings,
- writes a local JSONL catalogue.

Important tradeoff:

- a stricter filter yields a smaller but cleaner catalogue,
- a looser filter yields a larger but noisier one.

That tradeoff is visible in this repo and is intentional: it reflects a real recommender design tension rather than being polished away.

### Stage 2: Semantic enrichment

The current recommender no longer relies on the older keyword-weighting scorer.

Instead, it uses a semantic pipeline in which:

- books are enriched into structured profiles,
- those profiles are embedded,
- a user prompt is also turned into a structured profile and embedded,
- ranking is done by vector similarity.

I kept this design for a few reasons.

#### Why use structured LLM profiles at all?

A raw free-text prompt like:

> "I want something gentler than Nietzsche, still about meaning."

contains several useful signals that are not just keywords:

- author affinity,
- desired tone,
- implicit topic,
- implicit avoidance.

Turning prompts into a structured preference profile gives the system a more stable semantic representation than naive token matching.

Likewise for books, the Open Library metadata often needs normalization into fields such as:

- themes,
- traditions,
- reader moods,
- style,
- difficulty,
- era.

#### Why precompute book embeddings offline?

Because the expensive part is repeated book understanding, not per-query search.

If every recommendation request had to re-profile and re-embed books online, the system would be:

- slow,
- expensive,
- rate-limit prone,
- and much harder to reason about.

Precomputing a semantic index is a clean MVP compromise.

#### Gemini Batch API

The book-enrichment stage can involve thousands of records. Doing those calls one by one is both slow and unnecessarily expensive.

Batch API preserves the same pipeline shape but makes the offline stage more realistic by:

- reducing unit cost,
- avoiding free-tier request pressure,
- and making long-running enrichment jobs easier to manage explicitly.

## Recommendation Strategy

At the moment, recommendations are primarily driven by **semantic similarity** between:

- the embedded user preference profile,
- and the embedded book semantic profiles.

This means the system is stronge at:

- prompt-based cold start,
- concept-level matching,
- retrieving books that "feel right" even when wording differs.

## Project Phases

This repo has evolved in stages.

### Phase 1: Local retrieval baseline

Goal:

- get a working recommender over a local philosophy catalogue

Main choice:

- use Open Library dumps and a local search flow rather than a live API

Why:

- removes dependence on rate-limited external services,
- makes the corpus inspectable,
- creates a stable base for later recommendation improvements.

### Phase 2: Better book understanding, better user understanding, better matching

Goal:

- replace brittle lexical matching with a semantic pipeline

Main choices:

- LLM-based structured book profiling,
- LLM-based structured user preference profiling,
- embeddings for retrieval.

Why:

- philosophy requests are often nuanced and not well captured by keywords alone,
- metadata is sparse and benefits from normalization,
- semantic retrieval is a better fit for cold-start than collaborative methods.

### Phase 3: Interaction-aware recommendations

Planned next step:

- collect user interactions,
- add collaborative filtering or collaborative reranking.

Why this is Phase 3 and not earlier:

- collaborative filtering without interaction data is premature,
- semantic retrieval solves cold start better,
- and a hybrid design makes more sense once user behavior starts accumulating.

The likely direction is:

- log impressions, clicks, selections, and opens,
- create a lightweight interaction store,
- add item-to-item collaborative signals,
- blend them into the semantic ranking rather than replace it.

## Key Tradeoffs

### Local catalogue vs larger commercial dataset

I chose a public, imperfect corpus because the project is trying to demonstrate system thinking, not just vendor access.

The downside is obvious:

- Open Library metadata is noisy,
- subject labels are inconsistent,
- some good books are sparse,
- some bad books sneak in unless filtered carefully.

The upside is that it forced the recommendation design to deal with realistic data quality problems.

### Simplicity vs framework richness

The backend is a very small custom HTTP server rather than FastAPI or Flask.

That is a conscious choice for this stage:

- the project has one main endpoint,
- the recommendation logic matters more than framework surface area,
- and a smaller server keeps attention on the system design itself.

If this project were extended into a multi-user or production-oriented service, I would move it to a more standard framework.

### Strict catalogue vs large catalogue

This is probably the most important product tradeoff in the repo.

A strict philosophy filter gives:

- fewer books,
- less garbage,
- more trust in the recommendation space.

A permissive filter gives:

- more books,
- but more fiction, self-help, programming, and adjacent noise.

For a philosophy recommender, I currently prefer the cleaner core, even if that means the catalogue is smaller than the original 10k target.

### LLM enrichment vs cost discipline

LLM enrichment improves semantic structure, but it introduces:

- cost,
- latency,
- quota management,
- and dependency on model behavior.

This is why the design keeps LLM use concentrated in places where it adds the most value:

- offline book understanding,
- online user-profile extraction,
- sparse-description fallback enrichment for UI display.

## Current Repository Structure

Core files:

- `ui.html`: light-mode graph-oriented frontend
- `myrecsys/app.py`: local HTTP server and `/recommend` endpoint
- `myrecsys/recommendation.py`: high-level recommendation orchestration
- `myrecsys/phase2.py`: semantic profiling, embedding, and ranking logic
- `myrecsys/book_enrichment.py`: display-oriented fallback enrichment for sparse books
- `myrecsys/local_catalog.py`: local catalogue loading and normalization
- `myrecsys/schemas.py`: application data structures

Data and build scripts:

- `data/books.jsonl`: local catalogue
- `data/semantic_index.jsonl`: precomputed semantic index
- `scripts/build_openlibrary_catalog.py`: philosophy catalogue builder
- `scripts/build_semantic_index.py`: direct semantic index builder
- `scripts/build_semantic_index_batch.py`: cheaper batch-based semantic index builder
- `scripts/append_zero_rated_philosophers.py`: append zero-rated works by selected canonical philosophers

## What I Would Improve Next

Improvements I am currently working on include: 

### 1. Interaction logging

- assign stable anonymous users,
- log recommendation impressions and clicks, opens, selections
- possibilitate saves
- store interaction history,
- collaborative reranking.

### 2. Lightweight collaborative reranking

- people who explored X also explored Y co-occurrences

### 3. Diversity and Serendipity focus

- explore beyond perfect matching and echo chambers, provide reliable novel and different reccomendations


## Run 


Main entry points:

```bash
python3 -m myrecsys.app
python3 -m myrecsys.cli "I liked Nietzsche but want something more practical and less dark"
python3 scripts/build_openlibrary_catalog.py
python3 scripts/build_semantic_index_batch.py prepare-profiles
```

Environment:

```text
GOOGLE_API_KEY=your_api_key_here
MYRECSYS_PHASE2_CHAT_MODEL=gemini-2.5-flash-lite
MYRECSYS_PHASE2_EMBEDDING_MODEL=gemini-embedding-001
```

For this prototype, the key artifacts are:

- a curated local catalogue,
- a precomputed semantic index,
- and the recommendation service built on top of them.

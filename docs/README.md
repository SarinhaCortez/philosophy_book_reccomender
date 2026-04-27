# Philosophy Recommender

This project is a small recommendation system prototype built around the question:

> How can we recommend philosophy books from sparse, messy public metadata, while still keeping the system explainable and cheap enough to iterate on?

The current vrsion shows a progression from a basic retrieval baseline to a more semantic recommender, while keeping the architecture small enough to reason about end to end.

![UI demo](./mygif.gif)

## Idea

Being passionate for literature, I wanted to explore how I can make my bookworm personality come alive in my very scientific field of studies through this project. I am currently leaning towards Machine Learning Systems for large-scale data, and so I sought to explore the mechanisms underlying Recommender Systems, one of the most relevant applications of this paradigm nowadays.
The idea is a reading reccomender for users who want to start on philosophy but don't know where to start (me :)). A user writes a prompt such as:

> "I liked Nietzsche but now want something more practical and less dark about meaning."

And then the system:

1. understands what the user is looking for,
2. searches a local catalogue of philosophy books,
3. retrieves semantically related titles,
4. presents them in a graph-oriented interface that invites exploration.

This repo is about building a rcommendation pipeline under realistic constraints:

- public metadata is noisy,
- philosophy is a fuzzy domain,
- many books have weak or no descriptions,
- the catalogue has to be curated before retrieval quality can improve,
- and the reccomendation logic should evolve iteratively

## Design Goals

The implementation is guided by the following design decisions.

### 1. Keep the system inspectable

The pipeline is split into small stages:

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

One of the key choices in the current version is book understanding happening offline.

- **offline:** enrich books, embed books, build the semantic index
- **online:** extract one user preference profile, embed one query, retrieve nearest books

This keeps the online experience responsive while still allowing rich semantic matching.

### 4. Treat data quality as part of the recommender

For this project, the catalogue is part of the recommendation logic. In the beginning, noisy entries such as fiction, self-help, programming, or generally non-philosophy books would enter the corpus through permissive Open Library subject metadata.

So a large part of the engineering work here was:

- deciding what counts as "in-domain,"
- filtering,
- and estabilish the tradeoffs between catalogue size and catalogue quality.

## System Overview

Current high-level pipeline:

```text
User prompt -> LLM preference profile -> query embedding -> similarity search over pre-embedded books -> recommendation list + graph UI
```

The development happened in two stages.

### Stage 1: Web-App and Catalogue building

Goal:

- configure a local philosophy catalogue, design stubbed UI and simple single-endpoint backend


The project uses Open Library dumps rather than a live external books API. I chose a public, imperfect corpus because the project is trying to demonstrate system thinking. Some problems I faced were that Open Library metadata is noisy, subject labels are inconsistent, and a very small number of rated works, which would be usful to perform rating-based optimization.

The catalogue builder:

- scans Open Library works, authors, and ratings dumps,
- applies philosophy-oriented subject rules,
- resolves author names,
- ranks by number of ratings,
- writes a local JSONL catalogue.

A stricter filter yields a smaller but cleaner catalogue, a looser filter yields a larger but noisier one.

The backend is a very small custom HTTP server . That is a conscious choice since the project has one main endpoin. If this project were extended, I would move it to a more standard framework.

### Stage 2: Semantic enrichment

Goal:

- replace lexical matching for the definite semantic pipeline

I chose to use LLM-based structured book profiling, LLM-based structured user preference profiling and embeddings for retrieval. This helps capturing nuance, since philosophy requests are often nuanced and not well captured by keywords alone, metadata is sparse and benefits from normalization, and semantic retrieval is a good fit for cold-start.

The current recommender uses a semantic pipeline in which:

- books are enriched into structured profiles,
- those profiles are embedded,
- a user prompt is also turned into a structured profile and embedded,
- ranking is done by vector similarity.


#### Structured LLM profiles

A raw free-text prompt like:

> "I want something not as difficult or harsh as Nietzsche, still about meaning."

contains several useful signals:

- author affinity,
- desired tone,
- implicit topic,
- implicit avoidance.

Turning prompts into a structured preference profile gives the system a more stable semantic representation than naive token matching. Likewise for books, the Open Library metadata often needs normalization into fields such as:

- themes,
- traditions,
- reader moods,
- style,
- difficulty,
- era.

#### Offline computation of book embeddings

The expensive part is repeated book understanding, not per-query search. If every recommendation request had to re-profile and re-embed books online, the system would be slow, expensive, rate-limit prone.

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


## Current Repository Structure

Logic: 

- `ui.html`: light-mode graph-oriented frontend
- `myrecsys/app.py`: local HTTP server and `/recommend` endpoint
- `myrecsys/recommendation.py`: high-level recommendation orchestration
- `myrecsys/phase2.py`: semantic profiling, embedding, and ranking logic
- `myrecsys/book_enrichment.py`: display-oriented fallback enrichment for sparse books
- `myrecsys/local_catalog.py`: local catalogue loading and normalization
- `myrecsys/schemas.py`: application data structures

Data and build scripts:

- `data/semantic_index.jsonl`: precomputed semantic index
- `scripts/build_semantic_index.py`: direct semantic index builder
- `scripts/build_semantic_index_batch.py`: cheaper batch-based semantic index builder
- `scripts/append_zero_rated_philosophers.py`: append zero-rated works by selected canonical philosophers

## Further Improvements

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

### 4. Improve explanations

Future explanations should be:

- specific about the match,
- robust enough to justify surprising recommendations.

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


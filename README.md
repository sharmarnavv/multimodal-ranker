# multimodal-ranker

backend for ranking creatives. clip + qdrant.

## run

```bash
uv sync
uv run -m src.db  # init db
uv run uvicorn src.api:app --reload
```

## dev notes (read this)

### frontend
1. **dropdowns only**: for `role`/`location`. no free text. backend expects exact matches (e.g. "Model").
2. **search logic**:
   - `query`: the vibe (text).
   - `hard_filters`: must haves (e.g. `role=Model`).
   - `soft_filters`: nice to haves (e.g. `location=London`). result gets boosted if matches.
3. **images**: compress to 1024x1024 jpg before `/ingest`. dont send 20mb raws (too slow).
4. **ui**: if `boosted: true` in response, show a "📍 Near You" badge.
5. **empty search**: if search bar empty, call your sql db. dont call this api.

### backend
1. **sync**: strictly dual-write. if user updates profile, call `/ingest` here immediately.
2. **circuit breaker**: if `/search` takes >3s, kill it and fallback to your sql `LIKE` search. this is ai, it can be slow.

### ops
1. **hardware**: cpu optimized (c6i.xlarge) is fine. gpu overkill for now.
2. **ram**: need 4gb min. clip eats ram.
3. **env**: `QDRANT_URL`, `QDRANT_API_KEY`.

### future expansion
- **redis**: for async job queue (celery/bullmq). needed if handling massive bulk ingests or video processing. currently disabled.

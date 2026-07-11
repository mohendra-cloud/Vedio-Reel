# ReelForge backend

A small always-on service that does the part the browser can't safely do:
holds your API keys, and turns a scene's narration + visual prompt into a
real merged video clip using ffmpeg.

**Cost picture:** two of the three pieces are now free.
- **Text (script rewrite, scene breakdown, style/music suggestions)** — Gemini,
  free tier, no card required.
- **Voice** — Microsoft Edge's neural voices via the `msedge-tts` package.
  Free, no API key at all.
- **Video** — Runway. Still paid, no free tier. This is now the only cost
  in the whole pipeline.

This is intentionally minimal — an in-memory job queue instead of
Redis/BullMQ, local disk instead of Supabase Storage, no database yet. It's
the fastest path to *actually generating a real video end to end*.

## What it does

- `POST /ai/generate` — proxies a text prompt to Gemini and returns the
  response. This is what the frontend calls for script rewrite, scene
  breakdown, and every ✨ AI-refine feature — the Gemini key lives only here,
  never in the browser.
- `POST /uploads` — accepts a manually uploaded image or video (multipart
  field name `file`) and returns a filename. This is the **free** alternative
  to Runway: instead of an AI-generated clip, a scene can use your own photo
  or footage. Images get an automatic Ken Burns zoom; videos get looped or
  trimmed — both matched to the narration's length automatically.
- `POST /jobs/generate-scene` — given a scene's narration text and *either*
  a visual prompt (AI path, calls Runway) *or* a `visualFile` from `/uploads`
  (manual path, free), generates voiceover (always free, via Edge TTS) and
  the visual, then merges them with ffmpeg into one file.
- `POST /jobs/export` — concatenates a list of already-generated scene clips
  into one final video.
- `GET /jobs/:id` — poll a job's status/progress.
- Finished files are served back at `/files/<filename>`.

## 1. Local setup

```bash
npm install
cp .env.example .env
# edit .env — paste in your free Gemini key and your paid Runway key
npm start
```

You'll also need `ffmpeg` installed locally to test this before deploying —
`brew install ffmpeg` (Mac) or `apt install ffmpeg` (Linux). The Docker image
used for deployment already includes it, so this is only for local testing.

Get your free Gemini key at **aistudio.google.com/apikey** — no payment
method needed.

## 2. Try it

```bash
# Test the free text endpoint
curl -X POST http://localhost:8080/ai/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Say hello in five words."}'

# Test the full scene pipeline (needs Runway credits)
curl -X POST http://localhost:8080/jobs/generate-scene \
  -H "Content-Type: application/json" \
  -d '{
    "sceneId": "scene-1",
    "narration": "Deep beneath the ocean, pressure builds in ways we rarely see.",
    "imagePrompt": "Wide shot, deep ocean trench, bioluminescent particles drifting, cinematic lighting"
  }'
# -> { "jobId": "..." }

curl http://localhost:8080/jobs/<jobId>
# -> poll until "status": "complete", then open the resultUrl
```

## 3. Deploy to Railway

1. Push this folder to a GitHub repo (or use Railway's CLI to deploy directly
   from disk — `railway up` from inside this folder).
2. In Railway: **New Project → Deploy from GitHub repo**, pick the repo.
   Railway will detect the `Dockerfile` automatically and build from it —
   this is what gets you ffmpeg, so don't let it fall back to Nixpacks.
3. Under **Variables**, add:
   - `GEMINI_API_KEY` — free, from aistudio.google.com/apikey
   - `RUNWAYML_API_SECRET` — paid, from dev.runwayml.com
   - `BACKEND_ACCESS_TOKEN` — generate one with `openssl rand -hex 32`, and
     keep it secret. This is what stops random people from hitting your
     endpoints and spending your Runway credits.
4. Railway will give you a public URL like `reelforge-backend-production.up.railway.app`.
   Every request to it needs `Authorization: Bearer <your BACKEND_ACCESS_TOKEN>`.

## 4. Frontend setup

The `reelforge.html` file's Settings panel now only needs two things:
your backend URL and your backend access token. There's no separate
Anthropic/Gemini key field in the browser anymore — the Settings panel
talks to your backend's `/ai/generate` endpoint for all text features, and
your Gemini key stays safely server-side.

Voice picks (Warm/Crisp/Deep, etc.) in the app now map to real Microsoft
Edge neural voice names (like `en-US-AriaNeural`) instead of ElevenLabs
voice IDs — see `VOICE_ID_MAP` in `reelforge.html` if you want to swap in
different voices. Browse the full list of available Edge voices in the
`msedge-tts` package docs.

## Known limitations of this version

- **Files aren't persistent.** Railway's filesystem resets on redeploy.
  Fine for testing; for production, add Supabase Storage or S3 uploads.
- **One job at a time.** The in-memory queue processes sequentially. Fine
  for a solo user testing; for real traffic, move to BullMQ + Redis as
  described in the architecture doc.
- **No database.** Job state disappears if the service restarts mid-render.
  Swap the in-memory `Map` for the `render_jobs` Postgres table when ready.
- **Runway clips are 5 seconds each** (a `gen4.5` default) — good enough to
  prove the pipeline works end to end; scene pacing/duration tuning is a
  later refinement.
- **Video generation is still the one paid step.** Everything else in this
  pipeline is now free.

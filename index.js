import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { randomUUID } from "crypto";
import { spawn } from "child_process";
import fs from "fs";
import path from "path";
import RunwayML from "@runwayml/sdk";
import { MsEdgeTTS, OUTPUT_FORMAT } from "msedge-tts";
import multer from "multer";

dotenv.config();

const PORT = process.env.PORT || 8080;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const RUNWAY_KEY = process.env.RUNWAYML_API_SECRET;
const ACCESS_TOKEN = process.env.BACKEND_ACCESS_TOKEN;

const OUTPUT_DIR = path.resolve("./outputs");
fs.mkdirSync(OUTPUT_DIR, { recursive: true });

const runway = RUNWAY_KEY ? new RunwayML({ apiKey: RUNWAY_KEY }) : null;

const uploadStorage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, OUTPUT_DIR),
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname) || "";
    cb(null, `upload-${randomUUID()}${ext}`);
  },
});
const upload = multer({ storage: uploadStorage, limits: { fileSize: 150 * 1024 * 1024 } }); // 150MB cap

// --- in-memory job store (swap for the Postgres `render_jobs` table from the architecture doc when ready) ---
const jobs = new Map(); // jobId -> { status, progress, error, resultUrl, ... }
const queue = []; // job ids waiting to run

function createJob(type, payload) {
  const id = randomUUID();
  jobs.set(id, { id, type, payload, status: "queued", progress: 0, error: null, resultUrl: null, createdAt: Date.now() });
  queue.push(id);
  return id;
}

// --- free, server-side voice via Microsoft Edge's neural voices (no API key, no per-character cost) ---
const DEFAULT_VOICE_ID = "en-US-AriaNeural";

async function generateVoice(text, voiceId, outPath) {
  const tts = new MsEdgeTTS();
  await tts.setMetadata(voiceId || DEFAULT_VOICE_ID, OUTPUT_FORMAT.AUDIO_24KHZ_48KBITRATE_MONO_MP3);
  const { audioStream } = tts.toStream(text);
  await new Promise((resolve, reject) => {
    const chunks = [];
    audioStream.on("data", (chunk) => chunks.push(chunk));
    audioStream.on("close", () => {
      fs.writeFileSync(outPath, Buffer.concat(chunks));
      resolve();
    });
    audioStream.on("error", reject);
  });
}

async function generateVisual(promptText, outPath) {
  if (!runway) throw new Error("RUNWAYML_API_SECRET is not set on the backend");
  const task = await runway.textToVideo
    .create({ promptText, model: "gen4.5", ratio: "1280:720", duration: 5 })
    .waitForTaskOutput();
  const videoUrl = task.output?.[0];
  if (!videoUrl) throw new Error("Runway task finished with no output video");
  const res = await fetch(videoUrl);
  const buf = Buffer.from(await res.arrayBuffer());
  fs.writeFileSync(outPath, buf);
}

function ffprobeDuration(filePath) {
  return new Promise((resolve) => {
    const proc = spawn("ffprobe", [
      "-v", "error",
      "-show_entries", "format=duration",
      "-of", "default=noprint_wrappers=1:nokey=1",
      filePath,
    ]);
    let out = "";
    proc.stdout.on("data", (d) => (out += d.toString()));
    proc.on("close", () => {
      const seconds = parseFloat(out.trim());
      resolve(Number.isFinite(seconds) && seconds > 0 ? seconds : 5);
    });
    proc.on("error", () => resolve(5));
  });
}

const VIDEO_EXTENSIONS = new Set([".mp4", ".mov", ".webm", ".m4v", ".avi"]);

// Turns a manually uploaded image or video into a clip matching the narration's length —
// the free, no-API-key alternative to calling Runway for a scene's visual.
async function prepareUploadedVisual(filename, audioPath, outPath) {
  const srcPath = path.join(OUTPUT_DIR, filename);
  if (!fs.existsSync(srcPath)) throw new Error(`Uploaded file not found: ${filename}`);
  const duration = await ffprobeDuration(audioPath);
  const ext = path.extname(filename).toLowerCase();

  if (VIDEO_EXTENSIONS.has(ext)) {
    // Loop (if shorter) or trim (if longer) the uploaded video to match the narration.
    await ffmpeg(["-stream_loop", "-1", "-i", srcPath, "-t", String(duration), "-c:v", "libx264", "-an", outPath]);
  } else {
    // Still image — animate with a gentle Ken Burns zoom for the length of the narration.
    const frames = Math.max(25, Math.round(duration * 25));
    await ffmpeg([
      "-loop", "1", "-i", srcPath,
      "-vf", `zoompan=z='min(zoom+0.0008,1.15)':d=${frames}:s=1280x720,format=yuv420p`,
      "-t", String(duration), "-r", "25",
      outPath,
    ]);
  }
}

// --- free text generation via Gemini, called server-side so the key never reaches the browser ---
async function generateText(prompt) {
  if (!GEMINI_API_KEY) throw new Error("GEMINI_API_KEY is not set on the backend");
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${GEMINI_API_KEY}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }] }),
  });
  if (!res.ok) throw new Error(`Gemini error ${res.status}: ${await res.text()}`);
  const data = await res.json();
  const text = data.candidates?.[0]?.content?.parts?.map((p) => p.text || "").join("") || "";
  return text;
}

function ffmpeg(args) {
  return new Promise((resolve, reject) => {
    const proc = spawn("ffmpeg", ["-y", ...args]);
    let stderr = "";
    proc.stderr.on("data", (d) => (stderr += d.toString()));
    proc.on("close", (code) => (code === 0 ? resolve() : reject(new Error(`ffmpeg exited ${code}: ${stderr.slice(-800)}`))));
  });
}

// Escapes text for safe use inside an ffmpeg drawtext filter's quoted text value.
function escapeDrawtext(text) {
  return text
    .replace(/\\/g, "\\\\")
    .replace(/:/g, "\\:")
    .replace(/'/g, "\u2019") // swap apostrophes for a typographic quote — avoids fragile nested-quote escaping
    .replace(/%/g, "\\%");
}

// Wraps caption text to a readable line width for on-screen burned-in captions.
function wrapCaptionText(text, maxCharsPerLine = 42) {
  const words = text.split(/\s+/);
  const lines = [];
  let current = "";
  for (const word of words) {
    if ((current + " " + word).trim().length > maxCharsPerLine) {
      if (current) lines.push(current.trim());
      current = word;
    } else {
      current = (current + " " + word).trim();
    }
  }
  if (current) lines.push(current.trim());
  return lines.join("\n");
}

const CAPTION_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf";

// Merge one scene's video + narration audio into a single clip, audio trimmed/padded to video length.
// If captionText is provided, burns it in as on-screen captions (requires a re-encode; otherwise a fast copy).
async function mergeSceneAV(videoPath, audioPath, outPath, captionText) {
  if (captionText && captionText.trim()) {
    const drawtext = `drawtext=fontfile=${CAPTION_FONT}:text='${escapeDrawtext(wrapCaptionText(captionText.trim()))}':fontcolor=white:fontsize=26:line_spacing=6:box=1:boxcolor=black@0.55:boxborderw=12:x=(w-text_w)/2:y=h-th-36`;
    await ffmpeg([
      "-i", videoPath,
      "-i", audioPath,
      "-vf", drawtext,
      "-c:v", "libx264",
      "-c:a", "aac",
      "-shortest",
      outPath,
    ]);
  } else {
    await ffmpeg([
      "-i", videoPath,
      "-i", audioPath,
      "-c:v", "copy",
      "-c:a", "aac",
      "-shortest",
      outPath,
    ]);
  }
}

// Concatenate merged scene clips into the final export.
async function concatScenes(clipPaths, outPath) {
  const listFile = path.join(OUTPUT_DIR, `concat-${randomUUID()}.txt`);
  fs.writeFileSync(listFile, clipPaths.map((p) => `file '${p.replace(/'/g, "'\\''")}'`).join("\n"));
  await ffmpeg(["-f", "concat", "-safe", "0", "-i", listFile, "-c", "copy", outPath]);
  fs.unlinkSync(listFile);
}

// Mixes an optional sound effect under a scene's narration audio — a stinger, whoosh,
// ambience, etc. Trimmed to the narration's length so it never runs past the scene.
async function mixSceneSfx(narrationPath, sfxFilename, outPath) {
  const sfxPath = path.join(OUTPUT_DIR, sfxFilename);
  if (!fs.existsSync(sfxPath)) throw new Error(`Uploaded sound effect file not found: ${sfxFilename}`);
  const duration = await ffprobeDuration(narrationPath);
  await ffmpeg([
    "-i", narrationPath,
    "-i", sfxPath,
    "-filter_complex", `[1:a]volume=0.5,atrim=0:${duration}[sfx];[0:a][sfx]amix=inputs=2:duration=first:dropout_transition=1[aout]`,
    "-map", "[aout]",
    "-t", String(duration),
    outPath,
  ]);
}

// Mixes an uploaded music track under the final video's narration audio, looped/trimmed to
// match, at reduced volume so the narration stays clearly audible. This is the real
// implementation behind the "Background music" picker — a genuine uploaded track, not a
// preset name with nothing behind it.
async function mixBackgroundMusic(videoPath, musicFilename, outPath, volume) {
  const musicPath = path.join(OUTPUT_DIR, musicFilename);
  if (!fs.existsSync(musicPath)) throw new Error(`Uploaded music file not found: ${musicFilename}`);
  const duration = await ffprobeDuration(videoPath);
  const musicVolume = Number.isFinite(volume) ? Math.max(0, Math.min(1, volume)) : 0.18;
  await ffmpeg([
    "-i", videoPath,
    "-stream_loop", "-1", "-i", musicPath,
    "-filter_complex", `[1:a]volume=${musicVolume},atrim=0:${duration}[music];[0:a][music]amix=inputs=2:duration=first:dropout_transition=2[aout]`,
    "-map", "0:v",
    "-map", "[aout]",
    "-c:v", "copy",
    "-t", String(duration),
    outPath,
  ]);
}

// --- worker loop: pulls one job at a time. Swap for BullMQ/Redis if you need parallel workers later. ---
let working = false;
async function tick() {
  if (working || queue.length === 0) return;
  working = true;
  const jobId = queue.shift();
  const job = jobs.get(jobId);
  try {
    job.status = "processing";
    if (job.type === "scene_generate") {
      const { sceneId, narration, imagePrompt, voiceId, visualFile, showCaptions, sfxFile } = job.payload;
      const audioPath = path.join(OUTPUT_DIR, `${sceneId}-voice.mp3`);
      const audioMixedPath = path.join(OUTPUT_DIR, `${sceneId}-voice-mixed.mp3`);
      const videoPath = path.join(OUTPUT_DIR, `${sceneId}-visual.mp4`);
      const mergedPath = path.join(OUTPUT_DIR, `${sceneId}-merged.mp4`);

      await generateVoice(narration, voiceId, audioPath);
      let finalAudioPath = audioPath;
      if (sfxFile) {
        await mixSceneSfx(audioPath, sfxFile, audioMixedPath);
        finalAudioPath = audioMixedPath;
      }
      job.progress = 40;
      if (visualFile) {
        await prepareUploadedVisual(visualFile, finalAudioPath, videoPath);
      } else {
        await generateVisual(imagePrompt, videoPath);
      }
      job.progress = 75;
      await mergeSceneAV(videoPath, finalAudioPath, mergedPath, showCaptions ? narration : null);
      job.progress = 100;
      job.resultUrl = `/files/${path.basename(mergedPath)}`;
      job.status = "complete";
    } else if (job.type === "final_export") {
      const { sceneFiles, musicFile, musicVolume } = job.payload; // sceneFiles: filenames already in OUTPUT_DIR
      const concatPath = path.join(OUTPUT_DIR, `concat-${job.id}.mp4`);
      const finalPath = path.join(OUTPUT_DIR, `export-${job.id}.mp4`);
      const fullPaths = sceneFiles.map((f) => path.join(OUTPUT_DIR, f));
      await concatScenes(fullPaths, concatPath);
      job.progress = 60;
      if (musicFile) {
        await mixBackgroundMusic(concatPath, musicFile, finalPath, musicVolume);
        fs.unlinkSync(concatPath);
      } else {
        fs.renameSync(concatPath, finalPath);
      }
      job.progress = 100;
      job.resultUrl = `/files/${path.basename(finalPath)}`;
      job.status = "complete";
    }
  } catch (e) {
    job.status = "failed";
    job.error = e.message;
  } finally {
    working = false;
  }
}
setInterval(tick, 1000);

// --- API ---
const app = express();
app.use(cors());
app.use(express.json({ limit: "2mb" }));
app.use("/files", express.static(OUTPUT_DIR));

// Public — no token needed, so uptime checks and a quick browser visit both work.
app.get("/health", (req, res) => res.json({ ok: true }));

app.use((req, res, next) => {
  if (!ACCESS_TOKEN) return next(); // no token configured — open access, fine for local dev only
  const auth = req.headers.authorization || "";
  if (auth === `Bearer ${ACCESS_TOKEN}`) return next();
  res.status(401).json({ error: "missing or invalid Authorization header" });
});

// Text generation (script rewrite, scene breakdown, style/music suggestions, etc.)
// Runs server-side so the Gemini key never touches the browser.
app.post("/ai/generate", async (req, res) => {
  const { prompt } = req.body || {};
  if (!prompt) return res.status(400).json({ error: "prompt is required" });
  try {
    const text = await generateText(prompt);
    res.json({ text });
  } catch (e) {
    res.status(502).json({ error: e.message });
  }
});

// Manual video path: upload your own image or video for a scene instead of paying for Runway.
// Returns a filename to pass as visualFile in /jobs/generate-scene.
app.post("/uploads", upload.single("file"), (req, res) => {
  if (!req.file) return res.status(400).json({ error: "file is required (multipart field name: file)" });
  res.json({ filename: req.file.filename });
});

app.post("/jobs/generate-scene", (req, res) => {
  const { sceneId, narration, imagePrompt, voiceId, visualFile, showCaptions, sfxFile } = req.body || {};
  if (!sceneId || !narration || !(imagePrompt || visualFile)) {
    return res.status(400).json({ error: "sceneId, narration, and either imagePrompt (for AI generation) or visualFile (for a manual upload) are required" });
  }
  const id = createJob("scene_generate", { sceneId, narration, imagePrompt, voiceId, visualFile, showCaptions, sfxFile });
  res.json({ jobId: id });
});

app.post("/jobs/export", (req, res) => {
  const { sceneFiles, musicFile, musicVolume } = req.body || {};
  if (!Array.isArray(sceneFiles) || sceneFiles.length === 0) {
    return res.status(400).json({ error: "sceneFiles must be a non-empty array of filenames from prior scene_generate jobs" });
  }
  const id = createJob("final_export", { sceneFiles, musicFile, musicVolume });
  res.json({ jobId: id });
});

app.get("/jobs/:id", (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) return res.status(404).json({ error: "not found" });
  res.json(job);
});

app.listen(PORT, () => console.log(`ReelForge backend listening on :${PORT}`));

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

async function generateVoice(text, voiceOptions, outPath) {
  const opts = voiceOptions || {};
  const voiceId = opts.voiceId || DEFAULT_VOICE_ID;
  const tts = new MsEdgeTTS();
  await tts.setMetadata(voiceId, OUTPUT_FORMAT.AUDIO_24KHZ_48KBITRATE_MONO_MP3);
  const prosody = {};
  if (opts.pitch) prosody.pitch = opts.pitch;
  if (opts.rate) prosody.rate = opts.rate;
  const { audioStream } = tts.toStream(text, prosody);
  await new Promise((resolve, reject) => {
    const chunks = [];
    audioStream.on("data", (chunk) => chunks.push(chunk));
    audioStream.on("close", () => {
      fs.writeFileSync(outPath, Buffer.concat(chunks));
      resolve();
    });
    audioStream.on("error", reject);
  });
  // A real MP3 with any actual speech in it is at minimum a few KB. A file this small almost
  // always means the TTS engine returned little or no real audio — most commonly because the
  // narration's language doesn't match the selected voice (e.g. Hindi/Devanagari text sent to
  // an English-only voice). Fail here with a clear reason instead of letting a near-empty file
  // silently corrupt the merge step several stages later with a confusing ffmpeg error.
  const stats = fs.statSync(outPath);
  if (stats.size < 2000) {
    throw new Error(
      `Voice generation produced almost no audio (${stats.size} bytes) for voice "${voiceId}". ` +
      `This usually means the narration's language doesn't match the selected voice — for example, Hindi text sent to an English voice. Try a voice that matches the narration's actual language.`
    );
  }
}

// Common target frame — every scene's visual, regardless of source (Runway, uploaded video,
// or uploaded image), gets normalized to exactly this, so scenes never mismatch when concatenated.
const FRAME_W = 1280;
const FRAME_H = 720;
const FRAME_RATE = 25;

// Fits the FULL source into the target frame (nothing cropped, nothing cut off) and fills
// whatever empty space is left with a blurred, zoomed copy of the same image — the same
// technique Instagram/TikTok/CapCut use when converting between aspect ratios. This replaced
// two earlier attempts: plain padding (solid black bars, and compounded into a small boxed-in
// video when applied at both the per-scene and export resize stages) and plain cropping
// (fixed the black bars, but cut off real content — including the edges of burned-in
// captions — whenever the source aspect ratio didn't match the target).
function fitFillFilter(w, h, inLabel = "0:v", outLabel = "outv") {
  return (
    `[${inLabel}]split=2[bg][fg];` +
    `[bg]scale=${w}:${h}:force_original_aspect_ratio=increase,crop=${w}:${h},gblur=sigma=25[bgb];` +
    `[fg]scale=${w}:${h}:force_original_aspect_ratio=decrease[fgs];` +
    `[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1[${outLabel}]`
  );
}

// Loops (if the source is shorter) or trims (if longer) a video to an exact target duration,
// while normalizing it to the common frame size — used for both Runway output and uploaded
// video files, so every scene's visual duration matches its narration exactly, never truncating
// or overrunning it. Also forces a consistent frame rate: without this, a 30fps phone video
// and a Runway clip at its own native rate would produce scenes with different frame rates,
// which corrupts timestamps throughout the file the moment they're concatenated together
// (confirmed by reproducing it directly — this was the actual cause of the recurring,
// seemingly random ffmpeg crashes during export).
async function normalizeVideoToDuration(srcPath, outPath, targetDuration) {
  await ffmpeg([
    "-stream_loop", "-1", "-i", srcPath,
    "-filter_complex", `${fitFillFilter(FRAME_W, FRAME_H)};[outv]format=yuv420p[outv2]`,
    "-map", "[outv2]",
    "-t", String(targetDuration),
    "-r", String(FRAME_RATE),
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-an",
    outPath,
  ]);
}

async function generateVisual(promptText, outPath, targetDuration) {
  if (!runway) throw new Error("RUNWAYML_API_SECRET is not set on the backend");
  const task = await runway.textToVideo
    .create({ promptText, model: "gen4.5", ratio: "1280:720", duration: 5 })
    .waitForTaskOutput();
  const videoUrl = task.output?.[0];
  if (!videoUrl) throw new Error("Runway task finished with no output video");
  const res = await fetch(videoUrl);
  const buf = Buffer.from(await res.arrayBuffer());
  const rawPath = outPath.replace(/\.mp4$/, "-raw.mp4");
  fs.writeFileSync(rawPath, buf);
  // Runway always returns a fixed ~5s clip — loop/trim it to match this scene's actual
  // narration length exactly, so longer narration never gets cut off mid-sentence.
  await normalizeVideoToDuration(rawPath, outPath, targetDuration);
  fs.unlinkSync(rawPath);
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

// Turns manually uploaded images/videos into a single clip matching the narration's length —
// the free, no-API-key alternative to calling Runway for a scene's visual. Accepts one file,
// or an array — multiple files split the scene's duration evenly and play in sequence. Every
// file, whatever its original resolution or aspect ratio, gets normalized to the same common
// frame (letterboxed/pillarboxed, never stretched or distorted) so scenes always concatenate
// cleanly no matter what mix of photos and videos went into them.
async function prepareUploadedVisual(filenames, audioPath, outPath) {
  const list = Array.isArray(filenames) ? filenames : [filenames];
  if (list.length === 0) throw new Error("No uploaded visual files given for this scene");
  const duration = await ffprobeDuration(audioPath);
  const perClip = duration / list.length;
  const tempBase = outPath.replace(/\.mp4$/, "");
  const clipPaths = [];

  for (let i = 0; i < list.length; i++) {
    const filename = list[i];
    const srcPath = path.join(OUTPUT_DIR, filename);
    if (!fs.existsSync(srcPath)) throw new Error(`Uploaded file not found: ${filename}`);
    const ext = path.extname(filename).toLowerCase();
    const clipPath = `${tempBase}-part${i}.mp4`;

    if (VIDEO_EXTENSIONS.has(ext)) {
      await normalizeVideoToDuration(srcPath, clipPath, perClip);
    } else {
      // Still image — normalize to the common frame, then animate with a gentle Ken Burns
      // zoom for its share of the narration.
      const frames = Math.max(FRAME_RATE, Math.round(perClip * FRAME_RATE));
      await ffmpeg([
        "-loop", "1", "-i", srcPath,
        "-filter_complex", `${fitFillFilter(FRAME_W, FRAME_H)};[outv]zoompan=z='min(zoom+0.0008,1.15)':d=${frames}:s=${FRAME_W}x${FRAME_H},format=yuv420p[outv2]`,
        "-map", "[outv2]",
        "-t", String(perClip), "-r", String(FRAME_RATE),
        "-c:v", "libx264", "-preset", "veryfast",
        clipPath,
      ]);
    }
    clipPaths.push(clipPath);
  }

  if (clipPaths.length === 1) {
    fs.renameSync(clipPaths[0], outPath);
  } else {
    await concatScenes(clipPaths, outPath);
    clipPaths.forEach((p) => fs.existsSync(p) && fs.unlinkSync(p));
  }
}

// --- free text generation via Gemini, called server-side so the key never reaches the browser ---
async function generateText(prompt) {
  if (!GEMINI_API_KEY) throw new Error("GEMINI_API_KEY is not set on the backend");
  const url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent";
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY },
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

// Noto Sans Devanagari — chosen because it renders both Devanagari script (Hindi, Sanskrit)
// and standard Latin text correctly in the same font, confirmed by direct testing. DejaVu Sans
// (the previous font) has no Devanagari glyphs at all, which produced garbled boxes instead of
// text for any Hindi/Sanskrit captions.
// Bundled directly in the repo (fonts/NotoSansDevanagari-Bold.ttf) rather than relying on
// an apt-get install succeeding at deploy time — this removes an entire class of "did the
// package installer actually work on this platform" uncertainty. Renders both Devanagari
// (Hindi, Sanskrit) and standard Latin text correctly, confirmed by direct rendering test.
const CAPTION_FONT = path.join(process.cwd(), "fonts", "NotoSansDevanagari-Bold.ttf");
const CAPTION_SIZES = { small: 20, medium: 26, large: 34 };

// Merge one scene's video + narration audio into a single clip, audio trimmed/padded to video length.
// If captionText is provided, burns it in as on-screen captions (requires a re-encode; otherwise a fast copy).
async function mergeSceneAV(videoPath, audioPath, outPath, captionText, captionSize) {
  if (captionText && captionText.trim()) {
    const fontsize = CAPTION_SIZES[captionSize] || CAPTION_SIZES.medium;
    const drawtext = `drawtext=fontfile=${CAPTION_FONT}:text='${escapeDrawtext(wrapCaptionText(captionText.trim()))}':fontcolor=white:fontsize=${fontsize}:line_spacing=6:box=1:boxcolor=black@0.55:boxborderw=12:x=(w-text_w)/2:y=h-th-36`;
    await ffmpeg([
      "-i", videoPath,
      "-i", audioPath,
      "-vf", drawtext,
      "-c:v", "libx264",
      "-preset", "veryfast",
      "-c:a", "aac",
      "-shortest",
      "-movflags", "+faststart",
      outPath,
    ]);
  } else {
    await ffmpeg([
      "-i", videoPath,
      "-i", audioPath,
      "-c:v", "copy",
      "-c:a", "aac",
      "-shortest",
      "-movflags", "+faststart",
      outPath,
    ]);
  }
}

// Concatenate merged scene clips into the final export.
async function concatScenes(clipPaths, outPath) {
  const listFile = path.join(OUTPUT_DIR, `concat-${randomUUID()}.txt`);
  fs.writeFileSync(listFile, clipPaths.map((p) => `file '${p.replace(/'/g, "'\\''")}'`).join("\n"));
  // Video is fast-copied (all inputs are already normalized to the same codec/resolution by
  // this point); audio is re-encoded rather than copied, which avoids timestamp drift at the
  // splice points between scenes. +faststart lets the file start playing before it's fully
  // downloaded, so the preview players in the app show something immediately.
  await ffmpeg(["-f", "concat", "-safe", "0", "-i", listFile, "-c:v", "copy", "-c:a", "aac", "-movflags", "+faststart", outPath]);
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

// Resolution + aspect ratio targets for common platforms. `null` width/height means
// "leave the resolution as-is." Exported so the frontend can show the same list.
const PLATFORM_PRESETS = {
  none: { label: "Original — no resizing", width: null, height: null },
  instagram_reel: { label: "Instagram Reels / TikTok (9:16)", width: 1080, height: 1920 },
  instagram_feed: { label: "Instagram Feed post (1:1)", width: 1080, height: 1080 },
  youtube: { label: "YouTube (16:9)", width: 1920, height: 1080 },
  youtube_shorts: { label: "YouTube Shorts (9:16)", width: 1080, height: 1920 },
  facebook: { label: "Facebook video (16:9)", width: 1280, height: 720 },
  twitter: { label: "X / Twitter (16:9)", width: 1280, height: 720 },
};

// Plain 16:9 resolution tiers — used only when no platform preset (which has its own
// aspect-specific dimensions) is selected.
const RESOLUTION_TIERS = {
  "720p": { width: 1280, height: 720 },
  "1080p": { width: 1920, height: 1080 },
  "4K": { width: 3840, height: 2160 },
};

// One pass: resize (platform preset takes priority; otherwise the plain resolution tier),
// adjust brightness/contrast/saturation, and change playback speed — video and audio
// together so they stay in sync. Any combination can be a no-op; only the filters actually
// needed run. Resize uses the same fit-fill technique as scene normalization — full content
// preserved, no cropping, blurred fill instead of black bars.
async function applyFinalAdjustments(inputPath, outputPath, opts) {
  const { speed = 1, platformPreset = "none", resolution, brightness = 0, contrast = 1, saturation = 1 } = opts || {};
  const platform = PLATFORM_PRESETS[platformPreset] || PLATFORM_PRESETS.none;
  const target = platform.width && platform.height ? platform : (RESOLUTION_TIERS[resolution] || null);
  const speedClamped = Math.max(0.25, Math.min(4, Number(speed) || 1));

  const simpleFilters = [];
  if (brightness !== 0 || contrast !== 1 || saturation !== 1) {
    simpleFilters.push(`eq=brightness=${brightness}:contrast=${contrast}:saturation=${saturation}`);
  }
  if (speedClamped !== 1) {
    simpleFilters.push(`setpts=${(1 / speedClamped).toFixed(6)}*PTS`);
  }

  const audioFilters = [];
  if (speedClamped !== 1) {
    // ffmpeg's atempo filter only accepts 0.5–2.0 per instance; chain multiple to cover a wider range.
    let remaining = speedClamped;
    while (remaining > 2.0) { audioFilters.push("atempo=2.0"); remaining /= 2.0; }
    while (remaining < 0.5) { audioFilters.push("atempo=0.5"); remaining /= 0.5; }
    audioFilters.push(`atempo=${remaining.toFixed(6)}`);
  }

  const args = ["-i", inputPath];
  if (target && target.width && target.height) {
    // Resize needed — build via filter_complex (fit-fill requires it), chaining any
    // color/speed filters onto the same output label.
    let fc = fitFillFilter(target.width, target.height);
    if (simpleFilters.length) fc += `;[outv]${simpleFilters.join(",")}[outv2]`;
    const finalLabel = simpleFilters.length ? "[outv2]" : "[outv]";
    args.push("-filter_complex", fc, "-map", finalLabel, "-map", "0:a?");
  } else if (simpleFilters.length) {
    args.push("-vf", simpleFilters.join(","));
  }
  if (audioFilters.length) args.push("-af", audioFilters.join(","));
  args.push("-r", String(FRAME_RATE), "-c:v", "libx264", "-preset", "veryfast", "-threads", "2", "-x264-params", "threads=2:lookahead-threads=1", "-c:a", "aac", "-movflags", "+faststart", outputPath);
  await ffmpeg(args);
}

// Two-pass GIF conversion (palette generation, then reuse) — much better color quality
// than a naive single-pass GIF encode. GIFs have no audio track.
async function convertToGif(inputPath, outputPath) {
  const palettePath = outputPath.replace(/\.gif$/, "-palette.png");
  await ffmpeg(["-i", inputPath, "-vf", "fps=10,scale=480:-1:flags=lanczos,palettegen", palettePath]);
  await ffmpeg([
    "-i", inputPath, "-i", palettePath,
    "-filter_complex", "fps=10,scale=480:-1:flags=lanczos[x];[x][1:v]paletteuse",
    outputPath,
  ]);
  fs.unlinkSync(palettePath);
}

async function convertToWebm(inputPath, outputPath) {
  // Default libvpx-vp9 settings are extremely slow (confirmed: ~26s and dropping to 0.2x
  // realtime speed on a 5s clip in testing) — slow enough on a small server to risk hitting
  // a memory/time limit on longer videos. These flags trade a little quality for roughly an
  // 8x speedup, which is the right tradeoff for a background render job.
  await ffmpeg([
    "-i", inputPath,
    "-c:v", "libvpx-vp9",
    "-deadline", "realtime",
    "-cpu-used", "8",
    "-row-mt", "1",
    "-c:a", "libopus",
    "-b:v", "1M",
    outputPath,
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
      const { sceneId, narration, imagePrompt, voiceId, voicePitch, voiceRate, visualFiles, showCaptions, captionSize, sfxFile } = job.payload;
      const audioPath = path.join(OUTPUT_DIR, `${sceneId}-voice.mp3`);
      const audioMixedPath = path.join(OUTPUT_DIR, `${sceneId}-voice-mixed.mp3`);
      const videoPath = path.join(OUTPUT_DIR, `${sceneId}-visual.mp4`);
      const mergedPath = path.join(OUTPUT_DIR, `${sceneId}-merged.mp4`);

      await generateVoice(narration, { voiceId, pitch: voicePitch, rate: voiceRate }, audioPath);
      let finalAudioPath = audioPath;
      if (sfxFile) {
        await mixSceneSfx(audioPath, sfxFile, audioMixedPath);
        finalAudioPath = audioMixedPath;
      }
      job.progress = 40;
      if (visualFiles && visualFiles.length > 0) {
        await prepareUploadedVisual(visualFiles, finalAudioPath, videoPath);
      } else {
        const targetDuration = await ffprobeDuration(finalAudioPath);
        await generateVisual(imagePrompt, videoPath, targetDuration);
      }
      job.progress = 75;
      await mergeSceneAV(videoPath, finalAudioPath, mergedPath, showCaptions ? narration : null, captionSize);
      job.progress = 100;
      job.resultUrl = `/files/${path.basename(mergedPath)}`;
      job.status = "complete";
    } else if (job.type === "final_export") {
      const {
        sceneFiles, musicFile, musicVolume,
        speed, platformPreset, resolution, brightness, contrast, saturation,
        exportFormat,
      } = job.payload; // sceneFiles: filenames already in OUTPUT_DIR
      const concatPath = path.join(OUTPUT_DIR, `concat-${job.id}.mp4`);
      const musicedPath = path.join(OUTPUT_DIR, `music-${job.id}.mp4`);
      const adjustedPath = path.join(OUTPUT_DIR, `adjusted-${job.id}.mp4`);
      const finalExt = exportFormat === "GIF" ? "gif" : exportFormat === "WebM" ? "webm" : exportFormat === "MOV" ? "mov" : "mp4";
      const finalPath = path.join(OUTPUT_DIR, `export-${job.id}.${finalExt}`);
      const fullPaths = sceneFiles.map((f) => path.join(OUTPUT_DIR, f));

      await concatScenes(fullPaths, concatPath);
      job.progress = 35;

      let postMusicPath = concatPath;
      if (musicFile) {
        await mixBackgroundMusic(concatPath, musicFile, musicedPath, musicVolume);
        fs.unlinkSync(concatPath);
        postMusicPath = musicedPath;
      }
      job.progress = 55;

      await applyFinalAdjustments(postMusicPath, adjustedPath, { speed, platformPreset, resolution, brightness, contrast, saturation });
      fs.unlinkSync(postMusicPath);
      job.progress = 80;

      if (exportFormat === "GIF") {
        await convertToGif(adjustedPath, finalPath);
        fs.unlinkSync(adjustedPath);
      } else if (exportFormat === "WebM") {
        await convertToWebm(adjustedPath, finalPath);
        fs.unlinkSync(adjustedPath);
      } else if (exportFormat === "MOV") {
        fs.renameSync(adjustedPath, finalPath); // MOV can hold the same h264/aac stream as-is
      } else {
        fs.renameSync(adjustedPath, finalPath);
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

// Reports real memory numbers so we can confirm or rule out a container memory limit
// instead of guessing from crash symptoms alone. Reads the cgroup memory limit directly —
// /proc/meminfo reports the HOST's total memory, not this container's actual allocation,
// which is misleading on any containerized platform (Railway included).
function readContainerMemory() {
  const result = { limitMB: null, usedMB: null, source: null };
  try {
    const v2max = fs.readFileSync("/sys/fs/cgroup/memory.max", "utf8").trim();
    if (v2max !== "max") {
      result.limitMB = Math.round(Number(v2max) / 1024 / 1024);
      result.usedMB = Math.round(Number(fs.readFileSync("/sys/fs/cgroup/memory.current", "utf8").trim()) / 1024 / 1024);
      result.source = "cgroup v2";
      return result;
    }
  } catch (e) { /* fall through to v1 */ }
  try {
    const v1max = Number(fs.readFileSync("/sys/fs/cgroup/memory/memory.limit_in_bytes", "utf8").trim());
    const v1cur = Number(fs.readFileSync("/sys/fs/cgroup/memory/memory.usage_in_bytes", "utf8").trim());
    if (v1max > 0 && v1max < 1024 ** 4) {
      // A real, finite limit — under 1TB rules out the "effectively unlimited" sentinel value.
      result.limitMB = Math.round(v1max / 1024 / 1024);
      result.usedMB = Math.round(v1cur / 1024 / 1024);
      result.source = "cgroup v1";
    } else {
      result.source = "cgroup v1 (no limit set)";
      result.usedMB = Math.round(v1cur / 1024 / 1024);
    }
  } catch (e) {
    result.source = "unavailable: " + e.message;
  }
  return result;
}

app.get("/diagnostics", (req, res) => {
  const mem = process.memoryUsage();
  const toMB = (bytes) => Math.round(bytes / 1024 / 1024);
  res.json({
    processMemoryMB: { rss: toMB(mem.rss), heapUsed: toMB(mem.heapUsed) },
    containerMemory: readContainerMemory(),
    outputDirFileCount: fs.readdirSync(OUTPUT_DIR).length,
    activeJob: working,
    queueLength: queue.length,
  });
});

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
// Returns a filename to include in the visualFiles array in /jobs/generate-scene.
app.post("/uploads", upload.single("file"), (req, res) => {
  if (!req.file) return res.status(400).json({ error: "file is required (multipart field name: file)" });
  res.json({ filename: req.file.filename });
});

app.post("/jobs/generate-scene", (req, res) => {
  const { sceneId, narration, imagePrompt, voiceId, voicePitch, voiceRate, visualFiles, showCaptions, captionSize, sfxFile } = req.body || {};
  const hasVisualFiles = Array.isArray(visualFiles) && visualFiles.length > 0;
  if (!sceneId || !narration || !(imagePrompt || hasVisualFiles)) {
    return res.status(400).json({ error: "sceneId, narration, and either imagePrompt (for AI generation) or visualFiles (for manual uploads) are required" });
  }
  const id = createJob("scene_generate", { sceneId, narration, imagePrompt, voiceId, voicePitch, voiceRate, visualFiles, showCaptions, captionSize, sfxFile });
  res.json({ jobId: id });
});

app.get("/platform-presets", (req, res) => {
  const presets = Object.entries(PLATFORM_PRESETS).map(([id, p]) => ({ id, ...p }));
  res.json({ presets });
});

app.post("/jobs/export", (req, res) => {
  const { sceneFiles, musicFile, musicVolume, speed, platformPreset, resolution, brightness, contrast, saturation, exportFormat } = req.body || {};
  if (!Array.isArray(sceneFiles) || sceneFiles.length === 0) {
    return res.status(400).json({ error: "sceneFiles must be a non-empty array of filenames from prior scene_generate jobs" });
  }
  const id = createJob("final_export", {
    sceneFiles, musicFile, musicVolume,
    speed, platformPreset, resolution, brightness, contrast, saturation,
    exportFormat,
  });
  res.json({ jobId: id });
});

app.get("/jobs/:id", (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) return res.status(404).json({ error: "not found" });
  res.json(job);
});

if (fs.existsSync(CAPTION_FONT)) {
  console.log(`✓ Caption font found: ${CAPTION_FONT}`);
} else {
  console.error(`✗ WARNING: Caption font NOT FOUND at ${CAPTION_FONT} — Hindi/Sanskrit/English captions will fail or render incorrectly. Check that the fonts/ folder was actually pushed to the repo.`);
}

app.listen(PORT, () => console.log(`ReelForge backend listening on :${PORT}`));

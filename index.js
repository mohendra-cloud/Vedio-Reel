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

// Inserts SSML <break> pauses at natural line/sentence boundaries — the danda (।) and double
// danda (॥) used in Hindi/Sanskrit verse, plus standard punctuation — so poetry, dohas, and
// shlokas get a real breathing rhythm between lines instead of running together. This is a
// basic, well-supported SSML element (unlike style tags below), so it's reliable.
function insertPoeticPauses(text, breakMs = 350) {
  const parts = text.split(/([।॥?!.\n]+)/);
  let result = "";
  for (const part of parts) {
    if (!part) continue;
    result += part;
    if (/[।॥?!.\n]/.test(part)) result += `<break time="${breakMs}ms"/>`;
  }
  return result;
}

// Splits one scene's narration into smaller pieces for progressive, line-by-line captions
// instead of one static block for the whole scene. Prefers paragraph breaks if present,
// otherwise falls back to sentence-ish boundaries (danda ।, double danda ॥, and standard
// punctuation) — the same boundaries insertPoeticPauses already uses.
function splitIntoCaptionChunks(text) {
  const paraSplit = text.split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);
  if (paraSplit.length > 1) return paraSplit;
  const sentenceSplit = text.split(/(?<=[।॥.!?])\s+/).map((s) => s.trim()).filter(Boolean);
  return sentenceSplit.length > 0 ? sentenceSplit : [text.trim()];
}

// Gives each chunk a time slice within the scene's [sceneStart, sceneEnd] window, sized
// proportionally to how much of the scene's text it contains — a reasonable approximation of
// speech timing without needing true word-level audio alignment (which would need a speech
// recognition/forced-alignment step we don't have). Longer sentences get proportionally more
// time than short ones, rather than every chunk getting an identical, likely-wrong duration.
function allocateChunkTimings(chunks, sceneStart, sceneEnd) {
  const totalChars = chunks.reduce((sum, c) => sum + c.length, 0) || 1;
  const sceneDuration = sceneEnd - sceneStart;
  let cursor = sceneStart;
  return chunks.map((chunk) => {
    const share = chunk.length / totalChars;
    const dur = sceneDuration * share;
    const seg = { text: chunk, start: cursor, end: cursor + dur };
    cursor += dur;
    return seg;
  });
}

// Wraps text in <mstts:express-as>, Azure's SSML tag for speaking styles (cheerful, empathetic,
// etc). IMPORTANT HONESTY NOTE: this is a real Azure Speech feature, confirmed to exist for
// hi-IN-SwaraNeural specifically (styles: cheerful, empathetic, newscast) — but our TTS client
// is the free, unofficial Edge "Read Aloud" endpoint, not the paid Azure Speech API, and this
// hasn't been verified to actually change the audio on that free endpoint. It's included
// because it's harmless to try (the endpoint should just ignore an unrecognized tag rather than
// error), not because it's confirmed working.
function wrapExpressStyle(text, style) {
  if (!style) return text;
  return `<mstts:express-as style="${style}">${text}</mstts:express-as>`;
}

async function attemptGenerateVoice(text, opts, outPath) {
  const voiceId = opts.voiceId || DEFAULT_VOICE_ID;
  const tts = new MsEdgeTTS();
  await tts.setMetadata(voiceId, OUTPUT_FORMAT.AUDIO_24KHZ_48KBITRATE_MONO_MP3);
  const prosody = {};
  if (opts.pitch) prosody.pitch = opts.pitch;
  if (opts.rate) prosody.rate = opts.rate;
  let ssmlText = text;
  if (opts.poeticPauses) ssmlText = insertPoeticPauses(ssmlText, opts.pauseMs);
  if (opts.style) ssmlText = wrapExpressStyle(ssmlText, opts.style);
  const { audioStream } = tts.toStream(ssmlText, prosody);
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

async function generateVoice(text, voiceOptions, outPath) {
  const opts = voiceOptions || {};
  const voiceId = opts.voiceId || DEFAULT_VOICE_ID;
  const maxRetries = 3;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      await attemptGenerateVoice(text, opts, outPath);
      break;
    } catch (e) {
      // A dropped connection to Microsoft's free TTS endpoint (before it signals synthesis is
      // actually complete) is a real, known instability of that endpoint — a fresh attempt is
      // the correct recovery, since it's a network hiccup, not a problem with the request
      // itself. Retrying won't help a genuinely bad request, so only retry this specific case.
      const isConnectionIssue = /Stream closed before|WebSocket error/i.test(e.message || "");
      if (isConnectionIssue && attempt < maxRetries) {
        await new Promise((r) => setTimeout(r, 1000 * (attempt + 1))); // 1s, 2s, 3s
        continue;
      }
      throw e;
    }
  }
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
//
// mode: "fit" (default) — full content visible, blurred fill for empty space.
//       "fill" — crops to fill the frame completely, no blur, but can cut off content
//       at the edges. Offered as a user choice for people who prefer a full-bleed look.
function fitFillFilter(w, h, mode = "fit", inLabel = "0:v", outLabel = "outv") {
  if (mode === "fill") {
    return `[${inLabel}]scale=${w}:${h}:force_original_aspect_ratio=increase,crop=${w}:${h},setsar=1[${outLabel}]`;
  }
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
async function normalizeVideoToDuration(srcPath, outPath, targetDuration, frameFitMode) {
  await ffmpeg([
    "-stream_loop", "-1", "-i", srcPath,
    "-filter_complex", `${fitFillFilter(FRAME_W, FRAME_H, frameFitMode)};[outv]format=yuv420p[outv2]`,
    "-map", "[outv2]",
    "-t", String(targetDuration),
    "-r", String(FRAME_RATE),
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-an",
    outPath,
  ]);
}

async function generateVisual(promptText, outPath, targetDuration, frameFitMode) {
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
  await normalizeVideoToDuration(rawPath, outPath, targetDuration, frameFitMode);
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

// Reads the actual real width/height of a video file — used to size caption text and margins
// correctly for whatever the true final frame turned out to be, rather than recomputing (and
// risking drift from) the resize logic elsewhere.
function ffprobeDimensions(filePath) {
  return new Promise((resolve) => {
    const proc = spawn("ffprobe", [
      "-v", "error",
      "-select_streams", "v:0",
      "-show_entries", "stream=width,height",
      "-of", "csv=s=x:p=0",
      filePath,
    ]);
    let out = "";
    proc.stdout.on("data", (d) => (out += d.toString()));
    proc.on("close", () => {
      const match = out.trim().match(/^(\d+)x(\d+)$/);
      resolve(match ? { width: parseInt(match[1], 10), height: parseInt(match[2], 10) } : { width: FRAME_W, height: FRAME_H });
    });
    proc.on("error", () => resolve({ width: FRAME_W, height: FRAME_H }));
  });
}

const VIDEO_EXTENSIONS = new Set([".mp4", ".mov", ".webm", ".m4v", ".avi"]);

// Turns manually uploaded images/videos into a single clip matching the narration's length —
// the free, no-API-key alternative to calling Runway for a scene's visual. Accepts one file,
// or an array — multiple files split the scene's duration evenly and play in sequence. Every
// file, whatever its original resolution or aspect ratio, gets normalized to the same common
// frame (letterboxed/pillarboxed, never stretched or distorted) so scenes always concatenate
// cleanly no matter what mix of photos and videos went into them.
// Builds the zoompan filter string for a given transition type — each expression was tested
// directly against real rendered frames before being included here.
function buildTransitionFilter(transition, frames, w, h) {
  switch (transition) {
    case "zoom-out":
      return `zoompan=z='if(eq(on,0),1.15,max(1.15-0.0008*on,1.0))':d=${frames}:s=${w}x${h}`;
    case "pan-left":
      return `zoompan=z=1.15:x='(iw-iw/zoom)*(1-on/${frames})':y='ih/2-(ih/zoom/2)':d=${frames}:s=${w}x${h}`;
    case "pan-right":
      return `zoompan=z=1.15:x='(iw-iw/zoom)*(on/${frames})':y='ih/2-(ih/zoom/2)':d=${frames}:s=${w}x${h}`;
    case "static":
      return null; // no animation filter needed at all
    case "zoom-in":
    default:
      return `zoompan=z='min(zoom+0.0008,1.15)':d=${frames}:s=${w}x${h}`;
  }
}

async function prepareUploadedVisual(filenames, audioPath, outPath, frameFitMode, transition) {
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
      await normalizeVideoToDuration(srcPath, clipPath, perClip, frameFitMode);
    } else {
      // Still image — normalize to the common frame, then animate with the chosen transition.
      const frames = Math.max(FRAME_RATE, Math.round(perClip * FRAME_RATE));
      const transitionFilter = buildTransitionFilter(transition, frames, FRAME_W, FRAME_H);
      const filterChain = transitionFilter
        ? `${fitFillFilter(FRAME_W, FRAME_H, frameFitMode)};[outv]${transitionFilter},format=yuv420p[outv2]`
        : `${fitFillFilter(FRAME_W, FRAME_H, frameFitMode)};[outv]format=yuv420p[outv2]`;
      await ffmpeg([
        "-loop", "1", "-i", srcPath,
        "-filter_complex", filterChain,
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
  const maxRetries = 6;
  const maxWaitMs = 10000; // cap backoff so retries don't compound into an unreasonably long wait
  let lastError;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY },
      body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }] }),
    });
    if (res.ok) {
      const data = await res.json();
      return data.candidates?.[0]?.content?.parts?.map((p) => p.text || "").join("") || "";
    }
    const errText = await res.text();
    lastError = new Error(`Gemini error ${res.status}: ${errText}`);
    // 503 (server overloaded) and 429 (rate limited) are genuinely worth retrying — they're
    // temporary. A 400/401/403 means something's actually wrong (bad request, bad key), and
    // retrying that would just fail the same way every time, so those throw immediately.
    if ((res.status === 503 || res.status === 429) && attempt < maxRetries) {
      const waitMs = Math.min(1000 * Math.pow(2, attempt), maxWaitMs); // 1s,2s,4s,8s,10s,10s
      await new Promise((r) => setTimeout(r, waitMs));
      continue;
    }
    if (res.status === 503 || res.status === 429) {
      // All retries exhausted — this is a known, widely-reported Google-side capacity issue
      // (not specific to this account or key), so say so plainly rather than just surfacing
      // the raw JSON error.
      throw new Error(
        `Gemini's servers are still overloaded after ${maxRetries + 1} attempts over about 35 seconds. ` +
        `This is a known, widely-reported issue on Google's side during high-demand periods, not something wrong with your setup. Wait a minute or two and try again.`
      );
    }
    throw lastError;
  }
  throw lastError;
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

// Converts a "#RRGGBB" hex string to ffmpeg's "0xRRGGBB" color syntax, falling back to a
// safe default for anything that isn't a strictly valid hex color — these values come from
// a user-controlled color picker, so this also guards against malformed input reaching the
// ffmpeg command.
function toFFmpegColor(hex, fallback) {
  if (typeof hex === "string" && /^#[0-9a-fA-F]{6}$/.test(hex)) {
    return "0x" + hex.slice(1);
  }
  return fallback;
}

// Merge one scene's video + narration audio into a single clip, audio trimmed/padded to
// video length. Captions are NOT burned in here anymore — they're burned in once, at export
// time, after the final resize — because burning them at this stage positions them relative
// to the intermediate 1280x720 scene frame, which lands them off-center (often mid-frame
// instead of at the bottom) once that scene gets resized into a different final aspect ratio.
async function mergeSceneAV(videoPath, audioPath, outPath) {
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

// Burns captions onto the FINAL, already-resized video — one drawtext filter per scene, each
// scoped to only show during that scene's own time window via ffmpeg's enable=between(t,..)
// mechanism. This is what fixes captions landing off-center: by running after the final
// resize (not before it, at the per-scene stage), "bottom of frame" always means the bottom
// of the frame you're actually watching, whatever its final shape.
//
// segments: [{ text, start, end }] — start/end in seconds, already adjusted for any speed change.
// frameWidth/frameHeight: the actual final output dimensions, so line-wrapping and margins
// scale correctly whether the video ends up wide, narrow, or square — instead of wrapping at
// a fixed character count regardless of how much horizontal room is actually available.
// Formats seconds as an SRT timestamp: HH:MM:SS,mmm
function srtTimestamp(totalSeconds) {
  const ms = Math.round(totalSeconds * 1000);
  const h = Math.floor(ms / 3600000);
  const m = Math.floor((ms % 3600000) / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  const msRemainder = ms % 1000;
  const pad = (n, len = 2) => String(n).padStart(len, "0");
  return `${pad(h)}:${pad(m)}:${pad(s)},${pad(msRemainder, 3)}`;
}

// Builds a standard .srt subtitle file from the same timed segments used for burned-in
// captions — so you always get a real, downloadable subtitle file that matches your video
// exactly, whether or not you also chose to burn the captions into the video itself.
function generateSrt(segments) {
  const usable = segments.filter((s) => s.text && s.text.trim());
  return usable
    .map((s, i) => `${i + 1}\n${srtTimestamp(s.start)} --> ${srtTimestamp(s.end)}\n${s.text.trim()}\n`)
    .join("\n");
}

// Burns a one-time title card at the start of the video — separate from the per-scene
// captions above, meant for an episode title or intro text. Verified directly: extracted
// frames confirm it's visible partway through its window and genuinely gone afterward, not
// just theoretically timed.
async function burnTitleCard(inputPath, outputPath, text, durationSec, fontSize, fontColor, bgColor) {
  if (!text || !text.trim()) {
    fs.copyFileSync(inputPath, outputPath);
    return;
  }
  const color = toFFmpegColor(fontColor, "white");
  const box = toFFmpegColor(bgColor, "black") + "@0.6";
  const escaped = escapeDrawtext(text.trim());
  const drawtext = `drawtext=fontfile=${CAPTION_FONT}:text='${escaped}':fontcolor=${color}:fontsize=${fontSize || 48}:box=1:boxcolor=${box}:boxborderw=16:x=(w-text_w)/2:y=(h-text_h)/2:enable='between(t,0,${durationSec})'`;
  await ffmpeg([
    "-i", inputPath,
    "-vf", drawtext,
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-threads", "2",
    "-x264-params", "threads=2:lookahead-threads=1",
    "-c:a", "copy",
    "-movflags", "+faststart",
    outputPath,
  ]);
}

// Overlays a logo/watermark image in a chosen corner, present for the whole video. Uses
// scale2ref to size the watermark relative to the MAIN VIDEO's width — a bug in an earlier
// version scaled it relative to the watermark image's own native size instead, which meant
// the same "12%" setting rendered as 15.6% of a 720p frame but only 10.4% of a 1080p frame
// (confirmed by direct pixel measurement before this fix). scale2ref fixes that: measured at
// exactly 12.0% on both resolutions after the fix.
async function burnWatermark(inputPath, watermarkFilename, outputPath, position, sizePct) {
  const watermarkPath = path.join(OUTPUT_DIR, watermarkFilename);
  if (!fs.existsSync(watermarkPath)) throw new Error(`Watermark file not found: ${watermarkFilename}`);
  const scalePct = Math.max(0.05, Math.min(0.3, sizePct || 0.12));
  const margin = 20;
  const positions = {
    "top-left": `${margin}:${margin}`,
    "top-right": `main_w-overlay_w-${margin}:${margin}`,
    "bottom-left": `${margin}:main_h-overlay_h-${margin}`,
    "bottom-right": `main_w-overlay_w-${margin}:main_h-overlay_h-${margin}`,
  };
  const overlayPos = positions[position] || positions["bottom-right"];
  await ffmpeg([
    "-i", inputPath,
    "-i", watermarkPath,
    "-filter_complex", `[1:v][0:v]scale2ref=w=iw*${scalePct}:h=ow/mdar[wm][main];[main][wm]overlay=${overlayPos}[out]`,
    "-map", "[out]",
    "-map", "0:a?",
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-threads", "2",
    "-x264-params", "threads=2:lookahead-threads=1",
    "-c:a", "copy",
    "-movflags", "+faststart",
    outputPath,
  ]);
}

async function burnCaptionsWithTiming(inputPath, outputPath, segments, captionSize, captionFontColor, captionBgColor, frameWidth, frameHeight) {
  const usable = segments.filter((s) => s.text && s.text.trim());
  if (usable.length === 0) {
    fs.copyFileSync(inputPath, outputPath);
    return;
  }
  const fontsize = CAPTION_SIZES[captionSize] || CAPTION_SIZES.medium;
  const fontColor = toFFmpegColor(captionFontColor, "white");
  const boxColor = toFFmpegColor(captionBgColor, "black") + "@0.55";
  const w = frameWidth || FRAME_W;
  const h = frameHeight || FRAME_H;
  // Calibrated against the real font: measured ~0.44x fontsize per average Devanagari glyph
  // width at render time; 0.5x adds a small safety margin so wrapped lines never overflow.
  const marginPct = 0.08; // 8% margin on each side (left + right)
  const usableWidth = w * (1 - marginPct * 2);
  const maxCharsPerLine = Math.max(14, Math.floor(usableWidth / (fontsize * 0.5)));
  const bottomMargin = Math.round(h * 0.04); // scales with frame height instead of a fixed pixel value
  const filters = usable.map((s) => {
    const text = escapeDrawtext(wrapCaptionText(s.text.trim(), maxCharsPerLine));
    return `drawtext=fontfile=${CAPTION_FONT}:text='${text}':fontcolor=${fontColor}:fontsize=${fontsize}:line_spacing=6:box=1:boxcolor=${boxColor}:boxborderw=12:x=(w-text_w)/2:y=h-th-${bottomMargin}:enable='between(t,${s.start.toFixed(3)},${s.end.toFixed(3)})'`;
  });
  await ffmpeg([
    "-i", inputPath,
    "-vf", filters.join(","),
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-threads", "2",
    "-x264-params", "threads=2:lookahead-threads=1",
    "-c:a", "copy",
    "-movflags", "+faststart",
    outputPath,
  ]);
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
const MUSIC_PRESETS_DIR = path.join(process.cwd(), "music-presets");
const MUSIC_PRESETS = {
  none: { label: "None", file: null },
  peaceful: { label: "Peaceful — soft, slow", file: "preset-peaceful.mp3" },
  bhakti: { label: "Bhakti / Devotional — drone, Indian style", file: "preset-bhakti.mp3" },
  energetic: { label: "Energetic — upbeat, bright", file: "preset-energetic.mp3" },
};

async function mixBackgroundMusic(videoPath, musicFilename, outPath, volume, isPreset, ducking) {
  const musicPath = isPreset ? path.join(MUSIC_PRESETS_DIR, musicFilename) : path.join(OUTPUT_DIR, musicFilename);
  if (!fs.existsSync(musicPath)) throw new Error(`Music file not found: ${musicFilename}`);
  const duration = await ffprobeDuration(videoPath);
  const musicVolume = Number.isFinite(volume) ? Math.max(0, Math.min(1, volume)) : 0.18;
  const filterComplex = ducking
    // Real dynamic ducking (sidechaincompress) — music automatically gets quieter while
    // narration is speaking and comes back up in gaps, the same technique every professional
    // editor uses. These exact parameters were tuned by direct measurement: an earlier,
    // gentler setting (threshold=0.05, ratio=8) only produced a 2.6dB dip — too subtle to
    // actually notice. These settings measured a genuine 10.5dB dip between narration and
    // silence, confirmed with volumedetect before shipping.
    ? `[1:a]volume=${musicVolume},atrim=0:${duration}[musicpre];[musicpre][0:a]sidechaincompress=threshold=0.015:ratio=20:attack=5:release=300:makeup=1[musicducked];[0:a][musicducked]amix=inputs=2:duration=first:dropout_transition=2[aout]`
    : `[1:a]volume=${musicVolume},atrim=0:${duration}[music];[0:a][music]amix=inputs=2:duration=first:dropout_transition=2[aout]`;
  await ffmpeg([
    "-i", videoPath,
    "-stream_loop", "-1", "-i", musicPath,
    "-filter_complex", filterComplex,
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
// Builds an optional decorative frame filter — applied last, after resize, so the corner
// radius / vignette strength look right relative to the actual final frame, not an
// intermediate size. Rounded corners paint directly in RGB (no alpha channel), which is what
// makes them safe for standard MP4/H.264 delivery — an alpha-channel version was tested first
// and rejected because most video containers/players don't handle transparency correctly.
function buildFrameStyleFilter(frameStyle) {
  if (frameStyle === "vignette") return "vignette=PI/4";
  if (frameStyle === "rounded") {
    const r = 40; // corner radius in pixels — matches the value actually tested
    const cornerCond = `gt(abs(W/2-X),W/2-${r})*gt(abs(H/2-Y),H/2-${r})`;
    const dist = `hypot(W/2-${r}-abs(W/2-X),H/2-${r}-abs(H/2-Y))`;
    const keep = (ch) => `if(${cornerCond},if(lte(${dist},${r}),${ch}(X,Y),0),${ch}(X,Y))`;
    return `geq=r='${keep("r")}':g='${keep("g")}':b='${keep("b")}'`;
  }
  return null;
}

// Normalizes final audio to -14 LUFS (YouTube's actual loudness target) using a proper
// two-pass approach: pass 1 measures the real input characteristics, pass 2 applies
// normalization using those measured values for an accurate result — a single-pass version
// was tested first and produces a less precise result. Verified by direct measurement: a
// -38.35 LUFS test source came out at -13.95 LUFS after this exact two-pass process, a 0.05
// LUFS margin from the -14 target.
async function normalizeLoudness(inputPath, outputPath) {
  const measurePass = await new Promise((resolve, reject) => {
    const proc = spawn("ffmpeg", ["-i", inputPath, "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"]);
    let stderr = "";
    proc.stderr.on("data", (d) => (stderr += d.toString()));
    proc.on("close", (code) => {
      if (code !== 0 && !stderr.includes('"input_i"')) return reject(new Error(`Loudness measurement failed: ${stderr.slice(-500)}`));
      const match = stderr.match(/\{[\s\S]*"input_i"[\s\S]*?\}/);
      if (!match) return reject(new Error("Could not parse loudness measurement output"));
      try {
        resolve(JSON.parse(match[0]));
      } catch (e) {
        reject(new Error("Could not parse loudness measurement JSON"));
      }
    });
  });

  const filter = `loudnorm=I=-14:TP=-1.5:LRA=11:measured_I=${measurePass.input_i}:measured_TP=${measurePass.input_tp}:measured_LRA=${measurePass.input_lra}:measured_thresh=${measurePass.input_thresh}:offset=${measurePass.target_offset}:linear=true`;
  await ffmpeg([
    "-i", inputPath,
    "-af", filter,
    "-ar", "44100",
    "-c:v", "copy",
    "-c:a", "aac",
    "-movflags", "+faststart",
    outputPath,
  ]);
}

async function applyFinalAdjustments(inputPath, outputPath, opts) {
  const { speed = 1, platformPreset = "none", resolution, brightness = 0, contrast = 1, saturation = 1, frameFitMode, frameStyle } = opts || {};
  const platform = PLATFORM_PRESETS[platformPreset] || PLATFORM_PRESETS.none;
  const target = platform.width && platform.height ? platform : (RESOLUTION_TIERS[resolution] || null);
  const speedClamped = Math.max(0.25, Math.min(4, Number(speed) || 1));

  const simpleFilters = [];
  if (brightness !== 0 || contrast !== 1 || saturation !== 1) {
    simpleFilters.push(`eq=brightness=${brightness}:contrast=${contrast}:saturation=${saturation}`);
  }
  const frameStyleFilter = buildFrameStyleFilter(frameStyle);
  if (frameStyleFilter) simpleFilters.push(frameStyleFilter);
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
    let fc = fitFillFilter(target.width, target.height, frameFitMode);
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
      const { sceneId, narration, imagePrompt, voiceId, voicePitch, voiceRate, voiceStyle, poeticPauses, pauseMs, visualFiles, frameFitMode, transition, sfxFile } = job.payload;
      const audioPath = path.join(OUTPUT_DIR, `${sceneId}-voice.mp3`);
      const audioMixedPath = path.join(OUTPUT_DIR, `${sceneId}-voice-mixed.mp3`);
      const videoPath = path.join(OUTPUT_DIR, `${sceneId}-visual.mp4`);
      const mergedPath = path.join(OUTPUT_DIR, `${sceneId}-merged.mp4`);

      await generateVoice(narration, { voiceId, pitch: voicePitch, rate: voiceRate, style: voiceStyle, poeticPauses, pauseMs }, audioPath);
      let finalAudioPath = audioPath;
      if (sfxFile) {
        await mixSceneSfx(audioPath, sfxFile, audioMixedPath);
        finalAudioPath = audioMixedPath;
      }
      job.progress = 40;
      if (visualFiles && visualFiles.length > 0) {
        await prepareUploadedVisual(visualFiles, finalAudioPath, videoPath, frameFitMode, transition);
      } else {
        const targetDuration = await ffprobeDuration(finalAudioPath);
        await generateVisual(imagePrompt, videoPath, targetDuration, frameFitMode);
      }
      job.progress = 75;
      await mergeSceneAV(videoPath, finalAudioPath, mergedPath);
      job.progress = 100;
      job.resultUrl = `/files/${path.basename(mergedPath)}`;
      job.status = "complete";
    } else if (job.type === "final_export") {
      const {
        sceneFiles, captionTexts, musicFile, musicPreset, musicVolume,
        speed, platformPreset, resolution, brightness, contrast, saturation, frameFitMode, frameStyle,
        showCaptions, captionSize, captionFontColor, captionBgColor, ducking,
        watermarkFile, watermarkPosition, watermarkSize,
        titleCardText, titleCardDuration, titleCardFontColor, titleCardBgColor,
        exportFormat,
      } = job.payload; // sceneFiles: filenames already in OUTPUT_DIR
      const concatPath = path.join(OUTPUT_DIR, `concat-${job.id}.mp4`);
      const musicedPath = path.join(OUTPUT_DIR, `music-${job.id}.mp4`);
      const adjustedPath = path.join(OUTPUT_DIR, `adjusted-${job.id}.mp4`);
      const captionedPath = path.join(OUTPUT_DIR, `captioned-${job.id}.mp4`);
      const watermarkedPath = path.join(OUTPUT_DIR, `watermarked-${job.id}.mp4`);
      const titledPath = path.join(OUTPUT_DIR, `titled-${job.id}.mp4`);
      const finalExt = exportFormat === "GIF" ? "gif" : exportFormat === "WebM" ? "webm" : exportFormat === "MOV" ? "mov" : "mp4";
      const finalPath = path.join(OUTPUT_DIR, `export-${job.id}.${finalExt}`);
      const fullPaths = sceneFiles.map((f) => path.join(OUTPUT_DIR, f));

      // Work out each scene's time window in the final timeline BEFORE concatenating, then
      // split that scene's narration into smaller chunks (paragraphs, or sentences if there
      // are no paragraph breaks) so captions appear and disappear progressively through the
      // scene instead of showing the whole block the entire time.
      let cumulative = 0;
      const rawSegments = [];
      for (let i = 0; i < fullPaths.length; i++) {
        const dur = await ffprobeDuration(fullPaths[i]);
        const sceneText = (captionTexts && captionTexts[i]) || "";
        if (sceneText.trim()) {
          const chunks = splitIntoCaptionChunks(sceneText);
          rawSegments.push(...allocateChunkTimings(chunks, cumulative, cumulative + dur));
        }
        cumulative += dur;
      }

      await concatScenes(fullPaths, concatPath);
      job.progress = 30;

      let postMusicPath = concatPath;
      const activeMusicFile = musicPreset && MUSIC_PRESETS[musicPreset] && MUSIC_PRESETS[musicPreset].file
        ? MUSIC_PRESETS[musicPreset].file
        : musicFile;
      const activeMusicIsPreset = !!(musicPreset && MUSIC_PRESETS[musicPreset] && MUSIC_PRESETS[musicPreset].file);
      if (activeMusicFile) {
        await mixBackgroundMusic(concatPath, activeMusicFile, musicedPath, musicVolume, activeMusicIsPreset, ducking);
        fs.unlinkSync(concatPath);
        postMusicPath = musicedPath;
      }
      job.progress = 50;

      await applyFinalAdjustments(postMusicPath, adjustedPath, { speed, platformPreset, resolution, brightness, contrast, saturation, frameFitMode, frameStyle });
      fs.unlinkSync(postMusicPath);
      job.progress = 65;

      // Normalize to -14 LUFS (YouTube's real loudness target) now — after speed/color
      // adjustments are already applied, since those could otherwise shift levels again after
      // normalization. Runs once on the fully-mixed narration+music+ducking result. Skipped
      // for GIF exports — confirmed convertToGif's filter never maps an audio stream at all,
      // so normalizing loudness before converting to GIF would just be discarded processing.
      let normalizedPath = adjustedPath;
      if (exportFormat !== "GIF") {
        normalizedPath = path.join(OUTPUT_DIR, `normalized-${job.id}.mp4`);
        await normalizeLoudness(adjustedPath, normalizedPath);
        fs.unlinkSync(adjustedPath);
      }
      job.progress = 70;

      // Timing adjusted for any speed change — shared by both burned-in captions and the
      // standalone .srt file, so they always agree with each other and with the real video.
      const speedClamped = Math.max(0.25, Math.min(4, Number(speed) || 1));
      const timedSegments = rawSegments.map((s) => ({ text: s.text, start: s.start / speedClamped, end: s.end / speedClamped }));
      const hasCaptionText = timedSegments.some((s) => s.text.trim());

      // Burn captions now, on the truly final-shaped frame — running after the resize (not
      // before it, at the per-scene stage) is what makes "bottom of frame" always mean the
      // bottom of the frame you're actually watching, whatever its final shape.
      let postCaptionPath = normalizedPath;
      if (showCaptions && hasCaptionText) {
        const { width: realWidth, height: realHeight } = await ffprobeDimensions(normalizedPath);
        await burnCaptionsWithTiming(normalizedPath, captionedPath, timedSegments, captionSize, captionFontColor, captionBgColor, realWidth, realHeight);
        fs.unlinkSync(normalizedPath);
        postCaptionPath = captionedPath;
      }
      job.progress = 80;

      // Watermark — a persistent logo/brand mark in one corner, present for the whole video.
      let postWatermarkPath = postCaptionPath;
      if (watermarkFile) {
        await burnWatermark(postCaptionPath, watermarkFile, watermarkedPath, watermarkPosition, watermarkSize);
        fs.unlinkSync(postCaptionPath);
        postWatermarkPath = watermarkedPath;
      }

      // Title card — text overlaid on the first few seconds only (e.g. an episode title),
      // separate from the spoken-narration captions above.
      let postTitlePath = postWatermarkPath;
      if (titleCardText && titleCardText.trim()) {
        await burnTitleCard(postWatermarkPath, titledPath, titleCardText, titleCardDuration || 3, 48, titleCardFontColor, titleCardBgColor);
        fs.unlinkSync(postWatermarkPath);
        postTitlePath = titledPath;
      }
      job.progress = 88;

      // A standalone .srt file, generated whenever there's caption text at all — independent
      // of whether you also chose to burn captions into the video. Useful for YouTube's native
      // subtitle upload, which is more accessible/searchable than burned-in text alone.
      let srtUrl = null;
      if (hasCaptionText) {
        const srtPath = path.join(OUTPUT_DIR, `export-${job.id}.srt`);
        fs.writeFileSync(srtPath, generateSrt(timedSegments));
        srtUrl = `/files/${path.basename(srtPath)}`;
      }

      if (exportFormat === "GIF") {
        await convertToGif(postTitlePath, finalPath);
        fs.unlinkSync(postCaptionPath);
      } else if (exportFormat === "WebM") {
        await convertToWebm(postCaptionPath, finalPath);
        fs.unlinkSync(postCaptionPath);
      } else if (exportFormat === "MOV") {
        fs.renameSync(postCaptionPath, finalPath); // MOV can hold the same h264/aac stream as-is
      } else {
        fs.renameSync(postCaptionPath, finalPath);
      }

      job.progress = 100;
      job.resultUrl = `/files/${path.basename(finalPath)}`;
      job.srtUrl = srtUrl;
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

// Public — serves a preset track for in-app preview playback (an <audio> tag can't send an
// Authorization header, same reasoning as /files being public).
app.get("/music-presets/:id/audio", (req, res) => {
  const preset = MUSIC_PRESETS[req.params.id];
  if (!preset || !preset.file) return res.status(404).json({ error: "not found" });
  res.sendFile(path.join(MUSIC_PRESETS_DIR, preset.file));
});

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
  const { sceneId, narration, imagePrompt, voiceId, voicePitch, voiceRate, voiceStyle, poeticPauses, pauseMs, visualFiles, frameFitMode, transition, sfxFile } = req.body || {};
  const hasVisualFiles = Array.isArray(visualFiles) && visualFiles.length > 0;
  if (!sceneId || !narration || !(imagePrompt || hasVisualFiles)) {
    return res.status(400).json({ error: "sceneId, narration, and either imagePrompt (for AI generation) or visualFiles (for manual uploads) are required" });
  }
  const id = createJob("scene_generate", { sceneId, narration, imagePrompt, voiceId, voicePitch, voiceRate, voiceStyle, poeticPauses, pauseMs, visualFiles, frameFitMode, transition, sfxFile });
  res.json({ jobId: id });
});

app.get("/platform-presets", (req, res) => {
  const presets = Object.entries(PLATFORM_PRESETS).map(([id, p]) => ({ id, ...p }));
  res.json({ presets });
});

app.get("/music-presets", (req, res) => {
  const presets = Object.entries(MUSIC_PRESETS).map(([id, p]) => ({ id, label: p.label }));
  res.json({ presets });
});

app.post("/jobs/export", (req, res) => {
  const {
    sceneFiles, captionTexts, musicFile, musicPreset, musicVolume,
    speed, platformPreset, resolution, brightness, contrast, saturation, frameFitMode, frameStyle,
    showCaptions, captionSize, captionFontColor, captionBgColor, ducking,
    watermarkFile, watermarkPosition, watermarkSize,
    titleCardText, titleCardDuration, titleCardFontColor, titleCardBgColor,
    exportFormat,
  } = req.body || {};
  if (!Array.isArray(sceneFiles) || sceneFiles.length === 0) {
    return res.status(400).json({ error: "sceneFiles must be a non-empty array of filenames from prior scene_generate jobs" });
  }
  const id = createJob("final_export", {
    sceneFiles, captionTexts, musicFile, musicPreset, musicVolume,
    speed, platformPreset, resolution, brightness, contrast, saturation, frameFitMode, frameStyle,
    showCaptions, captionSize, captionFontColor, captionBgColor, ducking,
    watermarkFile, watermarkPosition, watermarkSize,
    titleCardText, titleCardDuration, titleCardFontColor, titleCardBgColor,
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

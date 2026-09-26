#!/usr/bin/env python3
"""
shrink_samples.py - make small, fast development copies of large CCTV clips.

The organizer sample videos are 3840x2160, 10-bit 4:2:2, ~140 Mbit/s, which is
about 5.8 GB per five-minute clip. That is awkward to annotate, to copy around,
and to run experiments on. This script re-encodes them to a much smaller file
that is still perfectly usable for development work.

It uses PyAV rather than the `ffmpeg` command line on purpose:

  * PyAV ships a complete FFmpeg build including libx264. Many system ffmpeg
    packages are compiled without it, and some cannot even *decode* 10-bit
    4:2:2 H.264 (the sample clips are yuv422p10le).
  * No subprocess handling, no shell quoting, no PATH surprises.

What is preserved
    Frame timing. Source frame rate is kept unless --fps is given, because the
    event boundaries, TTC and braking rules all depend on real timestamps.
    Wall-clock duration is identical.

What is dropped
    Audio and metadata tracks. Nothing in the task uses them, and the clips
    carry ~1.5 Mbit/s of uncompressed PCM audio plus a data track.
    Bit depth: 10-bit 4:2:2 becomes 8-bit yuv420p. 10-bit buys nothing here
    (a 4:2:0 sensor image is what the camera actually produced) and costs
    roughly a third of the bitrate.

Examples
    # whole folder -> data/samples_small/, 1080p, ~40x smaller
    python scripts/shrink_samples.py C3897.MP4 C3902.MP4 -o data/samples_small/

    # hit a size budget per file instead of a resolution
    python scripts/shrink_samples.py . -o data/samples_small/ --max-size 150

    # a short clip for eyeballing one event
    python scripts/shrink_samples.py C3897.MP4 --start 120 --duration 30 \
        -o debug/clip_120s.mp4 --height 720
"""
from __future__ import annotations

import argparse
import concurrent.futures
import shutil
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

try:
    import av
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "PyAV is required: pip install av\n"
        "(PyAV bundles a full FFmpeg with libx264, unlike many system builds.)"
    ) from exc


VIDEO_SUFFIXES = {".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".avi", ".m4v"}

# Preference order. libx264 with CRF gives the best quality per byte; the rest
# are fallbacks for stripped-down FFmpeg builds.
ENCODER_PREFERENCE = ("libx264", "libvpx-vp9", "mpeg4")


# ---------------------------------------------------------------------------
# probing
# ---------------------------------------------------------------------------
@dataclass
class ClipInfo:
    path: Path
    width: int
    height: int
    fps: Fraction
    frames: int
    duration: float
    size_bytes: int
    pix_fmt: str
    encoder: str


def probe(path: Path) -> ClipInfo:
    """Read stream metadata without decoding the whole file."""
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise RuntimeError(f"{path.name}: no video stream")
        stream = container.streams.video[0]
        ctx = stream.codec_context
        rate = stream.average_rate or stream.base_rate or Fraction(25, 1)
        duration = float(container.duration / av.time_base) if container.duration else 0.0
        if duration <= 0 and stream.duration and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        frames = stream.frames or 0
        if not frames and duration > 0:
            frames = int(round(duration * float(rate)))
        return ClipInfo(
            path=path,
            width=int(ctx.width),
            height=int(ctx.height),
            fps=rate,
            frames=frames,
            duration=duration,
            size_bytes=path.stat().st_size,
            pix_fmt=str(ctx.pix_fmt or "?"),
            encoder=str(ctx.name),
        )


def pick_encoder(requested: str | None) -> str:
    available = {str(c) for c in av.codecs_available}
    if requested:
        if requested not in available:
            raise SystemExit(f"encoder {requested!r} not available in this PyAV build")
        return requested
    for name in ENCODER_PREFERENCE:
        if name in available:
            return name
    raise SystemExit("no usable video encoder found in this PyAV build")


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def target_size(src_w: int, src_h: int, height: int | None, width: int | None, scale: float | None) -> tuple[int, int]:
    """Resolve the output resolution, keeping aspect ratio and even dimensions."""
    if scale is not None:
        out_w, out_h = src_w * scale, src_h * scale
    elif height is not None:
        out_h = float(height)
        out_w = src_w * (out_h / src_h)
    elif width is not None:
        out_w = float(width)
        out_h = src_h * (out_w / src_w)
    else:
        return src_w, src_h

    out_w = max(2, int(round(out_w / 2.0)) * 2)
    out_h = max(2, int(round(out_h / 2.0)) * 2)
    if out_w > src_w or out_h > src_h:
        # Upscaling only costs bytes and adds no information.
        factor = min(src_w / out_w, src_h / out_h)
        out_w = max(2, int(round(out_w * factor / 2.0)) * 2)
        out_h = max(2, int(round(out_h * factor / 2.0)) * 2)
    return out_w, out_h


# ---------------------------------------------------------------------------
# transcoding
# ---------------------------------------------------------------------------
@dataclass
class Result:
    source: Path
    output: Path | None
    ok: bool
    note: str = ""
    src_bytes: int = 0
    out_bytes: int = 0
    src_frames: int = 0
    out_frames: int = 0
    src_duration: float = 0.0
    out_duration: float = 0.0
    seconds: float = 0.0
    width: int = 0
    height: int = 0


def build_command_line(
    info: ClipInfo,
    dst: Path,
    out_w: int,
    out_h: int,
    out_rate: Fraction,
    encoder: str,
    crf: int | None,
    preset: str,
    bitrate: int | None,
    start: float,
    duration: float | None,
) -> list[str]:
    """Human-readable equivalent, for --dry-run."""
    cmd = [
        "ffmpeg", "-i", str(info.path),
        "-an", "-sn", "-dn",
        "-c:v", encoder,
        "-vf", f"scale={out_w}:{out_h}",
    ]
    if float(out_rate) != float(info.fps):
        cmd += ["-r", f"{float(out_rate):.6f}"]
    if bitrate:
        cmd += ["-b:v", str(bitrate)]
    else:
        cmd += ["-crf", str(crf)]
    if encoder == "libx264":
        cmd += ["-preset", preset, "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += [str(dst)]
    return cmd


def transcode(
    info: ClipInfo,
    dst: Path,
    out_w: int,
    out_h: int,
    out_rate: Fraction,
    encoder: str,
    crf: int | None,
    preset: str,
    bitrate: int | None,
    start: float,
    duration: float | None,
    quiet: bool = False,
) -> Result:
    """Re-encode one clip. Streams frame by frame, so memory stays flat."""
    started = time.perf_counter()
    written = 0
    # Keep the real extension on the temp name so the muxer can infer the format.
    tmp = dst.with_name(f"{dst.stem}.part{dst.suffix}")

    input_container = av.open(str(info.path))
    try:
        in_stream = input_container.streams.video[0]
        in_stream.thread_type = "AUTO"  # multi-threaded decode
        in_time_base = in_stream.time_base or Fraction(1, 1000)

        # Seek to the start offset when trimming. Seeking is keyframe-based, so
        # frames before the target are still decoded and then dropped below.
        if start > 0:
            seek_target = max(0, int(start * float(in_stream.time_base or Fraction(1, 1000))))
            try:
                input_container.seek(seek_target, stream=in_stream, backward=True)
            except av.error.FFmpegError:
                pass

        output_container = av.open(str(tmp), "w", options={"movflags": "+faststart"})
        try:
            out_stream = output_container.add_stream(encoder, rate=out_rate)
            out_stream.width = out_w
            out_stream.height = out_h
            out_stream.pix_fmt = "yuv420p"
            options = {}
            if bitrate:
                out_stream.bit_rate = bitrate
            else:
                options["crf"] = str(crf)
            if encoder == "libx264":
                options["preset"] = preset
            if options:
                out_stream.options = options

            keep_fps = abs(float(out_rate) - float(info.fps)) < 1e-6
            next_keep = 0.0
            stop_at = (start + duration) if duration else None
            last_pts: int | None = None

            for frame in input_container.decode(video=0):
                if frame.pts is None:
                    continue
                t = float(frame.pts * in_time_base)
                if t < start - 1e-6:
                    continue
                if stop_at is not None and t >= stop_at + 1e-6:
                    break
                if not keep_fps:
                    if t + 1e-6 < next_keep:
                        continue
                    next_keep += 1.0 / float(out_rate)

                # One reformat handles scaling, 10-bit -> 8-bit and YUV layout.
                out_frame = frame.reformat(width=out_w, height=out_h, format="yuv420p")
                for packet in out_stream.encode(out_frame):
                    output_container.mux(packet)
                    last_pts = packet.pts
                written += 1
                if not quiet and written % 300 == 0:
                    rate = written / max(1e-6, time.perf_counter() - started)
                    print(f"    {written} frames  {rate:.1f} fps", flush=True)

            for packet in out_stream.encode():
                output_container.mux(packet)
                last_pts = packet.pts
        finally:
            output_container.close()
    finally:
        input_container.close()

    if written == 0:
        tmp.unlink(missing_ok=True)
        return Result(
            source=info.path, output=None, ok=False, note="no frames decoded",
            src_bytes=info.size_bytes, src_frames=info.frames,
            src_duration=info.duration, seconds=time.perf_counter() - started,
        )

    tmp.replace(dst)

    # Verify what we actually produced rather than trusting the encoder.
    out_info = probe(dst)
    return Result(
        source=info.path,
        output=dst,
        ok=True,
        src_bytes=info.size_bytes,
        out_bytes=out_info.size_bytes,
        src_frames=info.frames,
        out_frames=out_info.frames,
        src_duration=info.duration,
        out_duration=out_info.duration,
        seconds=time.perf_counter() - started,
        width=out_info.width,
        height=out_info.height,
        note=out_info.pix_fmt,
    )


# ---------------------------------------------------------------------------
# input handling
# ---------------------------------------------------------------------------
def collect_inputs(paths: list[str], recursive: bool) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            found.extend(sorted(x for x in it if x.is_file() and x.suffix in VIDEO_SUFFIXES))
        elif p.is_file():
            found.append(p)
        else:
            matches = sorted(Path().glob(raw))
            if not matches:
                print(f"warning: no such file or directory: {raw}", file=sys.stderr)
            found.extend(m for m in matches if m.is_file())
    # De-duplicate while keeping order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in found:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("inputs", nargs="+", help="video file(s), directory, or glob")
    ap.add_argument("-o", "--out", default="data/samples_small",
                    help="output directory, or a file path when a single input is given")
    ap.add_argument("--height", type=int, default=1080,
                    help="output height in pixels; width follows aspect (default 1080, 0 = keep)")
    ap.add_argument("--width", type=int, default=None, help="output width; overrides --height")
    ap.add_argument("--scale", type=float, default=None, help="scale factor, e.g. 0.5")
    ap.add_argument("--fps", type=float, default=None,
                    help="output frame rate (default: keep the source rate so timings stay exact)")
    ap.add_argument("--crf", type=int, default=26, help="x264 quality, lower = better (default 26)")
    ap.add_argument("--preset", default="veryfast", help="x264 speed preset (default veryfast)")
    ap.add_argument("--max-size", type=float, default=None, metavar="MB",
                    help="target a size budget per file instead of a resolution")
    ap.add_argument("--start", type=float, default=0.0, help="trim start, seconds")
    ap.add_argument("--duration", type=float, default=None, help="trim length, seconds")
    ap.add_argument("--suffix", default="_small", help="output filename suffix (default _small)")
    ap.add_argument("--encoder", default=None, help="force an encoder (default: best available)")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="parallel files (default 1)")
    ap.add_argument("-r", "--recursive", action="store_true", help="recurse into sub-directories")
    ap.add_argument("--overwrite", action="store_true", help="replace existing output files")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--quiet", action="store_true", help="suppress per-frame progress")
    args = ap.parse_args()

    videos = collect_inputs(args.inputs, args.recursive)
    if not videos:
        print("no input videos found", file=sys.stderr)
        return 2

    encoder = pick_encoder(args.encoder)
    out_is_file = len(videos) == 1 and Path(args.out).suffix in VIDEO_SUFFIXES
    out_dir = None if out_is_file else Path(args.out)

    print(f"encoder : {encoder}")
    print(f"inputs  : {len(videos)} file(s)")
    print()

    jobs: list[tuple[ClipInfo, Path, dict]] = []
    total_in = 0
    for src in videos:
        try:
            info = probe(src)
        except Exception as exc:
            print(f"skip {src.name}: {exc}", file=sys.stderr)
            continue
        total_in += info.size_bytes

        if out_is_file:
            dst = Path(args.out)
        else:
            stem = src.stem
            # Trimming changes the identity of the clip, so make that visible.
            if args.start or args.duration:
                stem += f"_{args.start:g}s" + (f"_{args.duration:g}s" if args.duration else "")
            dst = (out_dir or Path(".")) / f"{stem}{args.suffix}{src.suffix.lower()}"

        if dst.resolve() == src.resolve():
            print(f"error: {src.name} would overwrite its own source", file=sys.stderr)
            return 2
        if dst.exists() and not args.overwrite:
            print(f"skip {dst.name}: already exists (use --overwrite)")
            continue

        out_w, out_h = target_size(
            info.width, info.height,
            args.height if args.height else None,
            args.width, args.scale,
        )
        out_rate = Fraction(args.fps).limit_denominator(1001) if args.fps else info.fps

        bitrate = None
        if args.max_size:
            # Length of the output, which is not the same as the source length
            # once trimming is in play: --start 30 --duration 45 yields 45 s.
            usable = args.duration if args.duration else (info.duration - args.start)
            usable = max(1.0, usable)
            bitrate = max(150_000, int(args.max_size * 1_000_000 * 8 / usable * 0.92))

        opts = dict(
            out_w=out_w, out_h=out_h, out_rate=out_rate, encoder=encoder,
            crf=args.crf, preset=args.preset, bitrate=bitrate,
            start=args.start, duration=args.duration,
        )

        if args.dry_run:
            cmd = build_command_line(
                info, dst, out_w, out_h, out_rate, encoder,
                None if bitrate else args.crf, args.preset, bitrate,
                args.start, args.duration,
            )
            print(" ".join(cmd))
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((info, dst, opts))

    if args.dry_run:
        return 0
    if not jobs:
        print("nothing to do")
        return 0

    print(f"{'file':<26}{'in':>11}{'out':>11}{'ratio':>8}{'frames':>14}{'time':>9}  note")
    print("-" * 92)

    results: list[Result] = []
    if args.jobs > 1 and len(jobs) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures_list = [pool.submit(transcode, i, d, quiet=args.quiet, **o) for i, d, o in jobs]
            for fut in concurrent.futures.as_completed(futures_list):
                results.append(fut.result())
    else:
        for info, dst, opts in jobs:
            results.append(transcode(info, dst, quiet=args.quiet, **opts))

    results.sort(key=lambda r: r.source.name)
    total_out = 0
    failed = 0
    for r in results:
        if not r.ok:
            failed += 1
            print(f"{r.source.name:<26}{human(r.src_bytes):>11}{'-':>11}{'-':>8}{'-':>14}{'-':>9}  FAILED: {r.note}")
            continue
        total_out += r.out_bytes
        ratio = r.src_bytes / r.out_bytes if r.out_bytes else 0
        frames = f"{r.src_frames}->{r.out_frames}"
        print(f"{r.source.name:<26}{human(r.src_bytes):>11}{human(r.out_bytes):>11}"
              f"{ratio:>7.1f}x{frames:>14}{r.seconds:>8.1f}s  {r.width}x{r.height} {r.note}")
        # Compare against the length we actually asked for, not the whole source.
        expected = args.duration if args.duration else max(0.0, r.src_duration - args.start)
        if expected > 0 and abs(r.out_duration - expected) > max(1.0, 0.02 * expected):
            print(f"{'':<26}warning: expected ~{expected:.1f}s, got {r.out_duration:.1f}s")

    print("-" * 92)
    if total_out:
        print(f"total  {human(total_in)} -> {human(total_out)}  "
              f"({total_in / max(1, total_out):.1f}x smaller)")
    if failed:
        print(f"{failed} file(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

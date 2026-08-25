# Why this fork exists

This is a fork of FFmpeg carrying one addition: a `libavcodec` encoder wrapper
for **next264**, a from-scratch H.264/AVC encoder
(<https://github.com/terranvigil/next264>).

**It is a staging ground, not a destination.** The wrapper is intended for
ffmpeg-devel once it has been proven here. Nothing about this fork is meant to
be a permanent parallel FFmpeg, and no other change to FFmpeg is planned or
wanted. If you have arrived here looking for FFmpeg, use upstream.

## How it is structured, and why that matters

The fork holds a **single commit** adding the wrapper, on top of an upstream
release tag. Everything else is stock FFmpeg at that tag.

That discipline is deliberate. It keeps rebasing onto a new FFmpeg release to
moving one commit, it means `git format-patch` produces the exact submission for
ffmpeg-devel, and it makes the diff against upstream reviewable in one sitting.
A fork that accumulates unrelated changes stops being upstreamable, and this one
is meant to graduate.

## What is being proven before submission

1. The wrapper satisfies the `AVCodec` contract properly: delay and flush at
   EOF, pts/dts under B-frame reordering, `extradata` for containers that want
   a global header, and the option mapping a user expects.
2. It is correct across formats and bit depths, gated by next264's own
   reconstruction-match test, where the encoder's reconstruction must equal an
   independent decoder's output bit for bit.
3. It carries its weight: encoding through FFmpeg should be measurably faster
   than piping Y4M between processes, which is the only way to use next264
   today and costs about a third of a machine on 1080p because the decode runs
   in a separate process.

## Licensing

next264 is BSD-2-Clause. It is therefore linkable into FFmpeg's default LGPL
build, and `--enable-libnext264` does not require `--enable-gpl` the way
`--enable-libx264` does. The wrapper file itself lives in `libavcodec` and
carries FFmpeg's licence, as it must.

next264 contains no code from FFmpeg, from x264, or from any other encoder. The
wrapper here is written against FFmpeg's public headers and documented encoder
API, not by adapting an existing wrapper.

## Building

```sh
./configure --enable-libnext264   # plus your usual options
make -j
ffmpeg -i input.mp4 -c:v next264 -crf 25 output.mp4
```

Requires libnext264 installed and discoverable through `pkg-config`.

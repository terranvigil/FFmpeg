/*
 * yah264 H.264 encoder wrapper
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <yah264.h>

#include "libavutil/cpu.h"
#include "libavutil/frame.h"
#include "libavutil/mastering_display_metadata.h"
#include "libavutil/opt.h"
#include "libavutil/mem.h"
#include "libavutil/pixdesc.h"
#include "avcodec.h"
#include "codec_internal.h"
#include "encode.h"
#include "internal.h"

typedef struct Yah264Context {
    AVClass        *class;
    yah264_param_t params;
    yah264_encoder_t *enc;

    /* yah264 hands back NAL units it owns, valid only until the next call in.
     * They are copied into the packet before returning. */
    char *preset;
    char *tune;
    float crf;
    int   subme;
    int   trellis_opt;
    float aq_strength;
    float psy_rd;
    char *mastering_display;
    char *cll;

    /* Input timestamps kept BY DISPLAY INDEX, because that is the key the
     * encoder reports its output against. The encoder reorders, so a packet's
     * timestamp cannot be inferred from arrival order: B frames are coded after
     * the anchor that follows them in display order, and pairing packets with
     * timestamps as they arrive gives every B the wrong one. */
    int64_t *pts_by_disp;
    int      pts_alloc;
    int      n_input;           /* display index of the next submitted frame */
    int     *emit_disp;         /* which frames the last encode call emitted */
    int      emit_alloc;
    int64_t  dts_next;
    int      delay_frames;
    int      eof;

    AVFrame *frame;
    /* SPS/PPS held for in-band emission. A muxer that wants a global header
     * takes them through extradata; one that does not, such as raw Annex-B,
     * needs them in the stream ahead of the first slice or the result does not
     * decode at all. */
    uint8_t *hdr;
    size_t   hdr_size;
    /* Without a global header, every IDR after the first carries the
     * parameter sets again, so a reader that starts or seeks there (a
     * transport stream, raw Annex-B) can decode it. The encoder writes them
     * once; its display-metadata SEI it repeats at every IDR itself. */
    uint8_t *rep_ps;
    size_t   rep_ps_size;
    int      first_out;
    int      hdr_warned;
    struct Yah264Out { uint8_t *data; size_t size; int key; int64_t pts; } *out;
    int      out_alloc;
    int      out_count;

    /* Input frames held back before any are submitted. yah264 parallelises
     * over a window: it needs a run of consecutive submits in flight before
     * the GOP and frame workers have anything to spread. lavc hands over one
     * frame per packet retrieved, so draining after every submit pins the
     * encoder near serial. Buffering `depth` frames and then feeding them
     * back-to-back gives it the run it needs. */
    AVFrame **inq;
    int       inq_alloc;
    int       inq_count;
    int       depth;
} Yah264Context;

typedef struct Yah264Out Yah264Out;

static av_cold int yah264_close(AVCodecContext *avctx)
{
    Yah264Context *ctx = avctx->priv_data;

    if (ctx->enc)
        yah264_encoder_close(ctx->enc);
    ctx->enc = NULL;
    av_freep(&ctx->pts_by_disp);
    av_freep(&ctx->emit_disp);
    av_frame_free(&ctx->frame);
    while (ctx->out_count > 0)
        av_freep(&ctx->out[--ctx->out_count].data);
    av_freep(&ctx->out);
    av_freep(&ctx->hdr);
    av_freep(&ctx->rep_ps);
    while (ctx->inq_count > 0)
        av_frame_free(&ctx->inq[--ctx->inq_count]);
    av_freep(&ctx->inq);
    return 0;
}

/* yah264's bit depth is fixed when the library is compiled, so one build takes
 * 8-bit planes and another takes 16-bit ones, and the sample type this file was
 * compiled against has to be the one the .so was. pkg-config carries the depth
 * in its Cflags, which is how configure gets it right; the run-time check in
 * init() is for the case where the library underneath us was swapped after
 * that. Advertising both depths from one build is not possible. */
static int yah264_csp_from_pixfmt(enum AVPixelFormat fmt)
{
    switch (fmt) {
#if Y264_BIT_DEPTH > 8
    case AV_PIX_FMT_YUV420P10: return YAH264_CSP_I420;
    case AV_PIX_FMT_YUV422P10: return YAH264_CSP_I422;
    case AV_PIX_FMT_YUV444P10: return YAH264_CSP_I444;
#else
    case AV_PIX_FMT_YUV420P: return YAH264_CSP_I420;
    case AV_PIX_FMT_YUV422P: return YAH264_CSP_I422;
    case AV_PIX_FMT_YUV444P: return YAH264_CSP_I444;
#endif
    default:                 return -1;
    }
}

/* lavc's colour enums carry the H.273 code points themselves, so they pass
 * through as numbers. lavc's extended values (256 and up) have no H.273
 * code and are left unsignalled, with a warning. */
static int h273_code(AVCodecContext *avctx, const char *what, int v, int unset)
{
    if (v == unset)
        return -1;
    if (v < 0 || v > 255) {
        av_log(avctx, AV_LOG_WARNING, "%s %d has no H.273 code; not signalled\n", what, v);
        return -1;
    }
    return v;
}

/* Nothing is signalled unless something was set. Limited range on its own
 * counts as nothing: it is what a decoder assumes without being told, so a
 * Y4M input tagged limited codes as an untagged one would. */
static int yah264_set_color(AVCodecContext *avctx, yah264_encoder_t *enc)
{
    yah264_video_signal_t vs = { 0, 2, 2, 2, -1 };
    int set = 0, v;

    if (avctx->color_range == AVCOL_RANGE_JPEG) {
        vs.full_range = 1;
        set = 1;
    }
    if ((v = h273_code(avctx, "color_primaries", avctx->color_primaries, AVCOL_PRI_UNSPECIFIED)) >= 0) {
        vs.primaries = v;
        set = 1;
    }
    if ((v = h273_code(avctx, "color_trc", avctx->color_trc, AVCOL_TRC_UNSPECIFIED)) >= 0) {
        vs.transfer = v;
        set = 1;
    }
    if ((v = h273_code(avctx, "colorspace", avctx->colorspace, AVCOL_SPC_UNSPECIFIED)) >= 0) {
        vs.matrix = v;
        set = 1;
    }
    /* lavc counts from 1 with 0 as unspecified; H.264's chroma_sample_loc_type
     * counts from 0 in the same order */
    if (avctx->chroma_sample_location != AVCHROMA_LOC_UNSPECIFIED) {
        vs.chroma_loc = avctx->chroma_sample_location - 1;
        set = 1;
    }
    if (!set)
        return 0;
    return yah264_encoder_set_video_signal(enc, &vs) < 0 ? AVERROR_EXTERNAL : 0;
}

static unsigned rescale_clip(AVRational q, int unit, int64_t max)
{
    int64_t v;

    if (q.den <= 0 || q.num < 0)
        return 0;
    v = av_rescale(q.num, unit, q.den);
    return v > max ? (unsigned)max : (unsigned)v;
}

/* Side data in the encoder's units, so a frame's can be compared with what
 * init took. Returns 0 when there's nothing usable. The SEI needs the
 * primaries and the luminance both, so side data with only one of them is
 * skipped with a warning rather than written with zeroes for the other. */
static int mdcv_units(AVCodecContext *avctx, const AVMasteringDisplayMetadata *m,
                      unsigned prim[6], unsigned wp[2], unsigned lum[2], int warn)
{
    /* the SEI lists green, blue, red; lavc stores red, green, blue */
    static const int order[3] = { 1, 2, 0 };

    if (!m->has_primaries || !m->has_luminance) {
        if (warn && (m->has_primaries || m->has_luminance))
            av_log(avctx, AV_LOG_WARNING, "mastering display side data has %s only; "
                   "the SEI needs both, so none is written\n",
                   m->has_primaries ? "primaries" : "luminance");
        return 0;
    }
    for (int i = 0; i < 3; i++) {
        prim[2 * i]     = rescale_clip(m->display_primaries[order[i]][0], 50000, 50000);
        prim[2 * i + 1] = rescale_clip(m->display_primaries[order[i]][1], 50000, 50000);
    }
    wp[0]  = rescale_clip(m->white_point[0], 50000, 50000);
    wp[1]  = rescale_clip(m->white_point[1], 50000, 50000);
    lum[0] = rescale_clip(m->max_luminance, 10000, UINT32_MAX);
    lum[1] = rescale_clip(m->min_luminance, 10000, UINT32_MAX);
    return 1;
}

/* HDR10 metadata from the side data the caller attached to the context.
 * ffmpeg's own tool copies a stream's mastering display and light level
 * there from the first decoded frame, so a tagged input carries through.
 * The encoder takes these at open; a later frame whose side data differs is
 * reported in submit_one. */
static void yah264_hdr_side_data(AVCodecContext *avctx, yah264_param_t *p)
{
    const AVFrameSideData *sd;
    unsigned prim[6], wp[2], lum[2];

    sd = av_frame_side_data_get(avctx->decoded_side_data, avctx->nb_decoded_side_data,
                                AV_FRAME_DATA_CONTENT_LIGHT_LEVEL);
    if (sd) {
        const AVContentLightMetadata *cll = (const AVContentLightMetadata *)sd->data;
        p->cll_max = FFMIN(cll->MaxCLL, 65535);
        p->cll_avg = FFMIN(cll->MaxFALL, 65535);
    }

    sd = av_frame_side_data_get(avctx->decoded_side_data, avctx->nb_decoded_side_data,
                                AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
    if (sd && mdcv_units(avctx, (const AVMasteringDisplayMetadata *)sd->data, prim, wp, lum, 1)) {
        memcpy(p->mastering_prim, prim, sizeof(prim));
        memcpy(p->mastering_wp, wp, sizeof(wp));
        p->mastering_max = lum[0];
        p->mastering_min = lum[1];
        p->mastering_set = 1;
    }
}

/* Once per encode: the first frame whose mastering display or light level
 * differs from what open took. Nothing can change it mid-stream, so it is
 * only reported. */
static void yah264_hdr_frame_check(AVCodecContext *avctx, Yah264Context *ctx, const AVFrame *f)
{
    const yah264_param_t *p = &ctx->params;
    const AVFrameSideData *sd;
    unsigned prim[6], wp[2], lum[2];
    int differs = 0;

    if (ctx->hdr_warned)
        return;
    sd = av_frame_get_side_data(f, AV_FRAME_DATA_CONTENT_LIGHT_LEVEL);
    if (sd) {
        const AVContentLightMetadata *cll = (const AVContentLightMetadata *)sd->data;
        differs |= (int)FFMIN(cll->MaxCLL, 65535) != p->cll_max ||
                   (int)FFMIN(cll->MaxFALL, 65535) != p->cll_avg;
    }
    sd = av_frame_get_side_data(f, AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
    if (sd && mdcv_units(avctx, (const AVMasteringDisplayMetadata *)sd->data, prim, wp, lum, 0))
        differs |= !p->mastering_set || memcmp(prim, p->mastering_prim, sizeof(prim)) ||
                   memcmp(wp, p->mastering_wp, sizeof(wp)) ||
                   lum[0] != p->mastering_max || lum[1] != p->mastering_min;
    if (differs) {
        av_log(avctx, AV_LOG_WARNING, "frame %"PRId64" carries mastering display or light "
               "level side data that differs from the stream's; the encoder writes what "
               "it took at init (put it on the codec context, or use -mastering-display "
               "and -cll)\n", f->pts);
        ctx->hdr_warned = 1;
    }
}

/* The private options take the command-line encoder's syntax and go through
 * the library's own parser, so they are checked the same way. They win over
 * side data. */
static int yah264_hdr_options(AVCodecContext *avctx, Yah264Context *ctx, yah264_param_t *p)
{
    char why[256];
    int r;

    if (ctx->mastering_display) {
        r = yah264_param_parse_mastering(p, ctx->mastering_display, why, sizeof(why));
        if (r < 0) {
            av_log(avctx, AV_LOG_ERROR, "mastering-display %s\n", why);
            return AVERROR(EINVAL);
        }
        if (r > 0)
            av_log(avctx, AV_LOG_WARNING, "mastering-display %s\n", why);
    }
    if (ctx->cll) {
        r = yah264_param_parse_cll(p, ctx->cll, why, sizeof(why));
        if (r < 0) {
            av_log(avctx, AV_LOG_ERROR, "cll %s\n", why);
            return AVERROR(EINVAL);
        }
    }
    return 0;
}

static int append_bytes(uint8_t **buf, size_t *size, const yah264_nal_t *n)
{
    uint8_t *p = av_realloc(*buf, *size + n->size);
    if (!p)
        return AVERROR(ENOMEM);
    memcpy(p + *size, n->payload, n->size);
    *buf = p;
    *size += n->size;
    return 0;
}

static av_cold int yah264_init(AVCodecContext *avctx)
{
    Yah264Context *ctx = avctx->priv_data;
    yah264_param_t *p = &ctx->params;
    yah264_nal_t *nal;
    int nnal, csp, i, ret;

    /* Both depths install under the same soname, so this mismatch links and
     * loads; it shows up as every plane being read at half or double its
     * stride. Fail here rather than encode garbage. */
    if (yah264_bit_depth() != Y264_BIT_DEPTH) {
        av_log(avctx, AV_LOG_ERROR,
               "libyah264 is a %d-bit build, this wrapper was compiled for "
               "%d-bit\n", yah264_bit_depth(), (int)Y264_BIT_DEPTH);
        return AVERROR(EINVAL);
    }

    csp = yah264_csp_from_pixfmt(avctx->pix_fmt);
    if (csp < 0) {
        av_log(avctx, AV_LOG_ERROR, "unsupported pixel format %s\n",
               av_get_pix_fmt_name(avctx->pix_fmt));
        return AVERROR(EINVAL);
    }

    yah264_param_default(p);

    if (ctx->preset && yah264_param_apply_preset(p, ctx->preset) < 0) {
        av_log(avctx, AV_LOG_ERROR, "unknown preset '%s'\n", ctx->preset);
        return AVERROR(EINVAL);
    }

    p->width  = avctx->width;
    p->height = avctx->height;
    p->csp    = csp;

    if (avctx->framerate.num > 0 && avctx->framerate.den > 0) {
        p->timebase.fps_num = avctx->framerate.num;
        p->timebase.fps_den = avctx->framerate.den;
    } else if (avctx->time_base.num > 0 && avctx->time_base.den > 0) {
        p->timebase.fps_num = avctx->time_base.den;
        p->timebase.fps_den = avctx->time_base.num;
    }

    /* The library takes a thread BUDGET and resolves it: 0 means "decide",
     * which is exactly what ffmpeg's thread_count 0 means, so it passes
     * straight through. No core counting and no clamping here -- both are the
     * encoder's business, and a wrapper carrying either would be carrying
     * encoder policy that upstream would rightly object to.
     *
     * This replaces a version that resolved auto itself via av_cpu_count(), and
     * before that one that ignored auto entirely and ran every ordinary command
     * line on a single core. */
    p->threads = avctx->thread_count;

    if (avctx->gop_size > 0)
        p->keyint = avctx->gop_size;
    if (avctx->max_b_frames >= 0)
        p->bframes = avctx->max_b_frames;
    /* refs, unlike gop_size and max_b_frames, has a POSITIVE default: an
     * untouched AVCodecContext reports 1, so "is it positive" cannot tell a
     * caller's choice from libavcodec's default, and the default won. Every
     * ffmpeg job ran yah264 at ref=1 no matter the preset, which cost about a
     * point of VMAF and made the encoder look faster than it is. Ask the option
     * machinery whether the value was actually assigned. One consequence is
     * that an explicit -refs 1 reads as untouched and leaves the preset alone.
     */
    if (av_opt_is_set_to_default_by_name(avctx, "refs", 0) == 0 && avctx->refs > 0)
        p->ref = avctx->refs;

    /* Rate control. An explicit bitrate selects ABR; otherwise CRF, which is
     * what -crf sets and what the encoder defaults to. */
    if (avctx->bit_rate > 0) {
        p->rc.method  = YAH264_RC_ABR;
        p->rc.bitrate = (int)(avctx->bit_rate / 1000);
    } else {
        p->rc.method = YAH264_RC_CRF;
        p->rc.rf     = ctx->crf;
    }
    if (avctx->rc_max_rate > 0)
        p->rc.vbv_maxrate = (int)(avctx->rc_max_rate / 1000);
    if (avctx->rc_buffer_size > 0)
        p->rc.vbv_bufsize = (int)(avctx->rc_buffer_size / 1000);

    if (ctx->subme >= 0)       p->subme       = ctx->subme;
    if (ctx->trellis_opt >= 0) p->trellis     = ctx->trellis_opt;
    if (ctx->aq_strength >= 0) p->aq_strength = ctx->aq_strength;
    if (ctx->psy_rd >= 0)      p->psy_rd      = ctx->psy_rd;

    if (avctx->sample_aspect_ratio.num > 0 && avctx->sample_aspect_ratio.den > 0) {
        p->sar_num = avctx->sample_aspect_ratio.num;
        p->sar_den = avctx->sample_aspect_ratio.den;
    }

    yah264_hdr_side_data(avctx, p);
    if ((ret = yah264_hdr_options(avctx, ctx, p)) < 0)
        return ret;

    ctx->enc = yah264_encoder_open(p);
    if (!ctx->enc) {
        av_log(avctx, AV_LOG_ERROR, "yah264_encoder_open failed\n");
        return AVERROR_EXTERNAL;
    }
    if ((ret = yah264_set_color(avctx, ctx->enc)) < 0)
        return ret;

    /* dts trails pts by the stream's reorder depth and no more: the pyramid's
     * depth, 1 for flat B, 0 without B. The encoder's input latency does not
     * belong here. Counting it put the first dts seven frames early at the
     * defaults, and a player that honours that offset stalls at the start.
     * A bitrate or VBV mode may emit anchors early, so those keep the wide
     * margin. */
    if (p->bframes <= 0) {
        ctx->delay_frames = 0;
    } else if (p->rc.method == YAH264_RC_ABR ||
               (p->rc.vbv_maxrate > 0 && p->rc.vbv_bufsize > 0)) {
        ctx->delay_frames = yah264_lookahead_delay(p) + p->bframes;
    } else if (p->b_pyramid && p->bframes >= 2) {
        int d = 0;
        while ((1 << d) < p->bframes + 1)
            d++;
        ctx->delay_frames = d;
    } else {
        ctx->delay_frames = 1;
    }

    if (yah264_encoder_headers(ctx->enc, &nal, &nnal) < 0)
        return AVERROR_EXTERNAL;

    /* A global header holds the parameter sets only: an MP4 or Matroska
     * muxer rebuilds it as a decoder configuration record, which has no
     * place for SEI. The encoder already writes its display metadata in-band
     * at every key; what SEI the headers still carry (the settings record, a
     * buffering period) rides in-band ahead of the first slice. Without a
     * global header everything goes in-band ahead of the first slice. */
    for (i = 0; i < nnal; i++) {
        int is_ps = nal[i].type == YAH264_NAL_SPS || nal[i].type == YAH264_NAL_PPS;

        if ((avctx->flags & AV_CODEC_FLAG_GLOBAL_HEADER) && is_ps) {
            uint8_t *x = av_realloc(avctx->extradata, avctx->extradata_size + nal[i].size +
                                    AV_INPUT_BUFFER_PADDING_SIZE);
            if (!x)
                return AVERROR(ENOMEM);
            memcpy(x + avctx->extradata_size, nal[i].payload, nal[i].size);
            avctx->extradata = x;
            avctx->extradata_size += nal[i].size;
            memset(x + avctx->extradata_size, 0, AV_INPUT_BUFFER_PADDING_SIZE);
        } else if ((ret = append_bytes(&ctx->hdr, &ctx->hdr_size, &nal[i])) < 0) {
            return ret;
        }
        if (is_ps && !(avctx->flags & AV_CODEC_FLAG_GLOBAL_HEADER) &&
            (ret = append_bytes(&ctx->rep_ps, &ctx->rep_ps_size, &nal[i])) < 0)
            return ret;
    }
    ctx->first_out = 1;

    ctx->pts_alloc = 64;
    ctx->pts_by_disp = av_malloc_array(ctx->pts_alloc, sizeof(*ctx->pts_by_disp));
    if (!ctx->pts_by_disp)
        return AVERROR(ENOMEM);
    ctx->frame = av_frame_alloc();
    if (!ctx->frame)
        return AVERROR(ENOMEM);
    /* Input frames are held one at a time. Buffering a deeper run was tried
     * and measured no difference at any depth from 1 to 256: a single encoder
     * instance parallelises on the row wavefront, which the thread budget
     * above drives, not on how many frames are queued ahead of it. */
    ctx->depth = 1;
    ctx->inq_alloc = 1;
    ctx->inq = av_calloc(ctx->inq_alloc, sizeof(*ctx->inq));
    if (!ctx->inq)
        return AVERROR(ENOMEM);
    /* dts starts far enough back that the reordered pts never precedes it. */
    ctx->dts_next = -ctx->delay_frames;

    av_log(avctx, AV_LOG_INFO,
           "yah264 %s cpu=%s preset=%s threads=%d subme=%d bframes=%d ref=%d "
           "rc=%d crf=%.1f keyint=%d\n",
           yah264_version(), yah264_cpu_features(),
           ctx->preset ? ctx->preset : "(none)",
           p->threads, p->subme, p->bframes, p->ref,
           p->rc.method, p->rc.rf, p->keyint);
    return 0;
}

/* Record a submitted frame's timestamp against its display index. */
static int pts_push(Yah264Context *ctx, int64_t pts)
{
    if (ctx->n_input == ctx->pts_alloc) {
        int n = ctx->pts_alloc * 2;
        int64_t *q = av_realloc_array(ctx->pts_by_disp, n, sizeof(*q));
        if (!q)
            return AVERROR(ENOMEM);
        ctx->pts_by_disp = q;
        ctx->pts_alloc = n;
    }
    ctx->pts_by_disp[ctx->n_input++] = pts;
    return 0;
}

/* Ask the encoder which frames the call just emitted, in coding order, and
 * label each stashed packet with that frame's timestamp. This is the step that
 * was missing: without it the wrapper handed out timestamps in sorted order and
 * the file's presentation timestamps ran backwards. */
static int burst_label(Yah264Context *ctx, int first, int nframes)
{
    int n, i;

    if (nframes <= 0)
        return 0;
    if (ctx->emit_alloc < nframes) {
        int *d = av_realloc_array(ctx->emit_disp, nframes, sizeof(*d));
        if (!d)
            return AVERROR(ENOMEM);
        ctx->emit_disp = d;
        ctx->emit_alloc = nframes;
    }
    /* Take exactly as many as we split. The encoder finalises frames and
     * appends their NALs on decoupled schedules, so asking per call would
     * mismatch; asking per packet cannot. */
    n = yah264_encoder_frame_order(ctx->enc, ctx->emit_disp, nframes);
    if (n != nframes) {
        av_log(ctx, AV_LOG_ERROR,
               "split %d packets but the encoder has %d frames to label\n",
               nframes, n);
        return AVERROR_EXTERNAL;
    }
    for (i = 0; i < nframes; i++) {
        int d = ctx->emit_disp[i];
        ctx->out[first + i].pts = (d >= 0 && d < ctx->n_input)
                                ? ctx->pts_by_disp[d] : AV_NOPTS_VALUE;
    }
    return 0;
}

/* The encoder returns NAL units in coding order and a single call can carry
 * several coded frames: an anchor plus the B frames that reference it. lavc
 * wants one packet per frame, so the burst is split on slice boundaries, with
 * any parameter-set or SEI units attached to the slice they precede.
 *
 * Each split frame is stashed here and handed out one call at a time. */
static int burst_stash(Yah264Context *ctx, yah264_nal_t *nal, int nnal)
{
    int i, start = 0;

    for (i = 0; i < nnal; i++) {
        int is_slice = nal[i].type == YAH264_NAL_SLICE ||
                       nal[i].type == YAH264_NAL_SLICE_IDR;
        if (!is_slice)
            continue;

        /* [start .. i] is one coded frame: leading non-slice units plus the
         * slice itself. */
        {
            size_t size = 0, ps = 0;
            int j, key = 0;
            uint8_t *buf;
            for (j = start; j <= i; j++) {
                size += nal[j].size;
                if (nal[j].type == YAH264_NAL_SLICE_IDR)
                    key = 1;
            }
            /* the first IDR takes the headers from burst_pop instead */
            if (key && !ctx->first_out)
                ps = ctx->rep_ps_size;
            if (key)
                ctx->first_out = 0;
            buf = av_malloc(size + ps);
            if (!buf)
                return AVERROR(ENOMEM);
            {
                uint8_t *d = buf;
                int done = !ps;
                for (j = start; j <= i; j++) {
                    /* after a delimiter, ahead of any SEI: a buffering
                     * period names the SPS it belongs to */
                    if (!done && nal[j].type != 9 /* AUD */) {
                        memcpy(d, ctx->rep_ps, ps);
                        d += ps;
                        done = 1;
                    }
                    memcpy(d, nal[j].payload, nal[j].size);
                    d += nal[j].size;
                }
            }
            if (ctx->out_count == ctx->out_alloc) {
                int n = ctx->out_alloc ? ctx->out_alloc * 2 : 16;
                Yah264Out *o = av_realloc_array(ctx->out, n, sizeof(*o));
                if (!o) { av_free(buf); return AVERROR(ENOMEM); }
                ctx->out = o;
                ctx->out_alloc = n;
            }
            ctx->out[ctx->out_count].data = buf;
            ctx->out[ctx->out_count].size = size + ps;
            ctx->out[ctx->out_count].key  = key;
            ctx->out_count++;
        }
        start = i + 1;
    }
    return 0;
}

static int burst_pop(AVCodecContext *avctx, AVPacket *pkt)
{
    Yah264Context *ctx = avctx->priv_data;
    Yah264Out *o;
    int ret;

    if (ctx->out_count <= 0)
        return AVERROR(EAGAIN);

    o = &ctx->out[0];
    ret = ff_get_encode_buffer(avctx, pkt, o->size + ctx->hdr_size, 0);
    if (ret < 0)
        return ret;
    if (ctx->hdr_size) {
        memcpy(pkt->data, ctx->hdr, ctx->hdr_size);
        memcpy(pkt->data + ctx->hdr_size, o->data, o->size);
        av_freep(&ctx->hdr);
        ctx->hdr_size = 0;
    } else {
        memcpy(pkt->data, o->data, o->size);
    }
    if (o->key)
        pkt->flags |= AV_PKT_FLAG_KEY;

    /* One packet per coded frame, and frames come out in coding order while
     * the queued pts are in display order. The earliest outstanding pts is
     * this packet's presentation time; dts trails it by the reorder depth so
     * dts <= pts holds for every packet. */
    pkt->pts = o->pts;
    pkt->dts = ctx->dts_next++;
    if (pkt->pts == AV_NOPTS_VALUE)
        pkt->dts = AV_NOPTS_VALUE;

    av_freep(&o->data);
    memmove(ctx->out, ctx->out + 1, (ctx->out_count - 1) * sizeof(*ctx->out));
    ctx->out_count--;
    return 0;
}

/* Submit one buffered frame and stash whatever comes back. */
static int submit_one(AVCodecContext *avctx, AVFrame *f)
{
    Yah264Context *ctx = avctx->priv_data;
    yah264_picture_t pic;
    yah264_nal_t *nal;
    int nnal = 0, i, ret;

    yah264_hdr_frame_check(avctx, ctx, f);
    memset(&pic, 0, sizeof(pic));
    pic.csp    = ctx->params.csp;
    pic.width  = avctx->width;
    pic.height = avctx->height;
    pic.pts    = f->pts;
    for (i = 0; i < 3; i++) {
        /* ABI 2 dropped the public `pixel` typedef: planes are void* and the
         * stride is in samples, so the byte width comes from the depth this
         * wrapper was compiled at. */
        pic.plane[i]  = f->data[i];
        pic.stride[i] = f->linesize[i] / (Y264_BIT_DEPTH > 8 ? 2 : 1);
    }
    ret = pts_push(ctx, f->pts);
    if (ret < 0)
        return ret;
    ret = yah264_encoder_encode(ctx->enc, &nal, &nnal, &pic);
    if (ret < 0)
        return AVERROR_EXTERNAL;
    if (nnal > 0)
        {
            int first = ctx->out_count, r = burst_stash(ctx, nal, nnal);
            if (r < 0)
                return r;
            return burst_label(ctx, first, ctx->out_count - first);
        }
    return 0;
}

static int drain_queue(AVCodecContext *avctx)
{
    Yah264Context *ctx = avctx->priv_data;
    int i, ret;

    for (i = 0; i < ctx->inq_count; i++) {
        ret = submit_one(avctx, ctx->inq[i]);
        av_frame_free(&ctx->inq[i]);
        if (ret < 0) {
            for (i++; i < ctx->inq_count; i++)
                av_frame_free(&ctx->inq[i]);
            ctx->inq_count = 0;
            return ret;
        }
    }
    ctx->inq_count = 0;
    return 0;
}

static int yah264_receive_packet(AVCodecContext *avctx, AVPacket *pkt)
{
    Yah264Context *ctx = avctx->priv_data;
    yah264_nal_t *nal;
    int nnal = 0, ret;

    ret = burst_pop(avctx, pkt);
    if (ret != AVERROR(EAGAIN))
        return ret;

    for (;;) {
        if (!ctx->eof) {
            ret = ff_encode_get_frame(avctx, ctx->frame);
            if (ret == AVERROR_EOF) {
                ctx->eof = 1;
            } else if (ret == AVERROR(EAGAIN)) {
                /* lavc has nothing more right now. Ask for another frame
                 * rather than forcing the encoder to work on a short queue. */
                return AVERROR(EAGAIN);
            } else if (ret < 0) {
                return ret;
            } else {
                AVFrame *held = av_frame_alloc();
                if (!held)
                    return AVERROR(ENOMEM);
                av_frame_move_ref(held, ctx->frame);
                if (ctx->inq_count == ctx->inq_alloc) {
                    int n = ctx->inq_alloc * 2;
                    AVFrame **q = av_realloc_array(ctx->inq, n, sizeof(*q));
                    if (!q) { av_frame_free(&held); return AVERROR(ENOMEM); }
                    ctx->inq = q;
                    ctx->inq_alloc = n;
                }
                ctx->inq[ctx->inq_count++] = held;

                /* Hold back until the run is deep enough to be worth starting. */
                if (ctx->inq_count < ctx->depth)
                    return AVERROR(EAGAIN);
            }
        }

        if (ctx->inq_count > 0) {
            ret = drain_queue(avctx);
            if (ret < 0)
                return ret;
            ret = burst_pop(avctx, pkt);
            if (ret != AVERROR(EAGAIN))
                return ret;
        }

        if (ctx->eof) {
            ret = yah264_encoder_encode(ctx->enc, &nal, &nnal, NULL);
            if (ret < 0)
                return AVERROR_EXTERNAL;
            if (nnal > 0) {
                int first = ctx->out_count;
                int r = burst_stash(ctx, nal, nnal);
                if (r >= 0)
                    r = burst_label(ctx, first, ctx->out_count - first);
                if (r < 0)
                    return r;
                return burst_pop(avctx, pkt);
            }
            if (ret <= 0 && nnal == 0)
                return AVERROR_EOF;
        }
    }
}

#define OFFSET(x) offsetof(Yah264Context, x)
#define VE AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_ENCODING_PARAM

static const AVOption options[] = {
    { "preset", "speed preset (ultrafast..placebo)", OFFSET(preset),
      AV_OPT_TYPE_STRING, { .str = "medium" }, 0, 0, VE },
    { "crf", "constant rate factor", OFFSET(crf),
      AV_OPT_TYPE_FLOAT, { .dbl = 23.0 }, 0, 51, VE },
    { "subme", "subpel/analysis level (-1 = preset default)", OFFSET(subme),
      AV_OPT_TYPE_INT, { .i64 = -1 }, -1, 11, VE },
    { "trellis", "RDOQ placement: 0 off, 1 final MB, 2 every decision", OFFSET(trellis_opt),
      AV_OPT_TYPE_INT, { .i64 = -1 }, -1, 2, VE },
    { "aq-strength", "variance AQ strength (-1 = default)", OFFSET(aq_strength),
      AV_OPT_TYPE_FLOAT, { .dbl = -1 }, -1, 4, VE },
    { "psy-rd", "psychovisual RD strength (-1 = default)", OFFSET(psy_rd),
      AV_OPT_TYPE_FLOAT, { .dbl = -1 }, -1, 4, VE },
    { "mastering-display", "mastering display SEI, G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min) "
      "(over side data)", OFFSET(mastering_display), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, VE },
    { "cll", "content light level SEI, MaxCLL,MaxFALL in cd/m^2 (over side data)",
      OFFSET(cll), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, VE },
    { NULL }
};

static const AVClass yah264_class = {
    .class_name = "libyah264",
    .item_name  = av_default_item_name,
    .option     = options,
    .version    = LIBAVUTIL_VERSION_INT,
};

static const FFCodecDefault yah264_defaults[] = {
    { "b",  "0"  },   /* no default bitrate: absent selects CRF */
    { "g",  "-1" },
    { "bf", "-1" },
    { NULL },
};

const FFCodec ff_libyah264_encoder = {
    .p.name         = "libyah264",
    CODEC_LONG_NAME("libyah264 H.264 / AVC"),
    .p.type         = AVMEDIA_TYPE_VIDEO,
    .p.id           = AV_CODEC_ID_H264,
    .p.capabilities = AV_CODEC_CAP_DR1 | AV_CODEC_CAP_DELAY |
                      AV_CODEC_CAP_OTHER_THREADS |
                      AV_CODEC_CAP_ENCODER_REORDERED_OPAQUE,
    .p.priv_class   = &yah264_class,
    .p.wrapper_name = "libyah264",
    .priv_data_size = sizeof(Yah264Context),
    .init           = yah264_init,
    FF_CODEC_RECEIVE_PACKET_CB(yah264_receive_packet),
    .close          = yah264_close,
    .defaults       = yah264_defaults,
    .caps_internal  = FF_CODEC_CAP_NOT_INIT_THREADSAFE | FF_CODEC_CAP_INIT_CLEANUP |
                      FF_CODEC_CAP_AUTO_THREADS,
#if Y264_BIT_DEPTH > 8
    CODEC_PIXFMTS(AV_PIX_FMT_YUV420P10, AV_PIX_FMT_YUV422P10,
                  AV_PIX_FMT_YUV444P10),
#else
    CODEC_PIXFMTS(AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV444P),
#endif
};

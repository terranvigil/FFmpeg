/*
 * next264 H.264 encoder wrapper
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

#include <next264.h>

#include "libavutil/opt.h"
#include "libavutil/mem.h"
#include "libavutil/pixdesc.h"
#include "avcodec.h"
#include "codec_internal.h"
#include "encode.h"
#include "internal.h"

typedef struct Next264Context {
    AVClass        *class;
    next264_param_t params;
    next264_encoder_t *enc;

    /* next264 hands back NAL units it owns, valid only until the next call in.
     * They are copied into the packet before returning. */
    char *preset;
    char *tune;
    float crf;
    int   subme;
    int   trellis_opt;
    float aq_strength;
    float psy_rd;

    /* Frames submitted but not yet emitted. The encoder reorders and buffers,
     * so a dts has to be reconstructed rather than taken from the frame that
     * happened to be submitted last. */
    int64_t *pts_queue;
    int      pts_alloc;
    int      pts_count;
    int64_t  dts_next;
    int      delay_frames;
    int      eof;

    AVFrame *frame;
    struct Next264Out { uint8_t *data; size_t size; int key; } *out;
    int      out_alloc;
    int      out_count;
} Next264Context;

typedef struct Next264Out Next264Out;

static av_cold int next264_close(AVCodecContext *avctx)
{
    Next264Context *ctx = avctx->priv_data;

    if (ctx->enc)
        next264_encoder_close(ctx->enc);
    ctx->enc = NULL;
    av_freep(&ctx->pts_queue);
    av_frame_free(&ctx->frame);
    while (ctx->out_count > 0)
        av_freep(&ctx->out[--ctx->out_count].data);
    av_freep(&ctx->out);
    return 0;
}

static int next264_csp_from_pixfmt(enum AVPixelFormat fmt)
{
    switch (fmt) {
    case AV_PIX_FMT_YUV420P: return NEXT264_CSP_I420;
    case AV_PIX_FMT_YUV422P: return NEXT264_CSP_I422;
    case AV_PIX_FMT_YUV444P: return NEXT264_CSP_I444;
    default:                 return -1;
    }
}

static av_cold int next264_init(AVCodecContext *avctx)
{
    Next264Context *ctx = avctx->priv_data;
    next264_param_t *p = &ctx->params;
    next264_nal_t *nal;
    int nnal, csp, i, size = 0;
    uint8_t *dst;

    csp = next264_csp_from_pixfmt(avctx->pix_fmt);
    if (csp < 0) {
        av_log(avctx, AV_LOG_ERROR, "unsupported pixel format %s\n",
               av_get_pix_fmt_name(avctx->pix_fmt));
        return AVERROR(EINVAL);
    }

    next264_param_default(p);

    if (ctx->preset && next264_param_apply_preset(p, ctx->preset) < 0) {
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

    /* The encoder threads itself; AV_CODEC_CAP_OTHER_THREADS tells lavc not to
     * wrap it, so avctx->thread_count is a request to pass straight through. */
    if (avctx->thread_count > 0)
        p->threads = avctx->thread_count;

    if (avctx->gop_size > 0)
        p->keyint = avctx->gop_size;
    if (avctx->max_b_frames >= 0)
        p->bframes = avctx->max_b_frames;
    if (avctx->refs > 0)
        p->ref = avctx->refs;

    /* Rate control. An explicit bitrate selects ABR; otherwise CRF, which is
     * what -crf sets and what the encoder defaults to. */
    if (avctx->bit_rate > 0) {
        p->rc.method  = NEXT264_RC_ABR;
        p->rc.bitrate = (int)(avctx->bit_rate / 1000);
    } else {
        p->rc.method = NEXT264_RC_CRF;
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

    ctx->enc = next264_encoder_open(p);
    if (!ctx->enc) {
        av_log(avctx, AV_LOG_ERROR, "next264_encoder_open failed\n");
        return AVERROR_EXTERNAL;
    }

    /* Input latency the encoder adds beyond B-frame reordering. Needed so the
     * first dts values are placed early enough that dts <= pts always holds. */
    ctx->delay_frames = next264_lookahead_delay(p) + p->bframes;

    if (avctx->flags & AV_CODEC_FLAG_GLOBAL_HEADER) {
        if (next264_encoder_headers(ctx->enc, &nal, &nnal) < 0)
            return AVERROR_EXTERNAL;
        for (i = 0; i < nnal; i++)
            size += nal[i].size;
        avctx->extradata = av_mallocz(size + AV_INPUT_BUFFER_PADDING_SIZE);
        if (!avctx->extradata)
            return AVERROR(ENOMEM);
        dst = avctx->extradata;
        for (i = 0; i < nnal; i++) {
            memcpy(dst, nal[i].payload, nal[i].size);
            dst += nal[i].size;
        }
        avctx->extradata_size = size;
    }

    ctx->pts_alloc = 64;
    ctx->pts_queue = av_malloc_array(ctx->pts_alloc, sizeof(*ctx->pts_queue));
    if (!ctx->pts_queue)
        return AVERROR(ENOMEM);
    ctx->frame = av_frame_alloc();
    if (!ctx->frame)
        return AVERROR(ENOMEM);
    /* dts starts far enough back that the reordered pts never precedes it. */
    ctx->dts_next = -ctx->delay_frames;

    av_log(avctx, AV_LOG_INFO,
           "next264 %s cpu=%s preset=%s threads=%d subme=%d bframes=%d ref=%d "
           "rc=%d crf=%.1f keyint=%d\n",
           next264_version(), next264_cpu_features(),
           ctx->preset ? ctx->preset : "(none)",
           p->threads, p->subme, p->bframes, p->ref,
           p->rc.method, p->rc.rf, p->keyint);
    return 0;
}

/* Record a submitted pts. Output is in coding order while these arrive in
 * display order, so the sorted queue is what makes a monotonic dts possible. */
static int pts_push(Next264Context *ctx, int64_t pts)
{
    int i;

    if (ctx->pts_count == ctx->pts_alloc) {
        int n = ctx->pts_alloc * 2;
        int64_t *q = av_realloc_array(ctx->pts_queue, n, sizeof(*q));
        if (!q)
            return AVERROR(ENOMEM);
        ctx->pts_queue = q;
        ctx->pts_alloc = n;
    }
    for (i = ctx->pts_count; i > 0 && ctx->pts_queue[i - 1] > pts; i--)
        ctx->pts_queue[i] = ctx->pts_queue[i - 1];
    ctx->pts_queue[i] = pts;
    ctx->pts_count++;
    return 0;
}

static int64_t pts_pop_smallest(Next264Context *ctx)
{
    int64_t v;

    if (ctx->pts_count <= 0)
        return AV_NOPTS_VALUE;
    v = ctx->pts_queue[0];
    memmove(ctx->pts_queue, ctx->pts_queue + 1,
            (ctx->pts_count - 1) * sizeof(*ctx->pts_queue));
    ctx->pts_count--;
    return v;
}

/* The encoder returns NAL units in coding order and a single call can carry
 * several coded frames: an anchor plus the B frames that reference it. lavc
 * wants one packet per frame, so the burst is split on slice boundaries, with
 * any parameter-set or SEI units attached to the slice they precede.
 *
 * Each split frame is stashed here and handed out one call at a time. */
static int burst_stash(Next264Context *ctx, next264_nal_t *nal, int nnal)
{
    int i, start = 0;

    for (i = 0; i < nnal; i++) {
        int is_slice = nal[i].type == NEXT264_NAL_SLICE ||
                       nal[i].type == NEXT264_NAL_SLICE_IDR;
        if (!is_slice)
            continue;

        /* [start .. i] is one coded frame: leading non-slice units plus the
         * slice itself. */
        {
            size_t size = 0;
            int j, key = 0;
            uint8_t *buf;
            for (j = start; j <= i; j++) {
                size += nal[j].size;
                if (nal[j].type == NEXT264_NAL_SLICE_IDR)
                    key = 1;
            }
            buf = av_malloc(size);
            if (!buf)
                return AVERROR(ENOMEM);
            {
                uint8_t *d = buf;
                for (j = start; j <= i; j++) {
                    memcpy(d, nal[j].payload, nal[j].size);
                    d += nal[j].size;
                }
            }
            if (ctx->out_count == ctx->out_alloc) {
                int n = ctx->out_alloc ? ctx->out_alloc * 2 : 16;
                Next264Out *o = av_realloc_array(ctx->out, n, sizeof(*o));
                if (!o) { av_free(buf); return AVERROR(ENOMEM); }
                ctx->out = o;
                ctx->out_alloc = n;
            }
            ctx->out[ctx->out_count].data = buf;
            ctx->out[ctx->out_count].size = size;
            ctx->out[ctx->out_count].key  = key;
            ctx->out_count++;
        }
        start = i + 1;
    }
    return 0;
}

static int burst_pop(AVCodecContext *avctx, AVPacket *pkt)
{
    Next264Context *ctx = avctx->priv_data;
    Next264Out *o;
    int ret;

    if (ctx->out_count <= 0)
        return AVERROR(EAGAIN);

    o = &ctx->out[0];
    ret = ff_get_encode_buffer(avctx, pkt, o->size, 0);
    if (ret < 0)
        return ret;
    memcpy(pkt->data, o->data, o->size);
    if (o->key)
        pkt->flags |= AV_PKT_FLAG_KEY;

    /* One packet per coded frame, and frames come out in coding order while
     * the queued pts are in display order. The earliest outstanding pts is
     * this packet's presentation time; dts trails it by the reorder depth so
     * dts <= pts holds for every packet. */
    pkt->pts = pts_pop_smallest(ctx);
    pkt->dts = ctx->dts_next++;
    if (pkt->pts == AV_NOPTS_VALUE)
        pkt->dts = AV_NOPTS_VALUE;

    av_freep(&o->data);
    memmove(ctx->out, ctx->out + 1, (ctx->out_count - 1) * sizeof(*ctx->out));
    ctx->out_count--;
    return 0;
}

static int next264_receive_packet(AVCodecContext *avctx, AVPacket *pkt)
{
    Next264Context *ctx = avctx->priv_data;
    next264_picture_t pic, *picp;
    next264_nal_t *nal;
    AVFrame *frame = ctx->frame;
    int nnal = 0, i, ret;

    /* Anything already split from a previous burst goes out first. */
    ret = burst_pop(avctx, pkt);
    if (ret != AVERROR(EAGAIN))
        return ret;

    for (;;) {
        picp = NULL;
        if (!ctx->eof) {
            ret = ff_encode_get_frame(avctx, frame);
            if (ret == AVERROR_EOF)
                ctx->eof = 1;
            else if (ret < 0)
                return ret;
        }

        if (!ctx->eof) {
            memset(&pic, 0, sizeof(pic));
            pic.csp    = ctx->params.csp;
            pic.width  = avctx->width;
            pic.height = avctx->height;
            pic.pts    = frame->pts;
            for (i = 0; i < 3; i++) {
                pic.plane[i]  = (pixel *)frame->data[i];
                pic.stride[i] = frame->linesize[i] / (int)sizeof(pixel);
            }
            picp = &pic;
            ret = pts_push(ctx, frame->pts);
            if (ret < 0) { av_frame_unref(frame); return ret; }
        }

        ret = next264_encoder_encode(ctx->enc, &nal, &nnal, picp);
        if (picp)
            av_frame_unref(frame);
        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "next264_encoder_encode failed\n");
            return AVERROR_EXTERNAL;
        }

        if (nnal > 0) {
            int r = burst_stash(ctx, nal, nnal);
            if (r < 0)
                return r;
            return burst_pop(avctx, pkt);
        }

        /* The flush contract is "call until it returns 0 bytes with 0 units",
         * so an empty flush call is not the end. Only a zero-byte, zero-unit
         * return after EOF finishes the stream. */
        if (ctx->eof && ret <= 0 && nnal == 0)
            return AVERROR_EOF;
    }
}

#define OFFSET(x) offsetof(Next264Context, x)
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
    { NULL }
};

static const AVClass next264_class = {
    .class_name = "libnext264",
    .item_name  = av_default_item_name,
    .option     = options,
    .version    = LIBAVUTIL_VERSION_INT,
};

static const FFCodecDefault next264_defaults[] = {
    { "b",  "0"  },   /* no default bitrate: absent selects CRF */
    { "g",  "-1" },
    { "bf", "-1" },
    { NULL },
};

const FFCodec ff_libnext264_encoder = {
    .p.name         = "libnext264",
    CODEC_LONG_NAME("libnext264 H.264 / AVC"),
    .p.type         = AVMEDIA_TYPE_VIDEO,
    .p.id           = AV_CODEC_ID_H264,
    .p.capabilities = AV_CODEC_CAP_DR1 | AV_CODEC_CAP_DELAY |
                      AV_CODEC_CAP_OTHER_THREADS |
                      AV_CODEC_CAP_ENCODER_REORDERED_OPAQUE,
    .p.priv_class   = &next264_class,
    .p.wrapper_name = "libnext264",
    .priv_data_size = sizeof(Next264Context),
    .init           = next264_init,
    FF_CODEC_RECEIVE_PACKET_CB(next264_receive_packet),
    .close          = next264_close,
    .defaults       = next264_defaults,
    .caps_internal  = FF_CODEC_CAP_NOT_INIT_THREADSAFE |
                      FF_CODEC_CAP_AUTO_THREADS,
    CODEC_PIXFMTS(AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV444P),
};

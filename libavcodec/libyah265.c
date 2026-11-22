/*
 * yah265 HEVC encoder wrapper
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

#include <stdint.h>
#include <yah265.h>

#include "libavutil/avstring.h"
#include "libavutil/frame.h"
#include "libavutil/mastering_display_metadata.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avcodec.h"
#include "codec_internal.h"
#include "encode.h"

/* The installed library carries both depths: the header names the 8-bit
 * entry points and the Main10 ones are the same calls with a _10 suffix.
 * The pixel format picks the table at init. */
void n265_param_default_10(n265_param *p);
n265_encoder *n265_encoder_open_10(const n265_param *p);
int n265_encoder_encode_10(n265_encoder *e, n265_nal **nal, int *nnal,
                           const n265_picture *pic);
void n265_encoder_close_10(n265_encoder *e);
const char *n265_version_str_10(void);
int n265_encoder_headers_10(n265_encoder *e, n265_nal **nal, int *nnal);
int n265_encoder_access_units_10(const n265_encoder *e, const n265_au **au);
int n265_encoder_reorder_depth_10(const n265_encoder *e);
n265_config *n265_config_new_10(void);
void n265_config_free_10(n265_config *c);
int n265_config_parse_10(n265_config *c, const char *name, const char *value);
int n265_config_parse_list_10(n265_config *c, const char *list);
int n265_config_apply_10(n265_config *c, n265_param *p);
int n265_config_get_10(const n265_config *c, const char *key, double *v);
const char *n265_config_get_str_10(const n265_config *c, const char *key);

typedef struct Yah265Api {
    int depth;
    void (*param_default)(n265_param *);
    n265_encoder *(*encoder_open)(const n265_param *);
    int  (*encoder_encode)(n265_encoder *, n265_nal **, int *, const n265_picture *);
    void (*encoder_close)(n265_encoder *);
    const char *(*version_str)(void);
    int  (*encoder_headers)(n265_encoder *, n265_nal **, int *);
    int  (*access_units)(const n265_encoder *, const n265_au **);
    int  (*reorder_depth)(const n265_encoder *);
    n265_config *(*config_new)(void);
    void (*config_free)(n265_config *);
    int  (*config_parse)(n265_config *, const char *, const char *);
    int  (*config_parse_list)(n265_config *, const char *);
    int  (*config_apply)(n265_config *, n265_param *);
    int  (*config_get)(const n265_config *, const char *, double *);
    const char *(*config_get_str)(const n265_config *, const char *);
} Yah265Api;

static const Yah265Api api8 = {
    8, n265_param_default, n265_encoder_open, n265_encoder_encode,
    n265_encoder_close, n265_version_str, n265_encoder_headers,
    n265_encoder_access_units, n265_encoder_reorder_depth,
    n265_config_new, n265_config_free, n265_config_parse,
    n265_config_parse_list, n265_config_apply, n265_config_get,
    n265_config_get_str,
};

static const Yah265Api api10 = {
    10, n265_param_default_10, n265_encoder_open_10, n265_encoder_encode_10,
    n265_encoder_close_10, n265_version_str_10, n265_encoder_headers_10,
    n265_encoder_access_units_10, n265_encoder_reorder_depth_10,
    n265_config_new_10, n265_config_free_10, n265_config_parse_10,
    n265_config_parse_list_10, n265_config_apply_10, n265_config_get_10,
    n265_config_get_str_10,
};

typedef struct Yah265Out {
    uint8_t *data;
    size_t   size;
    int      key;
    int64_t  pts;
} Yah265Out;

typedef struct Yah265Context {
    AVClass *class;
    const Yah265Api *api;
    n265_config  *cfg;
    n265_encoder *enc;

    char  *preset;
    char  *tune;
    double crf;
    int    qp;
    char  *params;

    /* input pts by display index: the encoder reports each access unit's
     * display index, and B pictures come out after the anchor that
     * follows them */
    int64_t *pts_by_disp;
    int      pts_alloc;
    int      n_input;
    int      reorder;
    int64_t  n_coded;

    AVFrame *frame;
    int      eof;

    /* HDR10 side data as init took it, in SEI units */
    int      md_set, cll_set;
    unsigned md_prim[6], md_wp[2], md_lum[2];
    unsigned cll[2];
    int      md_user, cll_user;
    int      hdr_on, hdr_warned;

    Yah265Out *out;
    int        out_alloc, out_count;
    /* NALs with no picture (--eos / --eob at the end of a flush), which
     * ride on the last packet */
    uint8_t   *tail;
    size_t     tail_size;
} Yah265Context;

static av_cold int yah265_close(AVCodecContext *avctx)
{
    Yah265Context *ctx = avctx->priv_data;

    if (ctx->enc)
        ctx->api->encoder_close(ctx->enc);
    ctx->enc = NULL;
    /* after the close: the encoder holds the config's strings */
    if (ctx->cfg)
        ctx->api->config_free(ctx->cfg);
    ctx->cfg = NULL;
    av_freep(&ctx->pts_by_disp);
    av_frame_free(&ctx->frame);
    while (ctx->out_count > 0)
        av_freep(&ctx->out[--ctx->out_count].data);
    av_freep(&ctx->out);
    av_freep(&ctx->tail);
    return 0;
}

static int yah265_csp(enum AVPixelFormat fmt, int *depth)
{
    switch (fmt) {
    case AV_PIX_FMT_YUV420P:   *depth = 8;  return N265_CSP_I420;
    case AV_PIX_FMT_YUV422P:   *depth = 8;  return N265_CSP_I422;
    case AV_PIX_FMT_YUV444P:   *depth = 8;  return N265_CSP_I444;
    case AV_PIX_FMT_YUV420P10: *depth = 10; return N265_CSP_I420;
    case AV_PIX_FMT_YUV422P10: *depth = 10; return N265_CSP_I422;
    case AV_PIX_FMT_YUV444P10: *depth = 10; return N265_CSP_I444;
    default:                   return -1;
    }
}

static int opt_set(AVCodecContext *avctx, const char *name, const char *val)
{
    Yah265Context *ctx = avctx->priv_data;
    int r = ctx->api->config_parse(ctx->cfg, name, val);

    if (r == 0)
        return 0;
    av_log(avctx, AV_LOG_ERROR, "yah265 refused %s%s%s\n", name,
           val ? "=" : "", val ? val : "");
    return AVERROR(EINVAL);
}

static int opt_setf(AVCodecContext *avctx, const char *name, double v)
{
    char buf[64];

    snprintf(buf, sizeof(buf), "%.17g", v);
    return opt_set(avctx, name, buf);
}

/* lavc's colour enums are the H.273 code points; its extended values have
 * none and are left unsignalled */
static int set_h273(AVCodecContext *avctx, const char *name, int v, int unset)
{
    char buf[16];

    if (v == unset)
        return 0;
    if (v < 0 || v > 255) {
        av_log(avctx, AV_LOG_WARNING, "%s %d has no H.273 code; not signalled\n",
               name, v);
        return 0;
    }
    snprintf(buf, sizeof(buf), "%d", v);
    return opt_set(avctx, name, buf);
}

static unsigned rescale_clip(AVRational q, int unit, int64_t max)
{
    int64_t v;

    if (q.den <= 0 || q.num < 0)
        return 0;
    v = av_rescale(q.num, unit, q.den);
    return v > max ? (unsigned)max : (unsigned)v;
}

/* The SEI needs the primaries and the luminance both, so side data with
 * only one of them is skipped rather than written with zeroes. */
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

/* whether the params list names key, so its value replaces the side data's */
static int params_name(const char *params, const char *key)
{
    size_t n = strlen(key);
    const char *q;

    for (const char *p = params; p; p = (q = strchr(p, ':')) ? q + 1 : NULL) {
        while (*p == '-')
            p++;
        if (!strncmp(p, key, n) && (p[n] == '=' || p[n] == ':' || !p[n]))
            return 1;
    }
    return 0;
}

/* HDR10 metadata from the side data on the codec context, which ffmpeg's
 * own tool copies there from the first decoded frame. Only a 10-bit PQ or
 * HLG stream takes it; the library would write it on any stream. */
static int yah265_hdr_side_data(AVCodecContext *avctx)
{
    Yah265Context *ctx = avctx->priv_data;
    const AVFrameSideData *md, *cll;
    char buf[160];
    int pq_hlg = avctx->color_trc == AVCOL_TRC_SMPTE2084 ||
                 avctx->color_trc == AVCOL_TRC_ARIB_STD_B67;
    int ret;

    md  = av_frame_side_data_get(avctx->decoded_side_data, avctx->nb_decoded_side_data,
                                 AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
    cll = av_frame_side_data_get(avctx->decoded_side_data, avctx->nb_decoded_side_data,
                                 AV_FRAME_DATA_CONTENT_LIGHT_LEVEL);
    if (!md && !cll)
        return 0;
    if (!pq_hlg || ctx->api->depth != 10) {
        av_log(avctx, pq_hlg ? AV_LOG_WARNING : AV_LOG_VERBOSE,
               "mastering display and light level side data not written: "
               "the stream is not 10-bit PQ or HLG\n");
        return 0;
    }
    ctx->hdr_on   = 1;
    ctx->md_user  = params_name(ctx->params, "master-display");
    ctx->cll_user = params_name(ctx->params, "max-cll");

    if (cll && !ctx->cll_user) {
        const AVContentLightMetadata *l = (const AVContentLightMetadata *)cll->data;
        ctx->cll[0] = FFMIN(l->MaxCLL, 65535);
        ctx->cll[1] = FFMIN(l->MaxFALL, 65535);
        ctx->cll_set = 1;
        snprintf(buf, sizeof(buf), "%u,%u", ctx->cll[0], ctx->cll[1]);
        if ((ret = opt_set(avctx, "max-cll", buf)) < 0)
            return ret;
    }
    if (md && !ctx->md_user &&
        mdcv_units(avctx, (const AVMasteringDisplayMetadata *)md->data,
                   ctx->md_prim, ctx->md_wp, ctx->md_lum, 1)) {
        const unsigned *g = ctx->md_prim;
        ctx->md_set = 1;
        snprintf(buf, sizeof(buf), "G(%u,%u)B(%u,%u)R(%u,%u)WP(%u,%u)L(%u,%u)",
                 g[0], g[1], g[2], g[3], g[4], g[5], ctx->md_wp[0], ctx->md_wp[1],
                 ctx->md_lum[0], ctx->md_lum[1]);
        if ((ret = opt_set(avctx, "master-display", buf)) < 0)
            return ret;
    }
    return 0;
}

/* Once per encode: the first frame whose side data differs from what init
 * took. Nothing can change these SEIs mid-stream, so it is only reported. */
static void yah265_hdr_frame_check(AVCodecContext *avctx, const AVFrame *f)
{
    Yah265Context *ctx = avctx->priv_data;
    const AVFrameSideData *sd;
    unsigned prim[6], wp[2], lum[2];
    int differs = 0;

    if (!ctx->hdr_on || ctx->hdr_warned)
        return;
    sd = ctx->cll_user ? NULL : av_frame_get_side_data(f, AV_FRAME_DATA_CONTENT_LIGHT_LEVEL);
    if (sd) {
        const AVContentLightMetadata *l = (const AVContentLightMetadata *)sd->data;
        differs |= !ctx->cll_set || FFMIN(l->MaxCLL, 65535) != ctx->cll[0] ||
                   FFMIN(l->MaxFALL, 65535) != ctx->cll[1];
    }
    sd = ctx->md_user ? NULL : av_frame_get_side_data(f, AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
    if (sd && mdcv_units(avctx, (const AVMasteringDisplayMetadata *)sd->data, prim, wp, lum, 0))
        differs |= !ctx->md_set || memcmp(prim, ctx->md_prim, sizeof(prim)) ||
                   memcmp(wp, ctx->md_wp, sizeof(wp)) ||
                   memcmp(lum, ctx->md_lum, sizeof(lum));
    if (differs) {
        av_log(avctx, AV_LOG_WARNING, "the frame at pts %"PRId64" carries mastering display or light "
               "level side data that differs from the stream's; the encoder writes what "
               "it took at init (put it on the codec context, or use -yah265-params)\n",
               f->pts);
        ctx->hdr_warned = 1;
    }
}

/* The options go through the library's own parser, in the order a command
 * line would name them, so `-c:v libyah265` and n265 given the same
 * options write the same stream, identification SEI included. */
static int yah265_configure(AVCodecContext *avctx, int csp)
{
    Yah265Context *ctx = avctx->priv_data;
    char buf[64];
    int ret;

    if (ctx->preset && (ret = opt_set(avctx, "preset", ctx->preset)) < 0)
        return ret;
    if (ctx->tune && (ret = opt_set(avctx, "tune", ctx->tune)) < 0)
        return ret;
    if (avctx->bit_rate > 0) {
        if ((ret = opt_setf(avctx, "bitrate", avctx->bit_rate / 1000.0)) < 0)
            return ret;
    } else if (ctx->crf >= 0) {
        if ((ret = opt_setf(avctx, "crf", ctx->crf)) < 0)
            return ret;
    } else if (ctx->qp >= 0) {
        snprintf(buf, sizeof(buf), "%d", ctx->qp);
        if ((ret = opt_set(avctx, "qp", buf)) < 0)
            return ret;
    }
    if (avctx->rc_max_rate > 0 &&
        (ret = opt_setf(avctx, "vbv-maxrate", avctx->rc_max_rate / 1000.0)) < 0)
        return ret;
    if (avctx->rc_buffer_size > 0 &&
        (ret = opt_setf(avctx, "vbv-bufsize", avctx->rc_buffer_size / 1000.0)) < 0)
        return ret;
    if (avctx->gop_size > 0) {
        snprintf(buf, sizeof(buf), "%d", avctx->gop_size);
        if ((ret = opt_set(avctx, "keyint", buf)) < 0)
            return ret;
    }
    if (avctx->max_b_frames >= 0) {
        snprintf(buf, sizeof(buf), "%d", avctx->max_b_frames);
        if ((ret = opt_set(avctx, "bframes", buf)) < 0)
            return ret;
    }
    if (avctx->refs > 0) {
        snprintf(buf, sizeof(buf), "%d", avctx->refs);
        if ((ret = opt_set(avctx, "ref", buf)) < 0)
            return ret;
    }
    /* 0 is "one per CPU" on both sides */
    snprintf(buf, sizeof(buf), "%d", avctx->thread_count);
    if ((ret = opt_set(avctx, "threads", buf)) < 0)
        return ret;

    /* limited range is what a decoder assumes untold, so only full is
     * signalled */
    if (avctx->color_range == AVCOL_RANGE_JPEG &&
        (ret = opt_set(avctx, "range", "full")) < 0)
        return ret;
    if ((ret = set_h273(avctx, "colorprim", avctx->color_primaries,
                        AVCOL_PRI_UNSPECIFIED)) < 0 ||
        (ret = set_h273(avctx, "transfer", avctx->color_trc,
                        AVCOL_TRC_UNSPECIFIED)) < 0 ||
        (ret = set_h273(avctx, "colormatrix", avctx->colorspace,
                        AVCOL_SPC_UNSPECIFIED)) < 0)
        return ret;
    /* lavc counts from 1 with 0 unspecified; type 0 (left) is what a
     * decoder assumes untold, and E.2.1 wants it absent off 4:2:0 */
    if (csp == N265_CSP_I420 &&
        avctx->chroma_sample_location > AVCHROMA_LOC_LEFT &&
        avctx->chroma_sample_location < AVCHROMA_LOC_NB) {
        snprintf(buf, sizeof(buf), "%d", avctx->chroma_sample_location - 1);
        if ((ret = opt_set(avctx, "chromaloc", buf)) < 0)
            return ret;
    }
    /* square pixels are what a player assumes untold */
    if (avctx->sample_aspect_ratio.num > 0 && avctx->sample_aspect_ratio.den > 0 &&
        avctx->sample_aspect_ratio.num != avctx->sample_aspect_ratio.den) {
        AVRational sar = avctx->sample_aspect_ratio;
        av_reduce(&sar.num, &sar.den, sar.num, sar.den, 65535);
        snprintf(buf, sizeof(buf), "%d:%d", sar.num, sar.den);
        if ((ret = opt_set(avctx, "sar", buf)) < 0)
            return ret;
    }

    if ((ret = yah265_hdr_side_data(avctx)) < 0)
        return ret;

    if (ctx->params && ctx->api->config_parse_list(ctx->cfg, ctx->params) != 0) {
        av_log(avctx, AV_LOG_ERROR, "yah265-params refused: %s\n", ctx->params);
        return AVERROR(EINVAL);
    }
    return 0;
}

static av_cold int yah265_init(AVCodecContext *avctx)
{
    Yah265Context *ctx = avctx->priv_data;
    n265_param p;
    n265_nal *nal;
    int nnal, depth, csp, ret;
    double v;

    csp = yah265_csp(avctx->pix_fmt, &depth);
    if (csp < 0) {
        av_log(avctx, AV_LOG_ERROR, "unsupported pixel format %s\n",
               av_get_pix_fmt_name(avctx->pix_fmt));
        return AVERROR(EINVAL);
    }
    ctx->api = depth == 10 ? &api10 : &api8;
    ctx->cfg = ctx->api->config_new();
    if (!ctx->cfg)
        return AVERROR(ENOMEM);
    if ((ret = yah265_configure(avctx, csp)) < 0)
        return ret;

    ctx->api->param_default(&p);
    p.width  = avctx->width;
    p.height = avctx->height;
    p.csp    = csp;
    if (avctx->framerate.num > 0 && avctx->framerate.den > 0) {
        p.fps_num = avctx->framerate.num;
        p.fps_den = avctx->framerate.den;
    } else if (avctx->time_base.num > 0 && avctx->time_base.den > 0) {
        p.fps_num = avctx->time_base.den;
        p.fps_den = avctx->time_base.num;
    }
    if (ctx->api->config_apply(ctx->cfg, &p) != 0) {
        av_log(avctx, AV_LOG_ERROR, "yah265 refused the configuration\n");
        return AVERROR(EINVAL);
    }
    /* the dynamic metadata files are the command line's to read */
    if (ctx->api->config_get_str(ctx->cfg, "dhdr10-info") ||
        ctx->api->config_get_str(ctx->cfg, "dolby-vision-rpu")) {
        av_log(avctx, AV_LOG_ERROR, "dhdr10-info and dolby-vision-rpu are "
               "n265 command-line options only\n");
        return AVERROR(EINVAL);
    }

    ctx->enc = ctx->api->encoder_open(&p);
    if (!ctx->enc) {
        av_log(avctx, AV_LOG_ERROR, "n265_encoder_open failed\n");
        return AVERROR_EXTERNAL;
    }
    ctx->reorder = ctx->api->reorder_depth(ctx->enc);
    if (ctx->reorder < 0)
        ctx->reorder = 0;
    avctx->has_b_frames = ctx->reorder;

    /* the access units carry the parameter sets in-band as well, so raw
     * Annex-B output decodes; a container takes them from extradata */
    if (avctx->flags & AV_CODEC_FLAG_GLOBAL_HEADER) {
        ret = ctx->api->encoder_headers(ctx->enc, &nal, &nnal);
        if (ret <= 0 || nnal <= 0)
            return AVERROR_EXTERNAL;
        avctx->extradata = av_mallocz(ret + AV_INPUT_BUFFER_PADDING_SIZE);
        if (!avctx->extradata)
            return AVERROR(ENOMEM);
        memcpy(avctx->extradata, nal[0].payload, ret);
        avctx->extradata_size = ret;
    }

    ctx->pts_alloc = 64;
    ctx->pts_by_disp = av_malloc_array(ctx->pts_alloc, sizeof(*ctx->pts_by_disp));
    if (!ctx->pts_by_disp)
        return AVERROR(ENOMEM);
    ctx->frame = av_frame_alloc();
    if (!ctx->frame)
        return AVERROR(ENOMEM);

    av_log(avctx, AV_LOG_VERBOSE, "yah265 %s %d-bit: %s\n",
           ctx->api->version_str(), depth,
           ctx->api->config_get_str(ctx->cfg, "info"));
    if (ctx->api->config_get(ctx->cfg, "bframes", &v) == 0)
        av_log(avctx, AV_LOG_VERBOSE, "bframes %d, reorder depth %d\n",
               (int)v, ctx->reorder);
    return 0;
}

static int pts_push(Yah265Context *ctx, int64_t pts)
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

static int is_irap(int type)
{
    return type >= 16 && type <= 23;
}

/* One packet per access unit; the library names each one's NALs and its
 * picture's display index. */
static int stash_output(AVCodecContext *avctx, n265_nal *nal, int nnal)
{
    Yah265Context *ctx = avctx->priv_data;
    const n265_au *au;
    int nau = ctx->api->access_units(ctx->enc, &au);

    if (nnal <= 0)
        return 0;
    if (nau < 0)
        return AVERROR_EXTERNAL;
    for (int i = 0; i < nau; i++) {
        const n265_nal *first = &nal[au[i].first_nal];
        size_t size = 0;
        int key = 0;
        uint8_t *buf;

        for (int j = 0; j < au[i].nnal; j++) {
            size += first[j].size;
            key |= is_irap(first[j].type);
        }
        if (au[i].display_idx < 0) {
            Yah265Out *last = ctx->out_count ? &ctx->out[ctx->out_count - 1] : NULL;
            uint8_t **dst = last ? &last->data : &ctx->tail;
            size_t *dsz = last ? &last->size : &ctx->tail_size;
            uint8_t *t = av_realloc(*dst, *dsz + size);
            if (!t)
                return AVERROR(ENOMEM);
            memcpy(t + *dsz, first[0].payload, size);
            *dst = t;
            *dsz += size;
            continue;
        }
        buf = av_malloc(ctx->tail_size + size);
        if (!buf)
            return AVERROR(ENOMEM);
        if (ctx->tail_size)
            memcpy(buf, ctx->tail, ctx->tail_size);
        /* an access unit's NALs are contiguous in the library's buffer */
        memcpy(buf + ctx->tail_size, first[0].payload, size);
        size += ctx->tail_size;
        av_freep(&ctx->tail);
        ctx->tail_size = 0;
        if (ctx->out_count == ctx->out_alloc) {
            int n = ctx->out_alloc ? ctx->out_alloc * 2 : 16;
            Yah265Out *o = av_realloc_array(ctx->out, n, sizeof(*o));
            if (!o) {
                av_free(buf);
                return AVERROR(ENOMEM);
            }
            ctx->out = o;
            ctx->out_alloc = n;
        }
        ctx->out[ctx->out_count].data = buf;
        ctx->out[ctx->out_count].size = size;
        ctx->out[ctx->out_count].key  = key;
        ctx->out[ctx->out_count].pts  = au[i].display_idx < ctx->n_input
                                      ? ctx->pts_by_disp[au[i].display_idx]
                                      : AV_NOPTS_VALUE;
        ctx->out_count++;
    }
    return 0;
}

/* dts trails by the stream's reorder depth: the k-th coded picture's dts
 * is the pts of display index k - reorder, extrapolated before the first */
static int64_t dts_for(Yah265Context *ctx, int64_t k)
{
    int64_t d = k - ctx->reorder, dur;

    if (d >= 0 && d < ctx->n_input)
        return ctx->pts_by_disp[d];
    dur = ctx->n_input > 1 ? ctx->pts_by_disp[1] - ctx->pts_by_disp[0] : 1;
    if (dur <= 0)
        dur = 1;
    return ctx->pts_by_disp[0] + d * dur;
}

static int pop_packet(AVCodecContext *avctx, AVPacket *pkt)
{
    Yah265Context *ctx = avctx->priv_data;
    Yah265Out *o;
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
    pkt->pts = o->pts;
    pkt->dts = o->pts == AV_NOPTS_VALUE ? AV_NOPTS_VALUE
                                        : dts_for(ctx, ctx->n_coded);
    ctx->n_coded++;
    av_freep(&o->data);
    memmove(ctx->out, ctx->out + 1, (ctx->out_count - 1) * sizeof(*ctx->out));
    ctx->out_count--;
    return 0;
}

static int submit(AVCodecContext *avctx, const AVFrame *f)
{
    Yah265Context *ctx = avctx->priv_data;
    const int bps = ctx->api->depth > 8 ? 2 : 1;
    n265_picture pic = { 0 };
    n265_nal *nal;
    int nnal = 0, ret;

    for (int i = 0; i < 3; i++) {
        pic.plane[i]  = f->data[i];
        pic.stride[i] = f->linesize[i] / bps;
    }
    pic.pts = f->pts;
    yah265_hdr_frame_check(avctx, f);
    if ((ret = pts_push(ctx, f->pts)) < 0)
        return ret;
    if (ctx->api->encoder_encode(ctx->enc, &nal, &nnal, &pic) < 0)
        return AVERROR_EXTERNAL;
    return stash_output(avctx, nal, nnal);
}

/* the whole tail at once, so a NAL-only access unit at the end has a
 * packet left to ride on */
static int flush_all(AVCodecContext *avctx)
{
    Yah265Context *ctx = avctx->priv_data;
    n265_nal *nal;
    int nnal, r, ret;

    for (;;) {
        nnal = 0;
        r = ctx->api->encoder_encode(ctx->enc, &nal, &nnal, NULL);
        if (r < 0)
            return AVERROR_EXTERNAL;
        if (nnal == 0)
            break;
        if ((ret = stash_output(avctx, nal, nnal)) < 0)
            return ret;
    }
    if (ctx->tail_size) {
        av_log(avctx, AV_LOG_WARNING, "%zu trailing bytes had no packet to "
               "ride on and were dropped\n", ctx->tail_size);
        av_freep(&ctx->tail);
        ctx->tail_size = 0;
    }
    return 0;
}

static int yah265_receive_packet(AVCodecContext *avctx, AVPacket *pkt)
{
    Yah265Context *ctx = avctx->priv_data;
    int ret;

    for (;;) {
        ret = pop_packet(avctx, pkt);
        if (ret != AVERROR(EAGAIN))
            return ret;
        if (ctx->eof)
            return AVERROR_EOF;
        ret = ff_encode_get_frame(avctx, ctx->frame);
        if (ret == AVERROR_EOF) {
            ctx->eof = 1;
            if ((ret = flush_all(avctx)) < 0)
                return ret;
            continue;
        }
        if (ret < 0)
            return ret;
        ret = submit(avctx, ctx->frame);
        av_frame_unref(ctx->frame);
        if (ret < 0)
            return ret;
    }
}

#define OFFSET(x) offsetof(Yah265Context, x)
#define VE AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_ENCODING_PARAM

static const AVOption options[] = {
    { "preset", "speed preset (fast, medium, slow, slower)", OFFSET(preset),
      AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, VE },
    { "tune", "tune (zerolatency)", OFFSET(tune),
      AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, VE },
    { "crf", "constant rate factor; unset with no bitrate is n265's default (CQP 26)",
      OFFSET(crf), AV_OPT_TYPE_DOUBLE, { .dbl = -1 }, -1, 51, VE },
    { "qp", "constant QP", OFFSET(qp),
      AV_OPT_TYPE_INT, { .i64 = -1 }, -1, 51, VE },
    { "yah265-params", "n265 options, key=value:key=value (the CLI's names "
      "without the dashes)", OFFSET(params),
      AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, VE },
    { NULL }
};

static const AVClass yah265_class = {
    .class_name = "libyah265",
    .item_name  = av_default_item_name,
    .option     = options,
    .version    = LIBAVUTIL_VERSION_INT,
};

/* every generic option the library has its own default for is left
 * unset, so the preset decides */
static const FFCodecDefault yah265_defaults[] = {
    { "b",    "0"  },
    { "g",    "-1" },
    { "bf",   "-1" },
    { "refs", "-1" },
    { NULL },
};

const FFCodec ff_libyah265_encoder = {
    .p.name         = "libyah265",
    CODEC_LONG_NAME("libyah265 H.265 / HEVC"),
    .p.type         = AVMEDIA_TYPE_VIDEO,
    .p.id           = AV_CODEC_ID_HEVC,
    .p.capabilities = AV_CODEC_CAP_DR1 | AV_CODEC_CAP_DELAY |
                      AV_CODEC_CAP_OTHER_THREADS,
    .p.priv_class   = &yah265_class,
    .p.wrapper_name = "libyah265",
    .priv_data_size = sizeof(Yah265Context),
    .init           = yah265_init,
    FF_CODEC_RECEIVE_PACKET_CB(yah265_receive_packet),
    .close          = yah265_close,
    .defaults       = yah265_defaults,
    /* the library's configuration is process-global until open */
    .caps_internal  = FF_CODEC_CAP_NOT_INIT_THREADSAFE | FF_CODEC_CAP_INIT_CLEANUP |
                      FF_CODEC_CAP_AUTO_THREADS,
    CODEC_PIXFMTS(AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV444P,
                  AV_PIX_FMT_YUV420P10, AV_PIX_FMT_YUV422P10,
                  AV_PIX_FMT_YUV444P10),
};

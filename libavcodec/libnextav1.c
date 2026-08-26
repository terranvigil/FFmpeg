/*
 * libnextav1 AV1 encoder
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

#include <string.h>

#include <nextav1.h>

#include "libavutil/avutil.h"
#include "libavutil/frame.h"
#include "libavutil/log.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavutil/pixfmt.h"
#include "libavutil/rational.h"

#include "avcodec.h"
#include "codec_internal.h"
#include "encode.h"
#include "packet.h"

typedef struct NextAV1Context {
    const AVClass *class;

    nextav1_encoder_t *enc;
    AVFrame *frame;             /* input handed over by ff_encode_get_frame() */
    int csp;                    /* nextav1_csp_t picked from avctx->pix_fmt */

    int qindex;
    char *preset;
} NextAV1Context;

static av_cold int nextav1_close(AVCodecContext *avctx)
{
    NextAV1Context *ctx = avctx->priv_data;

    if (ctx->enc) {
        nextav1_encoder_close(ctx->enc);
        ctx->enc = NULL;
    }
    av_frame_free(&ctx->frame);

    return 0;
}

static av_cold int nextav1_init(AVCodecContext *avctx)
{
    NextAV1Context *ctx = avctx->priv_data;
    nextav1_param_t param;
    AVRational fps;
    int ret;

    switch (avctx->pix_fmt) {
    case AV_PIX_FMT_YUV420P:
        ctx->csp = NEXTAV1_CSP_I420;
        break;
    case AV_PIX_FMT_GRAY8:
        ctx->csp = NEXTAV1_CSP_I400;
        break;
    default:
        av_log(avctx, AV_LOG_ERROR, "Unsupported pixel format %s\n",
               av_get_pix_fmt_name(avctx->pix_fmt));
        return AVERROR(EINVAL);
    }

    nextav1_param_default(&param);

    param.width  = avctx->width;
    param.height = avctx->height;
    param.csp    = ctx->csp;

    /* The library wants a frame rate, not a time base. Prefer the frame rate
     * the caller states; fall back to inverting the time base, which is what
     * a caller that only sets the latter means by it. */
    fps = avctx->framerate;
    if (fps.num <= 0 || fps.den <= 0)
        fps = av_inv_q(avctx->time_base);
    if (fps.num > 0 && fps.den > 0) {
        param.timebase.fps_num = fps.num;
        param.timebase.fps_den = fps.den;
    }

    if (avctx->gop_size > 0)
        param.keyint = avctx->gop_size;

    param.rc.qindex = ctx->qindex;

    if (ctx->preset && nextav1_param_apply_preset(&param, ctx->preset) < 0) {
        av_log(avctx, AV_LOG_ERROR, "Unknown preset '%s'\n", ctx->preset);
        return AVERROR(EINVAL);
    }

    ctx->enc = nextav1_encoder_open(&param);
    if (!ctx->enc) {
        av_log(avctx, AV_LOG_ERROR, "Could not open the nextav1 encoder\n");
        return AVERROR_EXTERNAL;
    }

    ctx->frame = av_frame_alloc();
    if (!ctx->frame)
        return AVERROR(ENOMEM);

    if (avctx->flags & AV_CODEC_FLAG_GLOBAL_HEADER) {
        const uint8_t *hdr;
        size_t hdr_size;

        ret = nextav1_encoder_headers(ctx->enc, &hdr, &hdr_size);
        if (ret < 0 || !hdr_size) {
            av_log(avctx, AV_LOG_ERROR, "Could not retrieve the sequence header\n");
            return AVERROR_EXTERNAL;
        }

        /* This is the bare sequence header OBU, not an
         * AV1CodecConfigurationRecord ("av1C"). FFmpeg's own AV1-in-ISOBMFF
         * writer takes either - it detects an OBU stream by the cleared marker
         * bit of the first byte and derives the record itself - so muxing to
         * MP4 through libavformat is correct as it stands. A caller that
         * copies extradata straight into a container as if it were already a
         * record would get a malformed av1C; building one here needs the
         * profile, level, tier and chroma fields parsed back out of the
         * sequence header, which is left for when the encoder signals more
         * than one configuration. */
        avctx->extradata = av_mallocz(hdr_size + AV_INPUT_BUFFER_PADDING_SIZE);
        if (!avctx->extradata)
            return AVERROR(ENOMEM);
        memcpy(avctx->extradata, hdr, hdr_size);
        avctx->extradata_size = hdr_size;
    }

    return 0;
}

static int nextav1_receive_packet(AVCodecContext *avctx, AVPacket *pkt)
{
    NextAV1Context *ctx = avctx->priv_data;
    AVFrame *frame = ctx->frame;
    nextav1_picture_t pic;
    nextav1_packet_t out;
    int ret, got, draining;

    ret = ff_encode_get_frame(avctx, frame);
    if (ret < 0 && ret != AVERROR_EOF)
        return ret;
    draining = ret == AVERROR_EOF;

    if (!draining) {
        memset(&pic, 0, sizeof(pic));
        /* csp, width and height are part of the contract, not just the
         * planes: the library validates the geometry against the one the
         * encoder was opened with. */
        pic.csp    = ctx->csp;
        pic.width  = frame->width;
        pic.height = frame->height;
        pic.pts    = frame->pts;
        for (int i = 0; i < (ctx->csp == NEXTAV1_CSP_I400 ? 1 : 3); i++) {
            pic.plane[i]  = frame->data[i];
            pic.stride[i] = frame->linesize[i];
        }
    }

    got = nextav1_encoder_encode(ctx->enc, &out, draining ? NULL : &pic);
    av_frame_unref(frame);
    if (got < 0) {
        av_log(avctx, AV_LOG_ERROR, "Error while encoding\n");
        return AVERROR_EXTERNAL;
    }

    /* A NULL picture asks the library to drain. Today it has no lookahead and
     * so never holds a frame back, which makes this loop terminate on the
     * first pass; it starts doing real work once the reference pyramid gives
     * the encoder something to buffer. */
    if (!got)
        return draining ? AVERROR_EOF : AVERROR(EAGAIN);

    ret = ff_get_encode_buffer(avctx, pkt, out.size, 0);
    if (ret < 0)
        return ret;
    memcpy(pkt->data, out.payload, out.size);

    pkt->pts = out.pts;
    /* No reordering: every temporal unit is a shown key frame, so composition
     * and decode order coincide. This has to become a real reordering delay
     * when the reference pyramid lands. */
    pkt->dts = pkt->pts;
    if (out.keyframe)
        pkt->flags |= AV_PKT_FLAG_KEY;

    return 0;
}

#define OFFSET(x) offsetof(NextAV1Context, x)
#define VE (AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_ENCODING_PARAM)

static const AVOption nextav1_options[] = {
    { "crf", "Constant quantizer index (1..255). Rate control is CQP only for "
             "now, so this is the AV1 qindex held for the whole stream and not "
             "a quality ladder; nothing targets a bit rate.",
      OFFSET(qindex), AV_OPT_TYPE_INT, { .i64 = 128 }, 1, 255, VE },
    { "preset", "Speed preset (slowest..fastest, or 0..10). Accepted and "
                "validated, but the speed ladder is not implemented yet.",
      OFFSET(preset), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, VE },
    { NULL },
};

static const AVClass nextav1_class = {
    .class_name = "libnextav1",
    .item_name  = av_default_item_name,
    .option     = nextav1_options,
    .version    = LIBAVUTIL_VERSION_INT,
};

static const enum AVPixelFormat nextav1_pix_fmts[] = {
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_GRAY8,
    AV_PIX_FMT_NONE,
};

const FFCodec ff_libnextav1_encoder = {
    .p.name         = "libnextav1",
    CODEC_LONG_NAME("nextav1 AV1"),
    .p.type         = AVMEDIA_TYPE_VIDEO,
    .p.id           = AV_CODEC_ID_AV1,
    /* No frame threading: the library encodes on the calling thread and has
     * no internal threading to declare either. */
    .p.capabilities = AV_CODEC_CAP_DELAY,
    .p.pix_fmts     = nextav1_pix_fmts,
    .p.priv_class   = &nextav1_class,
    .p.wrapper_name = "libnextav1",
    .priv_data_size = sizeof(NextAV1Context),
    .init           = nextav1_init,
    FF_CODEC_RECEIVE_PACKET_CB(nextav1_receive_packet),
    .close          = nextav1_close,
    .caps_internal  = FF_CODEC_CAP_INIT_CLEANUP,
};

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <x264.h>

// Structure to hold our command line configuration
typedef struct {
    const char *input_video;
    const char *output_mp4;
    const char *qpoffset_dir;
    double crf;
    const char *preset;
    const char *tune;
} EncoderConfig;

// Print usage instructions
static void print_usage(const char *program_name) {
    fprintf(stderr, "Usage: %s <input_video> <output.mp4> <qpoffset_dir> [options]\n", program_name);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --crf <value>       CRF value (default: 23.0)\n");
    fprintf(stderr, "  --preset <name>     x264 preset (default: \"medium\")\n");
    fprintf(stderr, "  --tune <name>       x264 tune (default: none)\n");
}

// Parse command line arguments
static int parse_arguments(int argc, char **argv, EncoderConfig *cfg) {
    if (argc < 4) {
        return -1;
    }
    cfg->input_video = argv[1];
    cfg->output_mp4 = argv[2];
    cfg->qpoffset_dir = argv[3];
    cfg->crf = 23.0;
    cfg->preset = "medium";
    cfg->tune = NULL;

    for (int i = 4; i < argc; i++) {                // Reads the non mandatory arguments
        if (strcmp(argv[i], "--crf") == 0 && i + 1 < argc) {
            cfg->crf = atof(argv[++i]);
        } else if (strcmp(argv[i], "--preset") == 0 && i + 1 < argc) {
            cfg->preset = argv[++i];
        } else if (strcmp(argv[i], "--tune") == 0 && i + 1 < argc) {
            cfg->tune = argv[++i];
        } else {
            fprintf(stderr, "Unknown option or missing value: %s\n", argv[i]);
            return -1;
        }
    }
    return 0;
}

typedef struct {
    const EncoderConfig *config;
    AVCodecContext *decoder;
    struct SwsContext *scaler;
    x264_t *encoder;
    AVFormatContext *output;
    AVStream *stream;
    AVRational time_base;
    int num_mbs;
    int frames_read;
    int frames_written;
} EncodingState;

// Used for both ordinary input frames and the x264 end-of-stream drain.
static int encode_picture(EncodingState *state, x264_picture_t *input) {
    x264_picture_t output;
    x264_nal_t *nals;
    int num_nals;
    int size = x264_encoder_encode(state->encoder, &nals, &num_nals, input, &output);
    if (size < 0) return AVERROR_EXTERNAL;
    if (size == 0) return 0;

    AVPacket *packet = av_packet_alloc();
    if (!packet) return AVERROR(ENOMEM);
    int ret = av_new_packet(packet, size);
    if (ret >= 0) {
        memcpy(packet->data, nals[0].p_payload, size);
        packet->pts = output.i_pts;
        packet->dts = output.i_dts;
        // Every encoded frame spans one frame-rate tick. A zero duration
        // makes the MP4 duration/rate wrong, especially with reordered frames.
        packet->duration = 1;
        packet->stream_index = state->stream->index;
        if (output.b_keyframe) packet->flags |= AV_PKT_FLAG_KEY;
        av_packet_rescale_ts(packet, state->time_base, state->stream->time_base);
        ret = av_interleaved_write_frame(state->output, packet);
        if (ret >= 0) state->frames_written++;
    }
    av_packet_free(&packet);
    return ret;
}

static int process_frame(EncodingState *state, AVFrame *frame) {
    x264_picture_t picture;
    x264_picture_init(&picture);
    if (x264_picture_alloc(&picture, X264_CSP_I420,
                           state->decoder->width, state->decoder->height) < 0)
        return AVERROR(ENOMEM);
    int ret = sws_scale(state->scaler, (const uint8_t * const *)frame->data,
                        frame->linesize, 0, state->decoder->height,
                        picture.img.plane, picture.img.i_stride);
    if (ret != state->decoder->height) {
        x264_picture_clean(&picture);
        return ret < 0 ? ret : AVERROR_EXTERNAL;
    }
    picture.i_pts = state->frames_read;
    float *offsets = calloc(state->num_mbs, sizeof(float));
    if (!offsets) {
        x264_picture_clean(&picture);
        return AVERROR(ENOMEM);
    }
    char path[4096];
    int length = snprintf(path, sizeof(path), "%s/qpoffset_%06d.bin",
                          state->config->qpoffset_dir, state->frames_read);
    if (length < 0 || length >= (int)sizeof(path)) {
        free(offsets);
        x264_picture_clean(&picture);
        return AVERROR(ENAMETOOLONG);
    }
    FILE *file = fopen(path, "rb");
    if (file) {
        size_t count = fread(offsets, sizeof(float), state->num_mbs, file);
        fclose(file);
        if (count != (size_t)state->num_mbs) {
            fprintf(stderr, "Warning: Incomplete QP offsets in %s. Using zeros.\n", path);
            memset(offsets, 0, state->num_mbs * sizeof(float));
        }
    } else if (state->frames_read < 10) {
        fprintf(stderr, "Warning: Could not open %s. Using zero QP offsets.\n", path);
    } else if (state->frames_read == 10) {
        fprintf(stderr, "Warning: Silencing further missing QP offset warnings.\n");
    }
    picture.prop.quant_offsets = offsets;
    picture.prop.quant_offsets_free = free;
    ret = encode_picture(state, &picture);
    x264_picture_clean(&picture);
    if (ret >= 0) {
        state->frames_read++;
        if (state->frames_read % 100 == 0)
            printf("Processed %d frames...\n", state->frames_read);
    }
    return ret;
}

// Drain decoded frames through exactly the same processing/QP path at EOF.
static int receive_frames(EncodingState *state, AVFrame *frame, int draining) {
    for (;;) {
        int ret = avcodec_receive_frame(state->decoder, frame);
        if (ret == AVERROR_EOF) return 0;
        if (ret == AVERROR(EAGAIN)) return draining ? ret : 0;
        if (ret < 0) return ret;
        ret = process_frame(state, frame);
        av_frame_unref(frame);
        if (ret < 0) return ret;
    }
}

int main(int argc, char **argv) {
    EncoderConfig cfg;
    if (parse_arguments(argc, argv, &cfg) < 0) {
        print_usage(argv[0]);
        return 1;
    }

    AVFormatContext *in_fmt_ctx = NULL;
    AVCodecContext *dec_ctx = NULL;
    int video_stream_idx = -1;
    int audio_stream_idx = -1;

    x264_param_t x264_param;
    x264_t *x264_encoder = NULL;

    AVFormatContext *out_fmt_ctx = NULL;
    AVStream *out_video_stream = NULL;
    AVStream *out_audio_stream = NULL;

    struct SwsContext *sws_ctx = NULL;

    int ret = 0;
    int exit_code = 1;
    AVPacket *in_pkt = NULL;
    AVFrame *dec_frame = NULL;

    // 1. Open input video
    if ((ret = avformat_open_input(&in_fmt_ctx, cfg.input_video, NULL, NULL)) < 0) {
        fprintf(stderr, "Error: Could not open input file %s (FFmpeg error: %d)\n", cfg.input_video, ret);
        return 1;
    }

    if ((ret = avformat_find_stream_info(in_fmt_ctx, NULL)) < 0) {
        fprintf(stderr, "Error: Could not find stream info (FFmpeg error: %d)\n", ret);
        avformat_close_input(&in_fmt_ctx);
        return 1;
    }

    // Find video and audio streams
    for (unsigned int i = 0; i < in_fmt_ctx->nb_streams; i++) {
        if (in_fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && video_stream_idx < 0) {
            video_stream_idx = i;
        } else if (in_fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO && audio_stream_idx < 0) {
            audio_stream_idx = i;
        }
    }

    if (video_stream_idx < 0) {
        fprintf(stderr, "Error: Could not find video stream in input file.\n");
        avformat_close_input(&in_fmt_ctx);
        return 1;
    }

    AVCodecParameters *in_video_par = in_fmt_ctx->streams[video_stream_idx]->codecpar;
    const AVCodec *decoder = avcodec_find_decoder(in_video_par->codec_id);
    if (!decoder) {
        fprintf(stderr, "Error: Video decoder not found.\n");
        avformat_close_input(&in_fmt_ctx);
        return 1;
    }

    dec_ctx = avcodec_alloc_context3(decoder);
    if (!dec_ctx) {
        fprintf(stderr, "Error: Could not allocate decoder context.\n");
        avformat_close_input(&in_fmt_ctx);
        return 1;
    }

    if ((ret = avcodec_parameters_to_context(dec_ctx, in_video_par)) < 0) {
        fprintf(stderr, "Error: Failed to copy decoder parameters (FFmpeg error: %d).\n", ret);
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&in_fmt_ctx);
        return 1;
    }

    if ((ret = avcodec_open2(dec_ctx, decoder, NULL)) < 0) {
        fprintf(stderr, "Error: Failed to open decoder (FFmpeg error: %d).\n", ret);
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&in_fmt_ctx);
        return 1;
    }

    // Guess frame rate from input video stream
    AVRational fps = av_guess_frame_rate(in_fmt_ctx, in_fmt_ctx->streams[video_stream_idx], NULL);
    if (fps.num == 0) {
        fprintf(stderr, "Warning: Could not determine frame rate, defaulting to 30fps.\n");
        fps = (AVRational){30, 1};
    }

    // 2. Initialize x264 encoder parameters
    if (x264_param_default_preset(&x264_param, cfg.preset, cfg.tune) < 0) {
        fprintf(stderr, "Error: Failed to set x264 preset/tune.\n");
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&in_fmt_ctx);
        return 1;
    }

    x264_param.i_width = dec_ctx->width;
    x264_param.i_height = dec_ctx->height;
    x264_param.i_fps_num = fps.num;
    x264_param.i_fps_den = fps.den;
    x264_param.i_csp = X264_CSP_I420;

    // CRF Rate Control
    x264_param.rc.i_rc_method = X264_RC_CRF;
    x264_param.rc.f_rf_constant = cfg.crf;

    // CRITICAL: Disable mb-tree (Macroblock tree rate control) to allow custom quant_offsets to work!
    // If mb-tree is enabled, x264 overwrites spatial QP offsets using its own temporal calculations.
    x264_param.rc.b_mb_tree = 0;

    // Timebase setup: 1 unit of PTS represents 1/framerate seconds
    x264_param.i_timebase_num = fps.den;
    x264_param.i_timebase_den = fps.num;

    x264_encoder = x264_encoder_open(&x264_param);
    if (!x264_encoder) {
        fprintf(stderr, "Error: Failed to open x264 encoder.\n");
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&in_fmt_ctx);
        return 1;
    }

    // 3. Initialize output muxer
    if ((ret = avformat_alloc_output_context2(&out_fmt_ctx, NULL, NULL, cfg.output_mp4)) < 0) {
        fprintf(stderr, "Error: Could not allocate output context (FFmpeg error: %d).\n", ret);
        x264_encoder_close(x264_encoder);
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&in_fmt_ctx);
        return 1;
    }

    // Create output video stream
    out_video_stream = avformat_new_stream(out_fmt_ctx, NULL);
    if (!out_video_stream) {
        fprintf(stderr, "Error: Could not create output video stream.\n");
        goto cleanup_all;
    }
    out_video_stream->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
    out_video_stream->codecpar->codec_id = AV_CODEC_ID_H264;
    out_video_stream->codecpar->width = dec_ctx->width;
    out_video_stream->codecpar->height = dec_ctx->height;
    out_video_stream->time_base = (AVRational){fps.den, fps.num};
    out_video_stream->avg_frame_rate = fps;

    // Extract H.264 headers (SPS/PPS) from x264 and set as codec extradata for MP4 container compliance
    x264_nal_t *headers;
    int i_headers;
    if (x264_encoder_headers(x264_encoder, &headers, &i_headers) < 0) {
        fprintf(stderr, "Error: Failed to get x264 headers.\n");
        goto cleanup_all;
    }
    int header_size = 0;
    for (int i = 0; i < i_headers; i++) {
        header_size += headers[i].i_payload;
    }
    out_video_stream->codecpar->extradata = av_mallocz(header_size + AV_INPUT_BUFFER_PADDING_SIZE);
    if (!out_video_stream->codecpar->extradata) {
        fprintf(stderr, "Error: Could not allocate memory for extradata.\n");
        goto cleanup_all;
    }
    out_video_stream->codecpar->extradata_size = header_size;
    uint8_t *p_extra = out_video_stream->codecpar->extradata;
    for (int i = 0; i < i_headers; i++) {
        memcpy(p_extra, headers[i].p_payload, headers[i].i_payload);
        p_extra += headers[i].i_payload;
    }

    // Copy audio stream parameters if audio stream exists
    if (audio_stream_idx >= 0) {
        AVCodecParameters *in_audio_par = in_fmt_ctx->streams[audio_stream_idx]->codecpar;
        out_audio_stream = avformat_new_stream(out_fmt_ctx, NULL);
        if (!out_audio_stream) {
            fprintf(stderr, "Error: Could not create output audio stream.\n");
            goto cleanup_all;
        }
        if ((ret = avcodec_parameters_copy(out_audio_stream->codecpar, in_audio_par)) < 0) {
            fprintf(stderr, "Error: Failed to copy audio parameters (FFmpeg error: %d).\n", ret);
            goto cleanup_all;
        }
        out_audio_stream->codecpar->codec_tag = 0; // Let FFmpeg choose appropriate tag
        out_audio_stream->time_base = in_fmt_ctx->streams[audio_stream_idx]->time_base;
        printf("Audio stream found and copy configured.\n");
    }

    // Open output file
    if (!(out_fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
        if ((ret = avio_open(&out_fmt_ctx->pb, cfg.output_mp4, AVIO_FLAG_WRITE)) < 0) {
            fprintf(stderr, "Error: Could not open output file %s (FFmpeg error: %d).\n", cfg.output_mp4, ret);
            goto cleanup_all;
        }
    }

    // Write output header
    if ((ret = avformat_write_header(out_fmt_ctx, NULL)) < 0) {
        fprintf(stderr, "Error: Failed to write output header (FFmpeg error: %d).\n", ret);
        goto cleanup_all;
    }

    // Prepare software scaler (convert decoded frame format to YUV420P for x264 input)
    sws_ctx = sws_getContext(dec_ctx->width, dec_ctx->height, dec_ctx->pix_fmt,
                             dec_ctx->width, dec_ctx->height, AV_PIX_FMT_YUV420P,
                             SWS_FAST_BILINEAR, NULL, NULL, NULL);
    if (!sws_ctx) {
        fprintf(stderr, "Error: Failed to initialize SwsContext scaling.\n");
        goto cleanup_all;
    }

    // Calculate macroblock dimensions
    int mb_width = (dec_ctx->width + 15) / 16;
    int mb_height = (dec_ctx->height + 15) / 16;
    int num_mbs_per_frame = mb_width * mb_height;
    printf("Grid size: %dx%d Macroblocks (%d MBs per frame)\n", mb_width, mb_height, num_mbs_per_frame);

    // 4. Decode packets, drain the decoder, then drain x264.
    in_pkt = av_packet_alloc();
    dec_frame = av_frame_alloc();
    if (!in_pkt || !dec_frame) {
        fprintf(stderr, "Error: Failed to allocate packet or frame structures.\n");
        goto cleanup_all;
    }
    EncodingState state = {
        .config = &cfg, .decoder = dec_ctx, .scaler = sws_ctx,
        .encoder = x264_encoder, .output = out_fmt_ctx,
        .stream = out_video_stream, .time_base = {fps.den, fps.num},
        .num_mbs = num_mbs_per_frame,
    };
    printf("Starting encoding loop...\n");
    while ((ret = av_read_frame(in_fmt_ctx, in_pkt)) >= 0) {
        if (in_pkt->stream_index == video_stream_idx) {
            ret = avcodec_send_packet(dec_ctx, in_pkt);
            if (ret >= 0) ret = receive_frames(&state, dec_frame, 0);
        } else if (in_pkt->stream_index == audio_stream_idx) {
            av_packet_rescale_ts(in_pkt, in_fmt_ctx->streams[audio_stream_idx]->time_base,
                                 out_audio_stream->time_base);
            in_pkt->stream_index = out_audio_stream->index;
            ret = av_interleaved_write_frame(out_fmt_ctx, in_pkt);
        }
        av_packet_unref(in_pkt);
        if (ret < 0) {
            fprintf(stderr, "Error processing input packet: %d\n", ret);
            goto cleanup_all;
        }
    }
    if (ret != AVERROR_EOF) {
        fprintf(stderr, "Error reading input: %d\n", ret);
        goto cleanup_all;
    }

    printf("Draining decoder delay...\n");
    ret = avcodec_send_packet(dec_ctx, NULL);
    if (ret >= 0) ret = receive_frames(&state, dec_frame, 1);
    if (ret < 0) {
        fprintf(stderr, "Error draining decoder: %d\n", ret);
        goto cleanup_all;
    }
    printf("Flushing encoder delay...\n");
    while (x264_encoder_delayed_frames(x264_encoder) > 0) {
        ret = encode_picture(&state, NULL);
        if (ret < 0) {
            fprintf(stderr, "Error draining encoder: %d\n", ret);
            goto cleanup_all;
        }
    }
    if (!state.frames_read || state.frames_read != state.frames_written) {
        fprintf(stderr, "Error: Read %d frames but wrote %d.\n",
                state.frames_read, state.frames_written);
        goto cleanup_all;
    }
    ret = av_write_trailer(out_fmt_ctx);
    if (ret < 0) {
        fprintf(stderr, "Error writing output trailer: %d\n", ret);
        goto cleanup_all;
    }
    printf("Encoding complete! Total video frames read: %d, written: %d\n",
            state.frames_read, state.frames_written);
    exit_code = 0;

cleanup_all:
    av_packet_free(&in_pkt);
    av_frame_free(&dec_frame);
    if (sws_ctx) sws_freeContext(sws_ctx);
    if (x264_encoder) x264_encoder_close(x264_encoder);
    if (dec_ctx) avcodec_free_context(&dec_ctx);
    if (in_fmt_ctx) avformat_close_input(&in_fmt_ctx);

    if (out_fmt_ctx) {
        if (!(out_fmt_ctx->oformat->flags & AVFMT_NOFILE) && out_fmt_ctx->pb) {
            avio_closep(&out_fmt_ctx->pb);
        }
        avformat_free_context(out_fmt_ctx);
    }

    return exit_code;
}

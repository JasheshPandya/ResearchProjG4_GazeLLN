#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

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

    // 4. Processing Loop
    AVPacket *in_pkt = av_packet_alloc();
    AVFrame *dec_frame = av_frame_alloc();
    if (!in_pkt || !dec_frame) {
        fprintf(stderr, "Error: Failed to allocate packet or frame structures.\n");
        if (in_pkt) av_packet_free(&in_pkt);
        if (dec_frame) av_frame_free(&dec_frame);
        goto cleanup_all;
    }

    int video_frame_count = 0;
    int encoded_frame_count = 0;

    printf("Starting encoding loop...\n");

    while (av_read_frame(in_fmt_ctx, in_pkt) >= 0) {
        if (in_pkt->stream_index == video_stream_idx) {
            // Decode video packet
            ret = avcodec_send_packet(dec_ctx, in_pkt);
            if (ret < 0) {
                fprintf(stderr, "Error sending packet for decoding: %d\n", ret);
                av_packet_unref(in_pkt);
                break;
            }

            while (ret >= 0) {
                ret = avcodec_receive_frame(dec_ctx, dec_frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                } else if (ret < 0) {
                    fprintf(stderr, "Error during decoding: %d\n", ret);
                    break;
                }

                // Process decoded frame
                x264_picture_t pic_in;
                x264_picture_t pic_out;
                x264_picture_init(&pic_in);

                // Allocate image planes in pic_in
                x264_picture_alloc(&pic_in, X264_CSP_I420, dec_ctx->width, dec_ctx->height);

                // Scale frame to YUV420P directly into pic_in planes
                sws_scale(sws_ctx, (const uint8_t * const *)dec_frame->data, dec_frame->linesize,
                          0, dec_ctx->height, pic_in.img.plane, pic_in.img.i_stride);

                pic_in.i_pts = video_frame_count;

                // Load QP offsets for this frame
                float *qp_offsets = malloc(num_mbs_per_frame * sizeof(float));
                if (!qp_offsets) {
                    fprintf(stderr, "Error: Could not allocate memory for QP offsets.\n");
                    x264_picture_clean(&pic_in);
                    break;
                }

                char qp_path[1024];
                snprintf(qp_path, sizeof(qp_path), "%s/qpoffset_%06d.bin", cfg.qpoffset_dir, video_frame_count);
                FILE *qp_file = fopen(qp_path, "rb");
                if (qp_file) {
                    size_t read_floats = fread(qp_offsets, sizeof(float), num_mbs_per_frame, qp_file);
                    fclose(qp_file);
                    if (read_floats != num_mbs_per_frame) {
                        fprintf(stderr, "Warning: Only read %zu/%d QP offsets from %s. Using zeros.\n",
                                read_floats, num_mbs_per_frame, qp_path);
                        memset(qp_offsets, 0, num_mbs_per_frame * sizeof(float));
                    }
                } else {
                    if (video_frame_count < 10) {
                        fprintf(stderr, "Warning: Could not open %s. Using zero QP offsets.\n", qp_path);
                    } else if (video_frame_count == 10) {
                        fprintf(stderr, "Warning: Silencing further missing QP offset file warnings.\n");
                    }
                    memset(qp_offsets, 0, num_mbs_per_frame * sizeof(float));
                }

                // Pass the QP offsets to the x264 structure
                pic_in.prop.quant_offsets = qp_offsets;
                pic_in.prop.quant_offsets_free = free; // Automatically freed by x264 when done

                // Encode H.264 frame
                x264_nal_t *nals;
                int i_nals;
                int frame_size = x264_encoder_encode(x264_encoder, &nals, &i_nals, &pic_in, &pic_out);
                if (frame_size > 0) {
                    // Create packet for output muxer
                    AVPacket *out_pkt = av_packet_alloc();
                    av_new_packet(out_pkt, frame_size);
                    memcpy(out_pkt->data, nals[0].p_payload, frame_size);

                    out_pkt->pts = pic_out.i_pts;
                    out_pkt->dts = pic_out.i_dts;
                    out_pkt->stream_index = out_video_stream->index;

                    // Rescale timestamps from x264 timebase to output stream timebase
                    AVRational x264_tb = {fps.den, fps.num};
                    av_packet_rescale_ts(out_pkt, x264_tb, out_video_stream->time_base);

                    av_interleaved_write_frame(out_fmt_ctx, out_pkt);
                    av_packet_free(&out_pkt);
                    encoded_frame_count++;
                }

                // Clean x264 input picture memory
                x264_picture_clean(&pic_in);
                video_frame_count++;

                if (video_frame_count % 100 == 0) {
                    printf("Processed %d frames...\n", video_frame_count);
                }
            }
        } else if (in_pkt->stream_index == audio_stream_idx) {
            // Passthrough audio packet directly
            av_packet_rescale_ts(in_pkt, in_fmt_ctx->streams[audio_stream_idx]->time_base, out_audio_stream->time_base);
            in_pkt->stream_index = out_audio_stream->index;
            av_interleaved_write_frame(out_fmt_ctx, in_pkt);
        }
        av_packet_unref(in_pkt);
    }

    // 5. Flush x264 encoder
    printf("Flushing encoder delay...\n");
    while (x264_encoder_delayed_frames(x264_encoder) > 0) {
        x264_picture_t pic_out;
        x264_nal_t *nals;
        int i_nals;
        int frame_size = x264_encoder_encode(x264_encoder, &nals, &i_nals, NULL, &pic_out);
        if (frame_size > 0) {
            AVPacket *out_pkt = av_packet_alloc();
            av_new_packet(out_pkt, frame_size);
            memcpy(out_pkt->data, nals[0].p_payload, frame_size);

            out_pkt->pts = pic_out.i_pts;
            out_pkt->dts = pic_out.i_dts;
            out_pkt->stream_index = out_video_stream->index;

            AVRational x264_tb = {fps.den, fps.num};
            av_packet_rescale_ts(out_pkt, x264_tb, out_video_stream->time_base);

            av_interleaved_write_frame(out_fmt_ctx, out_pkt);
            av_packet_free(&out_pkt);
            encoded_frame_count++;
        } else if (frame_size < 0) {
            break;
        }
    }

    av_write_trailer(out_fmt_ctx);
    printf("Encoding complete! Total video frames read: %d, written: %d\n", video_frame_count, encoded_frame_count);

    av_packet_free(&in_pkt);
    av_frame_free(&dec_frame);

cleanup_all:
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

    return 0;
}
#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id)*ids->ne[0]*ids->ne[1] + (i1)]

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

static void ggml_compute_forward_mul_mat_id_one_chunk(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    const int64_t cur_a,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end,
    const char * src0_cur,
    const struct mmid_row_mapping * matrix_rows,
    const size_t row_size,
    const bool src1_cont,
    const void * wdata,
    const void * compact_data,
    const void * mx_rhs,
    const size_t mx_stride,
    const uint64_t mx_safe_mask) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;

    ggml_vec_dot_t const vec_dot = compact_data ? (type == GGML_TYPE_IQ2_S ? xeon_iq2s_compact_dot : type == GGML_TYPE_IQ3_S ? xeon_iq3s_compact_dot : xeon_iq3_compact_dot) : type_traits_cpu[type].vec_dot;
    enum ggml_type const vec_dot_type = type_traits_cpu[type].vec_dot_type;
    const size_t weight_stride = compact_data ? (size_t)(ne00/256)*144 : nb01;
    const char * weight_data = compact_data ? (const char *) compact_data : src0_cur;

    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    float tmp[16];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
                const int64_t _i12 = ir1;                                     

                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
                const int id       = row_mapping.i1;                         

                const int64_t  i11 = id % ne11;
                const int64_t  i12 = row_mapping.i2;                     

                const int64_t  i1 = id;                          
                const int64_t  i2 = i12;       

                                                                                                                     
                                                                                                                              
                                                                                                      
                                                                                                     
                const char * src1_col = (const char *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*row_size
                    : (i11*nb11 + i12*nb12));

                float * dst_col = (float *) ((char *) dst->data + (i1*nb1 + i2*nb2));

                const void * prepared = mx_rhs && ((mx_safe_mask >> i11) & 1)
                    ? (const char *)mx_rhs + i11*mx_stride : NULL;
                if (prepared) {
                    for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                        xeon_mxfp4_rhs_dot_v1(ne00, &tmp[ir0-iir0], weight_data+ir0*weight_stride, src1_col, prepared);
                    }
                } else {
                    for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                        vec_dot(ne00, &tmp[ir0 - iir0], 0, weight_data + ir0*weight_stride, 0, src1_col, 0, 1);
                    }
                }

                memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir0_end) - iir0)*sizeof(float));
            }
        }
    }
}

#if defined(__AVX__) && !defined(__AVX2__)
                                                            
#if defined(_MSC_VER)
__declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
__attribute__((__noinline__))
#endif
static void ggml_compute_forward_mul_mat_id_mxfp4_cols2_chunk(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    const int64_t cur_a,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end,
    const char * src0_cur,
    const struct mmid_row_mapping * matrix_rows,
    const size_t row_size,
    const bool src1_cont,
    const void * wdata) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;

    ggml_vec_dot_t    const vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type    const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    float tmp[16];
    float tmp1[16];
    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        const int64_t row_end = MIN(iir1 + blck_1, ir1_end);
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            const int64_t col_end = MIN(iir0 + blck_0, ir0_end);
            for (int64_t ir1 = iir1; ir1 < row_end; ir1 += 2) {
                const int ncols = (int) MIN(2, row_end - ir1);
                const char * src1_cols[2];
                float * dst_cols[2];
                for (int c = 0; c < ncols; ++c) {
                    const struct mmid_row_mapping mapping = MMID_MATRIX_ROW(cur_a, ir1 + c);
                    const int64_t i11 = mapping.i1 % ne11;
                    const int64_t i12 = mapping.i2;
                    src1_cols[c] = (const char *) wdata +
                        (src1_cont || src1->type != vec_dot_type
                         ? (i11 + i12 * ne11) * row_size
                         : i11 * nb11 + i12 * nb12);
                    dst_cols[c] = (float *) ((char *) dst->data + mapping.i1 * nb1 + i12 * nb2);
                }
                if (ncols == 2) {
                    for (int64_t ir0 = iir0; ir0 < col_end; ++ir0) {
                        float sums[2];
                        ggml_vec_dot_mxfp4_q8_0_cols2(ne00, sums, src0_cur + ir0 * nb01, src1_cols[0], src1_cols[1]);
                        tmp [ir0 - iir0] = sums[0];
                        tmp1[ir0 - iir0] = sums[1];
                    }
                    memcpy(&dst_cols[1][iir0], tmp1, (col_end - iir0) * sizeof(float));
                } else {
                    for (int64_t ir0 = iir0; ir0 < col_end; ++ir0) {
                        vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0 * nb01, 0, src1_cols[0], 0, 1);
                    }
                }
                memcpy(&dst_cols[0][iir0], tmp, (col_end - iir0) * sizeof(float));
            }
        }
    }
}
#if defined(_MSC_VER)
__declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
__attribute__((__noinline__))
#endif
static void ggml_compute_forward_mul_mat_id_mxfp4_cols421_chunk(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    const int64_t cur_a,
    const int64_t ir0_start, const int64_t ir0_end,
    const int64_t ir1_start, const int64_t ir1_end,
    const char * src0_cur, const struct mmid_row_mapping * matrix_rows,
    const size_t row_size, const bool src1_cont, const void * wdata) {

    GGML_TENSOR_BINARY_OP_LOCALS
    const enum ggml_type type = src0->type;
    ggml_vec_dot_t const vec_dot = type_traits_cpu[type].vec_dot;
    const enum ggml_type vec_dot_type = type_traits_cpu[type].vec_dot_type;
    float tmp[4][16];
    for (int64_t iir1=ir1_start;iir1<ir1_end;iir1+=16) {
        const int64_t row_end=MIN(iir1+16,ir1_end);
        for (int64_t iir0=ir0_start;iir0<ir0_end;iir0+=16) {
            const int64_t col_end=MIN(iir0+16,ir0_end);
            for (int64_t ir1=iir1;ir1<row_end;) {
                const int64_t remaining=row_end-ir1;
                const int nc=remaining>=4 ? 4 : remaining>=2 ? 2 : 1;
                const char * input[4];
                float * output[4];
                for (int c=0;c<nc;++c) {
                    const struct mmid_row_mapping mapping=MMID_MATRIX_ROW(cur_a,ir1+c);
                    const int64_t i11=mapping.i1%ne11, i12=mapping.i2;
                    input[c]=(const char *)wdata+
                        (src1_cont || src1->type!=vec_dot_type
                         ? (i11+i12*ne11)*row_size : i11*nb11+i12*nb12);
                    output[c]=(float *)((char *)dst->data+mapping.i1*nb1+i12*nb2);
                }
                for (int64_t ir0=iir0;ir0<col_end;++ir0) {
                    float sums[4];
                    const char * weight=src0_cur+ir0*nb01;
                    if(nc==4) {
                        ggml_vec_dot_mxfp4_q8_0_cols4(ne00,sums,weight,input[0],input[1],input[2],input[3]);
                    } else if(nc==2) {
                        ggml_vec_dot_mxfp4_q8_0_cols2(ne00,sums,weight,input[0],input[1]);
                    } else {
                        vec_dot(ne00,sums,0,weight,0,input[0],0,1);
                    }
                    for(int c=0;c<nc;++c)tmp[c][ir0-iir0]=sums[c];
                }
                for(int c=0;c<nc;++c)memcpy(output[c]+iir0,tmp[c],(col_end-iir0)*sizeof(float));
                ir1+=nc;
            }
        }
    }
}

#endif

static void * incr_ptr_aligned(void ** p, size_t size, size_t align) {

    void * ptr = *p;
    ptr = (void *) GGML_PAD((uintptr_t) ptr, align);
    *p = (void *) ((char *) ptr + size);
    return ptr;
}

static void ggml_compute_forward_mul_mat_id(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const enum ggml_type type = src0->type;

    const bool src1_cont = ggml_is_contiguous(src1);

    enum ggml_type    const vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t const from_float      = type_traits_cpu[vec_dot_type].from_float;

                                             
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

                                           
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

                 
    const int n_ids = ids->ne[0];                 
    const int n_as  = ne02;                  

    void * wdata_cur = params->wdata;

    if (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_row_size(vec_dot_type, ggml_nelements(src1)), sizeof(int64_t));
    }

    int64_t * matrix_row_counts =          
        incr_ptr_aligned(&wdata_cur, n_as*sizeof(int64_t), sizeof(int64_t));

    struct mmid_row_mapping * matrix_rows =                                 
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping), sizeof(int64_t));

    char (*atomic_current_chunk)[CACHE_LINE_SIZE] =          
        incr_ptr_aligned(&wdata_cur, CACHE_LINE_SIZE * n_as, CACHE_LINE_SIZE);

                                                                                      
                                                                                           
    char * tiled_scratch = incr_ptr_aligned(&wdata_cur, ggml_tiled_wdata_size(nth, dst), 64);

    const bool mx_shape = xeon_mxfp4_rhs_shape(src0, src1, ids);
    const bool mx_enabled = XEON_MX_PREPACK && mx_shape && !params->use_ref;
    const size_t mx_stride = mx_shape ? xeon_mxfp4_rhs_stride(ne10) : 0;
    void * mx_rhs = mx_shape ? incr_ptr_aligned(&wdata_cur, mx_stride*ne11, 64) : NULL;
    uint8_t * mx_flags = mx_shape ? incr_ptr_aligned(&wdata_cur, ne11*nth, sizeof(int64_t)) : NULL;

    GGML_ASSERT(params->wsize >= (size_t)((char *) wdata_cur - (char *) params->wdata));

    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#else
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
#endif
    }

    if (mx_enabled) {
                                                                               
                                                                               
        const int first = (ith*(ne10/32))/nth;
        const int end = ((ith+1)*(ne10/32))/nth;
        const size_t qrow = ggml_row_size(vec_dot_type, ne10);
        for (int64_t input=0;input<ne11;++input) {
            const char * rhs = src1->type==vec_dot_type
                ? (const char *)src1->data + input*nb11
                : (const char *)params->wdata + input*qrow;
            mx_flags[input*nth+ith] = xeon_mxfp4_rhs_prepare_v1(ne10, rhs, first, end,
                (char *)mx_rhs + input*mx_stride);
        }
    }

    if (ith == 0) {
                                       
        memset(matrix_row_counts, 0, n_as*sizeof(int64_t));

                                    
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

                assert(i02 == -1 || (i02 >= 0 && i02 < n_as));

                if (i02 == -1) {
                    memset((char *) dst->data + id*nb1 + iid1*nb2, 0, ne0*sizeof(float));
                    continue;
                }

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (struct mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
    }

                                                                                  
    if (ith == 0) xeon_route37_record_v1(src0, ids);

                          
    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    ggml_barrier(params->threadpool);

    uint64_t mx_safe_mask = 0;
    if (mx_enabled) {
        for (int64_t input=0;input<ne11;++input) {
            bool safe=true;
            for (int worker=0;worker<nth;++worker) safe = safe && mx_flags[input*nth+worker]!=0;
            if (safe) mx_safe_mask |= UINT64_C(1)<<input;
        }
    }

    const void * const * compact_experts = (!params->use_ref && (type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_S) && ids->ne[1] == 1)
        ? xeon_iq3_cache_lookup_experts_v2(src0) : NULL;

    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];

        if (cne1 == 0) {
            continue;
        }

        const char * compact_data = compact_experts ? (const char *) compact_experts[cur_a] : NULL;

                                                                       
        if (!compact_data && ggml_compute_forward_mul_mat_id_tiled(params, dst, cur_a, cne1, (const int32_t *) &MMID_MATRIX_ROW(cur_a, 0), tiled_scratch)) {
            continue;
        }

        const char * src0_cur = (const char *) src0->data + cur_a * nb02;
        const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        const int64_t nr0 = ne01;
        const int64_t nr1 = cne1;

        int chunk_size = 16;
        if (nr0 == 1 || nr1 == 1) {
            chunk_size = 64;
        }

                           
        const bool disable_chunking = ggml_is_numa();

        int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

        if (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        int current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        while (current_chunk < nchunk0 * nchunk1) {
            const int64_t ith0 = current_chunk % nchunk0;
            const int64_t ith1 = current_chunk / nchunk0;

            const int64_t ir0_start = dr0 * ith0;
            const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

            const int64_t ir1_start = dr1 * ith1;
            const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

#if defined(__AVX__) && !defined(__AVX2__)
            if (!params->use_ref && type == GGML_TYPE_MXFP4 && ne00 >= 256 && ir1_end - ir1_start >= 2) {
                if (ir1_end - ir1_start >= 4) {
                    ggml_compute_forward_mul_mat_id_mxfp4_cols421_chunk(
                        dst, src0, src1, ids, cur_a,
                        ir0_start, ir0_end, ir1_start, ir1_end,
                        src0_cur, matrix_rows, row_size, src1_cont, wdata
                    );
                } else {
                ggml_compute_forward_mul_mat_id_mxfp4_cols2_chunk(
                    dst, src0, src1, ids, cur_a,
                    ir0_start, ir0_end, ir1_start, ir1_end,
                    src0_cur, matrix_rows, row_size, src1_cont, wdata
                );
                }
            } else
#endif
            {
                ggml_compute_forward_mul_mat_id_one_chunk(
                    dst, src0, src1, ids, cur_a,
                    ir0_start, ir0_end, ir1_start, ir1_end,
                    src0_cur, matrix_rows, row_size, src1_cont, wdata,
                    compact_data, mx_enabled ? mx_rhs : NULL, mx_stride, mx_safe_mask
                );
            }

            if (nth >= nchunk0 * nchunk1) {
                break;
            }

            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
}

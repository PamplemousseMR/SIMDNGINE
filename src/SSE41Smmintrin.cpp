#include "Simd.hpp"

#include <smmintrin.h>

/*  smmintrin
SSE4.1
Add a lot of instructions: Filling in a lot of the gaps by providing min and max and other operations for all integer data types (especially 32-bit integer had been lacking), where previously integer min was only available for unsigned bytes and signed 16-bit.
Also scaling, FP rounding, blending, linear algebra operation, text processing, comparisons.
Also a non temporal load for reading video memory, or copying it back to main memory.
(Previously only NT stores were available.)
*/
void smmintrin()
{
    __m128d m128d = _mm_setzero_pd();
    __m128i m128i = _mm_setzero_si128();
    __m128 m128 = _mm_setzero_ps();

    /*
    * Int8/int16/int32/int64 arithmetic
    */
    m128 = _mm_ceil_ps(m128);
    m128d = _mm_ceil_pd(m128d);
    m128 = _mm_ceil_ss(m128, m128);
    m128d = _mm_ceil_sd(m128d, m128d);
    m128 = _mm_floor_ps(m128);
    m128d = _mm_floor_pd(m128d);
    m128 = _mm_floor_ss(m128, m128);
    m128d = _mm_floor_sd(m128d, m128d);
    //m128 = _mm_round_ps(m128, M);
    //m128d = _mm_round_ss(m128d, m128d, M);
    m128d = _mm_round_pd(m128d, 0);
    m128d = _mm_round_sd(m128d, m128d, 0);
    m128d = _mm_blend_pd(m128d, m128d, 0);
    m128 = _mm_blend_ps(m128, m128, _MM_SHUFFLE2(2, 3));
    m128d = _mm_blendv_pd(m128d, m128d, m128d);
    m128 = _mm_blendv_ps(m128, m128, m128);
    m128i = _mm_blendv_epi8 (m128i, m128i, m128i);
    m128i = _mm_blend_epi16(m128i, m128i, _MM_SHUFFLE(2, 3, 0, 1));
    m128i = _mm_mullo_epi32 (m128i, m128i);
    m128i = _mm_mul_epi32 (m128i, m128i);
    m128 = _mm_dp_ps(m128, m128, _MM_SHUFFLE(2, 3, 0, 1));
    m128d = _mm_dp_pd(m128d, m128d, _MM_SHUFFLE(2, 3, 0, 1));
    m128i = _mm_stream_load_si128(&m128i);
    m128i = _mm_min_epi8(m128i, m128i);
    m128i = _mm_max_epi8(m128i, m128i);
    m128i = _mm_min_epu16(m128i, m128i);
    m128i = _mm_max_epu16(m128i, m128i);
    m128i = _mm_min_epi32(m128i, m128i);
    m128i = _mm_max_epi32(m128i, m128i);
    m128i = _mm_min_epu32(m128i, m128i);
    m128i = _mm_max_epu32(m128i, m128i);

    /*
    #define _mm_insert_ps(X, Y, N) __builtin_ia32_insertps128((X), (Y), (N))
    #define _mm_extract_ps(X, N) (__extension__                      \
    #define _MM_EXTRACT_FLOAT(D, X, N) (__extension__ ({ __v4sf __a = (__v4sf)(X); \
    #define _MM_MK_INSERTPS_NDX(X, Y, Z) (((X) << 6) | ((Y) << 4) | (Z))
    #define _MM_PICK_OUT_PS(X, N) _mm_insert_ps (_mm_setzero_ps(), (X),   \
    #define _mm_insert_epi8(X, I, N) (__extension__                           \
    #define _mm_insert_epi32(X, I, N) (__extension__                         \
    #define _mm_insert_epi64(X, I, N) (__extension__                         \
    #define _mm_extract_epi8(X, N) (__extension__                           \
    #define _mm_extract_epi32(X, N) (__extension__                         \
    #define _mm_extract_epi64(X, N) (__extension__                         \
    int _mm_testz_si128(__m128i __M, __m128i __V)
    int _mm_testc_si128(__m128i __M, __m128i __V)
    int _mm_testnzc_si128(__m128i __M, __m128i __V)
    #define _mm_test_all_ones(V) _mm_testc_si128((V), _mm_cmpeq_epi32((V), (V)))
    #define _mm_test_mix_ones_zeros(M, V) _mm_testnzc_si128((M), (V))
    #define _mm_test_all_zeros(M, V) _mm_testz_si128 ((M), (V))
    __m128i _mm_cmpeq_epi64(__m128i __V1, __m128i __V2)
    __m128i _mm_cvtepi8_epi16(__m128i __V)
    __m128i _mm_cvtepi8_epi32(__m128i __V)
    __m128i _mm_cvtepi8_epi64(__m128i __V)
    __m128i _mm_cvtepi16_epi32(__m128i __V)
    __m128i _mm_cvtepi16_epi64(__m128i __V)
    __m128i _mm_cvtepi32_epi64(__m128i __V)
    __m128i _mm_cvtepu8_epi16(__m128i __V)
    __m128i _mm_cvtepu8_epi32(__m128i __V)
    __m128i _mm_cvtepu8_epi64(__m128i __V)
    __m128i _mm_cvtepu16_epi32(__m128i __V)
    __m128i _mm_cvtepu16_epi64(__m128i __V)
    __m128i _mm_cvtepu32_epi64(__m128i __V)
    __m128i _mm_packus_epi32(__m128i __V1, __m128i __V2)
    #define _mm_mpsadbw_epu8(X, Y, M) __extension__ ({ \
    __m128i _mm_minpos_epu16(__m128i __V)
    #define _mm_cmpistrm(A, B, M) \
    #define _mm_cmpistri(A, B, M) \
    #define _mm_cmpestrm(A, LA, B, LB, M) \
    #define _mm_cmpestri(A, LA, B, LB, M) \
    #define _mm_cmpistra(A, B, M) \
    #define _mm_cmpistrc(A, B, M) \
    #define _mm_cmpistro(A, B, M) \
    #define _mm_cmpistrs(A, B, M) \
    #define _mm_cmpistrz(A, B, M) \
    #define _mm_cmpestra(A, LA, B, LB, M) \
    #define _mm_cmpestrc(A, LA, B, LB, M) \
    #define _mm_cmpestro(A, LA, B, LB, M) \
    #define _mm_cmpestrs(A, LA, B, LB, M) \
    #define _mm_cmpestrz(A, LA, B, LB, M) \
    __m128i _mm_cmpgt_epi64(__m128i __V1, __m128i __V2)
    unsigned int _mm_crc32_u8(unsigned int __C, unsigned char __D)
    unsigned int _mm_crc32_u16(unsigned int __C, unsigned short __D)
    unsigned int _mm_crc32_u32(unsigned int __C, unsigned int __D)
    unsigned long long _mm_crc32_u64(unsigned long long __C, unsigned long long __D)
    */
}

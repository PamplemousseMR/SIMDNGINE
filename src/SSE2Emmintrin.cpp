#include "simd.hpp"

#include <emmintrin.h>

/*  emmintrin
SSE2
Introduces instruction to work with 2 double precision floating point operands, and with packed int8/int16/int32/int64 integers in 128-bit xmm registers.
*/
void emmintrin()
{
    alignas(16) double arrDouble2[2] = {2.0, 3.0};
    __m128d m128d = _mm_setzero_pd();
    __m128i m128i = _mm_setzero_si128();
    __m128 m128 = _mm_setzero_ps();
    __m64 m64;
    double dou = 11.5;
    int in = 10;
    long long lolo = 28ll;
    short sh = 5;
    char ch = 'c';

    /*
    * Memory & initialization
    */
    m128d = _mm_set_sd(dou);                                        // r[0:63] = w, r[64:127] = 0
    m128d = _mm_set1_pd(dou);                                       // r[0:63] = r[64:127] = w
    //m128d = _mm_set_pd1(dou);                                       // r[0:63] = r[64:127] = w
    m128d = _mm_setzero_pd();                                       // r[0:127] = w
    m128d = _mm_set_pd(dou, dou);                                   // r[0:63] = y, r[64:127] = x
    m128d = _mm_setr_pd(dou, dou);                                  // r[0:63] = x, r[64:127] = y
    m128d = _mm_move_sd(m128d, m128d);                              // r[0:63] = b[0:63], r[64:127] = a[64:127]
    m128d = _mm_load_pd(arrDouble2);                                // r[0:127] = p[0:127]
    m128d = _mm_loadu_pd(arrDouble2);                               // r[0:127] = p[0:127]
    //m128i = _mm_loadu_si64(&lolo);                                  // r[0:63] = EM[mem_addr:mem_addr+63], r[64:127] = 0
    m128d = _mm_load1_pd(arrDouble2);                               // r[0:63] = p[0:63], r[64:127] = p[0:63]
    m128d = _mm_load_sd(arrDouble2);                                // r[0:63] = p[0:63], r[64:127] = 0
    m128d = _mm_loadr_pd(arrDouble2);                               // r[0:63] = p[0:63], r[64:127] = p[0:63]
    _mm_store_pd(arrDouble2, m128d);                                // p[0:127] = a[0:127]
    _mm_storeu_pd(arrDouble2, m128d);                               // p[0:127] = a[0:127]
    _mm_store_sd(arrDouble2, m128d);                                // p[0:63] = a[0:63]
    _mm_storel_pd(arrDouble2, m128d);                               // p[0:63] = a[0:63]
    _mm_storeh_pd(arrDouble2, m128d);                               // p[0:63] = a[64:127]
    _mm_store1_pd(arrDouble2, m128d);                               // p[0:63] = a[0:63], p[64:127] = a[0:63]
    //_mm_store_pd1(arrDouble2, m128d);                               // p[0:63] = a[0:63], p[64:127] = a[0:63]
    _mm_storer_pd(arrDouble2, m128d);                               // p[0:63] = a[64:127], p[64:127] = a[0:63]
    _mm_lfence();                                                   //
    _mm_mfence();                                                   //
    _mm_pause();                                                    //

    /*
    * Conversions
    */
    dou = _mm_cvtsd_f64(m128d);                                     // r = a[0:63]
    in = _mm_cvtsd_si32(m128d);                                     // r[0:31] = (Convert_FP64_To_Int32)a[0:63]
    lolo = _mm_cvtsd_si64(m128d);                                   // r[0:63] = (Convert_FP64_To_Int64)a[0:63]
    in = _mm_cvttsd_si32(m128d);                                    // r[0:31] = (Convert_FP64_To_Int32_Truncate)a[0:63]
    lolo = _mm_cvttsd_si64(m128d);                                  // r[0:63] = (Convert_FP64_To_Int64_Truncate)a[0:63]
    m128d = _mm_cvtsi32_sd(m128d, in);                              // r[0:63] = (Convert_Int32_To_FP64)b, r[64:127] = a[64:127]
    m128d = _mm_cvtsi64_sd(m128d, lolo);                            // r[0:63] = (Convert_Int64_To_FP64)b, r[64:127] = a[64:127]

    /*
    * Arithmetic
    */
    m128d = _mm_add_pd(m128d, m128d);                               // r[0:63] = a[0:63]+b[0:63], r[64:127] = a[64:127]+b[64:127]
    m128d = _mm_add_sd(m128d, m128d);                               // r[0:63] = a[0:63]+b[0:63], r[64:127] = a[64:127]
    m128d = _mm_sub_pd(m128d, m128d);                               // r[0:63] = a[0:63]-b[0:63], r[64:127] = a[64:127]-b[64:127]
    m128d = _mm_sub_sd(m128d, m128d);                               // r[0:63] = a[0:63]-b[0:63], r[64:127] = a[64:127]
    m128d = _mm_mul_pd(m128d, m128d);                               // r[0:63] = a[0:63]*b[0:63], r[64:127] = a[64:127]*b[64:127]
    m128d = _mm_mul_sd(m128d, m128d);                               // r[0:63] = a[0:63]*b[0:63], r[64:127] = a[64:127]
    m128d = _mm_div_pd(m128d, m128d);                               // r[0:63] = a[0:63]/b[0:63], r[64:127] = a[64:127]/b[64:127]
    m128d = _mm_div_sd(m128d, m128d);                               // r[0:63] = a[0:63]/b[0:63], r[64:127] = a[64:127]
    m128d = _mm_sqrt_pd(m128d);                                     // r[0:63] = sqrt(a[0:63]), r[64:127] = sqrt(a[64:127])
    m128d = _mm_sqrt_sd(m128d, m128d);                              // r[0:63] = sqrt(b[0:63]), r[64:127] = a[64:127]
    m128d = _mm_min_pd(m128d, m128d);                               // r[0:63] = min(a[0:63], b[0:63]), r[64:127] = min(a[64:127], b[64:127])
    m128d = _mm_min_sd(m128d, m128d);                               // r[0:63] = min(a[0:63], b[0:63]), r[64:127] = a[64:127]
    m128d = _mm_max_pd(m128d, m128d);                               // r[0:63] = max(a[0:63], b[0:63]), r[64:127] = max(a[64:127], b[64:127])
    m128d = _mm_max_sd(m128d, m128d);                               // r[0:63] = max(a[0:63], b[0:63]), r[64:127] = a[64:127]

    /*
     * Logical
     */
    m128d = _mm_and_pd(m128d, m128d);                               // r[0:63] = a[0:63]&b[0:63], r[64:127] = a[64:127]&b[64:127]
    m128d = _mm_andnot_pd(m128d, m128d);                            // r[0:63] = ~a[0:63]&b[0:63], r[64:127] = ~a[64:127]&b[64:127],
    m128d = _mm_or_pd(m128d, m128d);                                // r[0:63] = a[0:63]|b[0:63], r[64:127] = a[64:127]|b[64:127]
    m128d = _mm_xor_pd(m128d, m128d);                               // r[0:63] = a[0:63]^b[0:63], r[64:127] = a[64:127]^b[64:127]

    /*
     * Comparison
     */
    m128d = _mm_cmpeq_pd(m128d, m128d);                             // r[0:63] = (a[0:63]==b[0:63])?0xF:0x0, r[64:127] = a[64:127]
    m128d = _mm_cmplt_pd(m128d, m128d);                             // r[0:63] = (a[0:63]==b[0:63])?0xF:0x0, r[64:127] = (a[64:127]==b[64:127])?0xF:0x0
    m128d = _mm_cmple_pd(m128d, m128d);                             // r[0:63] = (a[0:63]<b[0:63])?0xF:0x0, r[64:127] = a[64:127]
    m128d = _mm_cmpgt_pd(m128d, m128d);                             // r[0:63] = (a[0:63]<b[0:63])?0xF:0x0, r[64:127] = (a[64:127]<b[64:127])?0xF:0x0
    m128d = _mm_cmpge_pd(m128d, m128d);                             // r[0:63] = (a[0:63]<=b[0:63])?0xF:0x0, r[64:127] = a[64:127]
    m128d = _mm_cmpneq_pd(m128d, m128d);                            // r[0:63] = (a[0:63]<=b[0:63])?0xF:0x0, r[64:127] = (a[64:127]<=b[64:127])?0xF:0x0
    m128d = _mm_cmpnlt_pd(m128d, m128d);                            // r[0:63] = (a[0:63]>b[0:63])?0xF:0x0, r[64:127] = a[64:127]
    m128d = _mm_cmpnle_pd(m128d, m128d);                            // r[0:63] = (a[0:63]>b[0:63])?0xF:0x0, r[64:127] = (a[64:127]>b[64:127])?0xF:0x0
    m128d = _mm_cmpngt_pd(m128d, m128d);                            // r[0:63] = (a[0:63]>=b[0:63])?0xF:0x0, r[64:127] = a[64:127]
    m128d = _mm_cmpnge_pd(m128d, m128d);                            // r[0:63] = (a[0:63]>=b[0:63])?0xF:0x0, r[64:127] = (a[64:127]>=b[64:127])?0xF:0x0
    m128d = _mm_cmpord_pd(m128d, m128d);                            // r[0:63] = (a[0:63]!=b[0:63])?0xF:0x0, r[64:127] = a[64:127]
    m128d = _mm_cmpunord_pd(m128d, m128d);                          // r[0:63] = (a[0:63]!=b[0:63])?0xF:0x0, r[64:127] = (a[64:127]!=b[64:127])?0xF:0x0
    m128d = _mm_cmpeq_sd(m128d, m128d);                             // r[0:63] = !(a[0:63]<b[0:63])?0xF:0x0, r[64:127] = a[64:127]
    m128d = _mm_cmplt_sd(m128d, m128d);                             // r[0:63] = !(a[0:63]<b[0:63])?0xF:0x0, r[64:127] = !(a[64:127]<b[64:127])?0xF:0x0
    m128d = _mm_cmple_sd(m128d, m128d);                             // r[0:63] = !(a[0:63]<=b[0:63])?0xF:0x0, r[64:127] = a[64:127]
    m128d = _mm_cmpgt_sd(m128d, m128d);                             // r[0:63] = !(a[0:63]<=b[0:63])?0xF:0x0, r[64:127] = !(a[64:127]<=b[64:127])?0xF:0x0
    m128d = _mm_cmpge_sd(m128d, m128d);                             // r[0:63] = !(a[0:63]>b[0:63])?0xF:0x0, r[64:127] = a[64:127]
    m128d = _mm_cmpneq_sd(m128d, m128d);                            // r[0:63] = !(a[0:63]>b[0:63])?0xF:0x0, r[64:127] = !(a[64:127]>b[64:127])?0xF:0x0
    m128d = _mm_cmpnlt_sd(m128d, m128d);                            // r[0:63] = !(a[0:63]>=b[0:63])?0xF:0x0, r[64:127] = a[64:127]
    m128d = _mm_cmpnle_sd(m128d, m128d);                            // r[0:63] = !(a[0:63]>=b[0:63])?0xF:0x0, r[64:127] = !(a[64:127]>=b[64:127])?0xF:0x0
    m128d = _mm_cmpngt_sd(m128d, m128d);                            // r[0:63] = (a[0:63]ord?b[0:63])?0xF:0x0, r[64:127] = a[64:127]
    m128d = _mm_cmpnge_sd(m128d, m128d);                            // r[0:63] = (a[0:63]ord?b[0:63])?0xF:0x0, r[64:127] = (a[64:127]ord?b[64:127])?0xF:0x0
    m128d = _mm_cmpord_sd(m128d, m128d);                            // r[0:63] = (a[0:63]unord?b[0:63])?0xF:0x0, r[64:127] = a[64:127]
    m128d = _mm_cmpunord_sd(m128d, m128d);                          // r[0:63] = (a[0:63]unord?b[0:63])?0xF:0x0, r[64:127] = (a[64:127]unord?b[64:127])?0xF:0x0
    in = _mm_comieq_sd(m128d, m128d);                               // r = (a[0:63]==b[0:63])?0x1:0x0
    in = _mm_comilt_sd(m128d, m128d);                               // r = (a[0:63]<b[0:63])?0x1:0x0
    in = _mm_comile_sd(m128d, m128d);                               // r = (a[0:63]<=b[0:63])?0x1:0x0
    in = _mm_comigt_sd(m128d, m128d);                               // r = (a[0:63]>b[0:63])?0x1:0x0
    in = _mm_comige_sd(m128d, m128d);                               // r = (a[0:63]>=b[0:63])?0x1:0x0
    in = _mm_comineq_sd(m128d, m128d);                              // r = (a[0:63]!=b[0:63])?0x1:0x0
    in = _mm_ucomieq_sd(m128d, m128d);                              // r = (a[0:63]==b[0:63])?0x1:0x0
    in = _mm_ucomilt_sd(m128d, m128d);                              // r = (a[0:63]<b[0:63])?0x1:0x0
    in = _mm_ucomile_sd(m128d, m128d);                              // r = (a[0:63]<=b[0:63])?0x1:0x0
    in = _mm_ucomigt_sd(m128d, m128d);                              // r = (a[0:63]>b[0:63])?0x1:0x0
    in = _mm_ucomige_sd(m128d, m128d);                              // r = (a[0:63]>=b[0:63])?0x1:0x0
    in = _mm_ucomineq_sd(m128d, m128d);                             // r = (a[0:63]!=b[0:63])?0x1:0x0

    /*
    * Miscellaneous
    */
    m128d = _mm_shuffle_pd(m128d, m128d, _MM_SHUFFLE2(0, 1));       // r[0:63] = (imm8[0]==0)?a[0:63]:a[64:127], r[64:127] = (imm8[1]==0)?b[0:63]:b[64:127]
    m128d = _mm_unpackhi_pd(m128d, m128d);                          // r[0:63] = a[64:127], r[64:127] = b[64:127]
    m128d = _mm_unpacklo_pd(m128d, m128d);                          // r[0:63] = a[0:63], r[64:127] = b[0:63]
    m128d = _mm_loadh_pd(m128d, arrDouble2);                        // r[0:63] = a[0:63], r[64:127] = p[0:63]
    m128d = _mm_loadl_pd(m128d, arrDouble2);                        // r[0:63] = p[0:63], r[64:127] = a[64:127]
    //m128d = _mm_undefined_pd();                                     // r[0:127] = undefined
    in = _mm_movemask_pd(m128d);                                    // r = sign(a[32:63])<<1|sign(a[0:31])

    /*
    * Int8/int16/int32/int64 (SSE) extensions
    */
    m128i = _mm_set_epi64x(lolo, lolo);                             // r[0:63] = y, r[64:127] = x
    //m128i = _mm_set_epi64(m64, m64);                              // r[0:63] = y, r[64:127] = x
    m128i = _mm_set_epi32(in, in, in, in);                          // r[0:31] = w, r[32:63] = z, r[64:95] = y, r[96:127] = x
    m128i = _mm_set_epi16(sh, sh, sh ,sh ,sh ,sh ,sh ,sh);          // r[0:15] = w, r[16:31] = z, r[32:47] = y, r[48:63] = x, r[64:79] = d, r[80:95] = c, r[96:111] = b, r[112:127] = a
    m128i = _mm_set_epi8(ch, ch, ch, ch, ch, ch, ch, ch,            // r[0:7] = w, r[8:15] = z, r[16:23] = y, r[24:31] = x, r[32:39] = l, r[40:47] = k, r[48:55] = j, r[56:63] = i,
                         ch, ch, ch, ch, ch, ch, ch, ch);           // r[64:71] = h, r[72:79] = g, r[80:87] = f, r[88:95] = e, r[96:103] = d, r[104:111] = c, r[112:119] = b, r[120:127] = a
    m128i = _mm_set1_epi64x(lolo);                                  // r[0:63] = w, r[64:127] = w
    //m128i = _mm_set1_epi64(m64);                                  // r[0:63] = w, r[64:127] = w
    m128i = _mm_set1_epi32(in);                                     // r[0:31] = w, r[32:63] = w, r[64:95] = w, r[96:127] = w
    m128i = _mm_set1_epi16(sh);                                     // r[0:15] = w, r[16:31] = w, r[32:47] = w, r[48:63] = w, r[64:79] = w, r[80:95] = w, r[96:111] = w, r[112:127] = w
    m128i = _mm_set1_epi8(ch);                                      // r[0:7] = w, r[8:15] = w, r[16:23] = w, r[24:31] = w, r[32:39] = w, r[40:47] = w, r[48:55] = w, r[56:63] = w, r[64:71] = w, r[72:79] = w, r[80:87] = w, r[88:95] = w, r[96:103] = w, r[104:111] = w, r[112:119] = w, r[120:127] = w
    //m128i = _mm_setr_epi64(m64, m64);                             // r[0:63] = x, r[64:127] = y
    m128i = _mm_setr_epi32(in, in, in, in);                         // r[0:31] = x, r[32:63] = y, r[64:95] = z, r[96:127] = w
    m128i = _mm_setr_epi16(sh, sh, sh ,sh ,sh ,sh ,sh ,sh);         // r[0:15] = a, r[16:31] = b, r[32:47] = c, r[48:63] = d, r[64:79] = x, r[80:95] = y, r[96:111] = z, r[112:127] = w
    m128i = _mm_setr_epi8(ch, ch, ch, ch, ch, ch, ch, ch,           // r[0:7] = a, r[8:15] = b, r[16:23] = c, r[24:31] = d, r[32:39] = e, r[40:47] = f, r[48:55] = g, r[56:63] = h,
                          ch, ch, ch, ch, ch, ch, ch, ch);          // r[64:71] = i, r[72:79] = j, r[80:87] = k, r[88:95] = l, r[96:103] = x, r[104:111] = y, r[112:119] = z, r[120:127] = w
    m128i = _mm_load_si128(&m128i);                                 // r[0:127] = MEM[mem_addr:mem_addr+127]
    m128i = _mm_loadu_si128(&m128i);                                // r[0:127] = MEM[mem_addr:mem_addr+127]
    m128i = _mm_loadl_epi64(&m128i);                                // r[0:63] = MEM[mem_addr:mem_addr+63], r[64:127] = 0
    //m128i = _mm_undefined_si128();                                  // r[0:127] = undefined
    _mm_store_si128(&m128i, m128i);                                 // r[0:127] = MEM[mem_addr:mem_addr+127]
    _mm_storeu_si128(&m128i, m128i);                                // r[0:127] = MEM[mem_addr:mem_addr+127]
    _mm_storel_epi64(&m128i, m128i);                                // r[0:63] = MEM[mem_addr:mem_addr+63]
    //m64 = _mm_movepi64_pi64(m128i);                               // r[0:63] = p[0:63]
    //m128i = _mm_movpi64_epi64(m64);                               // r[0:63] = p[0:63], r[64:127] = 0
    m128i = _mm_move_epi64(m128i);                                  // r[0:63] = p[0:63], r[64:127] = 0
    m128i = _mm_setzero_si128();                                    // r[0:127] = 0
    _mm_stream_si32(&in, in);                                       // MEM[mem_addr:mem_addr+31] = a[0:31]
    //_mm_stream_si64(&lolo, lolo);                                   // MEM[mem_addr:mem_addr+63] = a[63:0]
    _mm_stream_si128(&m128i, m128i);                                // MEM[mem_addr:mem_addr+127] = a[0:127]
    _mm_stream_pd(arrDouble2, m128d);                               // MEM[mem_addr:mem_addr+127] = a[0:127]
    in = _mm_extract_epi16(m128i, 0);                               // r[0:15] = (a[0:127]>>(imm8[0:2]*16))[0:15], r[16:31] = 0
    m128i = _mm_insert_epi16(m128i, in, 0);                         // r[0:127] = a[0:127], r[imm8[0:2]*16:imm8[0:2]*16+15] = i[0:15]
    m128i = _mm_mulhi_epu16(m128i, m128i);                          // r[0:15] = (a[0:15]*b[0:15])[16:31], r[16:31] = (a[16:31]*b[16:31])[16:31], r[32:47] = (a[32:47]*b[32:47])[16:31], r[48:63] = (a[48:63]*b[48:63])[16:31], r[64:79] = (a[64:79]*b[64:79])[16:31], r[80:95] = (a[80:95]*b[80:95])[16:31], r[96:111] = (a[96:111]*b[96:111])[16:31], r[112:127] = (a[112:127]*b[112:127])[16:31]
    m128i = _mm_sad_epu8(m128i, m128i);                             //

    /*
    * Int8/int16/int32/int64 convertion
    */
    m128d = _mm_cvtepi32_pd(m128i);                                 // r[0:63] = (Convert_Int32_To_FP64)a[0:31], r[64:127] = (Convert_Int32_To_FP64)a[32:63]
    m128 = _mm_cvtepi32_ps(m128i);                                  // r[0:31] = (Convert_Int32_To_FP32)a[0:31], r[32:63] = (Convert_Int32_To_FP32)a[32:63], r[64:95] = (Convert_Int32_To_FP32)a[64:95], r[96:127] = (Convert_Int32_To_FP32)a[96:127]
    m128i = _mm_cvtpd_epi32(m128d);                                 // r[0:31] = (Convert_FP64_To_Int32)a[0:63], r[32:63] = (Convert_FP64_To_Int32)a[63:127]
    //m64 = _mm_cvtpd_pi32(m128d);                                  // r[0:31] = (Convert_FP64_To_Int32)a[0:63], r[32:63] = (Convert_FP64_To_Int32)a[63:127]
    m128 = _mm_cvtpd_ps(m128d);                                     // r[0:31] = (Convert_FP64_To_FP32)a[0:63], r[32:63] = (Convert_FP64_To_FP32)a[63:127]
    m128i = _mm_cvttpd_epi32(m128d);                                // r[0:31] = (Convert_FP64_To_Int32_Truncate)a[0:63], r[32:63] = (Convert_FP64_To_Int32_Truncate)a[63:127]
    //m64 = _mm_cvttpd_pi32(m128d);                                 // r[0:31] = (Convert_FP64_To_Int32_Truncate)a[0:63], r[32:63] = (Convert_FP64_To_Int32_Truncate)a[63:127]
    //m128d = _mm_cvtpi32_pd(m64);                                  // r[0:63] = (Convert_Int32_To_FP64)a[0:31], r[64:127] = (Convert_Int32_To_FP64)a[32:63]
    m128i = _mm_cvtps_epi32(m128);                                  // r[0:31] = (Convert_FP32_To_Int32)a[0:31], r[32:63] = (Convert_FP32_To_Int32)a[32:63], r[64:95] = (Convert_FP32_To_Int32)a[64:95], r[96:127] = (Convert_FP32_To_Int32)a[96:127]
    m128i = _mm_cvttps_epi32(m128);                                 // r[0:31] = (Convert_FP32_To_Int32_Truncate)a[0:31], r[32:63] = (Convert_FP32_To_Int32_Truncate)a[32:63], r[64:95] = (Convert_FP32_To_Int32_Truncate)a[64:95], r[96:127] = (Convert_FP32_To_Int32_Truncate)a[96:127]
    m128d = _mm_cvtps_pd(m128);                                     // r[0:63] = (Convert_FP32_To_FP64)a[0:31], r[64:127] = (Convert_FP32_To_FP64)a[32:63]
    m128d = _mm_cvtss_sd(m128d, m128);                              // r[0:63] = (Convert_FP32_To_FP64)b[0:31], r[64:127] = a[64:127]
    m128 = _mm_cvtsd_ss(m128, m128d);                               // r[0:31] = (Convert_FP64_To_FP32)b[0:63], r[32:127] = a[32:127]
    in = _mm_cvtsi128_si32(m128i);                                  // r = (Convert_FP32_To_Int32)a[0:31]
    lolo = _mm_cvtsi128_si64(m128i);                                // r = (Convert_FP32_To_Int64)a[0:63]
    m128i = _mm_cvtsi32_si128(in);                                  // r[0:31] = w, r[32:127] = 0
    m128i = _mm_cvtsi64_si128(lolo);                                // r[0:63] = w, r[64:127] = 0
    //m128i = _mm_cvtsi64x_si128(lolo);                               // r[0:63] = w, r[64:127] = 0

    /*
    * Int8/int16/int32/int64 miscellaneous
    */
    m128i = _mm_packs_epi16(m128i, m128i);                          // r[0:7] = (Saturate_Int16_To_Int8)a[0:15], r[8:15] = (Saturate_Int16_To_Int8)a[16:31], r[16:23] = (Saturate_Int16_To_Int8)a[32:47], r[24:31] = (Saturate_Int16_To_Int8)a[48:63], r[32:39] = (Saturate_Int16_To_Int8)a[64:79], r[40:47] = (Saturate_Int16_To_Int8)a[80:95], r[48:55] = (Saturate_Int16_To_Int8)a[96:111], r[56:63] = (Saturate_Int16_To_Int8)a[112:127], r[64:71] = (Saturate_Int16_To_Int8)b[0:15], r[72:79] = (Saturate_Int16_To_Int8)b[16:31], r[80:87] = (Saturate_Int16_To_Int8)b[32:47], r[88:95] = (Saturate_Int16_To_Int8)b[48:63], r[96:103] = (Saturate_Int16_To_Int8)b[64:79], r[104:111] = (Saturate_Int16_To_Int8)b[80:95], r[112:119] = (Saturate_Int16_To_Int8)b[96:111], r[120:127] = (Saturate_Int16_To_Int8)b[112:127]
    m128i = _mm_packs_epi32(m128i, m128i);                          // r[0:15] = (Saturate_Int32_To_Int16)a[0:31], r[16:31] = (Saturate_Int32_To_Int16)a[32:63], r[32:47] = (Saturate_Int32_To_Int16)a[64:95], r[48:63] = (Saturate_Int32_To_Int16)a[96:127], r[64:79] = (Saturate_Int32_To_Int16)b[0:31], r[80:95] = (Saturate_Int32_To_Int16)b[32:63], r[96:111] = (Saturate_Int32_To_Int16)b[64:95], r[112:127] = (Saturate_Int32_To_Int16)b[96:127]
    m128i = _mm_packus_epi16(m128i, m128i);                         // r[0:7] = (Saturate_Int16_To_UnsignedInt8)a[0:15], r[8:15] = (Saturate_Int16_To_UnsignedInt8)a[16:31], r[16:23] = (Saturate_Int16_To_UnsignedInt8)a[32:47], r[24:31] = (Saturate_Int16_To_UnsignedInt8)a[48:63], r[32:39] = (Saturate_Int16_To_UnsignedInt8)a[64:79], r[40:47] = (Saturate_Int16_To_UnsignedInt8)a[80:95], r[48:55] = (Saturate_Int16_To_UnsignedInt8)a[96:111], r[56:63] = (Saturate_Int16_To_UnsignedInt8)a[112:127], r[64:71] = (Saturate_Int16_To_UnsignedInt8)b[0:15], r[72:79] = (Saturate_Int16_To_UnsignedInt8)b[16:31], r[80:87] = (Saturate_Int16_To_UnsignedInt8)b[32:47], r[88:95] = (Saturate_Int16_To_UnsignedInt8)b[48:63], r[96:103] = (Saturate_Int16_To_UnsignedInt8)b[64:79], r[104:111] = (Saturate_Int16_To_UnsignedInt8)b[80:95], r[112:119] = (Saturate_Int16_To_UnsignedInt8)b[96:111], r[120:127] = (Saturate_Int16_To_UnsignedInt8)b[112:127]
    m128i = _mm_unpackhi_epi8(m128i, m128i);                        // r[0:7] = a[64:71], r[8:15] = b[64:71], r[16:23] = a[72:79], r[24:31] = b[72:79], r[32:39] = a[80:87], r[40:47] = b[80:87], r[48:55] = a[88:95], r[56:63] = b[88:95], r[64:71] = a[96:103], r[72:79] = b[96:103], r[80:87] = a[104:111], r[88:95] = b[104:111], r[96:103] = a[112:119], r[104:111] = b[112:119], r[112:119] = a[120:127], r[120:127] = b[120:127]
    m128i = _mm_unpackhi_epi16(m128i, m128i);                       // r[0:15] = a[64:79], r[16:31] = b[64:79], r[32:47] = a[80:95], r[48:63] = b[80:95], r[64:79] = a[96:111], r[80:95] = b[96:111], r[96:111] = a[112:127], r[112:127] = b[112:127]
    m128i = _mm_unpackhi_epi32(m128i, m128i);                       // r[0:31] = a[64:95], r[32:63] = b[64:95], r[64:95] = a[96:127], r[96:127] = b[96:127]
    m128i = _mm_unpackhi_epi64(m128i, m128i);                       // r[0:63] = a[64:127], r[64:127] = b[64:127]
    m128i = _mm_unpacklo_epi8(m128i, m128i);                        // r[0:7] = a[0:7], r[8:15] = b[0:7], r[16:23] = a[8:15], r[24:31] = b[8:15], r[32:39] = a[16:23], r[40:47] = b[16:23], r[48:55] = a[24:31], r[56:63] = b[24:31], r[64:71] = a[32:39], r[72:79] = b[32:39], r[80:87] = a[40:47], r[88:95] = b[40:47], r[96:103] = a[48:55], r[104:111] = b[48:55], r[112:119] = a[56:63], r[120:127] = b[56:63]
    m128i = _mm_unpacklo_epi16(m128i, m128i);                       // r[0:15] = a[0:15], r[16:31] = b[0:15], r[32:47] = a[16:31], r[48:63] = b[16:31], r[64:79] = a[32:47], r[80:95] = b[32:47], r[96:111] = a[48:63], r[112:127] = b[48:63]
    m128i = _mm_unpacklo_epi32(m128i, m128i);                       // r[0:31] = a[0:31], r[32:63] = b[0:31], r[64:95] = a[32:63], r[96:127] = b[32:63]
    m128i = _mm_unpacklo_epi64(m128i, m128i);                       // r[0:63] = a[0:63], r[64:127] = b[0:63]
    m128i = _mm_shufflehi_epi16(m128i, _MM_SHUFFLE(2, 3, 0, 1));    // r[0:63] = a[63:0], r[64:79] = (a>>(imm8[0:1]*16))[64:79], r[80:95] = (a>>(imm8[2:3]*16))[64:79], r[96:111] = (a>>(imm8[4:5]*16))[64:79], r[112:127] = (a>>(imm8[6:7]*16))[64:79]
    m128i = _mm_shufflelo_epi16(m128i, _MM_SHUFFLE(2, 3, 0, 1));    // r[0:15] = (a>>(imm8[0:1]*16))[0:15], r[16:31] = (a>>(imm8[2:3]*16))[0:15], r[32:47] = (a>>(imm8[4:5]*16))[0:15], r[48:63] = (a>>(imm8[6:7]*16))[0:15], r[64:127] = a[127:64]
    m128i = _mm_shuffle_epi32(m128i, _MM_SHUFFLE(2, 3, 0, 1));      // r[0:31] = a[imm8[0:1]*32:imm8[0:1]*32+31], r[32:63] = a[imm8[2:3]*32:imm8[2:3]*32+31], r[64:95] = a[imm8[4:5]*32:imm8[4:5]*32+31], r[96:127] = a[imm8[6:7]*32:imm8[6:7]*32+31]
    _mm_maskmoveu_si128(m128i, m128i, &ch);                         //
    in = _mm_movemask_epi8(m128i);                                  // r[0] = a[7], r[1] = a[15], r[2] = a[23], r[3] = a[31], r[4] = a[39], r[5] = a[47], r[6] = a[55], r[7] = a[63], r[8] = a[71], r[9] = a[79], r[10] = a[87], r[11] = a[95], r[12] = a[103], r[13] = a[111], r[14] = a[119], r[15] = a[127]
    m128i  = _mm_slli_epi16(m128i, 0);                              // r[0:15] = (ZeroExtend)(a[64:79]<<imm8[0:7]), r[16:31] = (ZeroExtend)(a[64:79]<<imm8[0:7]), r[32:47] = (ZeroExtend)(a[80:95]<<imm8[0:7]), r[48:63] = (ZeroExtend)(a[80:95]<<imm8[0:7]), r[64:79] = (ZeroExtend)(a[96:111]<<imm8[0:7]), r[80:95] = (ZeroExtend)(a[96:111]<<imm8[0:7]), r[96:111] = (ZeroExtend)(a[112:127]<<imm8[0:7]), r[112:127] = (ZeroExtend)(a[112:127]<<imm8[0:7])
    m128i  = _mm_slli_epi32(m128i, 0);                              // r[0:31] = (ZeroExtend)(a[0:31]<<imm8[0:7]), r[32:63] = (ZeroExtend)(a[32:63]<<imm8[0:7]), r[64:95] = (ZeroExtend)(a[64:95]<<imm8[0:7]), r[96:127] = (ZeroExtend)(a[96:127]<<imm8[0:7])
    m128i  = _mm_slli_epi64(m128i, 0);                              // r[0:63] = (ZeroExtend)(a[0:63]<<imm8[0:7]), r[64:127] = (ZeroExtend)(a[64:127]<<imm8[0:7])
    m128i  = _mm_srai_epi16(m128i, 0);                              // r[0:15] = (ZeroExtend)(a[64:79]>>imm8[0:7]), r[16:31] = (ZeroExtend)(a[64:79]>>imm8[0:7]), r[32:47] = (ZeroExtend)(a[80:95]>>imm8[0:7]), r[48:63] = (ZeroExtend)(a[80:95]>>imm8[0:7]), r[64:79] = (ZeroExtend)(a[96:111]>>imm8[0:7]), r[80:95] = (ZeroExtend)(a[96:111]>>imm8[0:7]), r[96:111] = (ZeroExtend)(a[112:127]>>imm8[0:7]), r[112:127] = (ZeroExtend)(a[112:127]>>imm8[0:7])
    m128i  = _mm_srai_epi32(m128i, 0);                              // r[0:31] = (ZeroExtend)(a[0:31]>>imm8[0:7]), r[32:63] = (ZeroExtend)(a[32:63]>>imm8[0:7]), r[64:95] = (ZeroExtend)(a[64:95]>>imm8[0:7]), r[96:127] = (ZeroExtend)(a[96:127]>>imm8[0:7])
#if (!defined(__clang__)) || (defined(__clang__) && __clang_major__ >= 3 && __clang_minor__ >= 7)
    m128i  = _mm_bsrli_si128(m128i, 0);                             //
    m128i  = _mm_bslli_si128(m128i, 0);                             //
#endif
    m128i  = _mm_srli_si128(m128i, 0);                              //
    m128i  = _mm_slli_si128(m128i, 0);                              //
    m128i  = _mm_srli_epi16(m128i, 0);                              // r[0:7] = imm8[0:7]>15?0:(ZeroExtend)(a[64:71]>>imm8[0:7]), r[8:15] = imm8[0:7]>15?0:(ZeroExtend)(a[64:71]>>imm8[0:7]), r[16:23] = imm8[0:7]>15?0:(ZeroExtend)(a[72:79]>>imm8[0:7]), r[24:31] = imm8[0:7]>15?0:(ZeroExtend)(a[72:79]>>imm8[0:7]), r[32:39] = imm8[0:7]>15?0:(ZeroExtend)(a[80:87]>>imm8[0:7]), r[40:47] = imm8[0:7]>15?0:(ZeroExtend)(a[80:87]>>imm8[0:7]), r[48:55] = imm8[0:7]>15?0:(ZeroExtend)(a[88:95]>>imm8[0:7]), r[56:63] = imm8[0:7]>15?0:(ZeroExtend)(a[88:95]>>imm8[0:7]), r[64:71] = imm8[0:7]>15?0:(ZeroExtend)(a[96:103]>>imm8[0:7]), r[72:79] = imm8[0:7]>15?0:(ZeroExtend)(a[96:103]>>imm8[0:7]), r[80:87] = imm8[0:7]>15?0:(ZeroExtend)(a[104:111]>>imm8[0:7]), r[88:95] = imm8[0:7]>15?0:(ZeroExtend)(a[104:111]>>imm8[0:7]), r[96:103] = imm8[0:7]>15?0:(ZeroExtend)(a[112:119]>>imm8[0:7]), r[104:111] = imm8[0:7]>15?0:(ZeroExtend)(a[112:119]>>imm8[0:7]), r[112:119] = imm8[0:7]>15?0:(ZeroExtend)(a[120:127]>>imm8[0:7]), r[120:127] = imm8[0:7]>15?0:(ZeroExtend)(a[120:127]>>imm8[0:7])
    m128i  = _mm_srli_epi32(m128i, 0);                              // r[0:31] = (ZeroExtend)(a[0:31]>>imm8[0:7]), r[32:63] = (ZeroExtend)(a[32:63]>>imm8[0:7]), r[64:95] = (ZeroExtend)(a[64:95]>>imm8[0:7]), r[96:127] = (ZeroExtend)(a[96:127]>>imm8[0:7])
    m128i  = _mm_srli_epi64(m128i, 0);                              // r[0:63] = (ZeroExtend)(a[0:63]>>imm8[0:7]), r[64:127] = (ZeroExtend)(a[64:127]>>imm8[0:7])
    m128i  = _mm_sll_epi16(m128i, m128i);                           // r[0:15] = (ZeroExtend)(a[64:79]<<count[0:7]), r[16:31] = (ZeroExtend)(a[64:79]<<count[0:7]), r[32:47] = (ZeroExtend)(a[80:95]<<count[0:7]), r[48:63] = (ZeroExtend)(a[80:95]<<count[0:7]), r[64:79] = (ZeroExtend)(a[96:111]<<count[0:7]), r[80:95] = (ZeroExtend)(a[96:111]<<count[0:7]), r[96:111] = (ZeroExtend)(a[112:127]<<count[0:7]), r[112:127] = (ZeroExtend)(a[112:127]<<count[0:7])
    m128i  = _mm_sll_epi32(m128i, m128i);                           // r[0:31] = (ZeroExtend)(a[0:31]<<count[0:7]), r[32:63] = (ZeroExtend)(a[32:63]<<count[0:7]), r[64:95] = (ZeroExtend)(a[64:95]<<count[0:7]), r[96:127] = (ZeroExtend)(a[96:127]<<count[0:7])
    m128i  = _mm_sll_epi64(m128i, m128i);                           // r[0:63] = (ZeroExtend)(a[0:63]<<count[0:7]), r[64:127] = (ZeroExtend)(a[64:127]<<count[0:7])
    m128i  = _mm_sra_epi16(m128i, m128i);                           // r[0:15] = (ZeroExtend)(a[64:79]>>count[0:7]), r[16:31] = (ZeroExtend)(a[64:79]>>count[0:7]), r[32:47] = (ZeroExtend)(a[80:95]>>count[0:7]), r[48:63] = (ZeroExtend)(a[80:95]>>count[0:7]), r[64:79] = (ZeroExtend)(a[96:111]>>count[0:7]), r[80:95] = (ZeroExtend)(a[96:111]>>count[0:7]), r[96:111] = (ZeroExtend)(a[112:127]>>count[0:7]), r[112:127] = (ZeroExtend)(a[112:127]>>count[0:7])
    m128i  = _mm_sra_epi32(m128i, m128i);                           // r[0:31] = (ZeroExtend)(a[0:31]>>count[0:7]), r[32:63] = (ZeroExtend)(a[32:63]>>count[0:7]), r[64:95] = (ZeroExtend)(a[64:95]>>count[0:7]), r[96:127] = (ZeroExtend)(a[96:127]>>count[0:7])
    m128i  = _mm_srl_epi16(m128i, m128i);                           // r[0:7] = count[0:7]>15?0:(ZeroExtend)(a[64:71]>>count[0:7]), r[8:15] = count[0:7]>15?0:(ZeroExtend)(a[64:71]>>count[0:7]), r[16:23] = count[0:7]>15?0:(ZeroExtend)(a[72:79]>>count[0:7]), r[24:31] = count[0:7]>15?0:(ZeroExtend)(a[72:79]>>count[0:7]), r[32:39] = count[0:7]>15?0:(ZeroExtend)(a[80:87]>>count[0:7]), r[40:47] = count[0:7]>15?0:(ZeroExtend)(a[80:87]>>count[0:7]), r[48:55] = count[0:7]>15?0:(ZeroExtend)(a[88:95]>>count[0:7]), r[56:63] = count[0:7]>15?0:(ZeroExtend)(a[88:95]>>count[0:7]), r[64:71] = count[0:7]>15?0:(ZeroExtend)(a[96:103]>>count[0:7]), r[72:79] = count[0:7]>15?0:(ZeroExtend)(a[96:103]>>count[0:7]), r[80:87] = count[0:7]>15?0:(ZeroExtend)(a[104:111]>>count[0:7]), r[88:95] = count[0:7]>15?0:(ZeroExtend)(a[104:111]>>count[0:7]), r[96:103] = count[0:7]>15?0:(ZeroExtend)(a[112:119]>>count[0:7]), r[104:111] = count[0:7]>15?0:(ZeroExtend)(a[112:119]>>count[0:7]), r[112:119] = count[0:7]>15?0:(ZeroExtend)(a[120:127]>>count[0:7]), r[120:127] = count[0:7]>15?0:(ZeroExtend)(a[120:127]>>count[0:7])
    m128i  = _mm_srl_epi32(m128i, m128i);                           // r[0:31] = (ZeroExtend)(a[0:31]>>count[0:7]), r[32:63] = (ZeroExtend)(a[32:63]>>count[0:7]), r[64:95] = (ZeroExtend)(a[64:95]>>count[0:7]), r[96:127] = (ZeroExtend)(a[96:127]>>count[0:7])
    m128i  = _mm_srl_epi64(m128i, m128i);                           // r[0:63] = (ZeroExtend)(a[0:63]>>count[0:7]), r[64:127] = (ZeroExtend)(a[64:127]>>count[0:7])

    /*
    * Int8/int16/int32/int64 arithmetic
    */
    m128i = _mm_add_epi8(m128i, m128i);                             // r[0:7] = a[0:7]+b[0:7], r[8:15] = a[8:15]+b[8:15], r[16:23] = a[16:23]+b[16:23], r[24:31] = a[24:31]+b[24:31], r[32:39] = a[32:39]+b[32:39], r[40:47] = a[40:47]+b[40:47], r[48:55] = a[48:55]+b[48:55], r[56:63] = a[56:63]+b[56:63], r[64:71] = a[64:71]+b[64:71], r[72:79] = a[72:79]+b[72:79], r[80:87] = a[80:87]+b[80:87], r[88:95] = a[88:95]+b[88:95], r[96:103] = a[96:103]+b[96:103], r[104:111] = a[104:111]+b[104:111], r[112:119] = a[112:119]+b[112:119], r[120:127] = a[120:127]+b[120:127]
    m128i = _mm_add_epi16(m128i, m128i);                            // r[0:15] = a[0:15]+b[0:15], r[16:31] = a[16:31]+b[16:31], r[32:47] = a[32:47]+b[32:47], r[48:63] = a[48:63]+b[48:63], r[64:79] = a[64:79]+b[64:79], r[80:95] = a[80:95]+b[80:95], r[96:111] = a[96:111]+b[96:111], r[112:127] = a[112:127]+b[112:127]
    m128i = _mm_add_epi32(m128i, m128i);                            // r[0:31] = a[0:31]+b[0:31], r[32:63] = a[32:63]+b[32:63], r[64:95] = a[64:95]+b[64:95], r[96:127] = a[96:127]+b[96:127]
    //m64 = _mm_add_si64(m64, m64);                                   // r[0:63] = a[0:63]+b[0:63]
    m128i = _mm_add_epi64(m128i, m128i);                            // r[0:63] = a[0:63]+b[0:63], r[64:127] = a[64:127]+b[64:127]
    m128i = _mm_adds_epi8(m128i, m128i);                            // r[0:7] = (Saturate_To_Int8)a[0:7]+b[0:7], r[8:15] = (Saturate_To_Int8)a[8:15]+b[8:15], r[16:23] = (Saturate_To_Int8)a[16:23]+b[16:23], r[24:31] = (Saturate_To_Int8)a[24:31]+b[24:31], r[32:39] = (Saturate_To_Int8)a[32:39]+b[32:39], r[40:47] = (Saturate_To_Int8)a[40:47]+b[40:47], r[48:55] = (Saturate_To_Int8)a[48:55]+b[48:55], r[56:63] = (Saturate_To_Int8)a[56:63]+b[56:63], r[64:71] = (Saturate_To_Int8)a[64:71]+b[64:71], r[72:79] = (Saturate_To_Int8)a[72:79]+b[72:79], r[80:87] = (Saturate_To_Int8)a[80:87]+b[80:87], r[88:95] = (Saturate_To_Int8)a[88:95]+b[88:95], r[96:103] = (Saturate_To_Int8)a[96:103]+b[96:103], r[104:111] = (Saturate_To_Int8)a[104:111]+b[104:111], r[112:119] = (Saturate_To_Int8)a[112:119]+b[112:119], r[120:127] = (Saturate_To_Int8)a[120:127]+b[120:127]
    m128i = _mm_adds_epi16(m128i, m128i);                           // r[0:15] = (Saturate_To_Int16)a[0:15]+b[0:15], r[16:31] = (Saturate_To_Int16)a[16:31]+b[16:31], r[32:47] = (Saturate_To_Int16)a[32:47]+b[32:47], r[48:63] = (Saturate_To_Int16)a[48:63]+b[48:63], r[64:79] = (Saturate_To_Int16)a[64:79]+b[64:79], r[80:95] = (Saturate_To_Int16)a[80:95]+b[80:95], r[96:111] = (Saturate_To_Int16)a[96:111]+b[96:111], r[112:127] = (Saturate_To_Int16)a[112:127]+b[112:127]
    m128i = _mm_adds_epu8(m128i, m128i);                            // r[0:7] = (Saturate_To_UnsignedInt8)a[0:7]+b[0:7], r[8:15] = (Saturate_To_UnsignedInt8)a[8:15]+b[8:15], r[16:23] = (Saturate_To_UnsignedInt8)a[16:23]+b[16:23], r[24:31] = (Saturate_To_UnsignedInt8)a[24:31]+b[24:31], r[32:39] = (Saturate_To_UnsignedInt8)a[32:39]+b[32:39], r[40:47] = (Saturate_To_UnsignedInt8)a[40:47]+b[40:47], r[48:55] = (Saturate_To_UnsignedInt8)a[48:55]+b[48:55], r[56:63] = (Saturate_To_UnsignedInt8)a[56:63]+b[56:63], r[64:71] = (Saturate_To_UnsignedInt8)a[64:71]+b[64:71], r[72:79] = (Saturate_To_UnsignedInt8)a[72:79]+b[72:79], r[80:87] = (Saturate_To_UnsignedInt8)a[80:87]+b[80:87], r[88:95] = (Saturate_To_UnsignedInt8)a[88:95]+b[88:95], r[96:103] = (Saturate_To_UnsignedInt8)a[96:103]+b[96:103], r[104:111] = (Saturate_To_UnsignedInt8)a[104:111]+b[104:111], r[112:119] = (Saturate_To_UnsignedInt8)a[112:119]+b[112:119], r[120:127] = (Saturate_To_UnsignedInt8)a[120:127]+b[120:127]
    m128i = _mm_adds_epu16(m128i, m128i);                           // r[0:15] = (Saturate_To_UnsignedInt16)a[0:15]+b[0:15], r[16:31] = (Saturate_To_UnsignedInt16)a[16:31]+b[16:31], r[32:47] = (Saturate_To_UnsignedInt16)a[32:47]+b[32:47], r[48:63] = (Saturate_To_UnsignedInt16)a[48:63]+b[48:63], r[64:79] = (Saturate_To_UnsignedInt16)a[64:79]+b[64:79], r[80:95] = (Saturate_To_UnsignedInt16)a[80:95]+b[80:95], r[96:111] = (Saturate_To_UnsignedInt16)a[96:111]+b[96:111], r[112:127] = (Saturate_To_UnsignedInt16)a[112:127]+b[112:127]
    m128i = _mm_sub_epi8(m128i, m128i);                             // r[0:7] = a[0:7]-b[0:7], r[8:15] = a[8:15]-b[8:15], r[16:23] = a[16:23]-b[16:23], r[24:31] = a[24:31]-b[24:31], r[32:39] = a[32:39]-b[32:39], r[40:47] = a[40:47]-b[40:47], r[48:55] = a[48:55]-b[48:55], r[56:63] = a[56:63]-b[56:63], r[64:71] = a[64:71]-b[64:71], r[72:79] = a[72:79]-b[72:79], r[80:87] = a[80:87]-b[80:87], r[88:95] = a[88:95]-b[88:95], r[96:103] = a[96:103]-b[96:103], r[104:111] = a[104:111]-b[104:111], r[112:119] = a[112:119]-b[112:119], r[120:127] = a[120:127]-b[120:127]
    m128i = _mm_sub_epi16(m128i, m128i);                            // r[0:15] = a[0:15]-b[0:15], r[16:31] = a[16:31]-b[16:31], r[32:47] = a[32:47]-b[32:47], r[48:63] = a[48:63]-b[48:63], r[64:79] = a[64:79]-b[64:79], r[80:95] = a[80:95]-b[80:95], r[96:111] = a[96:111]-b[96:111], r[112:127] = a[112:127]-b[112:127]
    m128i = _mm_sub_epi32(m128i, m128i);                            // r[0:31] = a[0:31]-b[0:31], r[32:63] = a[32:63]-b[32:63], r[64:95] = a[64:95]-b[64:95], r[96:127] = a[96:127]-b[96:127]
    m128i = _mm_sub_epi64(m128i, m128i);                            // r[0:63] = a[0:63]-b[0:63], r[64:127] = a[64:127]-b[64:127]
    m128i = _mm_subs_epi8(m128i, m128i);                            // r[0:7] = (Saturate_To_Int8)a[0:7]-b[0:7], r[8:15] = (Saturate_To_Int8)a[8:15]-b[8:15], r[16:23] = (Saturate_To_Int8)a[16:23]-b[16:23], r[24:31] = (Saturate_To_Int8)a[24:31]-b[24:31], r[32:39] = (Saturate_To_Int8)a[32:39]-b[32:39], r[40:47] = (Saturate_To_Int8)a[40:47]-b[40:47], r[48:55] = (Saturate_To_Int8)a[48:55]-b[48:55], r[56:63] = (Saturate_To_Int8)a[56:63]-b[56:63], r[64:71] = (Saturate_To_Int8)a[64:71]-b[64:71], r[72:79] = (Saturate_To_Int8)a[72:79]-b[72:79], r[80:87] = (Saturate_To_Int8)a[80:87]-b[80:87], r[88:95] = (Saturate_To_Int8)a[88:95]-b[88:95], r[96:103] = (Saturate_To_Int8)a[96:103]-b[96:103], r[104:111] = (Saturate_To_Int8)a[104:111]-b[104:111], r[112:119] = (Saturate_To_Int8)a[112:119]-b[112:119], r[120:127] = (Saturate_To_Int8)a[120:127]-b[120:127]
    m128i = _mm_subs_epi16(m128i, m128i);                           // r[0:15] = (Saturate_To_Int16)a[0:15]-b[0:15], r[16:31] = (Saturate_To_Int16)a[16:31]-b[16:31], r[32:47] = (Saturate_To_Int16)a[32:47]-b[32:47], r[48:63] = (Saturate_To_Int16)a[48:63]-b[48:63], r[64:79] = (Saturate_To_Int16)a[64:79]-b[64:79], r[80:95] = (Saturate_To_Int16)a[80:95]-b[80:95], r[96:111] = (Saturate_To_Int16)a[96:111]-b[96:111], r[112:127] = (Saturate_To_Int16)a[112:127]-b[112:127]
    m128i = _mm_subs_epu8(m128i, m128i);                            // r[0:7] = (Saturate_To_UnsignedInt8)a[0:7]-b[0:7], r[8:15] = (Saturate_To_UnsignedInt8)a[8:15]-b[8:15], r[16:23] = (Saturate_To_UnsignedInt8)a[16:23]-b[16:23], r[24:31] = (Saturate_To_UnsignedInt8)a[24:31]-b[24:31], r[32:39] = (Saturate_To_UnsignedInt8)a[32:39]-b[32:39], r[40:47] = (Saturate_To_UnsignedInt8)a[40:47]-b[40:47], r[48:55] = (Saturate_To_UnsignedInt8)a[48:55]-b[48:55], r[56:63] = (Saturate_To_UnsignedInt8)a[56:63]-b[56:63], r[64:71] = (Saturate_To_UnsignedInt8)a[64:71]-b[64:71], r[72:79] = (Saturate_To_UnsignedInt8)a[72:79]-b[72:79], r[80:87] = (Saturate_To_UnsignedInt8)a[80:87]-b[80:87], r[88:95] = (Saturate_To_UnsignedInt8)a[88:95]-b[88:95], r[96:103] = (Saturate_To_UnsignedInt8)a[96:103]-b[96:103], r[104:111] = (Saturate_To_UnsignedInt8)a[104:111]-b[104:111], r[112:119] = (Saturate_To_UnsignedInt8)a[112:119]-b[112:119], r[120:127] = (Saturate_To_UnsignedInt8)a[120:127]-b[120:127]
    m128i = _mm_subs_epu16(m128i, m128i);                           // r[0:15] = (Saturate_To_UnsignedInt16)a[0:15]-b[0:15], r[16:31] = (Saturate_To_UnsignedInt16)a[16:31]-b[16:31], r[32:47] = (Saturate_To_UnsignedInt16)a[32:47]-b[32:47], r[48:63] = (Saturate_To_UnsignedInt16)a[48:63]-b[48:63], r[64:79] = (Saturate_To_UnsignedInt16)a[64:79]-b[64:79], r[80:95] = (Saturate_To_UnsignedInt16)a[80:95]-b[80:95], r[96:111] = (Saturate_To_UnsignedInt16)a[96:111]-b[96:111], r[112:127] = (Saturate_To_UnsignedInt16)a[112:127]-b[112:127]
    m128i = _mm_madd_epi16(m128i, m128i);                           // r[0:31] = a[16:31]*b[16:31]+a[0:15]*b[0:15], r[32:63] = a[48:63]*b[48:63]+a[32:47]*b[32:47], r[64:95] = a[80:95]*b[80:95]+a[64:79]*b[64:79], r[96:127] = a[112:127]*b[112:127]+a[96:111]*b[96:111]
    m128i = _mm_mulhi_epi16(m128i, m128i);                          // r[0:15] = (a[0:15]*b[0:15])[16:31], r[16:31] = (a[16:31]*b[16:31])[16:31], r[32:47] = (a[32:47]*b[32:47])[16:31], r[48:63] = (a[48:63]*b[48:63])[16:31], r[64:79] = (a[64:79]*b[64:79])[16:31], r[80:95] = (a[80:95]*b[80:95])[16:31], r[96:111] = (a[96:111]*b[96:111])[16:31], r[112:127] = (a[112:127]*b[112:127])[16:31]
    m128i = _mm_mullo_epi16(m128i, m128i);                          // r[0:15] = (a[0:15]*b[0:15])[0:15], r[16:31] = (a[16:31]*b[16:31])[0:15], r[32:47] = (a[32:47]*b[32:47])[0:15], r[48:63] = (a[48:63]*b[48:63])[0:15], r[64:79] = (a[64:79]*b[64:79])[0:15], r[80:95] = (a[80:95]*b[80:95])[0:15], r[96:111] = (a[96:111]*b[96:111])[0:15], r[112:127] = (a[112:127]*b[112:127])[0:15]
    //m64 = _mm_mul_su32(m64, m64);                                   // r[0:63] = a[0:31]*b[0:31]
    m128i = _mm_mul_epu32(m128i, m128i);                            // r[0:63] = a[0:31]*b[0:31], r[64:127] = a[64:95]*b[64:95]
    m128i = _mm_max_epi16(m128i, m128i);                            // r[0:15] = max(a[0:15], b[0:15]), r[16:31] = max(a[16:31], b[16:31]), r[32:47] = max(a[32:47], b[32:47]), r[48:63] = max(a[48:63], b[48:63]), r[64:79] = max(a[64:79], b[64:79]), r[80:95] = max(a[80:95], b[80:95]), r[96:111] = max(a[96:111], b[96:111]), r[112:127] = max(a[112:127], b[112:127])
    m128i = _mm_max_epu8(m128i, m128i);                             // r[0:7] = max(a[0:7], b[0:7]), r[8:15] = max(a[8:15], b[8:15]), r[16:23] = max(a[16:23], b[16:23]), r[24:31] = max(a[24:31], b[24:31]), r[32:39] = max(a[32:39], b[32:39]), r[40:47] = max(a[40:47], b[40:47]), r[48:55] = max(a[48:55], b[48:55]), r[56:63] = max(a[56:63], b[56:63]), r[64:71] = max(a[64:71], b[64:71]), r[72:79] = max(a[72:79], b[72:79]), r[80:87] = max(a[80:87], b[80:87]), r[88:95] = max(a[88:95], b[88:95]), r[96:103] = max(a[96:103], b[96:103]), r[104:111] = max(a[104:111], b[104:111]), r[112:119] = max(a[112:119], b[112:119]), r[120:127] = max(a[120:127], b[120:127])
    m128i = _mm_min_epi16(m128i, m128i);                            // r[0:15] = min(a[0:15], b[0:15]), r[16:31] = min(a[16:31], b[16:31]), r[32:47] = min(a[32:47], b[32:47]), r[48:63] = min(a[48:63], b[48:63]), r[64:79] = min(a[64:79], b[64:79]), r[80:95] = min(a[80:95], b[80:95]), r[96:111] = min(a[96:111], b[96:111]), r[112:127] = min(a[112:127], b[112:127])
    m128i = _mm_min_epu8(m128i, m128i);                             // r[0:7] = min(a[0:7], b[0:7]), r[8:15] = min(a[8:15], b[8:15]), r[16:23] = min(a[16:23], b[16:23]), r[24:31] = min(a[24:31], b[24:31]), r[32:39] = min(a[32:39], b[32:39]), r[40:47] = min(a[40:47], b[40:47]), r[48:55] = min(a[48:55], b[48:55]), r[56:63] = min(a[56:63], b[56:63]), r[64:71] = min(a[64:71], b[64:71]), r[72:79] = min(a[72:79], b[72:79]), r[80:87] = min(a[80:87], b[80:87]), r[88:95] = min(a[88:95], b[88:95]), r[96:103] = min(a[96:103], b[96:103]), r[104:111] = min(a[104:111], b[104:111]), r[112:119] = min(a[112:119], b[112:119]), r[120:127] = min(a[120:127], b[120:127])
    m128i = _mm_avg_epu16(m128i, m128i);                            // r[0:15] = (a[0:15]+b[0:15]+1)>>1, r[16:31] = (a[16:31]+b[16:31]+1)>>1, r[32:47] = (a[32:47]+b[32:47]+1)>>1, r[48:63] = (a[48:63]+b[48:63]+1)>>1, r[64:79] = (a[64:79]+b[64:79]+1)>>1, r[80:95] = (a[80:95]+b[80:95]+1)>>1, r[96:111] = (a[96:111]+b[96:111]+1)>>1, r[112:127] = (a[112:127]+b[112:127]+1)>>1
    m128i = _mm_avg_epu8(m128i, m128i);                             // r[0:7] = (a[0:7]+b[0:7]+1)>>1, r[8:15] = (a[8:15]+b[8:15]+1)>>1, r[16:23] = (a[16:23]+b[16:23]+1)>>1, r[24:31] = (a[24:31]+b[24:31]+1)>>1, r[32:39] = (a[32:39]+b[32:39]+1)>>1, r[40:47] = (a[40:47]+b[40:47]+1)>>1, r[48:55] = (a[48:55]+b[48:55]+1)>>1, r[56:63] = (a[56:63]+b[56:63]+1)>>1, r[64:71] = (a[64:71]+b[64:71]+1)>>1, r[72:79] = (a[72:79]+b[72:79]+1)>>1, r[80:87] = (a[80:87]+b[80:87]+1)>>1, r[88:95] = (a[88:95]+b[88:95]+1)>>1, r[96:103] = (a[96:103]+b[96:103]+1)>>1, r[104:111] = (a[104:111]+b[104:111]+1)>>1, r[112:119] = (a[112:119]+b[112:119]+1)>>1, r[120:127] = (a[120:127]+b[120:127]+1)>>1

    /*
    * Int8/int16/int32/int64 logical
    */
    m128i = _mm_and_si128(m128i, m128i);                            // r[0:127] = a[0:127]&b[0:127]
    m128i = _mm_andnot_si128(m128i, m128i);                         // r[0:127] = ~a[0:127]&b[0:127]
    m128i = _mm_or_si128(m128i, m128i);                             // r[0:127] = a[0:127]|b[0:127]
    m128i = _mm_xor_si128(m128i, m128i);                            // r[0:127] = a[0:127]^b[0:127]

    /*
     * Int8/int16/int32/int64 comparison
     */
    m128i = _mm_cmpeq_epi8(m128i, m128i);                           // r[0:7] = a[0:7]==b[0:7], r[8:15] = a[8:15]==b[8:15], r[16:23] = a[16:23]==b[16:23], r[24:31] = a[24:31]==b[24:31], r[32:39] = a[32:39]==b[32:39], r[40:47] = a[40:47]==b[40:47], r[48:55] = a[48:55]==b[48:55], r[56:63] = a[56:63]==b[56:63], r[64:71] = a[64:71]==b[64:71], r[72:79] = a[72:79]==b[72:79], r[80:87] = a[80:87]==b[80:87], r[88:95] = a[88:95]==b[88:95], r[96:103] = a[96:103]==b[96:103], r[104:111] = a[104:111]==b[104:111], r[112:119] = a[112:119]==b[112:119], r[120:127] = a[120:127]==b[120:127]
    m128i = _mm_cmpeq_epi16(m128i, m128i);                          // r[0:15] = a[0:15]==b[0:15], r[16:31] = a[16:31]==b[16:31], r[32:47] = a[32:47]==b[32:47], r[48:63] = a[48:63]==b[48:63], r[64:79] = a[64:79]==b[64:79], r[80:95] = a[80:95]==b[80:95], r[96:111] = a[96:111]==b[96:111], r[112:127] = a[112:127]==b[112:127]
    m128i = _mm_cmpeq_epi32(m128i, m128i);                          // r[0:31] = a[0:31]==b[0:31], r[32:63] = a[32:63]==b[32:63], r[64:95] = a[64:95]==b[64:95], r[96:127] = a[96:127]==b[96:127]
    m128i = _mm_cmplt_epi8(m128i, m128i);                           // r[0:7] = a[0:7]<b[0:7], r[8:15] = a[8:15]<b[8:15], r[16:23] = a[16:23]<b[16:23], r[24:31] = a[24:31]<b[24:31], r[32:39] = a[32:39]<b[32:39], r[40:47] = a[40:47]<b[40:47], r[48:55] = a[48:55]<b[48:55], r[56:63] = a[56:63]<b[56:63], r[64:71] = a[64:71]<b[64:71], r[72:79] = a[72:79]<b[72:79], r[80:87] = a[80:87]<b[80:87], r[88:95] = a[88:95]<b[88:95], r[96:103] = a[96:103]<b[96:103], r[104:111] = a[104:111]<b[104:111], r[112:119] = a[112:119]<b[112:119], r[120:127] = a[120:127]<b[120:127]
    m128i = _mm_cmplt_epi16(m128i, m128i);                          // r[0:15] = a[0:15]<b[0:15], r[16:31] = a[16:31]<b[16:31], r[32:47] = a[32:47]<b[32:47], r[48:63] = a[48:63]<b[48:63], r[64:79] = a[64:79]<b[64:79], r[80:95] = a[80:95]<b[80:95], r[96:111] = a[96:111]<b[96:111], r[112:127] = a[112:127]<b[112:127]
    m128i = _mm_cmplt_epi32(m128i, m128i);                          // r[0:31] = a[0:31]<b[0:31], r[32:63] = a[32:63]<b[32:63], r[64:95] = a[64:95]<b[64:95], r[96:127] = a[96:127]<b[96:127]
    m128i = _mm_cmpgt_epi8(m128i, m128i);                           // r[0:7] = a[0:7]>b[0:7], r[8:15] = a[8:15]>b[8:15], r[16:23] = a[16:23]>b[16:23], r[24:31] = a[24:31]>b[24:31], r[32:39] = a[32:39]>b[32:39], r[40:47] = a[40:47]>b[40:47], r[48:55] = a[48:55]>b[48:55], r[56:63] = a[56:63]>b[56:63], r[64:71] = a[64:71]>b[64:71], r[72:79] = a[72:79]>b[72:79], r[80:87] = a[80:87]>b[80:87], r[88:95] = a[88:95]>b[88:95], r[96:103] = a[96:103]>b[96:103], r[104:111] = a[104:111]>b[104:111], r[112:119] = a[112:119]>b[112:119], r[120:127] = a[120:127]>b[120:127]
    m128i = _mm_cmpgt_epi16(m128i, m128i);                          // r[0:15] = a[0:15]>b[0:15], r[16:31] = a[16:31]>b[16:31], r[32:47] = a[32:47]>b[32:47], r[48:63] = a[48:63]>b[48:63], r[64:79] = a[64:79]>b[64:79], r[80:95] = a[80:95]>b[80:95], r[96:111] = a[96:111]>b[96:111], r[112:127] = a[112:127]>b[112:127]
    m128i = _mm_cmpgt_epi32(m128i, m128i);                          // r[0:31] = a[0:31]>b[0:31], r[32:63] = a[32:63]>b[32:63], r[64:95] = a[64:95]>b[64:95], r[96:127] = a[96:127]>b[96:127]

    /*
     * Int8/int16/int32/int64 cast
     */
    m128 = _mm_castpd_ps(m128d);                                    // r[0:127] = (__m128)a[0:127]
    m128i = _mm_castpd_si128(m128d);                                // r[0:127] = (__m128i)a[0:127]
    m128d = _mm_castps_pd(m128);                                    // r[0:127] = (__m128d)a[0:127]
    m128i = _mm_castps_si128(m128);                                 // r[0:127] = (__m128i)a[0:127]
    m128 = _mm_castsi128_ps(m128i);                                 // r[0:127] = (__m128)a[0:127]
    m128d = _mm_castsi128_pd(m128i);                                // r[0:127] = (__m128da[0:127]
}


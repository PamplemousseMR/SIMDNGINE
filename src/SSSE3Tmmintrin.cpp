#include "Simd.hpp"

#include <tmmintrin.h>

/*  tmmintrin
SSSE3
Again a varied set of instructions, mostly integer.
The first shuffle that takes its control operand from a register instead of hard-coded (pshufb).
More horizontal processing, shuffle, packing/unpacking, mul+add on bytes, and some specialized integer add/mul stuff.
*/
void tmmintrin()
{
    __m128i m128i = _mm_setzero_si128();

    m128i = _mm_abs_epi8(m128i);                                    // r[0:7] = abs(a[0:7]), r[8:15] = abs(a[8:15]), r[16:23] = abs(a[16:23]), r[24:31] = abs(a[24:31]), r[32:39] = abs(a[32:39]), r[40:47] = abs(a[40:47]), r[48:55] = abs(a[48:55]), r[56:63] = abs(a[56:63]), r[64:71] = abs(a[64:71]), r[72:79] = abs(a[72:79]), r[80:87] = abs(a[80:87]), r[88:95] = abs(a[88:95]), r[96:103] = abs(a[96:103]), r[104:111] = abs(a[104:111]), r[112:119] = abs(a[112:119]), r[120:127] = abs(a[120:127])
    m128i = _mm_abs_epi16(m128i);                                   // r[0:15] = abs(a[0:15]), r[16:31] = abs(a[16:31]), r[32:47] = abs(a[32:47]), r[48:63] = abs(a[48:63]), r[64:79] = abs(a[64:79]), r[80:95] = abs(a[80:95]), r[96:111] = abs(a[96:111]), r[112:127] = abs(a[112:127])
    m128i = _mm_abs_epi32(m128i);                                   // r[0:31] = abs(a[0:31]), r[32:63] = abs(a[32:63]), r[64:95] = abs(a[64:95]), r[96:127] = abs(a[96:127])
    //_mm_alignr_epi8(a, b, n);                                     //
    //_mm_alignr_pi8(a, b, n);                                      //
    m128i = _mm_hadd_epi16(m128i, m128i);                           // r[0:15] = a[16:31]+a[0:15], r[16:31] = a[48:63]+a[32:47], r[32:47] = a[80:95]+a[64:79], r[48:63] = a[112:127]+a[96:111], r[64:79] = b[16:31]+b[0:15], r[80:95] = b[48:63]+b[32:47], r[96:111] = b[80:95]+b[64:79], r[112:127] = b[112:127]+b[96:111]
    m128i = _mm_hadd_epi32(m128i, m128i);                           // r[0:31] = a[32:63]+a[0:31], r[32:63] = a[96:127]+a[64:95], r[64:95] = b[32:63]+b[0:31], r[96:127] = b[96:127]+b[64:95]
    m128i = _mm_hadds_epi16(m128i, m128i);                          // r[0:15] = (Saturate_To_Int16)(a[16:31]+a[0:15]), r[16:31] = (Saturate_To_Int16)(a[48:63]+a[32:47]), r[32:47] = (Saturate_To_Int16)(a[80:95]+a[64:79]), r[48:63] = (Saturate_To_Int16)(a[112:127]+a[96:111]), r[64:79] = (Saturate_To_Int16)(b[16:31]+b[0:15]), r[80:95] = (Saturate_To_Int16)(b[48:63]+b[32:47]), r[96:111] = (Saturate_To_Int16)(b[80:95]+b[64:79]), r[112:127] = (Saturate_To_Int16)(b[112:127]+b[96:111])
    m128i = _mm_hsub_epi16(m128i, m128i);                           // r[0:15] = a[0:15]-a[16:31], r[16:31] = a[32:47]-a[48:63], r[32:47] = a[64:79]-a[80:95], r[48:63] = a[96:111]-a[112:127], r[64:79] = b[0:15]-b[16:31], r[80:95] = b[32:47]-b[48:63], r[96:111] = b[64:79]-b[80:95], r[112:127] = b[96:111]-b[112:127]
    m128i = _mm_hsub_epi32(m128i, m128i);                           // r[0:31] = a[0:31]-a[32:63], r[32:63] = a[64:95]-a[96:127], r[64:95] = b[0:31]-b[32:63], r[96:127] = b[64:95]-b[96:127]
    m128i = _mm_hsubs_epi16(m128i, m128i);                          // r[0:15] = (Saturate_To_Int16)(a[0:15]-a[16:31]), r[16:31] = (Saturate_To_Int16)(a[32:47]-a[48:63]), r[32:47] = (Saturate_To_Int16)(a[64:79]-a[80:95]), r[48:63] = (Saturate_To_Int16)(a[96:111]-a[112:127]), r[64:79] = (Saturate_To_Int16)(b[0:15]-b[16:31]), r[80:95] = (Saturate_To_Int16)(b[32:47]-b[48:63]), r[96:111] = (Saturate_To_Int16)(b[64:79]-b[80:95]), r[112:127] = (Saturate_To_Int16)(b[96:111]-b[112:127])
    m128i = _mm_maddubs_epi16(m128i, m128i);                        // r[0:15] = (Saturate_To_Int16)(a[8:15]*b[8:15]+a[0:7]*b[0:7]), r[16:31] = (Saturate_To_Int16)(a[24:31]*b[24:31]+a[16:23]*b[16:23]), r[32:47] = (Saturate_To_Int16)(a[40:47]*b[40:47]+a[32:39]*b[32:39]), r[48:63] = (Saturate_To_Int16)(a[56:63]*b[56:63]+a[48:55]*b[48:55]), r[64:79] = (Saturate_To_Int16)(a[64:79]*b[64:79]+a[63:71]*b[63:71]), r[80:95] = (Saturate_To_Int16)(a[88:95]*b[88:95]+a[80:87]*b[80:87]), r[96:111] = (Saturate_To_Int16)(a[104:111]*b[104:111]+a[96:103]*b[96:103]), r[112:127] = (Saturate_To_Int16)(a[120:127]*b[120:127]+a[112:119]*b[112:119])
    m128i = _mm_mulhrs_epi16(m128i, m128i);                         //
    m128i = _mm_shuffle_epi8(m128i, m128i);                         //
    m128i = _mm_sign_epi8(m128i, m128i);                            //
    m128i = _mm_sign_epi16(m128i, m128i);                           //
    m128i = _mm_sign_epi32(m128i, m128i);                           //

#if defined(_M_IX86) || defined(__unix)
    __m64 m64 = _mm_setzero_si64();

    m64 = _mm_abs_pi8(m64);                                         // r[0:7] = abs(a[0:7]), r[8:15] = abs(a[8:15]), r[16:23] = abs(a[16:23]), r[24:31] = abs(a[24:31]), r[32:39] = abs(a[32:39]), r[40:47] = abs(a[40:47]), r[48:55] = abs(a[48:55]), r[56:63] = abs(a[56:63])
    m64 = _mm_abs_pi16(m64);                                        // r[0:15] = abs(a[0:15]), r[16:31] = abs(a[16:31]), r[32:47] = abs(a[32:47]), r[48:63] = abs(a[48:63])
    m64 = _mm_abs_pi32(m64);                                        // r[0:31] = abs(a[0:31]), r[32:63] = abs(a[32:63])
    m64 = _mm_hadd_pi16(m64, m64);                                  // r[0:15] = a[16:31]+a[0:15], r[16:31] = a[48:63]+a[32:47], r[32:47] = b[16:31]+b[0:15], r[48:63] = b[48:63]+b[32:47]
    m64 = _mm_hadd_pi32(m64, m64);                                  // r[0:31] = a[32:63]+a[0:31], r[32:63] = b[32:63]+b[0:31]
    m64 = _mm_hadds_pi16(m64, m64);                                 // r[0:15] = (Saturate_To_Int16)(a[16:31]+a[0:15]), r[16:31] = (Saturate_To_Int16)(a[48:63]+a[32:47]), r[32:47] = (Saturate_To_Int16)(b[16:31]+b[0:15]), r[48:63] = (Saturate_To_Int16)(b[48:63]+b[32:47])
    m64 = _mm_hsub_pi16(m64, m64);                                  // r[0:15] = a[0:15]-a[16:31], r[16:31] = a[32:47]-a[48:63], r[32:47] = b[0:15]-b[16:31], r[48:63] = b[32:47]-b[48:63]
    m64 = _mm_hsub_pi32(m64, m64);                                  // r[0:31] = a[0:31]-a[32:63], r[32:63] = b[0:31]-b[32:63]
    m64 = _mm_hsubs_pi16(m64, m64);                                 // r[0:15] = (Saturate_To_Int16)(a[0:15]-a[16:31]), r[16:31] = (Saturate_To_Int16)(a[32:47]-a[48:63]), r[32:47] = (Saturate_To_Int16)(b[0:15]-b[16:31]), r[48:63] = (Saturate_To_Int16)(b[32:47]-b[48:63])
    m64 = _mm_maddubs_pi16(m64, m64);                               // r[0:15] = (Saturate_To_Int16)(a[8:15]*b[8:15]+a[0:7]*b[0:7]), r[16:31] = (Saturate_To_Int16)(a[24:31]*b[24:31]+a[16:23]*b[16:23]), r[32:47] = (Saturate_To_Int16)(a[40:47]*b[40:47]+a[32:39]*b[32:39]), r[48:63] = (Saturate_To_Int16)(a[56:63]*b[56:63]+a[48:55]*b[48:55])
    m64 = _mm_mulhrs_pi16(m64, m64);                                //
    m64 = _mm_shuffle_pi8(m64, m64);                                //
    m64 = _mm_sign_pi8(m64, m64);                                   //
    m64 = _mm_sign_pi16(m64, m64);                                  //
    m64 = _mm_sign_pi32(m64, m64);                                  //
#endif
}
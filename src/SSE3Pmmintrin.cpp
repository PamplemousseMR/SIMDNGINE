#include "simd.hpp"

#include <pmmintrin.h>

/*  pmmintrin
SSE3
Add a few varied instructions (mostly floating point), including a special kind of unaligned load (lddqu) that was better on Pentium 4, synchronization instruction, horizontal add/sub.
*/
void pmmintrin()
{
    __m128d m128d = _mm_setzero_pd();
    __m128i m128i = _mm_setzero_si128();
    __m128 m128 = _mm_setzero_ps();
    double dou = 1.2;

    /*
    * Miscellaneous
    */
    m128i = _mm_lddqu_si128(&m128i);                                // r[0:127] = MEM[mem_addr:mem_addr+127]
    m128 = _mm_addsub_ps(m128, m128);                               // r[0:31] = a[0:31]-b[0:31], r[32:63] = a[32:63]+b[32:63], r[64:95] = a[64:95]-b[64:95], r[96:127] = a[96:127]+b[96:127]
    m128 = _mm_hadd_ps(m128, m128);                                 // r[0:31] = a[32:63]+a[0:31], r[32:63] = a[96:127]+a[64:95], r[64:95] = b[32:63]+b[0:31], r[96:127] = b[96:127]+b[64:95]
    m128 = _mm_hsub_ps(m128, m128);                                 // r[0:31] = a[32:63]-a[0:31], r[32:63] = a[96:127]-a[64:95], r[64:95] = b[32:63]-b[0:31], r[96:127] = b[96:127]-b[64:95]
    m128 = _mm_movehdup_ps(m128);                                   // r[0:31] = a[32:63], r[32:63] = a[32:63], r[64:95] = a[96:127] , r[96:127] = a[96:127]
    m128 = _mm_moveldup_ps(m128);                                   // r[0:31] = a[0:31], r[32:63] = a[0:31], r[64:95] = a[64:95] , r[96:127] = a[64:95]
    m128d = _mm_addsub_pd(m128d, m128d);                            // r[0:63] = a[0:63]-b[0:63], r[64:127] = a[64:127]+b[64:127]
    m128d = _mm_hadd_pd(m128d, m128d);                              // r[0:63] = a[64:127]+a[0:63], r[64:127] = b[64:127]+b[0:63]
    m128d = _mm_hsub_pd(m128d, m128d);                              // r[0:63] = a[64:127]-a[0:63], r[64:127] = b[64:127]-b[0:63]
    m128d = _mm_loaddup_pd(&dou);                                   // r[0:63] = MEM[mem_addr:mem_addr+63], r[64:127] = MEM[mem_addr:mem_addr+63]
    m128d = _mm_movedup_pd(m128d);                                  // r[0:63] = a[0:63], r[64:127] = a[0:63]
    //_mm_monitor(void const* __p, 0, 0);                             //
    //_mm_mwait(0, 0);                                                //
}

#include "Simd.hpp"

#include <xmmintrin.h>

/*  xmmintrin
SSE
Introduce eight/sixteen 128 bit registers (XMM0-XMM7/15) and instruction to work with four single precision floating point operands.
Add integer operations on MMX registers too.
The MMX-integer part of SSE is sometimes called MMXEXT, and was implemented on a few non-Intel CPUs without xmm registers and the floating point part of SSE.
*/
void xmmintrin()
{
    alignas(16) float arrFloat4[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    __m128 m128 = _mm_setzero_ps();
    __m64 m64;
    const char cc = 'a';
    float fl = 10.5f;
    int in = 10;
    unsigned ui = 10;

    /*
    * Memory & initialization
    */
    m128 = _mm_set_ss(1.5f);                                        // r[0:31] = w, r[32:127] = 0.0
    m128 = _mm_set_ps1(1.5f);                                       // r[0:127] = w
    m128 = _mm_set_ps(1.5f, 2.5f, 3.5f, 4.5f);                      // r[0:31] = w, r[32:63] = x, r[64:95] = y, r[96:127] = z
    m128 = _mm_setr_ps(1.5f, 2.5f, 3.5f, 4.5f);                     // r[0:31] = x, r[32:63] = y, r[64:95] = z, r[96:127] = w
    m128 = _mm_setzero_ps();                                        // r[0:127] = 0.0
    m128 = _mm_load_ss(&arrFloat4[0]);                              // r[0:31] = MEM[mem_addr:mem_addr+31], r[32:127] = 0.0
    m128 = _mm_load_ps1(&arrFloat4[0]);                             // r[0:31] = MEM[mem_addr:mem_addr+31], r[32:63] = MEM[mem_addr:mem_addr+31], r[64:95] = MEM[mem_addr:mem_addr+31], r[96:127] = MEM[mem_addr:mem_addr+31]
    m128 = _mm_load_ps(&arrFloat4[0]);                              // r[0:127] = MEM[mem_addr:mem_addr+127]
    m128 = _mm_loadr_ps(&arrFloat4[0]);                             // r[0:31] = MEM[mem_addr+96:mem_addr+127], r[32:63] = MEM[mem_addr+64:mem_addr+95], r[64:95] = MEM[mem_addr+32:mem_addr+63], r[96:127] = MEM[mem_addr:mem_addr+31]
    //m128 = _mm_undefined_ps();                                      // r[0:127] = undefined
    m128 = _mm_loadu_ps(&arrFloat4[0]);                             // r[0:127] = MEM[mem_addr:mem_addr+127]
    _mm_store_ss(arrFloat4, m128);                                  // p[0:31] = a[0:31]
    _mm_store_ps1(arrFloat4, m128);                                 // p[0:31] = a[0:31], p[32:63] = a[0:31], p[64:95] = a[0:31], p[96:127] = a[0:31]
    _mm_store_ps(arrFloat4, m128);                                  // p[0:127] = a[0:127]
    _mm_storer_ps(arrFloat4, m128);                                 // p[0:31] = a[96:127], p[32:63] = a[64:95], p[64:95] = a[32:63], p[96:127] = a[0:31]
    _mm_storeu_ps(arrFloat4, m128);                                 // p[0:127] = a[0:127]
    _mm_prefetch(&cc, _MM_HINT_T0);                                 //
    _mm_stream_ps(arrFloat4, m128);                                 // p[0:127] = a[0:127]
    m128 = _mm_move_ss(m128, m128);                                 // r[0:31] = b[0:31], r[32:127] = a[32:127]
    _mm_sfence();                                                   //
    ui = _mm_getcsr();                                              // r = MXCSR
    _mm_setcsr(ui);                                                 // MXCSR = w
    m128 = _mm_set1_ps(1.5f);                                       // r[0:31] = w, r[32:63] = w, r[64:95] = w, r[96:127] = w
    m128 = _mm_load1_ps(&arrFloat4[0]);                             // r[0:31] = MEM[mem_addr:mem_addr+31], r[32:63] = MEM[mem_addr:mem_addr+31], r[64:95] = MEM[mem_addr:mem_addr+31], r[96:127] = MEM[mem_addr:mem_addr+31]
    _mm_store1_ps(arrFloat4, m128);                                 // p[0:31] = a[0:31], p[32:63] = a[0:31], p[64:95] = a[0:31], p[96:127] = a[0:31]

    /*
    * Arithmetic
    */
    m128 = _mm_add_ss(m128, m128);                                  // r[0:31] = a[0:31]+b[0:31], r[32:63] = a[32:63], r[64:95] = a[64:95], r[96:127] = a[96:127]
    m128 = _mm_add_ps(m128, m128);                                  // r[0:31] = a[0:31]+b[0:31], r[32:63] = a[32:63]+b[32:63], r[64:95] = a[64:95]+b[32:63], r[96:127] = a[96:127]+b[64:127]
    m128 = _mm_sub_ss(m128, m128);                                  // r[0:31] = a[0:31]-b[0:31], r[32:63] = a[32:63], r[64:95] = a[64:95], r[96:127] = a[96:127]
    m128 = _mm_sub_ps(m128, m128);                                  // r[0:31] = a[0:31]-b[0:31], r[32:63] = a[32:63]-b[32:63], r[64:95] = a[64:95]-b[64:95], r[96:127] = a[96:127]-b[64:127]
    m128 = _mm_mul_ss(m128, m128);                                  // r[0:31] = a[0:31]*b[0:31], r[32:63] = a[32:63], r[64:95] = a[64:95], r[96:127] = a[96:127]
    m128 = _mm_mul_ps(m128, m128);                                  // r[0:31] = a[0:31]*b[0:31], r[32:63] = a[32:63]*b[32:63], r[64:95] = a[64:95]*b[64:95], r[96:127] = a[96:127]*b[64:127]
    m128 = _mm_div_ss(m128, m128);                                  // r[0:31] = a[0:31]/b[0:31], r[32:63] = a[32:63], r[64:95] = a[64:95], r[96:127] = a[96:127]
    m128 = _mm_div_ps(m128, m128);                                  // r[0:31] = a[0:31]/b[0:31], r[32:63] = a[32:63]/b[32:63], r[64:95] = a[64:95]/b[64:95], r[96:127] = a[96:127]/b[64:127]
    m128 = _mm_sqrt_ss(m128);                                       // r[0:31] = sqrt(a[0:31]), r[32:63] = a[32:63], r[64:95] = a[64:95], r[96:127] = a[96:127]
    m128 = _mm_sqrt_ps(m128);                                       // r[0:31] = sqrt(a[0:31]), r[32:63] = sqrt(a[32:63]), r[64:95] = sqrt(a[64:95]), r[96:127] = sqrt(a[96:127])
    m128 = _mm_rcp_ss(m128);                                        // r[0:31] = recip(a[0:31]), r[32:63] = a[32:63], r[64:95] = a[64:95], r[96:127] = a[96:127]
    m128 = _mm_rcp_ps(m128);                                        // r[0:31] = recip(a[0:31]), r[32:63] = recip(a[32:63]), r[64:95] = recip(a[64:95]), r[96:127] = recip(a[96:127])
    m128 = _mm_rsqrt_ss(m128);                                      // r[0:31] = recip(sqrt(a[0:31])), r[32:63] = a[32:63], r[64:95] = a[64:95], r[96:127] = a[96:127]
    m128 = _mm_rsqrt_ps(m128);                                      // r[0:31] = recip(sqrt(a[0:31])), r[32:63] = recip(sqrt(a[32:63])), r[64:95] = recip(sqrt(a[64:95])), r[96:127] = recip(sqrt(a[96:127]))
    m128 = _mm_min_ss(m128, m128);                                  // r[0:31] = min(a[0:31], b[0:31]), r[32:63] = a[32:63], r[64:95] = a[64:95], r[96:127] = a[96:127]
    m128 = _mm_min_ps(m128, m128);                                  // r[0:31] = min(a[0:31], b[0:31]), r[32:63] =  min(a[32:63], b[32:63]), r[64:95] =  min(a[64:95], b[64:95]), r[96:127] =  min(a[96:127], b[64:127])
    m128 = _mm_max_ss(m128, m128);                                  // r[0:31] = max(a[0:31], b[0:31]), r[32:63] = a[32:63], r[64:95] = a[64:95], r[96:127] = a[96:127]
    m128 = _mm_max_ps(m128, m128);                                  // r[0:31] = max(a[0:31], b[0:31]), r[32:63] =  max(a[32:63], b[32:63]), r[64:95] =  max(a[64:95], b[64:95]), r[96:127] =  max(a[96:127], b[64:127])

    /*
    * Logical
    */
    m128 = _mm_and_ps(m128, m128);                                  // r[0:31] = a[0:31] & b[0:31], r[32:63] = a[32:63] & b[32:63], r[64:95] = a[64:95] & b[64:95], r[96:127] = a[96:127] & b[64:127]
    m128 = _mm_andnot_ps(m128, m128);                               // r[0:31] = ~a[0:31] & b[0:31], r[32:63] = ~a[32:63] & b[32:63], r[64:95] = ~a[64:95] & b[64:95], r[96:127] = ~a[96:127] & b[64:127]
    m128 = _mm_or_ps(m128, m128);                                   // r[0:31] = a[0:31] | b[0:31], r[32:63] = a[32:63] | b[32:63], r[64:95] = a[64:95] | b[64:95], r[96:127] = a[96:127] | b[64:127]
    m128 = _mm_xor_ps(m128, m128);                                  // r[0:31] = a[0:31] ^ b[0:31], r[32:63] = a[32:63] ^ b[32:63], r[64:95] = a[64:95] ^ b[64:95], r[96:127] = a[96:127] ^ b[64:127]

    /*
    * Comparison
    */
    m128 = _mm_cmpeq_ss(m128, m128);                                // r[0:31] = (a[0:31] == b[0:31]) ? 0xF : 0x0, r[32:63] = a[32:63], r[64:95] = a[64:95], r[96:127] = a[96:127]
    m128 = _mm_cmpeq_ps(m128, m128);                                // r[0:31] = (a[0:31] == b[0:31]) ? 0xF : 0x0, r[32:63] = (a[32:63] == b[32:63]) ? 0xF : 0x0, r[64:95] = (a[64:95] == b[64:95]) ? 0xF : 0x0, r[96:127] = (a[96:127] == b[64:127]) ? 0xF : 0x0
    m128 = _mm_cmplt_ss(m128, m128);                                // r[0:31] = (a[0:31] < b[0:31]) ? 0xF : 0x0, r[32:63] = a[32:63], r[64:95] = a[64:95], r[96:127] = a[96:127]
    m128 = _mm_cmplt_ps(m128, m128);                                // r[0:31] = (a[0:31] < b[0:31]) ? 0xF : 0x0, r[32:63] = (a[32:63] < b[32:63]) ? 0xF : 0x0, r[64:95] = (a[64:95] < b[64:95]) ? 0xF : 0x0, r[96:127] = (a[96:127] < b[64:127]) ? 0xF : 0x0
    m128 = _mm_cmple_ss(m128, m128);                                // r[0:31] = (a[0:31] <= b[0:31]) ? 0xF : 0x0, r[32:63] = a[32:63], r[64:95] = a[64:95], r[96:127] = a[96:127]
    m128 = _mm_cmple_ps(m128, m128);                                // r[0:31] = (a[0:31] <= b[0:31]) ? 0xF : 0x0, r[32:63] = (a[32:63] <= b[32:63]) ? 0xF : 0x0, r[64:95] = (a[64:95] <= b[64:95]) ? 0xF : 0x0, r[96:127] = (a[96:127] <= b[64:127]) ? 0xF : 0x0
    m128 = _mm_cmpgt_ss(m128, m128);                                // r[0:31] = (a[0:31] > b[0:31]) ? 0xF : 0x0, r[32:63] = a[32:63], r[64:95] = a[64:95], r[96:127] = a[96:127]
    m128 = _mm_cmpgt_ps(m128, m128);                                // r[0:31] = (a[0:31] > b[0:31]) ? 0xF : 0x0, r[32:63] = (a[32:63] > b[32:63]) ? 0xF : 0x0, r[64:95] = (a[64:95] > b[64:95]) ? 0xF : 0x0, r[96:127] = (a[96:127] > b[64:127]) ? 0xF : 0x0
    m128 = _mm_cmpge_ss(m128, m128);                                // r[0:31] = (a[0:31] >= b[0:31]) ? 0xF : 0x0, r[32:63] = a[32:63], r[64:95] = a[64:95], r[96:127] = a[96:127]
    m128 = _mm_cmpge_ps(m128, m128);                                // r[0:31] = (a[0:31] >= b[0:31]) ? 0xF : 0x0, r[32:63] = (a[32:63] >= b[32:63]) ? 0xF : 0x0, r[64:95] = (a[64:95] >= b[64:95]) ? 0xF : 0x0, r[96:127] = (a[96:127] >= b[64:127]) ? 0xF : 0x0
    m128 = _mm_cmpneq_ss(m128, m128);                               // r[0:31] = (a[0:31] != b[0:31]) ? 0xF : 0x0, r[32:63] = a[32:63], r[64:95] = a[64:95], r[96:127] = a[96:127]
    m128 = _mm_cmpneq_ps(m128, m128);                               // r[0:31] = (a[0:31] != b[0:31]) ? 0xF : 0x0, r[32:63] = (a[32:63] != b[32:63]) ? 0xF : 0x0, r[64:95] = (a[64:95] != b[64:95]) ? 0xF : 0x0, r[96:127] = (a[96:127] != b[64:127]) ? 0xF : 0x0
    m128 = _mm_cmpnlt_ss(m128, m128);                               // r[0:31] = !(a[0:31] < b[0:31]) ? 0xF : 0x0, r[32:63] = a[32:63], r[64:95] = a[64:95], r[96:127] = a[96:127]
    m128 = _mm_cmpnlt_ps(m128, m128);                               // r[0:31] = !(a[0:31] < b[0:31]) ? 0xF : 0x0, r[32:63] = !(a[32:63] < b[32:63]) ? 0xF : 0x0, r[64:95] = !(a[64:95] < b[64:95]) ? 0xF : 0x0, r[96:127] = !(a[96:127] < b[64:127]) ? 0xF : 0x0
    m128 = _mm_cmpnle_ss(m128, m128);                               // r[0:31] = !(a[0:31] <= b[0:31]) ? 0xF : 0x0, r[32:63] = a[32:63], r[64:95] = a[64:95], r[96:127] = a[96:127]
    m128 = _mm_cmpnle_ps(m128, m128);                               // r[0:31] = !(a[0:31] <= b[0:31]) ? 0xF : 0x0, r[32:63] = !(a[32:63] <= b[32:63]) ? 0xF : 0x0, r[64:95] = !(a[64:95] <= b[64:95]) ? 0xF : 0x0, r[96:127] = !(a[96:127] <= b[64:127]) ? 0xF : 0x0
    m128 = _mm_cmpngt_ss(m128, m128);                               // r[0:31] = !(a[0:31] > b[0:31]) ? 0xF : 0x0, r[32:63] = a[32:63], r[64:95] = a[64:95], r[96:127] = a[96:127]
    m128 = _mm_cmpngt_ps(m128, m128);                               // r[0:31] = !(a[0:31] > b[0:31]) ? 0xF : 0x0, r[32:63] = !(a[32:63] > b[32:63]) ? 0xF : 0x0, r[64:95] = !(a[64:95] > b[64:95]) ? 0xF : 0x0, r[96:127] = !(a[96:127] > b[64:127]) ? 0xF : 0x0
    m128 = _mm_cmpnge_ss(m128, m128);                               // r[0:31] = !(a[0:31] >= b[0:31]) ? 0xF : 0x0, r[32:63] = a[32:63], r[64:95] = a[64:95], r[96:127] = a[96:127]
    m128 = _mm_cmpnge_ps(m128, m128);                               // r[0:31] = !(a[0:31] >= b[0:31]) ? 0xF : 0x0, r[32:63] = !(a[32:63] >= b[32:63]) ? 0xF : 0x0, r[64:95] = !(a[64:95] >= b[64:95]) ? 0xF : 0x0, r[96:127] = !(a[96:127] >= b[64:127]) ? 0xF : 0x0
    m128 = _mm_cmpord_ss(m128, m128);                               // r[0:31] = (a[0:31] ord? b[0:31]) ? 0xF : 0x0, r[32:63] = a[32:63], r[64:95] = a[64:95], r[96:127] = a[96:127]
    m128 = _mm_cmpord_ps(m128, m128);                               // r[0:31] = (a[0:31] ord? b[0:31]) ? 0xF : 0x0, r[32:63] = (a[32:63] ord? b[32:63]) ? 0xF : 0x0, r[64:95] = (a[64:95] ord? b[64:95]) ? 0xF : 0x0, r[96:127] = (a[96:127] ord? b[64:127]) ? 0xF : 0x0
    m128 = _mm_cmpunord_ss(m128, m128);                             // r[0:31] = (a[0:31] unord? b[0:31]) ? 0xF : 0x0, r[32:63] = a[32:63], r[64:95] = a[64:95], r[96:127] = a[96:127]
    m128 = _mm_cmpunord_ps(m128, m128);                             // r[0:31] = (a[0:31] unord? b[0:31]) ? 0xF : 0x0, r[32:63] = (a[32:63] unord? b[32:63]) ? 0xF : 0x0, r[64:95] = (a[64:95] unord? b[64:95]) ? 0xF : 0x0, r[96:127] = (a[96:127] unord? b[64:127]) ? 0xF : 0x0
    in = _mm_comieq_ss(m128, m128);                                 // r = (a[0:31] == b[0:31]) ? 1 : 0
    in = _mm_comilt_ss(m128, m128);                                 // r = (a[0:31] < b[0:31]) ? 1 : 0
    in = _mm_comile_ss(m128, m128);                                 // r = (a[0:31] <= b[0:31]) ? 1 : 0
    in = _mm_comigt_ss(m128, m128);                                 // r = (a[0:31] > b[0:31]) ? 1 : 0
    in = _mm_comige_ss(m128, m128);                                 // r = (a[0:31] >= b[0:31]) ? 1 : 0
    in = _mm_comineq_ss(m128, m128);                                // r = (a[0:31] != b[0:31]) ? 1 : 0
    in = _mm_ucomieq_ss(m128, m128);                                // r = (a[0:31] == b[0:31]) ? 1 : 0
    in = _mm_ucomilt_ss(m128, m128);                                // r = (a[0:31] < b[0:31]) ? 1 : 0
    in = _mm_ucomile_ss(m128, m128);                                // r = (a[0:31] <= b[0:31]) ? 1 : 0
    in = _mm_ucomigt_ss(m128, m128);                                // r = (a[0:31] > b[0:31]) ? 1 : 0
    in = _mm_ucomige_ss(m128, m128);                                // r = (a[0:31] >= b[0:31]) ? 1 : 0
    in = _mm_ucomineq_ss(m128, m128);                               // r = (a[0:31] != b[0:31]) ? 1 : 0

    /*
    * Conversions
    */
    in = _mm_cvt_ss2si(m128);                                       // r = (Convert_FP32_To_Int32)a[0:31]
    in = _mm_cvtt_ss2si(m128);                                      // r = (Convert_FP32_To_Int32_Truncate)a[0:31]
    m128 = _mm_cvt_si2ss(m128, in);                                 // r[0:31] = (Convert_Int32_To_FP32)b, r[32:63] = a[32:63], r[64:95] = a[64:95], r[96:127] = a[96:127]
    fl = _mm_cvtss_f32(m128);                                       // r = a[0:31]
    in = _mm_cvtss_si32(m128);                                      // r = (Convert_FP32_To_Int32)a[0:31]
    in = _mm_cvttss_si32(m128);                                     // r = (Convert_FP32_To_Int32_Truncate)a[0:31]
    m128 = _mm_cvtsi32_ss(m128, in);                                // r[0:31] = (Convert_Int32_To_FP32)b, r[32:63] = a[32:63], r[64:95] = a[64:95], r[96:127] = a[96:127]

    /*
    * Miscellaneous
    */
    m128 = _mm_shuffle_ps(m128, m128, _MM_SHUFFLE(0, 1, 2, 3));     // r[0:31] = a[imm8[6:7]*16:imm8[0:7]*16+15], r[32:63] = a[imm8[4:5]*16:imm8[4:5]*16+15], r[64:95] = b[imm8[2:3]*16:imm8[2:3]*16+15], r[96:127] = b[imm8[0:1]*16:imm8[0:1]*16+15]
    m128 = _mm_unpackhi_ps(m128, m128);                             // r[0:31] = a[64:95], r[32:63] = b[64:95], r[64:95] = a[96:127], r[96:127] = b[64:127]
    m128 = _mm_unpacklo_ps(m128, m128);                             // r[0:31] = a[0:31], r[32:63] = b[0:31], r[64:95] = a[32:63], r[96:127] = b[32:63]
    m128 = _mm_loadh_pi(m128, &m64);                                // r[0:31] = a[0:31], r[32:63] = a[32:63], r[64:95] = MEM[mem_addr:mem_addr+31], r[96:127] = MEM[mem_addr+32:mem_addr+63]
    m128 = _mm_movehl_ps(m128, m128);                               // r[0:31] = b[64:95], r[32:63] = b[64:127], r[64:95] = a[64:95], r[96:127] = a[96:127]
    m128 = _mm_movelh_ps(m128, m128);                               // r[0:31] = a[0:31], r[32:63] = a[32:63], r[64:95] = b[0:31], r[96:127] = b[32:63]
    _mm_storeh_pi(&m64, m128);                                      // MEM[mem_addr:mem_addr+31] = a[64:95], MEM[mem_addr+32:mem_addr+63] = a[96:127]
    m128 = _mm_loadl_pi(m128, &m64);                                // r[0:31] = MEM[mem_addr:mem_addr+31], r[32:63] = MEM[mem_addr+32:mem_addr+63], r[64:95] = a[64:95], r[96:127] = a[96:127]
    _mm_storel_pi(&m64, m128);                                      // MEM[mem_addr:mem_addr+31] = a[0:31], MEM[mem_addr+32:mem_addr+63] = a[32:63]
    in = _mm_movemask_ps(m128);                                     // r = sign(a[96:127])<<3 | sign(a[64:95])<<2 | sign(a[32:63])<<1 | sign(a[0:31])

#if defined(_M_X64) || defined(__x86_64__)
    long long i64;

    /*
    * Support for 64-bit intrinsics
    */
    i64 = _mm_cvtss_si64(m128);                                     // r = (Convert_FP32_To_Int64)a[0:31]
    i64 = _mm_cvttss_si64(m128);                                    // r = (Convert_FP64_To_Int32_Truncate)a[0:31]
    m128 = _mm_cvtsi64_ss(m128, i64);                               // r[0:31] = (Convert_Int64_To_FP32)b[0:63], r[32:127] = a[32:127]
#endif

    /*
    * Support for MMX extension intrinsics
    */
#if defined(_M_IX86) || defined(__unix)
    alignas(16) char arrChar4[4];

    /*
    * Conversions
    */
    m64 = _mm_cvt_ps2pi(m128);                                      // r[0:31] = (Convert_FP32_To_Int32)a[0:31], r[32:63] = (Convert_FP32_To_Int32)a[32:63]
    m64 = _mm_cvtt_ps2pi(m128);                                     // r[0:31] = (Convert_FP32_To_Int32_Truncate)a[0:31], r[32:63] = (Convert_FP32_To_Int32_Truncate)a[32:63]
    m128 = _mm_cvt_pi2ps(m128, m64);                                // r[0:31] = (Convert_Int32_To_FP32)b[0:31], r[32:63] = (Convert_Int32_To_FP32)b[32:63], r[64:95] = a[64:95], r[96:127] = a[96:127]

    /*
    * Integer (MMX) extensions
    */
    in = _m_pextrw(m64, 0);                                         // r[0:15] = (a[0:63] >> (imm8[0:1] * 16))[0:15], r[16:31] = 0
    m64 = _m_pinsrw(m64, in, 0);                                    // r[0:63] = a[0:63], r[imm8[0:1]*16:imm8[0:1]*16+15] = i[0:15]
    m64 = _m_pmaxsw(m64, m64);                                      // r[0:15] = max(a[0:15], b[0:15]), r[16:31] = max(a[16:31], b[16:31]), r[32:47] = max(a[32:47], b[32:47]), r[48:63] = max(a[48:63], b[48:63])
    m64 = _m_pmaxub(m64, m64);                                      // r[0:7] = max(a[0:7], b[0:7]), r[8:15] = max(a[8:15], b[8:15]), r[16:23] = max(a[16:23], b[16:23]), r[24:31] = max(a[24:31], b[24:31]), r[32:39] = max(a[32:39], b[32:39]), r[40:47] = max(a[40:47], b[40:47]), r[48:55] = max(a[48:55], b[48:55]), r[56:63] = max(a[56:63], b[56:63])
    m64 = _m_pminsw(m64, m64);                                      // r[0:15] = min(a[0:15], b[0:15]), r[16:31] = min(a[16:31], b[16:31]), r[32:47] = min(a[32:47], b[32:47]), r[48:63] = min(a[48:63], b[48:63])
    m64 = _m_pminub(m64, m64);                                      // r[0:7] = min(a[0:7], b[0:7]), r[8:15] = min(a[8:15], b[8:15]), r[16:23] = min(a[16:23], b[16:23]), r[24:31] = min(a[24:31], b[24:31]), r[32:39] = min(a[32:39], b[32:39]), r[40:47] = min(a[40:47], b[40:47]), r[48:55] = min(a[48:55], b[48:55]), r[56:63] = min(a[56:63], b[56:63])
    in = _m_pmovmskb(m64);                                          // r[0] = a[0:7], r[1] = a[8:15], r[2] = a[16:23], r[3] = a[24:31], r[4] = a[32:39], r[5] = a[40:47], r[7] = a[48:55], r[8] = a[56:63]
    m64 = _m_pmulhuw(m64, m64);                                     // r[0:15] = a[0:15]*b[0:15], r[16:31] = a[16:31]*b[16:31], r[32:47] = a[32:47]*b[32:47], r[48:63] = a[48:63]*b[48:63]
    m64 = _m_pshufw(m64, _MM_SHUFFLE(2, 3, 0, 1));                  // r[0:15] = a[imm8[0:1]*16:imm8[0:1]*16+15], r[16:31] = a[imm8[2:3]*16:imm8[2:3]*16+15], r[32:47] = a[imm8[4:5]*16:imm8[4:5]*16+15], r[48:63] = a[imm8[6:7]*16:imm8[6:7]*16+15]
    _m_maskmovq(m64, m64, arrChar4);                                // MEM[mem_addr:mem_addr+:7] = a[0:7], MEM[mem_addr+8:mem_addr+15] = a[8:15], MEM[mem_addr+16:mem_addr+23] = a[16:23], MEM[mem_addr+24:mem_addr+31] = a[24:31], MEM[mem_addr+32:mem_addr+39] = a[32:39], MEM[mem_addr+40:mem_addr+47] = a[40:47], MEM[mem_addr+48:mem_addr+55] = a[48:55], MEM[mem_addr+56:mem_addr+63] = a[56:63]
    m64 = _m_pavgb(m64, m64);                                       // r[0:7] = (a[0:7]+b[0:7]+1)>>1, r[8:15] = (a[8:15]+b[8:15]+1)>>1, r[16:23] = (a[16:23]+b[16:23]+1)>>1, r[24:31] = (a[24:31]+b[24:31]+1)>>1, r[32:39] = (a[32:39]+b[32:39]+1)>>1, r[40:47] = (a[40:47]+b[40:47]+1)>>1, r[48:55] = (a[48:55]+b[48:55]+1)>>1, r[56:63] = (a[56:63]+b[56:63]+1)>>1
    m64 = _m_pavgw(m64, m64);                                       // r[0:15] = (a[0:15]+b[0:15]+1) >> 1, r[16:31] = (a[16:31]+b[16:31]+1) >> 1, r[32:47] = (a[32:47]+b[32:47]+1) >> 1, r[48:63] = (a[48:63]+b[48:63]+1) >> 1
    m64 = _m_psadbw(m64, m64);                                      // r[0:7] = abs(a[0:7]-b[0:7]), r[8:15] = abs(a[8:15]-b[8:15]), r[16:23] = abs(a[16:23]-b[16:23]), r[24:31] = abs(a[24:31]-b[24:31]), r[32:39] = abs(a[32:39]-b[32:39]), r[40:47] = abs(a[40:47]-b[40:47]), r[48:55] = abs(a[48:55]-b[48:55]), r[56:63] = abs(a[56:63]-b[56:63])

    /*
    * Memory & initialization
    */
    _mm_stream_pi(&m64, m64);                                       // MEM[mem_addr:mem_addr+63] = b[0:63]

    /*
    * Alternate intrinsic names definition
    */
    m64 = _mm_cvtps_pi32(m128);                                     // r[0:31] = (Convert_FP32_To_Int32)a[0:31], r[32:63] = (Convert_FP32_To_Int32)a[32:63]
    m64 = _mm_cvttps_pi32(m128);                                    // r[0:31] = (Convert_FP32_To_Int32_Truncate)a[0:31], r[32:63] = (Convert_FP32_To_Int32_Truncate)a[32:63]
    m128 = _mm_cvtpi32_ps(m128, m64);                               // r[0:31] = (Convert_Int32_To_FP32)b[0:31], r[32:63] = (Convert_Int32_To_FP32)b[32:63], r[64:95] = a[64:95], r[96:127] = a[96:127]
    in = _mm_extract_pi16(m64, 0);                                  // r[0:15] = (a[0:63] >> (imm8[0:1] * 16))[0:15], r[16:31] = 0
    m64 = _mm_insert_pi16(m64, in, 0);                              // r[0:63] = a[0:63], r[imm8[0:1]*16:imm8[0:1]*16+15] = i[0:15]
    m64 = _mm_max_pi16(m64, m64);                                   // r[0:15] = max(a[0:15], b[0:15]), r[16:31] = max(a[16:31], b[16:31]), r[32:47] = max(a[32:47], b[32:47]), r[48:63] = max(a[48:63], b[48:63])
    m64 = _mm_max_pu8(m64, m64);                                    // r[0:7] = max(a[0:7], b[0:7]), r[8:15] = max(a[8:15], b[8:15]), r[16:23] = max(a[16:23], b[16:23]), r[24:31] = max(a[24:31], b[24:31]), r[32:39] = max(a[32:39], b[32:39]), r[40:47] = max(a[40:47], b[40:47]), r[48:55] = max(a[48:55], b[48:55]), r[56:63] = max(a[56:63], b[56:63])
    m64 = _mm_min_pi16(m64, m64);                                   // r[0:15] = min(a[0:15], b[0:15]), r[16:31] = min(a[16:31], b[16:31]), r[32:47] = min(a[32:47], b[32:47]), r[48:63] = min(a[48:63], b[48:63])
    m64 = _mm_min_pu8(m64, m64);                                    // r[0:7] = min(a[0:7], b[0:7]), r[8:15] = min(a[8:15], b[8:15]), r[16:23] = min(a[16:23], b[16:23]), r[24:31] = min(a[24:31], b[24:31]), r[32:39] = min(a[32:39], b[32:39]), r[40:47] = min(a[40:47], b[40:47]), r[48:55] = min(a[48:55], b[48:55]), r[56:63] = min(a[56:63], b[56:63])
    in = _mm_movemask_pi8(m64);                                     // r[0] = a[0:7], r[1] = a[8:15], r[2] = a[16:23], r[3] = a[24:31], r[4] = a[32:39], r[5] = a[40:47], r[7] = a[48:55], r[8] = a[56:63]
    m64 = _mm_mulhi_pu16(m64, m64);                                 // r[0:15] = a[0:15]*b[0:15], r[16:31] = a[16:31]*b[16:31], r[32:47] = a[32:47]*b[32:47], r[48:63] = a[48:63]*b[48:63]
    m64 = _mm_shuffle_pi16(m64, _MM_SHUFFLE(2, 3, 0, 1));           // r[0:15] = a[imm8[0:1]*16:imm8[0:1]*16+15], r[16:31] = a[imm8[2:3]*16:imm8[2:3]*16+15], r[32:47] = a[imm8[4:5]*16:imm8[4:5]*16+15], r[48:63] = a[imm8[6:7]*16:imm8[6:7]*16+15]
    _mm_maskmove_si64(m64, m64, arrChar4);                          // MEM[mem_addr:mem_addr+:7] = a[0:7], MEM[mem_addr+8:mem_addr+15] = a[8:15], MEM[mem_addr+16:mem_addr+23] = a[16:23], MEM[mem_addr+24:mem_addr+31] = a[24:31], MEM[mem_addr+32:mem_addr+39] = a[32:39], MEM[mem_addr+40:mem_addr+47] = a[40:47], MEM[mem_addr+48:mem_addr+55] = a[48:55], MEM[mem_addr+56:mem_addr+63] = a[56:63]
    m64 = _mm_avg_pu8(m64, m64);                                    // r[0:7] = (a[0:7]+b[0:7]+1)>>1, r[8:15] = (a[8:15]+b[8:15]+1)>>1, r[16:23] = (a[16:23]+b[16:23]+1)>>1, r[24:31] = (a[24:31]+b[24:31]+1)>>1, r[32:39] = (a[32:39]+b[32:39]+1)>>1, r[40:47] = (a[40:47]+b[40:47]+1)>>1, r[48:55] = (a[48:55]+b[48:55]+1)>>1, r[56:63] = (a[56:63]+b[56:63]+1)>>1
    m64 = _mm_avg_pu16(m64, m64);                                   // r[0:15] = (a[0:15]+b[0:15]+1) >> 1, r[16:31] = (a[16:31]+b[16:31]+1) >> 1, r[32:47] = (a[32:47]+b[32:47]+1) >> 1, r[48:63] = (a[48:63]+b[48:63]+1) >> 1
    m64 = _mm_sad_pu8(m64, m64);                                    // r[0:7] = abs(a[0:7]-b[0:7]), r[8:15] = abs(a[8:15]-b[8:15]), r[16:23] = abs(a[16:23]-b[16:23]), r[24:31] = abs(a[24:31]-b[24:31]), r[32:39] = abs(a[32:39]-b[32:39]), r[40:47] = abs(a[40:47]-b[40:47]), r[48:55] = abs(a[48:55]-b[48:55]), r[56:63] = abs(a[56:63]-b[56:63])

    /*
    * Utility intrinsics function
    */
    m128 = _mm_cvtpi16_ps(m64);                                     // r[0:31] = (Convert_Int16_To_FP32)a[0:15], r[32:63] = (Convert_Int16_To_FP32)a[16:31], r[64:95] = (Convert_Int16_To_FP32)a[32:47], r[96:127] = (Convert_Int16_To_FP32)a[48:63]
    m128 = _mm_cvtpu16_ps(m64);                                     // r[0:31] = (Convert_UnsignedInt16_To_FP32)a[0:15], r[32:63] = (Convert_UnsignedInt16_To_FP32)a[16:31], r[64:95] = (Convert_UnsignedInt16_To_FP32)a[32:47], r[96:127] = (Convert_UnsignedInt16_To_FP32)a[48:63]
    m64 = _mm_cvtps_pi16(m128);                                     // r[0:15] = (Convert_FP32_To_Int16)a[0:31], r[16:31] = (Convert_FP32_To_Int16)a[32:63], r[32:47] = (Convert_FP32_To_Int16)a[64:95], r[48:63] = (Convert_FP32_To_Int16)a[96:127]
    m128 = _mm_cvtpi8_ps(m64);                                      // r[0:31] = (Convert_Int8_To_FP32)a[0:7], r[32:63] = (Convert_Int8_To_FP32)a[8:15], r[64:95] = (Convert_Int8_To_FP32)a[16:23], r[96:127] = (Convert_Int8_To_FP32)a[24:31]
    m128 = _mm_cvtpu8_ps(m64);                                      // r[0:31] = (Convert_UnsignedInt8_To_FP32)a[0:7], r[32:63] = (Convert_UnsignedInt8_To_FP32)a[8:15], r[64:95] = (Convert_UnsignedInt8_To_FP32)a[16:23], r[96:127] = (Convert_UnsignedInt8_To_FP32)a[24:31]
    m64 = _mm_cvtps_pi8(m128);                                      // r[0:7] = (Convert_FP32_To_Int8)a[0:31], r[8:15] = (Convert_FP32_To_Int8)a[32:63], r[16:23] = (Convert_FP32_To_Int8)a[64:95], r[24:31] = (Convert_FP32_To_Int8)a[96:127]
    m128 = _mm_cvtpi32x2_ps(m64, m64);                              // r[0:31] = (Convert_Int32_To_FP32)a[0:31], r[32:63] = (Convert_Int32_To_FP32)a[32:63], r[64:95] = (Convert_Int32_To_FP32)b[0:31], r[96:127] = (Convert_Int32_To_FP32)b[32:63]
#endif
}

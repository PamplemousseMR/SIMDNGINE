#include "simd.hpp"
#include <mmintrin.h>

/*  mmintrin
MMX
Introduce eight 64 bit registers(MM0 - MM7) and instructions to work with eight signed / unsigned bytes, four signed / unsigned words, two signed / unsigned dwords.
*/
void mmintrin()
{
#if defined(_M_IX86) || defined(__unix)
    __m64 m64;
    int in = 10;
    char ch = 'c';
    short sh = 5;

    /*
    * General support intrinsics
    */
    _m_empty();                                                     // CleanMMXstate
    m64 = _m_from_int(in);                                          // r[0:31] = a[0:31], r[32:63] = 0
    in = _m_to_int(m64);                                            // r = a[0:31]
    m64 = _m_packsswb(m64, m64);                                    // r[0:7] = (Saturate_Int16_To_Int8)a[0:15], r[8:15] = (Saturate_Int16_To_Int8)a[16:31], r[16:23] = (Saturate_Int16_To_Int8)a[32:47], r[24:31] = (Saturate_Int16_To_Int8)a[48:63], r[32:39] = (Saturate_Int16_To_Int8)b[0:15], r[40:47] = (Saturate_Int16_To_Int8)b[16:31], r[48:55] = (Saturate_Int16_To_Int8)b[32:47], r[56:63] = (Saturate_Int16_To_Int8)b[48:63]
    m64 = _m_packssdw(m64, m64);                                    // r[0:15] = (Saturate_Int32_To_Int16)a[0:31], r[16:31] = (Saturate_Int32_To_Int16)a[32:63], r[32:47] = (Saturate_Int32_To_Int16)b[0:31], r[48:63] = (Saturate_Int32_To_Int16)b[32:63]
    m64 = _m_packuswb(m64, m64);                                    // r[0:7] = (Saturate_Int16_To_UnsignedInt8)a[0:15], r[8:15] = (Saturate_Int16_To_UnsignedInt8)a[16:31], r[16:23] = (Saturate_Int16_To_UnsignedInt8)a[32:47], r[24:31] = (Saturate_Int16_To_UnsignedInt8)a[48:63], r[32:39] = (Saturate_Int16_To_UnsignedInt8)b[0:15], r[40:47] = (Saturate_Int16_To_UnsignedInt8)b[16:31], r[48:55] = (Saturate_Int16_To_UnsignedInt8)b[32:47], r[56:63] = (Saturate_Int16_To_UnsignedInt8)b[48:63]
    m64 = _m_punpckhbw(m64, m64);                                   // r[0:7] = a[32:39], r[8:15] = b[32:39], r[16:23] = a[40:47], r[24:31] = b[40:47], r[32:39] = a[48:55], r[40:47] = b[48:55], r[48:55] = a[56:63], r[56:63] = b[56:63]
    m64 = _m_punpckhwd(m64, m64);                                   // r[0:15] = a[32:47], r[16:31] = b[32:47], r[32:47] = a[48:63], r[48:63] = b[48:63]
    m64 = _m_punpckhdq(m64, m64);                                   // r[0:31] = a[32:63], r[32:63] = b[32:63]
    m64 = _m_punpcklbw(m64, m64);                                   // r[0:7] = a[0:7], r[8:15] = b[0:7], r[16:23] = a[8:15], r[24:31] = b[8:15], r[32:39] = a[16:23], r[40:47] = b[16:23], r[48:55] = a[24:31], r[56:63] = b[24:31]
    m64 = _m_punpcklwd(m64, m64);                                   // r[0:15] = a[0:15], r[16:31] = b[0:15], r[32:47] = a[16:31], r[48:63] = b[16:31]
    m64 = _m_punpckldq(m64, m64);                                   // r[0:31] = a[0:31], r[32:63] = b[0:31]

    /*
    * Packed arithmetic intrinsics
    */
    m64 = _m_paddb(m64, m64);                                       // r[0:7] = a[0:7]+b[0:7], r[8:15] = a[8:15]+b[8:15], r[16:23] = a[16:23]+b[16:23], r[24:31] = a[24:31]+b[24:31], r[32:39] = a[32:39]+b[32:39], r[40:47] = a[40:47]+b[40:47], r[48:55] = a[48:55]+b[48:55], r[56:63] = a[56:63]+b[56:63]
    m64 = _m_paddw(m64, m64);                                       // r[0:15] = a[0:15]+b[0:15], r[16:31] = a[16:31]+b[16:31], r[32:47] = a[32:47]+b[32:47], r[48:63] = a[48:63]+b[48:63]
    m64 = _m_paddd(m64, m64);                                       // r[0:31] = a[0:31]+b[0:31], r[32:63] = a[32:63]+b[32:63]
    m64 = _m_paddsb(m64, m64);                                      // r[0:7] = (Saturate_To_Int8)a[0:7]+b[0:7], r[8:15] = (Saturate_To_Int8)a[8:15]+b[8:15], r[16:23] = (Saturate_To_Int8)a[16:23]+b[16:23], r[24:31] = (Saturate_To_Int8)a[24:31]+b[24:31], r[32:39] = (Saturate_To_Int8)a[32:39]+b[32:39], r[40:47] = (Saturate_To_Int8)a[40:47]+b[40:47], r[48:55] = (Saturate_To_Int8)a[48:55]+b[48:55], r[56:63] = (Saturate_To_Int8)a[56:63]+b[56:63]
    m64 = _m_paddsw(m64, m64);                                      // r[0:15] = (Saturate_To_Int16)a[0:15]+b[0:15], r[16:31] = (Saturate_To_Int16)a[16:31]+b[16:31], r[32:47] = (Saturate_To_Int16)a[32:47]+b[32:47], r[48:63] = (Saturate_To_Int16)a[48:63]+b[48:63]
    m64 = _m_paddusb(m64, m64);                                     // r[0:7] = (Saturate_To_UnsignedInt8)a[0:7]+b[0:7], r[8:15] = (Saturate_To_UnsignedInt8)a[8:15]+b[8:15], r[16:23] = (Saturate_To_UnsignedInt8)a[16:23]+b[16:23], r[24:31] = (Saturate_To_UnsignedInt8)a[24:31]+b[24:31], r[32:39] = (Saturate_To_UnsignedInt8)a[32:39]+b[32:39], r[40:47] = (Saturate_To_UnsignedInt8)a[40:47]+b[40:47], r[48:55] = (Saturate_To_UnsignedInt8)a[48:55]+b[48:55], r[56:63] = (Saturate_To_UnsignedInt8)a[56:63]+b[56:63]
    m64 = _m_paddusw(m64, m64);                                     // r[0:15] = (Saturate_To_UnsignedInt16)a[0:15]+b[0:15], r[16:31] = (Saturate_To_UnsignedInt16)a[16:31]+b[16:31], r[32:47] = (Saturate_To_UnsignedInt16)a[32:47]+b[32:47], r[48:63] = (Saturate_To_UnsignedInt16)a[48:63]+b[48:63]
    m64 = _m_psubb(m64, m64);                                       // r[0:7] = a[0:7]-b[0:7], r[8:15] = a[8:15]-b[8:15], r[16:23] = a[16:23]-b[16:23], r[24:31] = a[24:31]-b[24:31], r[32:39] = a[32:39]-b[32:39], r[40:47] = a[40:47]-b[40:47], r[48:55] = a[48:55]-b[48:55], r[56:63] = a[56:63]-b[56:63]
    m64 = _m_psubw(m64, m64);                                       // r[0:15] = a[0:15]-b[0:15], r[16:31] = a[16:31]-b[16:31], r[32:47] = a[32:47]-b[32:47], r[48:63] = a[48:63]-b[48:63]
    m64 = _m_psubd(m64, m64);                                       // r[0:31] = a[0:31]-b[0:31], r[32:63] = a[32:63]-b[32:63]
    m64 = _m_psubsb(m64, m64);                                      // r[0:7] = (Saturate_To_Int8)a[0:7]-b[0:7], r[8:15] = (Saturate_To_Int8)a[8:15]-b[8:15], r[16:23] = (Saturate_To_Int8)a[16:23]-b[16:23], r[24:31] = (Saturate_To_Int8)a[24:31]-b[24:31], r[32:39] = (Saturate_To_Int8)a[32:39]-b[32:39], r[40:47] = (Saturate_To_Int8)a[40:47]-b[40:47], r[48:55] = (Saturate_To_Int8)a[48:55]-b[48:55], r[56:63] = (Saturate_To_Int8)a[56:63]-b[56:63]
    m64 = _m_psubsw(m64, m64);                                      // r[0:15] = (Saturate_To_Int16)a[0:15]-b[0:15], r[16:31] = (Saturate_To_Int16)a[16:31]-b[16:31], r[32:47] = (Saturate_To_Int16)a[32:47]-b[32:47], r[48:63] = (Saturate_To_Int16)a[48:63]-b[48:63]
    m64 = _m_psubusb(m64, m64);                                     // r[0:7] = (Saturate_To_UnsignedInt8)a[0:7]-b[0:7], r[8:15] = (Saturate_To_UnsignedInt8)a[8:15]-b[8:15], r[16:23] = (Saturate_To_UnsignedInt8)a[16:23]-b[16:23], r[24:31] = (Saturate_To_UnsignedInt8)a[24:31]-b[24:31], r[32:39] = (Saturate_To_UnsignedInt8)a[32:39]-b[32:39], r[40:47] = (Saturate_To_UnsignedInt8)a[40:47]-b[40:47], r[48:55] = (Saturate_To_UnsignedInt8)a[48:55]-b[48:55], r[56:63] = (Saturate_To_UnsignedInt8)a[56:63]-b[56:63]
    m64 = _m_psubusw(m64, m64);                                     // r[0:15] = (Saturate_To_UnsignedInt16)a[0:15]-b[0:15], r[16:31] = (Saturate_To_UnsignedInt16)a[16:31]-b[16:31], r[32:47] = (Saturate_To_UnsignedInt16)a[32:47]-b[32:47], r[48:63] = (Saturate_To_UnsignedInt16)a[48:63]-b[48:63]
    m64 = _m_pmaddwd(m64, m64);                                     // r[0:31] = a[16:31]*b[16:31]+a[0:15]*b[0:15], r[32:63] = a[48:63]*b[48:63]+a[32:47]*b[32:47]
    m64 = _m_pmulhw(m64, m64);                                      // r[0:15] = (a[0:15]*b[0:15])[16:31], r[16:31] = (a[16:31]*b[16:31])[16:31], r[32:47] = (a[32:47]*b[32:47])[16:31], r[48:63] = (a[48:63]*b[48:63])[16:31]
    m64 = _m_pmullw(m64, m64);                                      // r[0:15] = (a[0:15]*b[0:15])[0:15], r[16:31] = (a[16:31]*b[16:31])[0:15], r[32:47] = (a[32:47]*b[32:47])[0:15], r[48:63] = (a[48:63]*b[48:63])[0:15]

    /*
    * Shift intrinsics
    */
    m64 = _m_psllw(m64, m64);                                       // r[0:15] = count[0:63]>15?0:(ZeroExtend)(a[0:15]<<count[0:63]), r[16:31] = count[0:63]>15?0:(ZeroExtend)(a[16:31]<<count[0:63]), r[32:47] = count[0:63]>15?0:(ZeroExtend)(a[32:47]<<count[0:63]), r[48:63] = count[0:63]>15?0:(ZeroExtend)(a[48:63]<<count[0:63])
    m64 = _m_psllwi(m64, in);                                       // r[0:15] = imm8[0:7]>15?0:(ZeroExtend)(a[0:15]<<imm8[0:7]), r[16:31] = imm8[0:7]>15?0:(ZeroExtend)(a[16:31]<<imm8[0:7]), r[32:47] = imm3[0:7]>15?0:(ZeroExtend)(a[32:47]<<imm8[0:7]), r[48:63] = imm8[0:7]>15?0:(ZeroExtend)(a[48:63]<<imm8[0:7])
    m64 = _m_pslld(m64, m64);                                       // r[0:31] = count[0:63]>31?0:(ZeroExtend)(a[0:31]<<count[0:63]), r[32:63] = count[0:63]>31?0:(ZeroExtend)(a[32:63]<<count[0:63])
    m64 = _m_pslldi(m64, in);                                       // r[0:31] = imm8[0:7]>31?0:(ZeroExtend)(a[0:31]<<imm8[0:7]), r[32:63] = imm8[0:7]>31?0:(ZeroExtend)(a[32:63]<<imm8[0:7])
    m64 = _m_psllq(m64, m64);                                       // r[0:63] = count[0:63]>63?0:(ZeroExtend)(a[0:63]<<count[0:63])
    m64 = _m_psllqi(m64, in);                                       // r[0:63] = imm8[0:7]>63?0:(ZeroExtend)(a[0:63]<<imm8[0:7])
    m64 = _m_psraw(m64, m64);                                       // r[0:15] = count[0:63]>15?SignBit:(SignExtend)(a[0:15]>>count[0:63]), r[16:31] = count[0:63]>15?SignBit:(SignExtend)(a[16:31]>>count[0:63]), r[32:47] = count[0:63]>15?SignBit:(SignExtend)(a[32:47]>>count[0:63]), r[48:63] = count[0:63]>15?SignBit:(SignExtend)(a[48:63]>>count[0:63])
    m64 = _m_psrawi(m64, in);                                       // r[0:15] = imm8[0:7]>15?SignBit:(SignExtend)(a[0:15]>>imm8[0:7]), r[16:31] = imm8[0:7]>15?SignBit:(SignExtend)(a[16:31]>>imm8[0:7]), r[32:47] = imm3[0:7]>15?SignBit:(SignExtend)(a[32:47]>>imm8[0:7]), r[48:63] = imm8[0:7]>15?SignBit:(SignExtend)(a[48:63]>>imm8[0:7])
    m64 = _m_psrad(m64, m64);                                       // r[0:31] = count[0:63]>31?SignBit:(SignExtend)(a[0:31]>>count[0:63]), r[32:63] = count[0:63]>31?SignBit:(SignExtend)(a[32:63]>>count[0:63])
    m64 = _m_psradi(m64, in);                                       // r[0:31] = imm8[0:7]>31?SignBit:(SignExtend)(a[0:31]>>imm8[0:7]), r[32:63] = imm8[0:7]>31?SignBit:(SignExtend)(a[32:63]>>imm8[0:7])
    m64 = _m_psrlw(m64, m64);                                       // r[0:15] = count[0:63]>15?0:(ZeroExtend)(a[0:15]>>count[0:63]), r[16:31] = count[0:63]>15?0:(ZeroExtend)(a[16:31]>>count[0:63]), r[32:47] = count[0:63]>15?0:(ZeroExtend)(a[32:47]>>count[0:63]), r[48:63] = count[0:63]>15?0:(ZeroExtend)(a[48:63]>>count[0:63])
    m64 = _m_psrlwi(m64, in);                                       // r[0:15] = imm8[0:7]>15?0:(ZeroExtend)(a[0:15]>>imm8[0:7]), r[16:31] = imm8[0:7]>15?0:(ZeroExtend)(a[16:31]>>imm8[0:7]), r[32:47] = imm3[0:7]>15?0:(ZeroExtend)(a[32:47]>>imm8[0:7]), r[48:63] = imm8[0:7]>15?0:(ZeroExtend)(a[48:63]>>imm8[0:7])
    m64 = _m_psrld(m64, m64);                                       // r[0:31] = count[0:63]>31?0:(ZeroExtend)(a[0:31]>>count[0:63]), r[32:63] = count[0:63]>31?0:(ZeroExtend)(a[32:63]>>count[0:63])
    m64 = _m_psrldi(m64, in);                                       // r[0:31] = imm8[0:7]>31?0:(ZeroExtend)(a[0:31]>>imm8[0:7]), r[32:63] = imm8[0:7]>31?0:(ZeroExtend)(a[32:63]>>imm8[0:7])
    m64 = _m_psrlq(m64, m64);                                       // count[0:63]>63>0:r[0:63] = (ZeroExtend)(a[0:63]>>count[0:63])
    m64 = _m_psrlqi(m64, in);                                       // imm8[0:7]>63>0:r[0:63] = (ZeroExtend)(a[0:63]>>imm8[0:7])

    /*
    * Logical intrinsics
    */
    m64 = _m_pand(m64, m64);                                        // r[0:63] = a[0:63]&b[0:63]
    m64 = _m_pandn(m64, m64);                                       // r[0:63] = ~a[0:63]&b[0:63]
    m64 = _m_por(m64, m64);                                         // r[0:63] = a[0:63]|b[0:63]
    m64 = _m_pxor(m64, m64);                                        // r[0:63] = a[0:63]^b[0:63]

    /*
    * Comparison intrinsics
    */
    m64 = _m_pcmpeqb(m64, m64);                                     // r[0:7] = (a[0:7]==b[0:7])?0xF:0x0, r[8:15] = (a[8:15]==b[8:15])?0xF:0x0, r[16:23] = (a[16:23]==b[16:23])?0xF:0x0, r[24:31] = (a[24:31]==b[24:31])?0xF:0x0, r[32:39] = (a[32:39]==b[32:39])?0xF:0x0, r[40:47] = (a[40:47]==b[40:47])?0xF:0x0, r[48:55] = (a[48:55]==b[48:55])?0xF:0x0, r[56:63] = (a[56:63]==b[56:63])?0xF:0x0
    m64 = _m_pcmpeqw(m64, m64);                                     // r[0:15] = (a[0:15]==b[0:15])?0xF:0x0, r[16:31] = (a[16:31]==b[16:31])?0xF:0x0, r[32:47] = (a[32:47]==b[32:47])?0xF:0x0, r[48:63] = (a[48:63]==b[48:63])?0xF:0x0
    m64 = _m_pcmpeqd(m64, m64);                                     // r[0:31] = (a[0:31]==b[0:31])?0xF:0x0, r[32:63] = (a[32:63]==b[32:63])?0xF:0x0
    m64 = _m_pcmpgtb(m64, m64);                                     // r[0:7] = (a[0:7]>b[0:7])?0xF:0x0, r[8:15] = (a[8:15]>b[8:15])?0xF:0x0, r[16:23] = (a[16:23]>b[16:23])?0xF:0x0, r[24:31] = (a[24:31]>b[24:31])?0xF:0x0, r[32:39] = (a[32:39]>b[32:39])?0xF:0x0, r[40:47] = (a[40:47]>b[40:47])?0xF:0x0, r[48:55] = (a[48:55]>b[48:55])?0xF:0x0, r[56:63] = (a[56:63]>b[56:63])?0xF:0x0
    m64 = _m_pcmpgtw(m64, m64);                                     // r[0:15] = (a[0:15]>b[0:15])?0xF:0x0, r[16:31] = (a[16:31]>b[16:31])?0xF:0x0, r[32:47] = (a[32:47]>b[32:47])?0xF:0x0, r[48:63] = (a[48:63]>b[48:63])?0xF:0x0
    m64 = _m_pcmpgtd(m64, m64);                                     // r[0:31] = (a[0:31]>b[0:31])?0xF:0x0, r[32:63] = (a[32:63]>b[32:63])?0xF:0x0

    /*
    * Utility intrinsics
    */
    m64 = _mm_setzero_si64();                                       // r[0:63] = 0
    m64 = _mm_set_pi32(in, in);                                     // r[0:31] = y, r[32:63] = x
    m64 = _mm_set_pi16(sh, sh, sh, sh);                             // r[0:15] = w, r[16:31] = z, r[32:47] = y, r[48:63] = x
    m64 = _mm_set_pi8(ch, ch, ch, ch, ch, ch, ch, ch);              // r[0:7] = w, r[8:15] = z, r[16:23] = y, r[24:31] = x, r[32:39] = d, r[40:47] = c, r[48:55] = b, r[56:63] = a
    m64 = _mm_set1_pi32(in);                                        // r[0:31] = w, r[32:63] = w
    m64 = _mm_set1_pi16(sh);                                        // r[0:15] = w, r[16:31] = w, r[32:47] = w, r[48:63] = w
    m64 = _mm_set1_pi8(ch);                                         // r[0:7] = w, r[8:15] = w, r[16:23] = w, r[24:31] = w, r[32:39] = w, r[40:47] = w, r[48:55] = w, r[56:63] = w
    m64 = _mm_setr_pi32(in, in);                                    // r[0:31] = x, r[32:63] = y
    m64 = _mm_setr_pi16(sh, sh, sh, sh);                            // r[0:15] = x, r[16:31] = y, r[32:47] = z, r[48:63] = w
    m64 = _mm_setr_pi8(ch, ch, ch, ch, ch, ch, ch, ch);             // r[0:7] = a, r[8:15] = b, r[16:23] = c, r[24:31] = d, r[32:39] = x, r[40:47] = y, r[48:55] = z, r[56:63] = w

    /*
    * Alternate intrinsic name definitions
    */
    _mm_empty();                                                    // CleanMMXstate
    m64 = _mm_cvtsi32_si64(in);                                     // r[0:31] = a[0:31], r[32:63] = 0
    in = _mm_cvtsi64_si32(m64);                                     // r = a[0:31]
    m64 = _mm_packs_pi16(m64, m64);                                 // r[0:7] = (Saturate_Int16_To_Int8)a[0:15], r[8:15] = (Saturate_Int16_To_Int8)a[16:31], r[16:23] = (Saturate_Int16_To_Int8)a[32:47], r[24:31] = (Saturate_Int16_To_Int8)a[48:63], r[32:39] = (Saturate_Int16_To_Int8)b[0:15], r[40:47] = (Saturate_Int16_To_Int8)b[16:31], r[48:55] = (Saturate_Int16_To_Int8)b[32:47], r[56:63] = (Saturate_Int16_To_Int8)b[48:63]
    m64 = _mm_packs_pi32(m64, m64);                                 // r[0:15] = (Saturate_Int32_To_Int16)a[0:31], r[16:31] = (Saturate_Int32_To_Int16)a[32:63], r[32:47] = (Saturate_Int32_To_Int16)b[0:31], r[48:63] = (Saturate_Int32_To_Int16)b[32:63]
    m64 = _mm_packs_pu16(m64, m64);                                 // r[0:7] = (Saturate_Int16_To_UnsignedInt8)a[0:15], r[8:15] = (Saturate_Int16_To_UnsignedInt8)a[16:31], r[16:23] = (Saturate_Int16_To_UnsignedInt8)a[32:47], r[24:31] = (Saturate_Int16_To_UnsignedInt8)a[48:63], r[32:39] = (Saturate_Int16_To_UnsignedInt8)b[0:15], r[40:47] = (Saturate_Int16_To_UnsignedInt8)b[16:31], r[48:55] = (Saturate_Int16_To_UnsignedInt8)b[32:47], r[56:63] = (Saturate_Int16_To_UnsignedInt8)b[48:63]
    m64 = _mm_unpackhi_pi8(m64, m64);                               // r[0:7] = a[32:39], r[8:15] = b[32:39], r[16:23] = a[40:47], r[24:31] = b[40:47], r[32:39] = a[48:55], r[40:47] = b[48:55], r[48:55] = a[56:63], r[56:63] = b[56:63]
    m64 = _mm_unpackhi_pi16(m64, m64);                              // r[0:15] = a[32:47], r[16:31] = b[32:47], r[32:47] = a[48:63], r[48:63] = b[48:63]
    m64 = _mm_unpackhi_pi32(m64, m64);                              // r[0:31] = a[32:63], r[32:63] = b[32:63]
    m64 = _mm_unpacklo_pi8(m64, m64);                               // r[0:7] = a[0:7], r[8:15] = b[0:7], r[16:23] = a[8:15], r[24:31] = b[8:15], r[32:39] = a[16:23], r[40:47] = b[16:23], r[48:55] = a[24:31], r[56:63] = b[24:31]
    m64 = _mm_unpacklo_pi16(m64, m64);                              // r[0:15] = a[0:15], r[16:31] = b[0:15], r[32:47] = a[16:31], r[48:63] = b[16:31]
    m64 = _mm_unpacklo_pi32(m64, m64);                              // r[0:31] = a[0:31], r[32:63] = b[0:31]
    m64 = _mm_add_pi8(m64, m64);                                    // r[0:7] = a[0:7]+b[0:7], r[8:15] = a[8:15]+b[8:15], r[16:23] = a[16:23]+b[16:23], r[24:31] = a[24:31]+b[24:31], r[32:39] = a[32:39]+b[32:39], r[40:47] = a[40:47]+b[40:47], r[48:55] = a[48:55]+b[48:55], r[56:63] = a[56:63]+b[56:63]
    m64 = _mm_add_pi16(m64, m64);                                   // r[0:15] = a[0:15]+b[0:15], r[16:31] = a[16:31]+b[16:31], r[32:47] = a[32:47]+b[32:47], r[48:63] = a[48:63]+b[48:63]
    m64 = _mm_add_pi32(m64, m64);                                   // r[0:31] = a[0:31]+b[0:31], r[32:63] = a[32:63]+b[32:63]
    m64 = _mm_adds_pi8(m64, m64);                                   // r[0:7] = (Saturate_To_Int8)a[0:7]+b[0:7], r[8:15] = (Saturate_To_Int8)a[8:15]+b[8:15], r[16:23] = (Saturate_To_Int8)a[16:23]+b[16:23], r[24:31] = (Saturate_To_Int8)a[24:31]+b[24:31], r[32:39] = (Saturate_To_Int8)a[32:39]+b[32:39], r[40:47] = (Saturate_To_Int8)a[40:47]+b[40:47], r[48:55] = (Saturate_To_Int8)a[48:55]+b[48:55], r[56:63] = (Saturate_To_Int8)a[56:63]+b[56:63]
    m64 = _mm_adds_pi16(m64, m64);                                  // r[0:15] = (Saturate_To_Int16)a[0:15]+b[0:15], r[16:31] = (Saturate_To_Int16)a[16:31]+b[16:31], r[32:47] = (Saturate_To_Int16)a[32:47]+b[32:47], r[48:63] = (Saturate_To_Int16)a[48:63]+b[48:63]
    m64 = _mm_adds_pu8(m64, m64);                                   // r[0:7] = (Saturate_To_UnsignedInt8)a[0:7]+b[0:7], r[8:15] = (Saturate_To_UnsignedInt8)a[8:15]+b[8:15], r[16:23] = (Saturate_To_UnsignedInt8)a[16:23]+b[16:23], r[24:31] = (Saturate_To_UnsignedInt8)a[24:31]+b[24:31], r[32:39] = (Saturate_To_UnsignedInt8)a[32:39]+b[32:39], r[40:47] = (Saturate_To_UnsignedInt8)a[40:47]+b[40:47], r[48:55] = (Saturate_To_UnsignedInt8)a[48:55]+b[48:55], r[56:63] = (Saturate_To_UnsignedInt8)a[56:63]+b[56:63]
    m64 = _mm_adds_pu16(m64, m64);                                  // r[0:15] = (Saturate_To_UnsignedInt16)a[0:15]+b[0:15], r[16:31] = (Saturate_To_UnsignedInt16)a[16:31]+b[16:31], r[32:47] = (Saturate_To_UnsignedInt16)a[32:47]+b[32:47], r[48:63] = (Saturate_To_UnsignedInt16)a[48:63]+b[48:63]
    m64 = _mm_sub_pi8(m64, m64);                                    // r[0:7] = a[0:7]-b[0:7], r[8:15] = a[8:15]-b[8:15], r[16:23] = a[16:23]-b[16:23], r[24:31] = a[24:31]-b[24:31], r[32:39] = a[32:39]-b[32:39], r[40:47] = a[40:47]-b[40:47], r[48:55] = a[48:55]-b[48:55], r[56:63] = a[56:63]-b[56:63]
    m64 = _mm_sub_pi16(m64, m64);                                   // r[0:15] = a[0:15]-b[0:15], r[16:31] = a[16:31]-b[16:31], r[32:47] = a[32:47]-b[32:47], r[48:63] = a[48:63]-b[48:63]
    m64 = _mm_sub_pi32(m64, m64);                                   // r[0:31] = a[0:31]-b[0:31], r[32:63] = a[32:63]-b[32:63]
    m64 = _mm_subs_pi8(m64, m64);                                   // r[0:7] = (Saturate_To_Int8)a[0:7]-b[0:7], r[8:15] = (Saturate_To_Int8)a[8:15]-b[8:15], r[16:23] = (Saturate_To_Int8)a[16:23]-b[16:23], r[24:31] = (Saturate_To_Int8)a[24:31]-b[24:31], r[32:39] = (Saturate_To_Int8)a[32:39]-b[32:39], r[40:47] = (Saturate_To_Int8)a[40:47]-b[40:47], r[48:55] = (Saturate_To_Int8)a[48:55]-b[48:55], r[56:63] = (Saturate_To_Int8)a[56:63]-b[56:63]
    m64 = _mm_subs_pi16(m64, m64);                                  // r[0:15] = (Saturate_To_Int16)a[0:15]-b[0:15], r[16:31] = (Saturate_To_Int16)a[16:31]-b[16:31], r[32:47] = (Saturate_To_Int16)a[32:47]-b[32:47], r[48:63] = (Saturate_To_Int16)a[48:63]-b[48:63]
    m64 = _mm_subs_pu8(m64, m64);                                   // r[0:7] = (Saturate_To_UnsignedInt8)a[0:7]-b[0:7], r[8:15] = (Saturate_To_UnsignedInt8)a[8:15]-b[8:15], r[16:23] = (Saturate_To_UnsignedInt8)a[16:23]-b[16:23], r[24:31] = (Saturate_To_UnsignedInt8)a[24:31]-b[24:31], r[32:39] = (Saturate_To_UnsignedInt8)a[32:39]-b[32:39], r[40:47] = (Saturate_To_UnsignedInt8)a[40:47]-b[40:47], r[48:55] = (Saturate_To_UnsignedInt8)a[48:55]-b[48:55], r[56:63] = (Saturate_To_UnsignedInt8)a[56:63]-b[56:63]
    m64 = _mm_subs_pu16(m64, m64);                                  // r[0:15] = (Saturate_To_UnsignedInt16)a[0:15]-b[0:15], r[16:31] = (Saturate_To_UnsignedInt16)a[16:31]-b[16:31], r[32:47] = (Saturate_To_UnsignedInt16)a[32:47]-b[32:47], r[48:63] = (Saturate_To_UnsignedInt16)a[48:63]-b[48:63]
    m64 = _mm_madd_pi16(m64, m64);                                  // r[0:31] = a[16:31]*b[16:31]+a[0:15]*b[0:15], r[32:63] = a[48:63]*b[48:63]+a[32:47]*b[32:47]
    m64 = _mm_mulhi_pi16(m64, m64);                                 // r[0:15] = (a[0:15]*b[0:15])[16:31], r[16:31] = (a[16:31]*b[16:31])[16:31], r[32:47] = (a[32:47]*b[32:47])[16:31], r[48:63] = (a[48:63]*b[48:63])[16:31]
    m64 = _mm_mullo_pi16(m64, m64);                                 // r[0:15] = (a[0:15]*b[0:15])[0:15], r[16:31] = (a[16:31]*b[16:31])[0:15], r[32:47] = (a[32:47]*b[32:47])[0:15], r[48:63] = (a[48:63]*b[48:63])[0:15]
    m64 = _mm_sll_pi16(m64, m64);                                   // r[0:15] = count[0:63]>15?0:(ZeroExtend)(a[0:15]<<count[0:63]), r[16:31] = count[0:63]>15?0:(ZeroExtend)(a[16:31]<<count[0:63]), r[32:47] = count[0:63]>15?0:(ZeroExtend)(a[32:47]<<count[0:63]), r[48:63] = count[0:63]>15?0:(ZeroExtend)(a[48:63]<<count[0:63])
    m64 = _mm_slli_pi16(m64, in);                                   // r[0:15] = imm8[0:7]>15?0:(ZeroExtend)(a[0:15]<<imm8[0:7]), r[16:31] = imm8[0:7]>15?0:(ZeroExtend)(a[16:31]<<imm8[0:7]), r[32:47] = imm3[0:7]>15?0:(ZeroExtend)(a[32:47]<<imm8[0:7]), r[48:63] = imm8[0:7]>15?0:(ZeroExtend)(a[48:63]<<imm8[0:7])
    m64 = _mm_sll_pi32(m64, m64);                                   // r[0:31] = count[0:63]>31?0:(ZeroExtend)(a[0:31]<<count[0:63]), r[32:63] = count[0:63]>31?0:(ZeroExtend)(a[32:63]<<count[0:63])
    m64 = _mm_slli_pi32(m64, in);                                   // r[0:31] = imm8[0:7]>31?0:(ZeroExtend)(a[0:31]<<imm8[0:7]), r[32:63] = imm8[0:7]>31?0:(ZeroExtend)(a[32:63]<<imm8[0:7])
    m64 = _mm_sll_si64(m64, m64);                                   // r[0:63] = count[0:63]>63?0:(ZeroExtend)(a[0:63]<<count[0:63])
    m64 = _mm_slli_si64(m64, in);                                   // r[0:63] = imm8[0:7]>63?0:(ZeroExtend)(a[0:63]<<imm8[0:7])
    m64 = _mm_sra_pi16(m64, m64);                                   // r[0:15] = count[0:63]>15?SignBit:(SignExtend)(a[0:15]>>count[0:63]), r[16:31] = count[0:63]>15?SignBit:(SignExtend)(a[16:31]>>count[0:63]), r[32:47] = count[0:63]>15?SignBit:(SignExtend)(a[32:47]>>count[0:63]), r[48:63] = count[0:63]>15?SignBit:(SignExtend)(a[48:63]>>count[0:63])
    m64 = _mm_srai_pi16(m64, in);                                   // r[0:15] = imm8[0:7]>15?SignBit:(SignExtend)(a[0:15]>>imm8[0:7]), r[16:31] = imm8[0:7]>15?SignBit:(SignExtend)(a[16:31]>>imm8[0:7]), r[32:47] = imm3[0:7]>15?SignBit:(SignExtend)(a[32:47]>>imm8[0:7]), r[48:63] = imm8[0:7]>15?SignBit:(SignExtend)(a[48:63]>>imm8[0:7])
    m64 = _mm_sra_pi32(m64, m64);                                   // r[0:31] = count[0:63]>31?SignBit:(SignExtend)(a[0:31]>>count[0:63]), r[32:63] = count[0:63]>31?SignBit:(SignExtend)(a[32:63]>>count[0:63])
    m64 = _mm_srai_pi32(m64, in);                                   // r[0:31] = imm8[0:7]>31?SignBit:(SignExtend)(a[0:31]>>imm8[0:7]), r[32:63] = imm8[0:7]>31?SignBit:(SignExtend)(a[32:63]>>imm8[0:7])m64 = _mm_srl_pi16(m64, m64);//
    m64 = _mm_srl_pi16(m64, m64);                                   // r[0:15] = count[0:63]>15?0:(ZeroExtend)(a[0:15]>>count[0:63]), r[16:31] = count[0:63]>15?0:(ZeroExtend)(a[16:31]>>count[0:63]), r[32:47] = count[0:63]>15?0:(ZeroExtend)(a[32:47]>>count[0:63]), r[48:63] = count[0:63]>15?0:(ZeroExtend)(a[48:63]>>count[0:63])
    m64 = _mm_srli_pi16(m64, in);                                   // r[0:15] = imm8[0:7]>15?0:(ZeroExtend)(a[0:15]>>imm8[0:7]), r[16:31] = imm8[0:7]>15?0:(ZeroExtend)(a[16:31]>>imm8[0:7]), r[32:47] = imm3[0:7]>15?0:(ZeroExtend)(a[32:47]>>imm8[0:7]), r[48:63] = imm8[0:7]>15?0:(ZeroExtend)(a[48:63]>>imm8[0:7])
    m64 = _mm_srl_pi32(m64, m64);                                   // r[0:31] = count[0:63]>31?0:(ZeroExtend)(a[0:31]>>count[0:63]), r[32:63] = count[0:63]>31?0:(ZeroExtend)(a[32:63]>>count[0:63])
    m64 = _mm_srli_pi32(m64, in);                                   // r[0:31] = imm8[0:7]>31?0:(ZeroExtend)(a[0:31]>>imm8[0:7]), r[32:63] = imm8[0:7]>31?0:(ZeroExtend)(a[32:63]>>imm8[0:7])
    m64 = _mm_srl_si64(m64, m64);                                   // count[0:63]>63>0:r[0:63] = (ZeroExtend)(a[0:63]>>count[0:63])
    m64 = _mm_srli_si64(m64, in);                                   // imm8[0:7]>63>0:r[0:63] = (ZeroExtend)(a[0:63]>>imm8[0:7])
    m64 = _mm_and_si64(m64, m64);                                   // r[0:63] = a[0:63]&b[0:63]
    m64 = _mm_andnot_si64(m64, m64);                                // r[0:63] = ~a[0:63]&b[0:63]
    m64 = _mm_or_si64(m64, m64);                                    // r[0:63] = a[0:63]|b[0:63]
    m64 = _mm_xor_si64(m64, m64);                                   // r[0:63] = a[0:63]^b[0:63]
    m64 = _mm_cmpeq_pi8(m64, m64);                                  // r[0:7] = (a[0:7]==b[0:7])?0xF:0x0, r[8:15] = (a[8:15]==b[8:15])?0xF:0x0, r[16:23] = (a[16:23]==b[16:23])?0xF:0x0, r[24:31] = (a[24:31]==b[24:31])?0xF:0x0, r[32:39] = (a[32:39]==b[32:39])?0xF:0x0, r[40:47] = (a[40:47]==b[40:47])?0xF:0x0, r[48:55] = (a[48:55]==b[48:55])?0xF:0x0, r[56:63] = (a[56:63]==b[56:63])?0xF:0x0
    m64 = _mm_cmpeq_pi16(m64, m64);                                 // r[0:15] = (a[0:15]==b[0:15])?0xF:0x0, r[16:31] = (a[16:31]==b[16:31])?0xF:0x0, r[32:47] = (a[32:47]==b[32:47])?0xF:0x0, r[48:63] = (a[48:63]==b[48:63])?0xF:0x0
    m64 = _mm_cmpeq_pi32(m64, m64);                                 // r[0:31] = (a[0:31]==b[0:31])?0xF:0x0, r[32:63] = (a[32:63]==b[32:63])?0xF:0x0
    m64 = _mm_cmpgt_pi8(m64, m64);                                  // r[0:7] = (a[0:7]>b[0:7])?0xF:0x0, r[8:15] = (a[8:15]>b[8:15])?0xF:0x0, r[16:23] = (a[16:23]>b[16:23])?0xF:0x0, r[24:31] = (a[24:31]>b[24:31])?0xF:0x0, r[32:39] = (a[32:39]>b[32:39])?0xF:0x0, r[40:47] = (a[40:47]>b[40:47])?0xF:0x0, r[48:55] = (a[48:55]>b[48:55])?0xF:0x0, r[56:63] = (a[56:63]>b[56:63])?0xF:0x0
    m64 = _mm_cmpgt_pi16(m64, m64);                                 // r[0:15] = (a[0:15]>b[0:15])?0xF:0x0, r[16:31] = (a[16:31]>b[16:31])?0xF:0x0, r[32:47] = (a[32:47]>b[32:47])?0xF:0x0, r[48:63] = (a[48:63]>b[48:63])?0xF:0x0
    m64 = _mm_cmpgt_pi32(m64, m64);                                 // r[0:31] = (a[0:31]>b[0:31])?0xF:0x0, r[32:63] = (a[32:63]>b[32:63])?0xF:0x0
#endif
}

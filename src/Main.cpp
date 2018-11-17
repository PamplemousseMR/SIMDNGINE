#include "Simd.hpp"

#include <assert.h>

/*
    <mmintrin.h>  MMX       Introduce eight 64 bit registers (MM0-MM7) and instructions to work with eight signed/unsigned bytes, four signed/unsigned words, two signed/unsigned dwords.
    <xmmintrin.h> SSE       Introduce eight/sixteen 128 bit registers (XMM0-XMM7/15) and instruction to work with four single precision floating point operands. Add integer operations on MMX registers too. (The MMX-integer part of SSE is sometimes called MMXEXT, and was implemented on a few non-Intel CPUs without xmm registers and the floating point part of SSE.)
    <emmintrin.h> SSE2      Introduces instruction to work with 2 double precision floating point operands, and with packed byte/word/dword/qword integers in 128-bit xmm registers.
    <pmmintrin.h> SSE3      Add a few varied instructions (mostly floating point), including a special kind of unaligned load (lddqu) that was better on Pentium 4, synchronization instruction, horizontal add/sub.
    <tmmintrin.h> SSSE3     Again a varied set of instructions, mostly integer. The first shuffle that takes its control operand from a register instead of hard-coded (pshufb). More horizontal processing, shuffle, packing/unpacking, mul+add on bytes, and some specialized integer add/mul stuff.
    <smmintrin.h> SSE4.1    Add a lot of instructions: Filling in a lot of the gaps by providing min and max and other operations for all integer data types (especially 32-bit integer had been lacking), where previously integer min was only available for unsigned bytes and signed 16-bit. Also scaling, FP rounding, blending, linear algebra operation, text processing, comparisons. Also a non temporal load for reading video memory, or copying it back to main memory. (Previously only NT stores were available.)
    <nmmintrin.h> SSE4.2    Add a lot of instructions: Filling in a lot of the gaps by providing min and max and other operations for all integer data types (especially 32-bit integer had been lacking), where previously integer min was only available for unsigned bytes and signed 16-bit. Also scaling, FP rounding, blending, linear algebra operation, text processing, comparisons. Also a non temporal load for reading video memory, or copying it back to main memory. (Previously only NT stores were available.)
    <ammintrin.h> SSE4A     Add extension for AMD
    <wmmintrin.h> AES       Add support for accelerating AES symmetric encryption/decryption.
    <immintrin.h> AVX       Add eight/sixteen 256 bit registers (YMM0-YMM7/15). Support all previous floating point datatype. Three operand instructions.
    <immintrin.h> FMA       Add Fused Multiply Add and correlated instructions.
    <immintrin.h> AVX2      Add support for integer data types.
    <zmmintrin.h> AVX512    Add eight/thirty-two 512 bit registers (ZMM0-ZMM7/31) and eight 64-bit mask register (k0-k7). Promote most previous instruction to 512 bit wide. Optional parts of AVX512 add instruction for exponentials & reciprocals (AVX512ER), scatter/gather prefetching (AVX512PF), scatter conflict detection (AVX512CD), compress, expand.
    https://software.intel.com/sites/landingpage/IntrinsicsGuide
*/

int main()
{
	mmintrin();
	xmmintrin();
    emmintrin();
    pmmintrin();
    tmmintrin();

	return 0;
}

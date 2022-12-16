
#include <stdlib.h>
#include "intrin.h"

int8_t Saturate8(int16_t a)
{
	int8_t res;
	if (a > INT8_MAX) res = INT8_MAX;
	else if (a < INT8_MIN) res = INT8_MIN;
	else res = (int8_t)a;
	return res;
}
int16_t Saturate16(int32_t a)
{
	int16_t res;
	if (a > INT16_MAX) res = INT16_MAX;
	else if (a < INT16_MIN) res = INT16_MIN;
	else res = (int16_t)a;
	return res;
}

int32_t SignExtend32(int32_t a)
{
	//int32_t(X << (32 - B)) >> (32 - B)
	return a; //checkit SignExtend32 
}

int16_t SignExtend16(int16_t a)
{
	return a; //checkit SignExtend16 
}

int16_t ZeroExtend16(int16_t a)
{
	//return a & 0x7fff; //checkit ZeroExtend16 
	return a;
}

int32_t ZeroExtend32(int32_t a)
{
	//return a & 0x7fff; //checkit ZeroExtend16 
	return a;
}

int64_t ZeroExtend64(int64_t a)
{
	//return a & 0x7fffffffffffffff; //checkit ZeroExtend64 
	return a;
}

__m128i _mm_madd_epi16_(__m128i a, __m128i b){
	/* 
	Synopsis:
		__m128i _mm_madd_epi16 (__m128i a, __m128i b)
		#include <emmintrin.h>
	Instruction: pmaddwd xmm, xmm
	CPUID Flags: SSE2
	Description:
		Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Horizontally add adjacent pairs of intermediate 32-bit integers, and pack the results in dst.
	Operation:
		FOR j := 0 to 3
			i := j*32
			dst[i+31:i] := SignExtend32(a[i+31:i+16]*b[i+31:i+16]) + SignExtend32(a[i+15:i]*b[i+15:i])
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			5		0.5
		Broadwell		5		1
		Haswell			5		1
		Ivy Bridge		5		1
	*/
	__m128i res;
	for (int i=0; i <= 3; i++)
	{
		res.m128i_i32[i] = SignExtend32(a.m128i_i16[2 * i] * b.m128i_i16[2 * i]) + SignExtend32(a.m128i_i16[2 * i + 1] * b.m128i_i16[2 * i + 1]); 
	}

	return res;
}

__m128i _mm_sign_epi16_(__m128i a, __m128i b){
	/*
	Synopsis:
		__m128i _mm_sign_epi16(__m128i a, __m128i b)
		#include <tmmintrin.h>
	Instruction: psignw xmm, xmm
	CPUID Flags : SSSE3
	Description:
		Negate packed 16 - bit integers in a when the corresponding signed 16 - bit integer in b is negative, and store the results in dst.Element in dst are zeroed out when the corresponding element in b is zero.
	Operation:
		FOR j : = 0 to 7
			i : = j * 16
			IF b[i + 15:i] < 0
				dst[i + 15:i] : = -(a[i + 15:i])
			ELSE IF b[i + 15:i] == 0
				dst[i + 15:i] : = 0
			ELSE
				dst[i + 15:i] : = a[i + 15:i]
			FI
		ENDFOR
	Performance:
		Architecture	Latency	Throughput(CPI)
		Skylake			1		0.5
		Broadwell		1		0.5
		Haswell			1		0.5
		Ivy Bridge		1		0.5
	*/
	__m128i res;
	for (int i=0; i <= 7; i++)
	{
		if (b.m128i_i16[i] < 0) 
			res.m128i_i16[i] = -a.m128i_i16[i];
		else if (b.m128i_i16[i] == 0) 
			res.m128i_i16[i] = 0;
		else 
			res.m128i_i16[i] = a.m128i_i16[i];
	}
	return res;
}

__m64 _mm_sign_pi16_(__m64 a, __m64 b)
{
	/*
	Synopsis:
		__m64 _mm_sign_pi16 (__m64 a, __m64 b)
		#include <tmmintrin.h>
	Instruction: psignw mm, mm
	CPUID Flags: SSSE3
	Description:
		Negate packed 16-bit integers in a when the corresponding signed 16-bit integer in b is negative, and store the results in dst. Element in dst are zeroed out when the corresponding element in b is zero.
	Operation:
		FOR j := 0 to 3
			i := j*16
			IF b[i+15:i] < 0
				dst[i+15:i] := -(a[i+15:i])
			ELSE IF b[i+15:i] == 0
				dst[i+15:i] := 0
			ELSE
				dst[i+15:i] := a[i+15:i]
			FI
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			1		-
		Skylake			1		0.5
		Broadwell		1		0.5
		Haswell			1		0.5
		Ivy Bridge		1		0.5
	*/
	__m64 res;
	for (int i = 0; i <= 3; i++)
	{
		if (b.m64_i16[i] < 0)
			res.m64_i16[i] = -a.m64_i16[i];
		else if (b.m64_i16[i] == 0)
			res.m64_i16[i] = 0;
		else
			res.m64_i16[i] = a.m64_i16[i];
	}
	return res;
}

__m128i _mm_slli_epi16_(__m128i a, int imm8){
	/*
	Synopsis:
		__m128i _mm_slli_epi16 (__m128i a, int imm8)
		#include <emmintrin.h>
	Instruction: psllw xmm, imm8
	CPUID Flags: SSE2
	Description:
		Shift packed 16-bit integers in a left by imm8 while shifting in zeros, and store the results in dst.
	Operation:
		FOR j := 0 to 7
			i := j*16
			IF imm8[7:0] > 15
				dst[i+15:i] := 0
			ELSE
				dst[i+15:i] := ZeroExtend16(a[i+15:i] << imm8[7:0])
			FI
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		0.5
		Broadwell		1		1
		Haswell			1		1
		Ivy Bridge		1		1	
	*/
	__m128i res;
	for (int i=0; i <= 7; i++)
	{
		if (imm8 > 15) 
			res.m128i_i16[i] = 0;
		else 
			res.m128i_i16[i] = ZeroExtend16(a.m128i_i16[i] << imm8); 
	}
	return res;
}

__m128i _mm_slli_epi32_(__m128i a, int imm8)
{
	/*
	Synopsis:
		__m128i _mm_slli_epi32 (__m128i a, int imm8)
		#include <emmintrin.h>
	Instruction: pslld xmm, imm8
	CPUID Flags: SSE2
	Description:
		Shift packed 32-bit integers in a left by imm8 while shifting in zeros, and store the results in dst.
	Operation:
		FOR j := 0 to 3
			i := j*32
			IF imm8[7:0] > 31
				dst[i+31:i] := 0
			ELSE
				dst[i+31:i] := ZeroExtend32(a[i+31:i] << imm8[7:0])
			FI
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		0.5
		Broadwell		1		1
		Haswell			1		1
		Ivy Bridge		1		1
	*/
	__m128i res;
	for (int i = 0; i <= 3; i++)
	{
		if (imm8 > 31)
			res.m128i_i32[i] = 0;
		else
			res.m128i_i32[i] = ZeroExtend32(a.m128i_i16[i] << imm8);
	}
	return res;
}

__m128i _mm_mullo_epi16_(__m128i a, __m128i b){
	/*
	Synopsis:
		__m128i _mm_mullo_epi16 (__m128i a, __m128i b)
		#include <emmintrin.h>
	Instruction: pmullw xmm, xmm
	CPUID Flags: SSE2
	Description:
		Multiply the packed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the low 16 bits of the intermediate integers in dst.
	Operation:
		FOR j := 0 to 7
			i := j*16
			tmp[31:0] := SignExtend32(a[i+15:i]) * SignExtend32(b[i+15:i])
			dst[i+15:i] := tmp[15:0]
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			5		0.5
		Broadwell		5		1
		Haswell			5		1
		Ivy Bridge		5		1
	*/
	__m128i res;
	for (int i=0; i <= 7; i++)
	{
		int tmp;
		tmp = SignExtend32(a.m128i_i16[i]) * SignExtend32(b.m128i_i16[i]);
		res.m128i_i16[i] = tmp;
	}
	return res;
}

__m128i _mm_mulhi_epi16_(__m128i a, __m128i b)
{
	/*
	Synopsis:
		__m128i _mm_mulhi_epi16 (__m128i a, __m128i b)
		#include <emmintrin.h>
	Instruction: pmulhw xmm, xmm
	CPUID Flags: SSE2
	Description:
		Multiply the packed signed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst.
	Operation:
		FOR j := 0 to 7
			i := j*16
			tmp[31:0] := SignExtend32(a[i+15:i]) * SignExtend32(b[i+15:i])
			dst[i+15:i] := tmp[31:16]
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			5		0.5
		Broadwell		5		1
		Haswell			5		1
		Ivy Bridge		5		1
	*/
	__m128i res;
	for (int i=0; i <= 7; i++)
	{
		int tmp;
		tmp = SignExtend32(a.m128i_i16[i]) * SignExtend32(b.m128i_i16[i]); 
		tmp = tmp >> 16;
		res.m128i_i16[i] = tmp;
	}
	return res;
}

__m128i _mm_mulhrs_epi16_(__m128i a, __m128i b)
{
	/*
	Synopsis
		__m128i _mm_mulhrs_epi16 (__m128i a, __m128i b)
		#include <tmmintrin.h>
	Instruction: pmulhrsw xmm, xmm
	CPUID Flags: SSSE3
	Description:
		Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Truncate each intermediate integer to the 18 most significant bits, round by adding 1, and store bits [16:1] to dst.
	Operation:
		FOR j := 0 to 7
			i := j*16
			tmp[31:0] := ((SignExtend32(a[i+15:i]) * SignExtend32(b[i+15:i])) >> 14) + 1
			dst[i+15:i] := tmp[16:1]
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			5		0.5
		Broadwell		5		1
		Haswell			5		1
		Ivy Bridge		5		1	
	*/
	__m128i res;
	for (int i=0; i <= 7; i++)
	{
		int tmp;
		tmp = SignExtend32(a.m128i_i16[i]) * SignExtend32(b.m128i_i16[i]);
		tmp = ((tmp >> 14) + 1) >> 1;
		res.m128i_i16[i] = tmp;
	}
	return res;
}

__m128i _mm_setzero_si128_()
{
	/*
	Synopsis:
		__m128i _mm_setzero_si128 ()
		#include <emmintrin.h>
	Instruction: pxor xmm, xmm
	CPUID Flags: SSE2
	Description:
		Return vector of type __m128i with all elements set to zero.
	Operation:
		dst[MAX:0] := 0
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		0.33
		Broadwell		1		0.33
		Haswell			1		0.33
		Ivy Bridge		1		0.33
	*/
	__m128i res = {0};
	return res;
}

__m64 _mm_setzero_si64_()
{
	/*
	Synopsis:
		__m64 _mm_setzero_si64 (void)
		#include <mmintrin.h>
	Instruction: pxor mm, mm
	CPUID Flags: MMX
	Description:
		Return vector of type __m64 with all elements set to zero.
	Operation:
		dst[MAX:0] := 0
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			1		0.5
		Skylake			1		0.5
	*/
	__m64 res = { 0 };
	return res;
}

__m128 _mm_setzero_ps_()
{
	/*
	Synopsis:
		__m128 _mm_setzero_ps (void)
		#include <xmmintrin.h>
	Instruction: xorps xmm, xmm
	CPUID Flags: SSE
	Description:
		Return vector of type __m128 with all elements set to zero.
	Operation:
		dst[MAX:0] := 0
	Performance:
		Architecture	Latency	Throughput (CPI)
		Broadwell		1		1
		Haswell			1		1
		Ivy Bridge		1		1
	*/
	__m128 res = { 0 };
	return res;
}

__m256i _mm256_setzero_si256_(void)
{
	/*
	Synopsis:
		__m256i _mm256_setzero_si256 (void)
	#include <immintrin.h>
	Instruction: vpxor ymm, ymm, ymm
	CPUID Flags: AVX
	Description:
		Return vector of type __m256i with all elements set to zero.
	Operation:
		dst[MAX:0] := 0
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			1		0.33
		Skylake			1		0.33

	*/
	__m256i res = { 0 };
	return res;
}

__m128i _mm_set1_epi16_(short a)
{
	/*
	Synopsis:
		__m128i _mm_set1_epi16 (short a)
		#include <emmintrin.h>
	Instruction: Sequence
	CPUID Flags: SSE2
	Description:
		Broadcast 16-bit integer a to all all elements of dst. This intrinsic may generate vpbroadcastw.
	Operation:
		FOR j := 0 to 7
			i := j*16
			dst[i+15:i] := a[15:0]
		ENDFOR
	*/
	__m128i res;
	for (int i=0; i <= 7; i++)
	{
		res.m128i_i16[i] = a;
	}
	return res;
}

__m128i _mm_set1_epi32_(int a)
{
	/*
	Synopsis:
		__m128i _mm_set1_epi32 (int a)
		#include <emmintrin.h>
	Instruction: Sequence
	CPUID Flags: SSE2
	Description:
		Broadcast 32-bit integer a to all elements of dst. This intrinsic may generate vpbroadcastd.
	Operation:
		FOR j := 0 to 3
			i := j*32
			dst[i+31:i] := a[31:0]
		ENDFOR
	*/
	__m128i res;
	for (int i=0; i <= 3; i++)
	{
		res.m128i_i32[i] = a;
	}
	return res;
}

__m256i _mm256_set1_epi8_(char a)
{
	/*
	Synopsis:
		__m256i _mm256_set1_epi8 (char a)
		#include <immintrin.h>
	Instruction: Sequence
	CPUID Flags: AVX
	Description:
		Broadcast 8-bit integer a to all elements of dst. This intrinsic may generate the vpbroadcastb.
	Operation:
		FOR j := 0 to 31
			i := j*8
			dst[i+7:i] := a[7:0]
		ENDFOR
		dst[MAX:256] := 0
	*/
	__m256i res = { 0 };
	for (int i = 0; i <= 31; i++)
	{
		res.m256i_i8[i] = a;
	}
	return res;
}

__m256i _mm256_abs_epi8_(__m256i a)
{
	/*
	Synopsis:
		__m256i _mm256_abs_epi8 (__m256i a)
		#include <immintrin.h>
	Instruction: vpabsb ymm, ymm
	CPUID Flags: AVX2
	Description:
		Compute the absolute value of packed signed 8-bit integers in a, and store the unsigned results in dst.
	Operation:
		FOR j := 0 to 31
			i := j*8
			dst[i+7:i] := ABS(a[i+7:i])
		ENDFOR
		dst[MAX:256] := 0
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			1		0.5
		Skylake			1		0.5
	*/
	__m256i res = { 0 };
	for (int i = 0; i <= 31; i++)
	{
		res.m256i_i8[i] = Saturate8(abs(a.m256i_i8[i]));
	}
	return res;
}

int _mm256_movemask_epi8_(__m256i a)
{
	/*
	Synopsis:
		int _mm256_movemask_epi8 (__m256i a)
		#include <immintrin.h>
	Instruction: vpmovmskb r32, ymm
	CPUID Flags: AVX2
	Description:
		Create mask from the most significant bit of each 8-bit element in a, and store the result in dst.
	Operation:
		FOR j := 0 to 31
			i := j*8
			dst[j] := a[i+7]
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			2		1
	*/
	int res = 0;
	for (int i = 0; i <= 31; i++)
	{
		res = res | ((a.m256i_u8[i] >> 7) << i);
	}
	return res;
}

__m256i _mm256_cvtepi8_epi16_(__m128i a)
{
	/*
	Synopsis:
		__m256i _mm256_cvtepi8_epi16 (__m128i a)
		#include <immintrin.h>
	Instruction: vpmovsxbw ymm, xmm
	CPUID Flags: AVX2
	Description:
		Sign extend packed 8-bit integers in a to packed 16-bit integers, and store the results in dst.
	Operation:
		FOR j := 0 to 15
			i := j*8
			l := j*16
			dst[l+15:l] := SignExtend16(a[i+7:i])
		ENDFOR
		dst[MAX:256] := 0
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			3		1
		Skylake			3		1
	*/
	__m256i res = { 0 };
	for (int i = 0; i <= 15; i++)
	{
		res.m256i_i16[i] = SignExtend16(a.m128i_i8[i]);
	}
	return res;
}

__m256i _mm256_set1_epi32_(int a)
{
	/*
	Synopsis:
		__m256i _mm256_set1_epi32 (int a)
		#include <immintrin.h>
	Instruction: Sequence
	CPUID Flags: AVX
	Description:
		Broadcast 32-bit integer a to all elements of dst. This intrinsic may generate the vpbroadcastd.
	Operation:
		FOR j := 0 to 7
			i := j*32
			dst[i+31:i] := a[31:0]
		ENDFOR
		dst[MAX:256] := 0
	*/
	__m256i res = { 0 };
	for (int i = 0; i <= 7; i++)
	{
		res.m256i_i32[i] = a;
	}
	return res;
}

__m256i _mm256_set1_epi64x_(long long a)
{
	/*
	Synopsis:
		__m256i _mm256_set1_epi64x (long long a)
		#include <immintrin.h>
	Instruction: Sequence
	CPUID Flags: AVX
	Description:
		Broadcast 64-bit integer a to all elements of dst. This intrinsic may generate the vpbroadcastq.
	Operation:
		FOR j := 0 to 3
			i := j*64
			dst[i+63:i] := a[63:0]
		ENDFOR
		dst[MAX:256] := 0
	*/
	__m256i res = {0};
	for (int i = 0; i <= 3; i++)
	{
		res.m256i_i64[i] = a;
	}
	return res;
}

long long select4(__m256i src, int ctl){
	long long res = 0;
	ctl = ctl & 3;
	res = src.m256i_i64[ctl];
	return res;
}

__m256i _mm256_permute4x64_epi64_(__m256i a, const int imm8)
{
	/*
	Synopsis:
		__m256i _mm256_permute4x64_epi64 (__m256i a, const int imm8)
		#include <immintrin.h>
	Instruction: vpermq ymm, ymm, imm8
	CPUID Flags: AVX2
	Description:
		Shuffle 64-bit integers in a across lanes using the control in imm8, and store the results in dst.
	Operation:
		DEFINE SELECT4(src, control) {
			CASE(control[1:0]) OF
				0:	tmp[63:0] := src[63:0]
				1:	tmp[63:0] := src[127:64]
				2:	tmp[63:0] := src[191:128]
				3:	tmp[63:0] := src[255:192]
			ESAC
			RETURN tmp[63:0]
		}
		dst[63:0] := SELECT4(a[255:0], imm8[1:0])
		dst[127:64] := SELECT4(a[255:0], imm8[3:2])
		dst[191:128] := SELECT4(a[255:0], imm8[5:4])
		dst[255:192] := SELECT4(a[255:0], imm8[7:6])
		dst[MAX:256] := 0
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			3		1
		Skylake			3		1
	*/
	__m256i res = { 0 };
	res.m256i_i64[0] = select4(a, imm8);
	res.m256i_i64[1] = select4(a, imm8 >> 2);
	res.m256i_i64[2] = select4(a, imm8 >> 4);
	res.m256i_i64[3] = select4(a, imm8 >> 6);
	return res;
}

__m128i _mm_srai_epi16_(__m128i a, int imm8)
{
	/*
	Synopsis:
		__m128i _mm_srai_epi16 (__m128i a, int imm8)
		#include <emmintrin.h>
	Instruction: psraw xmm, imm8
	CPUID Flags: SSE2
	Description:
		Shift packed 16-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst.
	Operation:
		FOR j := 0 to 7
			i := j*16
			IF imm8[7:0] > 15
				dst[i+15:i] := (a[i+15] ? 0xFFFF : 0x0)
			ELSE
				dst[i+15:i] := SignExtend16(a[i+15:i] >> imm8[7:0])
			FI
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		0.5
	*/
	__m128i res;
	for (int i=0; i <= 7; i++)
	{
		if (imm8 > 15)
			res.m128i_i16[i] = a.m128i_i16[i] < 0 ? 0xFFFF : 0x0;
		else
			res.m128i_i16[i] = SignExtend16(a.m128i_i16[i] >> imm8);
	}
	return res;
}

__m256i _mm256_srai_epi16_(__m256i a, int imm8)
{
	/*
	Synopsis:
		__m256i _mm256_srai_epi16 (__m256i a, int imm8)
		#include <immintrin.h>
	Instruction: vpsraw ymm, ymm, imm8
	CPUID Flags: AVX2
	Description:
		Shift packed 16-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst.
	Operation:
		FOR j := 0 to 15
			i := j*16
			IF imm8[7:0] > 15
				dst[i+15:i] := (a[i+15] ? 0xFFFF : 0x0)
			ELSE
				dst[i+15:i] := SignExtend16(a[i+15:i] >> imm8[7:0])
			FI
		ENDFOR
		dst[MAX:256] := 0
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			1		0.5
		Skylake			1		0.5
		Broadwell		1		1
		Haswell			1		1
	*/
	__m256i res;
	for (int i = 0; i <= 16; i++)
	{
		if (imm8 > 15)
			res.m256i_i16[i] = a.m256i_i16[i] < 0 ? 0xFFFF : 0x0;
		else
			res.m256i_i16[i] = SignExtend16(a.m256i_i16[i] >> imm8);
	}
	return res;
}

__m128i _mm_srai_epi32_(__m128i a, int imm8){
/*
	Synopsis:
		__m128i _mm_srai_epi32 (__m128i a, int imm8)
		#include <emmintrin.h>
	Instruction: psrad xmm, imm8
	CPUID Flags: SSE2
	Description:
		Shift packed 32-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst.
	Operation:
		FOR j := 0 to 3
			i := j*32
			IF imm8[7:0] > 31
				dst[i+31:i] := (a[i+31] ? 0xFFFFFFFF : 0x0)
			ELSE
				dst[i+31:i] := SignExtend32(a[i+31:i] >> imm8[7:0])
			FI
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		0.5
	*/
	__m128i res;
	for (int i=0; i <= 3; i++)
	{
		if (imm8 > 31)
			res.m128i_i32[i] = a.m128i_i32[i] < 0 ? 0xFFFFFFFF : 0;
		else
			res.m128i_i32[i] = SignExtend32(a.m128i_i32[i] >> imm8); 
	}
	return res;
}

__m128i _mm_sra_epi32_(__m128i a, __m128i count)
{
	/*
	Synopsis:
		__m128i _mm_sra_epi32 (__m128i a, __m128i count)
		#include <emmintrin.h>
	Instruction: psrad xmm, xmm
	CPUID Flags: SSE2
	Description:
		Shift packed 32-bit integers in a right by count while shifting in sign bits, and store the results in dst.
	Operation:
		FOR j := 0 to 3
			i := j*32
			IF count[63:0] > 31
				dst[i+31:i] := (a[i+31] ? 0xFFFFFFFF : 0x0)
			ELSE
				dst[i+31:i] := SignExtend32(a[i+31:i] >> count[63:0])
			FI
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			2		1
		Broadwell		2		1
		Haswell			2		1
		Ivy Bridge		2		1
	*/
	__m128i res;
	for (int i=0; i <= 3; i++)
	{
		if (count.m128i_i64[0] > 31)
			res.m128i_i32[i] = a.m128i_i32[i] < 0 ? 0xFFFFFFFF : 0;
		else
			res.m128i_i32[i] = SignExtend32(a.m128i_i32[i] >> count.m128i_i64[0]);
	}
	return res;
}

__m128i _mm_packs_epi32_(__m128i a, __m128i b){
	/*
	Synopsis:
		__m128i _mm_packs_epi32 (__m128i a, __m128i b)
		#include <emmintrin.h>
	Instruction: packssdw xmm, xmm
	CPUID Flags: SSE2
	Description:
		Convert packed signed 32-bit integers from a and b to packed 16-bit integers using signed saturation, and store the results in dst.
	Operation:
		dst[15:0] := Saturate16(a[31:0])
		dst[31:16] := Saturate16(a[63:32])
		dst[47:32] := Saturate16(a[95:64])
		dst[63:48] := Saturate16(a[127:96])
		dst[79:64] := Saturate16(b[31:0])
		dst[95:80] := Saturate16(b[63:32])
		dst[111:96] := Saturate16(b[95:64])
		dst[127:112] := Saturate16(b[127:96])
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		1
		Broadwell		1		1
		Haswell			1		1
		Ivy Bridge		1		0.5
	*/
	__m128i res;
	res.m128i_i16[0] = Saturate16(a.m128i_i32[0]);
	res.m128i_i16[1] = Saturate16(a.m128i_i32[1]);
	res.m128i_i16[2] = Saturate16(a.m128i_i32[2]);
	res.m128i_i16[3] = Saturate16(a.m128i_i32[3]);
	res.m128i_i16[4] = Saturate16(b.m128i_i32[0]);
	res.m128i_i16[5] = Saturate16(b.m128i_i32[1]);
	res.m128i_i16[6] = Saturate16(b.m128i_i32[2]);
	res.m128i_i16[7] = Saturate16(b.m128i_i32[3]);	
	return res;
}

__m128i _mm_packs_epi16_(__m128i a, __m128i b){
	/*
	Synopsis:
		__m128i _mm_packs_epi16 (__m128i a, __m128i b)
		#include <emmintrin.h>
	Instruction: packsswb xmm, xmm
	CPUID Flags: SSE2
	Description:
		Convert packed signed 16-bit integers from a and b to packed 8-bit integers using signed saturation, and store the results in dst.
	Operation:
		dst[7:0] := Saturate8(a[15:0])
		dst[15:8] := Saturate8(a[31:16])
		dst[23:16] := Saturate8(a[47:32])
		dst[31:24] := Saturate8(a[63:48])
		dst[39:32] := Saturate8(a[79:64])
		dst[47:40] := Saturate8(a[95:80])
		dst[55:48] := Saturate8(a[111:96])
		dst[63:56] := Saturate8(a[127:112])
		dst[71:64] := Saturate8(b[15:0])
		dst[79:72] := Saturate8(b[31:16])
		dst[87:80] := Saturate8(b[47:32])
		dst[95:88] := Saturate8(b[63:48])
		dst[103:96] := Saturate8(b[79:64])
		dst[111:104] := Saturate8(b[95:80])
		dst[119:112] := Saturate8(b[111:96])
		dst[127:120] := Saturate8(b[127:112])
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		1
		Broadwell		1		1
		Haswell			1		1
		Ivy Bridge		1		0.5
	*/
	__m128i res;
	res.m128i_i8[0] = Saturate8(a.m128i_i16[0]);
	res.m128i_i8[1] = Saturate8(a.m128i_i16[1]);
	res.m128i_i8[2] = Saturate8(a.m128i_i16[2]);
	res.m128i_i8[3] = Saturate8(a.m128i_i16[3]);
	res.m128i_i8[4] = Saturate8(a.m128i_i16[4]);
	res.m128i_i8[5] = Saturate8(a.m128i_i16[5]);
	res.m128i_i8[6] = Saturate8(a.m128i_i16[6]);
	res.m128i_i8[7] = Saturate8(a.m128i_i16[7]);
	res.m128i_i8[8] = Saturate8(b.m128i_i16[0]);
	res.m128i_i8[9] = Saturate8(b.m128i_i16[1]);
	res.m128i_i8[10] = Saturate8(b.m128i_i16[2]);
	res.m128i_i8[11] = Saturate8(b.m128i_i16[3]);
	res.m128i_i8[12] = Saturate8(b.m128i_i16[4]);
	res.m128i_i8[13] = Saturate8(b.m128i_i16[5]);
	res.m128i_i8[14] = Saturate8(b.m128i_i16[6]);
	res.m128i_i8[15] = Saturate8(b.m128i_i16[7]);
	return res;
}

__m128i _mm_unpacklo_epi16_(__m128i a, __m128i b){
	/*	
	Synopsis:
		__m128i _mm_unpacklo_epi16 (__m128i a, __m128i b)
		#include <emmintrin.h>
	Instruction: punpcklwd xmm, xmm
	CPUID Flags: SSE2
	Description:
		Unpack and interleave 16-bit integers from the low half of a and b, and store the results in dst.
	Operation:
		DEFINE INTERLEAVE_WORDS(src1[127:0], src2[127:0]) {
			dst[15:0] := src1[15:0]
			dst[31:16] := src2[15:0]
			dst[47:32] := src1[31:16]
			dst[63:48] := src2[31:16]
			dst[79:64] := src1[47:32]
			dst[95:80] := src2[47:32]
			dst[111:96] := src1[63:48]
			dst[127:112] := src2[63:48]
			RETURN dst[127:0]
		}
		dst[127:0] := INTERLEAVE_WORDS(a[127:0], b[127:0])
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		1
		Broadwell		1		1
		Haswell			1		1
		Ivy Bridge		1		0.5
	*/
	__m128i res;
	res.m128i_i16[0] = a.m128i_i16[0];
	res.m128i_i16[1] = b.m128i_i16[0];
	res.m128i_i16[2] = a.m128i_i16[1]; 
	res.m128i_i16[3] = b.m128i_i16[1];
	res.m128i_i16[4] = a.m128i_i16[2];
	res.m128i_i16[5] = b.m128i_i16[2];
	res.m128i_i16[6] = a.m128i_i16[3];
	res.m128i_i16[7] = b.m128i_i16[3];
	return res;
}

__m128i _mm_unpackhi_epi16_(__m128i a, __m128i b){
	/*
	Synopsis:
		__m128i _mm_unpackhi_epi16 (__m128i a, __m128i b)
		#include <emmintrin.h>
	Instruction: punpckhwd xmm, xmm
	CPUID Flags: SSE2
	Description:
		Unpack and interleave 16-bit integers from the high half of a and b, and store the results in dst.
	Operation:
		DEFINE INTERLEAVE_HIGH_WORDS(src1[127:0], src2[127:0]) {
			dst[15:0] := src1[79:64]
			dst[31:16] := src2[79:64]
			dst[47:32] := src1[95:80]
			dst[63:48] := src2[95:80]
			dst[79:64] := src1[111:96]
			dst[95:80] := src2[111:96]
			dst[111:96] := src1[127:112]
			dst[127:112] := src2[127:112]
			RETURN dst[127:0]
		}
		dst[127:0] := INTERLEAVE_HIGH_WORDS(a[127:0], b[127:0])
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		1
		Broadwell		1		1
		Haswell			1		1
		Ivy Bridge		1		0.5
	*/
	__m128i res;
	res.m128i_i16[0] = a.m128i_i16[4];
	res.m128i_i16[1] = b.m128i_i16[4];
	res.m128i_i16[2] = a.m128i_i16[5];
	res.m128i_i16[3] = b.m128i_i16[5];
	res.m128i_i16[4] = a.m128i_i16[6];
	res.m128i_i16[5] = b.m128i_i16[6];
	res.m128i_i16[6] = a.m128i_i16[7];
	res.m128i_i16[7] = b.m128i_i16[7];
	return res;
}

__m128i _mm_adds_epi16_(__m128i a, __m128i b){
	/*	
	Synopsis:
		__m128i _mm_adds_epi16 (__m128i a, __m128i b)
		#include <emmintrin.h>
	Instruction: paddsw xmm, xmm
	CPUID Flags: SSE2
	Description:
		Add packed signed 16-bit integers in a and b using saturation, and store the results in dst.
	Operation:
		FOR j := 0 to 7
			i := j*16
			dst[i+15:i] := Saturate16( a[i+15:i] + b[i+15:i] )
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		0.5
		Broadwell		1		0.5
		Haswell			1		0.5
		Ivy Bridge		1		0.5
	*/
	__m128i res;
	for (int i=0; i <= 7; i++)
	{
		//res.m128i_i16[i] = a.m128i_i16[i] + b.m128i_i16[i]; //checkit now
		res.m128i_i16[i] = Saturate16(a.m128i_i16[i] + b.m128i_i16[i]);
	}
	return res;
}

__m128i _mm_and_si128_(__m128i a, __m128i b)
{
	/*
	Synopsis:
		__m128i _mm_and_si128 (__m128i a, __m128i b)
		#include <emmintrin.h>
	Instruction: pand xmm, xmm
	CPUID Flags: SSE2
	Description:
		Compute the bitwise AND of 128 bits (representing integer data) in a and b, and store the result in dst.
	Operation:
		dst[127:0] := (a[127:0] AND b[127:0])
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		0.33
		Broadwell		1		0.33
		Haswell			1		0.33
		Ivy Bridge		1		0.33
	*/
	__m128i res;
	for (int i=0; i <= 2; i++)
	{
		res.m128i_i64[i] = a.m128i_i64[i] & b.m128i_i64[i];
	}
	return res;
}

__m256i _mm256_and_si256_(__m256i a, __m256i b)
{
	/*
	Synopsis:
	__m256i _mm256_and_si256 (__m256i a, __m256i b)
		#include <immintrin.h>
		Instruction: vpand ymm, ymm, ymm
	CPUID Flags: AVX2
	Description:
		Compute the bitwise AND of 256 bits (representing integer data) in a and b, and store the result in dst.
	Operation:
		dst[255:0] := (a[255:0] AND b[255:0])
		dst[MAX:256] := 0
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			1		0.33
		Skylake			1		0.33
	*/

	__m256i res;
	for (int i = 0; i <= 3; i++)
	{
		res.m256i_u64[i] = a.m256i_u64[i] & b.m256i_u64[i];
	}
	return res;
}

__m256i _mm256_andnot_si256_(__m256i a, __m256i b)
{
	/*
	Synopsis:
		__m256i _mm256_andnot_si256 (__m256i a, __m256i b)
		#include <immintrin.h>
	Instruction: vpandn ymm, ymm, ymm
	CPUID Flags: AVX2
	Description:
		Compute the bitwise NOT of 256 bits (representing integer data) in a and then AND with b, and store the result in dst.
	Operation:
		dst[255:0] := ((NOT a[255:0]) AND b[255:0])
		dst[MAX:256] := 0
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			1		0.33
		Skylake			1		0.33
	*/

	__m256i res;
	for (int i = 0; i <= 3; i++)
	{
		res.m256i_u64[i] = (~a.m256i_u64[i]) & b.m256i_u64[i];
	}
	return res;
}

__m256i _mm256_or_si256_(__m256i a, __m256i b)
{
	/*
	Synopsis:
		__m256i _mm256_or_si256 (__m256i a, __m256i b)
		#include <immintrin.h>
	Instruction: vpor ymm, ymm, ymm
	CPUID Flags: AVX2
	Description:
		Compute the bitwise OR of 256 bits (representing integer data) in a and b, and store the result in dst.
	Operation:
		dst[255:0] := (a[255:0] OR b[255:0])
		dst[MAX:256] := 0
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			1		0.33
		Skylake			1		0.33
		Broadwell		1		0.33
		Haswell			1		0.33
	*/

	__m256i res;
	for (int i = 0; i <= 3; i++)
	{
		res.m256i_u64[i] = a.m256i_u64[i] | b.m256i_u64[i];
	}
	return res;
}

__m128i _mm_setr_epi16_(short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0){
	/*
	Synopsis:
		__m128i _mm_setr_epi16 (short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0)
		#include <emmintrin.h>
	Instruction: Sequence
	CPUID Flags: SSE2
	Description:
		Set packed 16-bit integers in dst with the supplied values in reverse order.
	Operation:
		dst[15:0] := e7
		dst[31:16] := e6
		dst[47:32] := e5
		dst[63:48] := e4
		dst[79:64] := e3
		dst[95:80] := e2
		dst[111:96] := e1
		dst[127:112] := e0
	*/
	__m128i res;
	res.m128i_i16[0] = e7;
	res.m128i_i16[1] = e6;
	res.m128i_i16[2] = e5;
	res.m128i_i16[3] = e4;
	res.m128i_i16[4] = e3;
	res.m128i_i16[5] = e2;
	res.m128i_i16[6] = e1;
	res.m128i_i16[7] = e0;
	return res;
}


__m128i _mm_set_epi16_(short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0)
{
	/*
	Synopsis:
		__m128i _mm_set_epi16 (short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0)
		#include <emmintrin.h>
	Instruction: Sequence
	CPUID Flags: SSE2
	Description:
		Set packed 16-bit integers in dst with the supplied values.
	Operation:
		dst[15:0] := e0
		dst[31:16] := e1
		dst[47:32] := e2
		dst[63:48] := e3
		dst[79:64] := e4
		dst[95:80] := e5
		dst[111:96] := e6
		dst[127:112] := e7
	*/
	__m128i res;
	res.m128i_i16[0] = e0;
	res.m128i_i16[1] = e1;
	res.m128i_i16[2] = e2;
	res.m128i_i16[3] = e3;
	res.m128i_i16[4] = e4;
	res.m128i_i16[5] = e5;
	res.m128i_i16[6] = e6;
	res.m128i_i16[7] = e7;
	return res;
}

__m128i _mm_setr_epi32_(int e3, int e2, int e1, int e0)
{
	/*
	Synopsis:
		__m128i _mm_setr_epi32 (int e3, int e2, int e1, int e0)
		#include <emmintrin.h>
	Instruction: Sequence
	CPUID Flags: SSE2
	Description:
		Set packed 32-bit integers in dst with the supplied values in reverse order.
	Operation:
		dst[31:0] := e3
		dst[63:32] := e2
		dst[95:64] := e1
		dst[127:96] := e0
	*/
	__m128i res;
	res.m128i_i32[0] = e3;
	res.m128i_i32[1] = e2;
	res.m128i_i32[2] = e1;
	res.m128i_i32[3] = e0;
	return res;
}

__m256i _mm256_set_epi64x_(__int64 e3, __int64 e2, __int64 e1, __int64 e0)
{
	/*
	Synopsis:
		__m256i _mm256_set_epi64x (__int64 e3, __int64 e2, __int64 e1, __int64 e0)
	#include <immintrin.h>
	Instruction: Sequence
	CPUID Flags: AVX
	Description:
		Set packed 64-bit integers in dst with the supplied values.
	Operation:
		dst[63:0] := e0
		dst[127:64] := e1
		dst[191:128] := e2
		dst[255:192] := e3
		dst[MAX:256] := 0
	*/
	__m256i res;
	res.m256i_i64[0] = e0;
	res.m256i_i64[1] = e1;
	res.m256i_i64[2] = e2;
	res.m256i_i64[3] = e3;

	return res;
}

__m128i _mm_shufflelo_epi16_(__m128i a, int imm8){
	/*
	Synopsis:
		__m128i _mm_shufflelo_epi16 (__m128i a, int imm8)
		#include <emmintrin.h>
	Instruction: pshuflw xmm, xmm, imm8
	CPUID Flags: SSE2
	Description:
		Shuffle 16-bit integers in the low 64 bits of a using the control in imm8. Store the results in the low 64 bits of dst, with the high 64 bits being copied from from a to dst.
	Operation:
		dst[15:0] := (a >> (imm8[1:0] * 16))[15:0]
		dst[31:16] := (a >> (imm8[3:2] * 16))[15:0]
		dst[47:32] := (a >> (imm8[5:4] * 16))[15:0]
		dst[63:48] := (a >> (imm8[7:6] * 16))[15:0]
		dst[127:64] := a[127:64]
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		1
		Broadwell		1		1
		Haswell			1		1
		Ivy Bridge		1		0.5
	*/
	__m128i res;
	res.m128i_i16[0] = a.m128i_i16[(imm8 & 0x3)];
	res.m128i_i16[1] = a.m128i_i16[((imm8 >> 2) & 0x3)];
	res.m128i_i16[2] = a.m128i_i16[((imm8 >> 4) & 0x3)];
	res.m128i_i16[3] = a.m128i_i16[((imm8 >> 6) & 0x3)];
	res.m128i_i64[1] = a.m128i_i64[1]; 
	return res;
}

__m128i _mm_shufflehi_epi16_(__m128i a, int imm8){
	/*
	Synopsis:
		__m128i _mm_shufflehi_epi16 (__m128i a, int imm8)
		#include <emmintrin.h>
	Instruction: pshufhw xmm, xmm, imm8
	CPUID Flags: SSE2
	Description:
		Shuffle 16-bit integers in the high 64 bits of a using the control in imm8. Store the results in the high 64 bits of dst, with the low 64 bits being copied from from a to dst.
	Operation:
		dst[63:0] := a[63:0]
		dst[79:64] := (a >> (imm8[1:0] * 16))[79:64]
		dst[95:80] := (a >> (imm8[3:2] * 16))[79:64]
		dst[111:96] := (a >> (imm8[5:4] * 16))[79:64]
		dst[127:112] := (a >> (imm8[7:6] * 16))[79:64]
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		1
		Broadwell		1		1
		Haswell			1		1
		Ivy Bridge		1		0.5
	*/
	__m128i res;
	res.m128i_i64[0] = a.m128i_i64[0]; 
	res.m128i_i16[4] = a.m128i_i16[4 + (imm8 & 0x3)];
	res.m128i_i16[5] = a.m128i_i16[4 + ((imm8 >> 2) & 0x3)];
	res.m128i_i16[6] = a.m128i_i16[4 + ((imm8 >> 4) & 0x3)];
	res.m128i_i16[7] = a.m128i_i16[4 + ((imm8 >> 6) & 0x3)];
	return res;
}

__m128i _mm_add_epi32_(__m128i a, __m128i b){
	/*
	Synopsis:
		__m128i _mm_add_epi32 (__m128i a, __m128i b)
		#include <emmintrin.h>
	Instruction: paddd xmm, xmm
	CPUID Flags: SSE2
	Description:
		Add packed 32-bit integers in a and b, and store the results in dst.
	Operation:
		FOR j := 0 to 3
			i := j*32
			dst[i+31:i] := a[i+31:i] + b[i+31:i]
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		0.33
		Broadwell		1		0.5
		Haswell			1		0.5
		Ivy Bridge		1		0.5
	*/
	__m128i res;
	for (int i=0; i <= 3; i++)
	{
		res.m128i_i32[i] = a.m128i_i32[i] + b.m128i_i32[i];
	}
	return res;
}

__m128i _mm_add_epi64_(__m128i a, __m128i b)
{
	/*
	Synopsis:
		__m128i _mm_add_epi64 (__m128i a, __m128i b)
		#include <emmintrin.h>
	Instruction: paddq xmm, xmm
	CPUID Flags: SSE2
	Description:
		Add packed 64-bit integers in a and b, and store the results in dst.
	Operation:
		FOR j := 0 to 1
			i := j*64
			dst[i+63:i] := a[i+63:i] + b[i+63:i]
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		0.33
		Broadwell		1		0.5
		Haswell			1		0.5
		Ivy Bridge		1		0.5
	*/
	__m128i res;
	for (int i = 0; i <= 2; i++)
	{
		res.m128i_i64[i] = a.m128i_i64[i] + b.m128i_i64[i];
	}
	return res;
}

__m128i _mm_sub_epi64_(__m128i a, __m128i b)
{
	/*
	Synopsis:
		__m128i _mm_sub_epi64 (__m128i a, __m128i b)
		#include <emmintrin.h>
	Instruction: psubq xmm, xmm
	CPUID Flags: SSE2
	Description:
		Subtract packed 64-bit integers in b from packed 64-bit integers in a, and store the results in dst.
	Operation:
		FOR j := 0 to 1
			i := j*64
			dst[i+63:i] := a[i+63:i] - b[i+63:i]
		ENDFOR
	Performance:
	Architecture	Latency	Throughput (CPI)
	Skylake			1		0.33
	*/
	__m128i res;
	for (int i = 0; i <= 2; i++)
	{
		res.m128i_i64[i] = a.m128i_i64[i] - b.m128i_i64[i];
	}
	return res;
}

__m128i _mm_add_epi16_(__m128i a, __m128i b){
	/*
	Synopsis:
		__m128i _mm_add_epi16 (__m128i a, __m128i b)
		#include <emmintrin.h>
	Instruction: paddw xmm, xmm
	CPUID Flags: SSE2
	Description:
		Add packed 16-bit integers in a and b, and store the results in dst.
	Operation:
		FOR j := 0 to 7
			i := j*16
			dst[i+15:i] := a[i+15:i] + b[i+15:i]
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		0.33
		Broadwell		1		0.5
		Haswell			1		0.5
		Ivy Bridge		1		0.5
	*/
	__m128i res;
	for (int i = 0; i <= 7; i++)
	{
		res.m128i_i16[i] = a.m128i_i16[i] + b.m128i_i16[i];
	}
	return res;
}

__m128i _mm_unpacklo_epi32_(__m128i a, __m128i b){
	/*
	Synopsis:
		__m128i _mm_unpacklo_epi32 (__m128i a, __m128i b)
		#include <emmintrin.h>
	Instruction: punpckldq xmm, xmm
	CPUID Flags: SSE2
	Description:
		Unpack and interleave 32-bit integers from the low half of a and b, and store the results in dst.
	Operation:
		DEFINE INTERLEAVE_DWORDS(src1[127:0], src2[127:0]) {
			dst[31:0] := src1[31:0]
			dst[63:32] := src2[31:0]
			dst[95:64] := src1[63:32]
			dst[127:96] := src2[63:32]
			RETURN dst[127:0]
		}
		dst[127:0] := INTERLEAVE_DWORDS(a[127:0], b[127:0])
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		1
		Broadwell		1		1
		Haswell			1		1
		Ivy Bridge		1		0.5
	*/
	__m128i res;
	for (int i=0; i <= 7; i++)
	{
		res.m128i_i32[0] = a.m128i_i32[0];
		res.m128i_i32[1] = b.m128i_i32[0];
		res.m128i_i32[2] = a.m128i_i32[1];
		res.m128i_i32[3] = b.m128i_i32[1];
	}
	return res;
}

__m128i _mm_unpackhi_epi32_(__m128i a, __m128i b){
	/*
	Synopsis:
		__m128i _mm_unpackhi_epi32 (__m128i a, __m128i b)
		#include <emmintrin.h>
	Instruction: punpckhdq xmm, xmm
	CPUID Flags: SSE2
	Description:
		Unpack and interleave 32-bit integers from the high half of a and b, and store the results in dst.
	Operation:
		DEFINE INTERLEAVE_HIGH_DWORDS(src1[127:0], src2[127:0]) {
			dst[31:0] := src1[95:64]
			dst[63:32] := src2[95:64]
			dst[95:64] := src1[127:96]
			dst[127:96] := src2[127:96]
			RETURN dst[127:0]
		}
	dst[127:0] := INTERLEAVE_HIGH_DWORDS(a[127:0], b[127:0])
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		1
		Broadwell		1		1
		Haswell			1		1
		Ivy Bridge		1		0.5
	*/
	__m128i res;
	for (int i=0; i <= 7; i++)
	{
		res.m128i_i32[0] = a.m128i_i32[2];
		res.m128i_i32[1] = b.m128i_i32[2];
		res.m128i_i32[2] = a.m128i_i32[3];
		res.m128i_i32[3] = b.m128i_i32[3];
	}
	return res;
}

__m128i _mm_sub_epi32_(__m128i a, __m128i b){
	/*
	Synopsis:
		__m128i _mm_sub_epi32 (__m128i a, __m128i b)
		#include <emmintrin.h>
	Instruction: psubd xmm, xmm
	CPUID Flags: SSE2
	Description:
		Subtract packed 32-bit integers in b from packed 32-bit integers in a, and store the results in dst.
	Operation:
		FOR j := 0 to 3
			i := j*32
			dst[i+31:i] := a[i+31:i] - b[i+31:i]
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		0.33
		Broadwell		1		0.5
		Haswell			1		0.5
		Ivy Bridge		1		0.5
	*/
	__m128i res;
	for (int i=0; i <= 3; i++)
	{
		res.m128i_i32[i] = a.m128i_i32[i] - b.m128i_i32[i];
	}
	return res;
}

__m128i _mm_abs_epi32_(__m128i a)
{
	/*
	Synopsis:
		__m128i _mm_abs_epi32 (__m128i a)
		#include <tmmintrin.h>
	Instruction: pabsd xmm, xmm
	CPUID Flags: SSSE3
	Description:
		Compute the absolute value of packed signed 32-bit integers in a, and store the unsigned results in dst.
	Operation:
		FOR j := 0 to 3
			i := j*32
			dst[i+31:i] := ABS(a[i+31:i])
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		0.5
		Broadwell		1		0.5
		Haswell			1		0.5
		Ivy Bridge		1		0.5
	*/
	__m128i res;
	for (int i = 0; i <= 3; i++)
	{
		res.m128i_i32[i] = abs(a.m128i_i32[i]);
	}
	return res;
}

__m128i _mm_sub_epi16_(__m128i a, __m128i b){
	/*
	Synopsis:
		__m128i _mm_sub_epi16 (__m128i a, __m128i b)
		#include <emmintrin.h>
	Instruction: psubw xmm, xmm
	CPUID Flags: SSE2
	Description:
		Subtract packed 16-bit integers in b from packed 16-bit integers in a, and store the results in dst.
	Operation:
		FOR j := 0 to 7
			i := j*16
			dst[i+15:i] := a[i+15:i] - b[i+15:i]
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		0.33
		Broadwell		1		0.5
		Haswell			1		0.5
		Ivy Bridge		1		0.5
	*/
	__m128i res;
	for (int i=0; i <= 7; i++)
	{
		res.m128i_i16[i] = a.m128i_i16[i] - b.m128i_i16[i];
	}
	return res;
}

__m128i _mm_subs_epi16_(__m128i a, __m128i b){
	/*
	Synopsis:
		__m128i _mm_subs_epi16 (__m128i a, __m128i b)
		#include <emmintrin.h>
	Instruction: psubsw xmm, xmm
	CPUID Flags: SSE2
	Description:
		Subtract packed signed 16-bit integers in b from packed 16-bit integers in a using saturation, and store the results in dst.
	Operation:
		FOR j := 0 to 7
			i := j*16
			dst[i+15:i] := Saturate16(a[i+15:i] - b[i+15:i])
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		0.5
		Broadwell		1		0.5
		Haswell			1		0.5
		Ivy Bridge		1		0.5
	*/
	__m128i res;
	for (int i=0; i <= 7; i++)
	{
		res.m128i_i16[i] = Saturate16(a.m128i_i16[i] - b.m128i_i16[i]);
	}
	return res;
}

__m128i _mm_shuffle_epi8_(__m128i a, __m128i b)
{
	/*
	Synopsis:
		__m128i _mm_shuffle_epi8 (__m128i a, __m128i b)
		#include <tmmintrin.h>
	Instruction: pshufb xmm, xmm
	CPUID Flags: SSSE3
	Description:
		Shuffle packed 8-bit integers in a according to shuffle control mask in the corresponding 8-bit element of b, and store the results in dst.
	Operation:
		FOR j := 0 to 15
			i := j*8
			IF b[i+7] == 1
				dst[i+7:i] := 0
			ELSE
				index[3:0] := b[i+3:i]
				dst[i+7:i] := a[index*8+7:index*8]
			FI
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		1
		Broadwell		1		1
		Haswell			1		1
		Ivy Bridge		1		0.5
	*/
	__m128i res;
	for (int i=0; i <= 15; i++)
	{
		if (b.m128i_i8[i] & 0x80)
			res.m128i_i8[i] = 0;
		else
		{
			int index = b.m128i_i8[i] & 0xf;
			res.m128i_i8[i] = a.m128i_i8[index];
		}
	}
	return res;
}

__m256i _mm256_shuffle_epi8_(__m256i a, __m256i b)
{
	/*
	Synopsis:
		__m256i _mm256_shuffle_epi8 (__m256i a, __m256i b)
		#include <immintrin.h>
	Instruction: vpshufb ymm, ymm, ymm
	CPUID Flags: AVX2
	Description:
		Shuffle 8-bit integers in a within 128-bit lanes according to shuffle control mask in the corresponding 8-bit element of b, and store the results in dst.
	Operation:
		FOR j := 0 to 15
			i := j*8
			IF b[i+7] == 1
				dst[i+7:i] := 0
			ELSE
				index[3:0] := b[i+3:i]
				dst[i+7:i] := a[index*8+7:index*8]
			FI
			IF b[128+i+7] == 1
				dst[128+i+7:128+i] := 0
			ELSE
				index[3:0] := b[128+i+3:128+i]
				dst[128+i+7:128+i] := a[128+index*8+7:128+index*8]
			FI
		ENDFOR
		dst[MAX:256] := 0
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			1		0.5
		Skylake			1		1
		Broadwell		1		1
		Haswell			1		1
	*/
	__m256i res = {0};
	for (int i = 0; i <= 15; i++)
	{
		if (b.m256i_i8[i] & 0x80)
			res.m256i_i8[i] = 0;
		else
		{
			int index = b.m256i_i8[i] & 0xf; //chekit
			res.m256i_i8[i] = a.m256i_i8[index];
		}
		if (b.m256i_i8[i+16] & 0x80)
			res.m256i_i8[i+16] = 0;
		else
		{
			int index = b.m256i_i8[i+16] & 0xf; //chekit
			res.m256i_i8[i+16] = a.m256i_i8[index+16];
		}
	}
	return res;
}

__m128i _mm_set_epi8_(char e15, char e14, char e13, char e12, char e11, char e10, char e9, char e8, char e7, char e6, char e5, char e4, char e3, char e2, char e1, char e0){
	/*
	Synopsis:
		__m128i _mm_set_epi8 (char e15, char e14, char e13, char e12, char e11, char e10, char e9, char e8, char e7, char e6, char e5, char e4, char e3, char e2, char e1, char e0)
		#include <emmintrin.h>
	Instruction: Sequence
	CPUID Flags: SSE2
	Description:
		Set packed 8-bit integers in dst with the supplied values.
	Operation:
		dst[7:0] := e0
		dst[15:8] := e1
		dst[23:16] := e2
		dst[31:24] := e3
		dst[39:32] := e4
		dst[47:40] := e5
		dst[55:48] := e6
		dst[63:56] := e7
		dst[71:64] := e8
		dst[79:72] := e9
		dst[87:80] := e10
		dst[95:88] := e11
		dst[103:96] := e12
		dst[111:104] := e13
		dst[119:112] := e14
		dst[127:120] := e15
	*/
	__m128i res;
	res.m128i_i8[0] = e0;
	res.m128i_i8[1] = e1;
	res.m128i_i8[2] = e2;
	res.m128i_i8[3] = e3;
	res.m128i_i8[4] = e4;
	res.m128i_i8[5] = e5;
	res.m128i_i8[6] = e6;
	res.m128i_i8[7] = e7;
	res.m128i_i8[8] = e8;
	res.m128i_i8[9] = e9;
	res.m128i_i8[10] = e10;
	res.m128i_i8[11] = e11;
	res.m128i_i8[12] = e12;
	res.m128i_i8[13] = e13;
	res.m128i_i8[14] = e14;
	res.m128i_i8[15] = e15;
	return res;
}

__m128i _mm_unpacklo_epi64_(__m128i a, __m128i b){
	/*
	Synopsis:
		__m128i _mm_unpacklo_epi64 (__m128i a, __m128i b)
		#include <emmintrin.h>
	Instruction: punpcklqdq xmm, xmm
	CPUID Flags: SSE2
	Description:
		Unpack and interleave 64-bit integers from the low half of a and b, and store the results in dst.
	Operation:
		DEFINE INTERLEAVE_QWORDS(src1[127:0], src2[127:0]) {
			dst[63:0] := src1[63:0]
			dst[127:64] := src2[63:0]
			RETURN dst[127:0]
		}
		dst[127:0] := INTERLEAVE_QWORDS(a[127:0], b[127:0])
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		1
		Broadwell		1		1
		Haswell			1		1
		Ivy Bridge		1		0.5
	*/
	__m128i res;
	res.m128i_i64[0] = a.m128i_i64[0];
	res.m128i_i64[1] = b.m128i_i64[0];
	return res;
}

__m128i _mm_unpackhi_epi64_(__m128i a, __m128i b){
	/*
	Synopsis:
		__m128i _mm_unpackhi_epi64 (__m128i a, __m128i b)
		#include <emmintrin.h>
	Instruction: punpckhqdq xmm, xmm
	CPUID Flags: SSE2
	Description:
		Unpack and interleave 64-bit integers from the high half of a and b, and store the results in dst.
	Operation:
		DEFINE INTERLEAVE_HIGH_QWORDS(src1[127:0], src2[127:0]) {
			dst[63:0] := src1[127:64]
			dst[127:64] := src2[127:64]
			RETURN dst[127:0]
		}
		dst[127:0] := INTERLEAVE_HIGH_QWORDS(a[127:0], b[127:0])
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		1
		Broadwell		1		1
		Haswell			1		1
		Ivy Bridge		1		0.5
	*/
	__m128i res;
	res.m128i_i64[0] = a.m128i_i64[1];
	res.m128i_i64[1] = b.m128i_i64[1];
	return res;
}

__m128i _mm_abs_epi16_(__m128i a)
{
	/*
	Synopsis:
		__m128i _mm_abs_epi16 (__m128i a)
		#include <tmmintrin.h>
	Instruction: pabsw xmm, xmm
	CPUID Flags: SSSE3
	Description:
		Compute the absolute value of packed signed 16-bit integers in a, and store the unsigned results in dst.
	Operation:
		FOR j := 0 to 7
			i := j*16
			dst[i+15:i] := ABS(a[i+15:i])
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		0.5
		Broadwell		1		0.5
		Haswell			1		0.5
		Ivy Bridge		1		0.5
	*/
	__m128i res;
	for (int i=0; i <= 7; i++)
	{
		res.m128i_i16[i] = abs(a.m128i_i16[i]);
	}
	return res;
}

__m64 _mm_abs_pi16_(__m64 a)
{
	/*
	Synopsis:
		__m64 _mm_abs_pi16 (__m64 a)
		#include <tmmintrin.h>
	Instruction: pabsw mm, mm
	CPUID Flags: SSSE3
	Description:
		Compute the absolute value of packed signed 16-bit integers in a, and store the unsigned results in dst.
	Operation:
		FOR j := 0 to 3
			i := j*16
			dst[i+15:i] := ABS(Int(a[i+15:i]))
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			1		-
		Skylake			1		0.5
		Broadwell		1		0.5
		Haswell			1		0.5
		Ivy Bridge		1		0.5
	*/
	__m64 res;
	for (int i = 0; i <= 3; i++)
	{
		res.m64_i16[i] = abs(a.m64_i16[i]);
	}
	return res;
}

__m64 _mm_unpacklo_pi32_(__m64 a, __m64 b)
{
	/*
	Synopsis:
		__m64 _mm_unpacklo_pi32 (__m64 a, __m64 b)
		#include <mmintrin.h>
	Instruction: punpckldq mm, mm
	CPUID Flags: MMX
	Description:
		Unpack and interleave 32-bit integers from the low half of a and b, and store the results in dst.
	Operation:
		dst[31:0] := a[31:0]
		dst[63:32] := b[31:0]
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			1		1
		Skylake			1		1
	*/
	__m64 res;
	res.m64_i32[0] = a.m64_i32[0];
	res.m64_i32[1] = b.m64_i32[0];
	return res;
}

__m64 _mm_unpackhi_pi32_(__m64 a, __m64 b)
{
	/*
	Synopsis:
		__m64 _mm_unpackhi_pi32 (__m64 a, __m64 b)
		#include <mmintrin.h>
	Instruction: punpckhdq mm, mm
	CPUID Flags: MMX
	Description:
		Unpack and interleave 32-bit integers from the high half of a and b, and store the results in dst.
	Operation:
		dst[31:0] := a[63:32]
		dst[63:32] := b[63:32]
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			1		1
		Skylake			1		1
	*/
	__m64 res;
	res.m64_i32[0] = a.m64_i32[1];
	res.m64_i32[1] = b.m64_i32[1];
	return res;
}

int _mm_extract_epi16_(__m128i a, int imm8)
{
	/*
	Synopsis:
		int _mm_extract_epi16 (__m128i a, int imm8)
		#include <emmintrin.h>
	Instruction: pextrw r32, xmm, imm8
	CPUID Flags: SSE2
	Description:
		Extract a 16-bit integer from a, selected with imm8, and store the result in the lower element of dst.
	Operation:
		dst[15:0] := (a[127:0] >> (imm8[2:0] * 16))[15:0]
		dst[31:16] := 0
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			3		1
	*/
	int res;
	res = a.m128i_i16[imm8 & 0x3];
	return res;
}

int _mm_extract_epi32_(__m128i a, int imm8)
{
	/*
	Synopsis:
		int _mm_extract_epi32 (__m128i a, const int imm8)
		#include <smmintrin.h>
	Instruction: pextrd r32, xmm, imm8
	CPUID Flags: SSE4.1
	Description:
		Extract a 32-bit integer from a, selected with imm8, and store the result in dst.
	Operation:
		dst[31:0] := (a[127:0] >> (imm8[1:0] * 32))[31:0]
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			3		1
	*/
	int res;
	res = a.m128i_i32[imm8 & 0x3];
	return res;
}

__int64 _mm_extract_epi64_(__m128i a, const int imm8)
{
	/*
	Synopsis
		__int64 _mm_extract_epi64 (__m128i a, const int imm8)
		#include <smmintrin.h>
	Instruction: pextrq r64, xmm, imm8
	CPUID Flags: SSE4.1
	Description
		Extract a 64-bit integer from a, selected with imm8, and store the result in dst.
	Operation
		dst[63:0] := (a[127:0] >> (imm8[0] * 64))[63:0]
	Performance
		Architecture	Latency	Throughput (CPI)
		Skylake			3		1
	*/
	__int64 res;
	res = a.m128i_i64[imm8];
	return res;
}

__m128 _mm_cvtepi32_ps_(__m128i a)
{
	/*
	Synopsis:
		__m128 _mm_cvtepi32_ps (__m128i a)
		#include <emmintrin.h>
	Instruction: cvtdq2ps xmm, xmm
	CPUID Flags: SSE2
	Description:
		Convert packed signed 32-bit integers in a to packed single-precision (32-bit) floating-point elements, and store the results in dst.
	Operation:
		FOR j := 0 to 3
			i := 32*j
			dst[i+31:i] := Convert_Int32_To_FP32(a[i+31:i])
		ENDFOR
	*/
	__m128 res;
	for (int i=0; i <= 3; i++)
	{
		res.m128_f32[i] = (float) a.m128i_i32[i];
	}
	return res;
}

__m128 _mm_rcp_ps_(__m128 a)
{
	/*
	Synopsis:
		__m128 _mm_rcp_ps (__m128 a)
		#include <xmmintrin.h>
	Instruction: rcpps xmm, xmm
	CPUID Flags: SSE
	Description:
		Compute the approximate reciprocal of packed single-precision (32-bit) floating-point elements in a, and store the results in dst. The maximum relative error for this approximation is less than 1.5*2^-12.
	Operation:
		FOR j := 0 to 3
			i := j*32
			dst[i+31:i] := (1.0 / a[i+31:i])
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			4		1
		Broadwell		5		1
		Haswell			5		1
		Ivy Bridge		5		1
	*/
	__m128 res;
	for (int i=0; i <= 3; i++)
	{
		res.m128_f32[i] = (1 / a.m128_f32[i]);
	}
	return res;
}

__m128 _mm_mul_ps_(__m128 a, __m128 b)
{
	/*
	Synopsis:
		__m128 _mm_mul_ps (__m128 a, __m128 b)
		#include <xmmintrin.h>
	Instruction: mulps xmm, xmm
	CPUID Flags: SSE
	Description:
		Multiply packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.
	Operation:
		FOR j := 0 to 3
			i := j*32
			dst[i+31:i] := a[i+31:i] * b[i+31:i]
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			4		0.5
		Broadwell		3		0.5
		Haswell			5		0.5
		Ivy Bridge		5		1
	*/
	__m128 res;
	for (int i=0; i <= 3; i++)
	{
		res.m128_f32[i] = a.m128_f32[i] * b.m128_f32[i];
	}
	return res;
}

__m128 _mm_add_ps_(__m128 a, __m128 b)
{
	/*
	Synopsis:
		__m128 _mm_add_ps (__m128 a, __m128 b)
		#include <xmmintrin.h>
	Instruction: addps xmm, xmm
	CPUID Flags: SSE
	Description:
		Add packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.
	Operation:
		FOR j := 0 to 3
			i := j*32
			dst[i+31:i] := a[i+31:i] + b[i+31:i]
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			4		0.5
		Broadwell		3		1
		Haswell			3		1
		Ivy Bridge		3		1
	*/
	__m128 res;
	for (int i=0; i <= 3; i++)
	{
		res.m128_f32[i] = a.m128_f32[i] + b.m128_f32[i];
	}
	return res;
}

__m128 _mm_sub_ps_(__m128 a, __m128 b)
{
	/*
	Synopsis:
		__m128 _mm_sub_ps (__m128 a, __m128 b)
		#include <xmmintrin.h>
	Instruction: subps xmm, xmm
	CPUID Flags: SSE
	Description:
		Subtract packed single-precision (32-bit) floating-point elements in b from packed single-precision (32-bit) floating-point elements in a, and store the results in dst.
	Operation:
		FOR j := 0 to 3
			i := j*32
			dst[i+31:i] := a[i+31:i] - b[i+31:i]
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			4		0.5
		Broadwell		3		1
		Haswell			3		1
		Ivy Bridge		3		1
	*/
	__m128 res;
	for (int i=0; i <= 3; i++)
	{
		res.m128_f32[i] = a.m128_f32[i] - b.m128_f32[i];
	}
	return res;
}

__m128 _mm_dp_ps(__m128 a, __m128 b, const int imm8)
{
	/*
	
	Synopsis:
		__m128 _mm_dp_ps (__m128 a, __m128 b, const int imm8)
		#include <smmintrin.h>
	Instruction: dpps xmm, xmm, imm8
	CPUID Flags: SSE4.1
	Description:
		Conditionally multiply the packed single-precision (32-bit) floating-point elements in a and b using the high 4 bits in imm8, sum the four products, and conditionally store the sum in dst using the low 4 bits of imm8.
	Operation:
		DEFINE DP(a[127:0], b[127:0], imm8[7:0]) {
			FOR j := 0 to 3
				i := j*32
				IF imm8[(4+j)%8]
					temp[i+31:i] := a[i+31:i] * b[i+31:i]
				ELSE
					temp[i+31:i] := 0
				FI
			ENDFOR

			sum[31:0] := (temp[127:96] + temp[95:64]) + (temp[63:32] + temp[31:0])

			FOR j := 0 to 3
				i := j*32
				IF imm8[j%8]
					tmpdst[i+31:i] := sum[31:0]
				ELSE
					tmpdst[i+31:i] := 0
				FI
			ENDFOR
			RETURN tmpdst[127:0]
		}
		dst[127:0] := DP(a[127:0], b[127:0], imm8[7:0])

	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			11		1.5
	*/
	__m128 res;
	__m128 tmp;
	for (int i=0; i <= 3; i++)
	{
		if ((imm8 >> (4 + i)) & 0x1)
			tmp.m128_f32[i] = a.m128_f32[i] * b.m128_f32[i];
		else
			tmp.m128_f32[i] = 0;

	}
	float sum = (tmp.m128_f32[3] + tmp.m128_f32[2]) + (tmp.m128_f32[1] + tmp.m128_f32[0]);
	for (int i=0; i <= 3; i++)
	{
		if ((imm8 >> (i)) & 0x1)
			res.m128_f32[i] = sum;
		else
			res.m128_f32[i] = 0;

	}	return res;
}

__m128i _mm_cmpeq_epi16_(__m128i a, __m128i b)
{
	/*
	Synopsis:
		__m128i _mm_cmpeq_epi16 (__m128i a, __m128i b)
		#include <emmintrin.h>
	Instruction: pcmpeqw xmm, xmm
	CPUID Flags: SSE2
	Description:
		Compare packed 16-bit integers in a and b for equality, and store the results in dst.
	Operation:
		FOR j := 0 to 7
		i := j*16
		dst[i+15:i] := ( a[i+15:i] == b[i+15:i] ) ? 0xFFFF : 0
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		0.5
		Broadwell		1		0.5
		Haswell			1		0.5
		Ivy Bridge		1		0.5
	*/
	__m128i res;
	for (int i=0; i <= 7; i++)
	{
		res.m128i_i16[i] = (a.m128i_i16[i] == b.m128i_i16[i]) ? 0xffff : 0;
	}
	return res;
}

__m128i _mm_cmplt_epi16_(__m128i a, __m128i b)
{
	/*
	Synopsis:
		__m128i _mm_cmplt_epi16 (__m128i a, __m128i b)
		#include <emmintrin.h>
	Instruction: pcmpgtw xmm, xmm
	CPUID Flags: SSE2
	Description:
		Compare packed signed 16-bit integers in a and b for less-than, and store the results in dst. Note: This intrinsic emits the pcmpgtw instruction with the order of the operands switched.
	Operation:
		FOR j := 0 to 7
			i := j*16
			dst[i+15:i] := ( a[i+15:i] < b[i+15:i] ) ? 0xFFFF : 0
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		0.5
		Broadwell		1		0.5
		Haswell			1		0.5
		Ivy Bridge		1		0.5
	*/
	__m128i res;
	for (int i = 0; i <= 7; i++)
	{
		res.m128i_i16[i] = (a.m128i_i16[i] < b.m128i_i16[i]) ? 0xffff : 0;
	}
	return res;
}

__m128i _mm_cmpgt_epi16_(__m128i a, __m128i b)
{
	/*
	Synopsis:
		__m128i _mm_cmpgt_epi16 (__m128i a, __m128i b)
		#include <emmintrin.h>
	Instruction: pcmpgtw xmm, xmm
	CPUID Flags: SSE2
	Description:
		Compare packed signed 16-bit integers in a and b for greater-than, and store the results in dst.
	Operation:
		FOR j := 0 to 7
			i := j*16
			dst[i+15:i] := ( a[i+15:i] > b[i+15:i] ) ? 0xFFFF : 0
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		0.5
		Broadwell		1		0.5
		Haswell			1		0.5
		Ivy Bridge		1		0.5
	*/
	__m128i res;
	for (int i = 0; i <= 7; i++)
	{
		res.m128i_i16[i] = (a.m128i_i16[i] > b.m128i_i16[i]) ? 0xffff : 0;
	}
	return res;
}

__m64 _mm_cmpeq_pi16_(__m64 a, __m64 b)
{
	/*
	Synopsis:
	__m64 _mm_cmpeq_pi16 (__m64 a, __m64 b)
		#include <mmintrin.h>
		Instruction: pcmpeqw mm, mm
	CPUID Flags: MMX
	Description:
		Compare packed 16-bit integers in a and b for equality, and store the results in dst.
	Operation:
		FOR j := 0 to 3
			i := j*16
			dst[i+15:i] := ( a[i+15:i] == b[i+15:i] ) ? 0xFFFF : 0
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			1		1
		Skylake			1		1
	*/
	__m64 res;
	for (int i = 0; i <= 3; i++)
	{
		res.m64_i16[i] = (a.m64_i16[i] == b.m64_i16[i]) ? 0xffff : 0;
	}
	return res;
}

__m256i _mm256_cmpeq_epi8_(__m256i a, __m256i b)
{
	/*
	Synopsis:
		__m256i _mm256_cmpeq_epi8 (__m256i a, __m256i b)
		#include <immintrin.h>
	Instruction: vpcmpeqb ymm, ymm, ymm
	CPUID Flags: AVX2
	Description:
		Compare packed 8-bit integers in a and b for equality, and store the results in dst.
	Operation:
		FOR j := 0 to 31
			i := j*8
			dst[i+7:i] := ( a[i+7:i] == b[i+7:i] ) ? 0xFF : 0
		ENDFOR
		dst[MAX:256] := 0
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			1		0.5
		Skylake			1		0.5
		Broadwell		1		0.5
		Haswell			1		0.5

	*/
	__m256i res;
	for (int i = 0; i <= 31; i++)
	{
		res.m256i_i8[i] = (a.m256i_i8[i] == b.m256i_i8[i]) ? 0xff : 0;
	}
	return res;
}

__m256i _mm256_sign_epi8_(__m256i a, __m256i b)
{
	/*
	Synopsis:
		__m256i _mm256_sign_epi8 (__m256i a, __m256i b)
		#include <immintrin.h>
	Instruction: vpsignb ymm, ymm, ymm
	CPUID Flags: AVX2
	Description:
		Negate packed signed 8-bit integers in a when the corresponding signed 8-bit integer in b is negative, and store the results in dst. Element in dst are zeroed out when the corresponding element in b is zero.
	Operation:
		FOR j := 0 to 31
			i := j*8
			IF b[i+7:i] < 0
				dst[i+7:i] := -(a[i+7:i])
			ELSE IF b[i+7:i] == 0
				dst[i+7:i] := 0
			ELSE
				dst[i+7:i] := a[i+7:i]
			FI
		ENDFOR
		dst[MAX:256] := 0
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			1		0.5
		Skylake			1		0.5
		Broadwell		1		0.5
		Haswell			1		0.5
	*/
	__m256i res;
	for (int i = 0; i <= 31; i++)
	{
		if (b.m256i_i8[i]<0)
			res.m256i_i8[i] = -a.m256i_i8[i];
		else if (b.m256i_i8[i]==0)
			res.m256i_i8[i] = 0;
		else 
			res.m256i_i8[i] = a.m256i_i8[i];
	}
	return res;
}

__m256i _mm256_min_epu8_(__m256i a, __m256i b)
{
	/*
	Synopsis:
		__m256i _mm256_min_epu8 (__m256i a, __m256i b)
		#include <immintrin.h>
	Instruction: vpminub ymm, ymm, ymm
	CPUID Flags: AVX2
	Description:
		Compare packed unsigned 8-bit integers in a and b, and store packed minimum values in dst.
	Operation:
		FOR j := 0 to 31
			i := j*8
			dst[i+7:i] := MIN(a[i+7:i], b[i+7:i])
		ENDFOR
		dst[MAX:256] := 0
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			1		0.5
		Skylake			1		0.5
	*/
	__m256i res;
	for (int i = 0; i <= 31; i++)
	{
		res.m256i_i8[i] = min(a.m256i_i8[i], b.m256i_i8[i]);
	}
	return res;
}

__m256i _mm256_adds_epi8_(__m256i a, __m256i b)
{
	/*
	Synopsis:
		__m256i _mm256_adds_epi8 (__m256i a, __m256i b)
		#include <immintrin.h>
	Instruction: vpaddsb ymm, ymm, ymm
	CPUID Flags: AVX2
	Description:
		Add packed 8-bit integers in a and b using saturation, and store the results in dst.
	Operation:
		FOR j := 0 to 31
			i := j*8
			dst[i+7:i] := Saturate8( a[i+7:i] + b[i+7:i] )
		ENDFOR
		dst[MAX:256] := 0
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			1		0.5
		Skylake			1		0.5
		Broadwell		1		0.5
		Haswell			1		0.5
	*/
	__m256i res;
	for (int i = 0; i <= 31; i++)
	{
		res.m256i_i8[i] = Saturate8(a.m256i_i8[i] +  b.m256i_i8[i]);
	}
	return res;
}

__m256i _mm256_subs_epi8_(__m256i a, __m256i b)
{
	/*
	Synopsis:
		__m256i _mm256_subs_epi8 (__m256i a, __m256i b)
		#include <immintrin.h>
	Instruction: vpsubsb ymm, ymm, ymm
	CPUID Flags: AVX2
	Description:
		Subtract packed signed 8-bit integers in b from packed 8-bit integers in a using saturation, and store the results in dst.
	Operation:
		FOR j := 0 to 31
			i := j*8
			dst[i+7:i] := Saturate8(a[i+7:i] - b[i+7:i])
		ENDFOR
		dst[MAX:256] := 0
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			1		0.5
		Skylake			1		0.5
	*/
	__m256i res;
	for (int i = 0; i <= 31; i++)
	{
		res.m256i_i8[i] = Saturate8(a.m256i_i8[i] - b.m256i_i8[i]);
	}
	return res;
}

__m256i _mm256_cmpgt_epi8_(__m256i a, __m256i b)
{
	/*
	Synopsis:
		__m256i _mm256_cmpgt_epi8 (__m256i a, __m256i b)
		#include <immintrin.h>
	Instruction: vpcmpgtb ymm, ymm, ymm
	CPUID Flags: AVX2
	Description:
		Compare packed signed 8-bit integers in a and b for greater-than, and store the results in dst.
	Operation:
		FOR j := 0 to 31
			i := j*8
			dst[i+7:i] := ( a[i+7:i] > b[i+7:i] ) ? 0xFF : 0
		ENDFOR
		dst[MAX:256] := 0
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			1		0.5
		Skylake			1		0.5
	*/
	__m256i res;
	for (int i = 0; i <= 31; i++)
	{
		res.m256i_i8[i] = (a.m256i_i8[i] > b.m256i_i8[i]) ? 0xff : 0;
	}
	return res;
}

int _mm_cvtsi128_si32_(__m128i a)
{
	/*
	Synopsis:
		int _mm_cvtsi128_si32 (__m128i a)
		#include <emmintrin.h>
	Instruction: movd r32, xmm
	CPUID Flags: SSE2
	Description:
		Copy the lower 32-bit integer in a to dst.
	Operation:
		dst[31:0] := a[31:0]
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			2		1
		Broadwell		1		1
		Haswell			1		1
		Ivy Bridge		1		1
	*/
	int res;
	res = a.m128i_i32[0];
	return res;
}

__m128i _mm_cvtsi32_si128_(int a)
{
	/*
	Synopsis:
		__m128i _mm_cvtsi32_si128 (int a)
		#include <emmintrin.h>
	Instruction: movd xmm, r32
	CPUID Flags: SSE2
	Description:
		Copy 32-bit integer a to the lower elements of dst, and zero the upper elements of dst.
	Operation:
		dst[31:0] := a[31:0]
		dst[127:32] := 0
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			2		1
		Broadwell		1		1
		Haswell			1		1
		Ivy Bridge		1		1
	*/
	__m128i res = {0};
	res.m128i_i32[0] = a;
	return res;
}
__m128i _mm_cvttps_epi32_(__m128 a)
{
	/*
	Synopsis:
		__m128i _mm_cvttps_epi32 (__m128 a)
		#include <emmintrin.h>
	Instruction: cvttps2dq xmm, xmm
	CPUID Flags: SSE2
	Description:
		Convert packed single-precision (32-bit) floating-point elements in a to packed 32-bit integers with truncation, and store the results in dst.
	Operation:
		FOR j := 0 to 3
			i := 32*j
			dst[i+31:i] := Convert_FP32_To_Int32_Truncate(a[i+31:i])
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			4		0.5
		Broadwell		3		1
		Haswell			3		1
		Ivy Bridge		3		1
	*/
	__m128i res;
	for (int i=0; i <= 3; i++)
	{
		res.m128i_i32[i] = (int) a.m128_f32[i];
	}
	return res;
}

__m64 _m_pmaddwd_(__m64 a, __m64 b)
{
	/*
	Synopsis:
		__m64 _m_pmaddwd (__m64 a, __m64 b)
		#include <mmintrin.h>
	Instruction: pmaddwd mm, mm
	CPUID Flags: MMX
	Description:
		Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Horizontally add adjacent pairs of intermediate 32-bit integers, and pack the results in dst.
	Operation:
		FOR j := 0 to 1
			i := j*32
			dst[i+31:i] := SignExtend32(a[i+31:i+16]*b[i+31:i+16]) + SignExtend32(a[i+15:i]*b[i+15:i])
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			-		1
		Skylake			4		1
	*/
	__m64 res;
	for (int i=0; i <= 1; i++)
	{
		res.m64_i32[i] = SignExtend32(a.m64_i16[i * 2 + 1] * b.m64_i16[i * 2 + 1]) + SignExtend32(a.m64_i16[i * 2] * b.m64_i16[i * 2]);
	}
	return res;
}

__m64 _m_paddd_(__m64 a, __m64 b)
{
	/*
	Synopsis:
		__m64 _m_paddd (__m64 a, __m64 b)
		#include <mmintrin.h>
	Instruction: paddd mm, mm
	CPUID Flags: MMX
	Description:
		Add packed 32-bit integers in a and b, and store the results in dst.
	Operation:
		FOR j := 0 to 1
			i := j*32
			dst[i+31:i] := a[i+31:i] + b[i+31:i]
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			1		0.5
		Skylake			1		0.5
	*/
	__m64 res;
	for (int i=0; i <= 1; i++)
	{
		res.m64_i32[i] = a.m64_i32[i] + b.m64_i32[i];
	}
	return res;
}

__m64 _m_paddw_(__m64 a, __m64 b)
{
	/*
	Synopsis:
		__m64 _m_paddw (__m64 a, __m64 b)
		#include <mmintrin.h>
	Instruction: paddw mm, mm
	CPUID Flags: MMX
	Description:
		Add packed 16-bit integers in a and b, and store the results in dst.
	Operation:
		FOR j := 0 to 3
			i := j*16
			dst[i+15:i] := a[i+15:i] + b[i+15:i]
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			1		0.5
		Skylake			1		0.5
	*/
	__m64 res;
	for (int i=0; i <= 3; i++)
	{
		res.m64_i16[i] = a.m64_i16[i] + b.m64_i16[i];
	}
	return res;
}

__m64 _m_psradi_(__m64 a, int imm8)
{
	/*
	Synopsis:
		__m64 _m_psradi (__m64 a, int imm8)
		#include <mmintrin.h>
	Instruction: psrad mm, imm8
	CPUID Flags: MMX
	Description:
		Shift packed 32-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst.
	Operation:
		FOR j := 0 to 1
			i := j*32
			IF imm8[7:0] > 31
				dst[i+31:i] := (a[i+31] ? 0xFFFFFFFF : 0x0)
			ELSE
				dst[i+31:i] := SignExtend32(a[i+31:i] >> imm8[7:0])
			FI
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		1
	*/
	__m64 res;
	for (int i=0; i <= 1; i++)
	{
		if (imm8 > 31)
			res.m64_i32[i] = (a.m64_i32[i] < 0 ? 0xFFFFFFFF : 0x0);
		else
			res.m64_i32[i] = SignExtend32(a.m64_i16[i] >> imm8);
	}
	return res;
}

__m64 _m_psrlqi_(__m64 a, int imm8)
{
	/*
	Synopsis:
		__m64 _m_psrlqi (__m64 a, int imm8)
		#include <mmintrin.h>
	Instruction: psrlq mm, imm8
	CPUID Flags: MMX
	Description:
		Shift 64-bit integer a right by imm8 while shifting in zeros, and store the result in dst.
	Operation:
		IF imm8[7:0] > 63
			dst[63:0] := 0
		ELSE
			dst[63:0] := ZeroExtend64(a[63:0] >> imm8[7:0])
		FI
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		1
	*/
	__m64 res;
	if (imm8 > 63)
		res.m64_i64 = 0;
	else
		res.m64_i64 = ZeroExtend64(a.m64_i64 >> imm8);
	return res;
}

int _m_to_int_(__m64 a)
{
	/*
	Synopsis:
		int _m_to_int (__m64 a)
		#include <mmintrin.h>
	Instruction: movd r32, mm
	CPUID Flags: MMX
	Description:
		Copy the lower 32-bit integer in a to dst.
	Operation:
		dst[31:0] := a[31:0]
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			-		1
		Skylake			2		1
	*/
	int res = a.m64_i32[0];

	return res;
}

__m128i _mm_cvtepi16_epi32_(__m128i a)
{
	/*
	Synopsis:
		__m128i _mm_cvtepi16_epi32 (__m128i a)
		#include <smmintrin.h>
	Instruction: pmovsxwd xmm, xmm
	CPUID Flags: SSE4.1
	Description:
		Sign extend packed 16-bit integers in a to packed 32-bit integers, and store the results in dst.
	Operation:
		FOR j := 0 to 3
			i := 32*j
			k := 16*j
			dst[i+31:i] := SignExtend32(a[k+15:k])
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		1
	*/
	__m128i res;
	for (int i=0; i <= 3; i++)
	{
		res.m128i_i32[i] = SignExtend32(a.m128i_i16[i]);
	}
	return res;
}

__m128i _mm_loadu_si128_(__m128i const* mem_addr)
{
	/*
	Synopsis:
		__m128i _mm_loadu_si128 (__m128i const* mem_addr)
		#include <emmintrin.h>
	Instruction: movdqu xmm, m128
	CPUID Flags: SSE2
	Description:
		Load 128-bits of integer data from memory into dst. mem_addr does not need to be aligned on any particular boundary.
	Operation:
		dst[127:0] := MEM[mem_addr+127:mem_addr]
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			6		0.5
		Broadwell		1		0.33
		Haswell			1		0.33
		Ivy Bridge		1		0.5
	*/
	__m128i res = *mem_addr;

	return res;
}

void _mm_storeu_si128_(__m128i* mem_addr, __m128i a)
{
	/*
	Synopsis:
		void _mm_storeu_si128 (__m128i* mem_addr, __m128i a)
		#include <emmintrin.h>
	Instruction: movdqu m128, xmm
	CPUID Flags: SSE2
	Description:
		Store 128-bits of integer data from a into memory. mem_addr does not need to be aligned on any particular boundary.
	Operation:
		MEM[mem_addr+127:mem_addr] := a[127:0]
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			5		1
		Broadwell		1		0.33
		Haswell			1		0.33
		Ivy Bridge		1		0.5
	*/
	*mem_addr = a;

	return;
}

__m128i _mm_shuffle_epi32_(__m128i a, int imm8)
{
	/*
	Synopsis:
		__m128i _mm_shuffle_epi32 (__m128i a, int imm8)
		#include <emmintrin.h>
	Instruction: pshufd xmm, xmm, imm8
	CPUID Flags: SSE2
	Description:
		Shuffle 32-bit integers in a using the control in imm8, and store the results in dst.
	Operation:
		DEFINE SELECT4(src, control) {
			CASE(control[1:0]) OF
			0:	tmp[31:0] := src[31:0]
			1:	tmp[31:0] := src[63:32]
			2:	tmp[31:0] := src[95:64]
			3:	tmp[31:0] := src[127:96]
			ESAC
			RETURN tmp[31:0]
		}
		dst[31:0] := SELECT4(a[127:0], imm8[1:0])
		dst[63:32] := SELECT4(a[127:0], imm8[3:2])
		dst[95:64] := SELECT4(a[127:0], imm8[5:4])
		dst[127:96] := SELECT4(a[127:0], imm8[7:6])
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		1
		Broadwell		1		1
		Haswell			1		1
		Ivy Bridge		1		0.5
	*/
	__m128i res;
	for (int i=0; i <= 3; i++)
	{
		res.m128i_i32[i] = a.m128i_i32[(imm8 >> (2 * i)) & 0x3];
	}
	return res;
}

void _mm_empty_(void)
{
	/*
	Synopsis:
		void _mm_empty (void)
		#include <mmintrin.h>
	Instruction: emms
	CPUID Flags: MMX
	Description:
		Empty the MMX state, which marks the x87 FPU registers as available for use by x87 instructions. This instruction must be used at the end of all MMX technology procedures.
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			10		4.5
	*/
	;
	return;
}

__m64 _mm_insert_pi16_(__m64 a, int i, int imm8)
{
	/*
	Synopsis:
		__m64 _mm_insert_pi16 (__m64 a, int i, int imm8)
		#include <xmmintrin.h>
	Instruction: pinsrw mm, r32, imm8
	CPUID Flags: SSE
	Description:
		Copy a to dst, and insert the 16-bit integer i into dst at the location specified by imm8.
	Operation:
		dst[63:0] := a[63:0]
		sel := imm8[1:0]*16
		dst[sel+15:sel] := i[15:0]
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			2		2
	*/
	__m64 res;
	res = a;
	res.m64_i16[imm8 & 3] = (int16_t)i;

	return res;
}

__m128i _mm_insert_epi16_(__m128i a, int i, int imm8)
{
	/*	
	Synopsis:
		__m128i _mm_insert_epi16 (__m128i a, int i, int imm8)
		#include <emmintrin.h>
	Instruction: pinsrw xmm, r32, imm8
	CPUID Flags: SSE2
	Description:
		Copy a to dst, and insert the 16-bit integer i into dst at the location specified by imm8.
	Operation:
		dst[127:0] := a[127:0]
		sel := imm8[2:0]*16
		dst[sel+15:sel] := i[15:0]
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			2		2
	*/
	__m128i res;
	res = a;
	res.m128i_i16[imm8 & 7] = (int16_t) i;
	return res;
}


__m64 _mm_xor_si64_(__m64 a, __m64 b)
{
	/*
	Synopsis
		__m64 _mm_xor_si64 (__m64 a, __m64 b)
		#include <mmintrin.h>
	Instruction: pxor mm, mm
	CPUID Flags: MMX
	Description
		Compute the bitwise XOR of 64 bits (representing integer data) in a and b, and store the result in dst.
	Operation
		dst[63:0] := (a[63:0] XOR b[63:0])
	Performance
		Architecture	Latency	Throughput (CPI)
		Icelake			1		0.5
		Skylake			1		0.5
	*/
	__m64 res;
	res.m64_i64 = a.m64_i64 ^ b.m64_i64;
	return res;
}

__m128i _mm_xor_si128_(__m128i a, __m128i b)
{
	/*
	Synopsis:
		__m128i _mm_xor_si128 (__m128i a, __m128i b)
		#include <emmintrin.h>
	Instruction: pxor xmm, xmm
	CPUID Flags: SSE2
	Description:
		Compute the bitwise XOR of 128 bits (representing integer data) in a and b, and store the result in dst.
	Operation:
		dst[127:0] := (a[127:0] XOR b[127:0])
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		0.33
		Broadwell		1		0.33
		Haswell			1		0.33
		Ivy Bridge		1		0.33
	*/
	__m128i res;
	res.m128i_i64[0] = a.m128i_i64[0] ^ b.m128i_i64[0];
	res.m128i_i64[1] = a.m128i_i64[1] ^ b.m128i_i64[1];
	return res;
}

__m128i _mm_or_si128_(__m128i a, __m128i b)
{
	/*
	Synopsis:
		__m128i _mm_or_si128 (__m128i a, __m128i b)
		#include <emmintrin.h>
	Instruction: por xmm, xmm
	CPUID Flags: SSE2
	Description:
		Compute the bitwise OR of 128 bits (representing integer data) in a and b, and store the result in dst.
	Operation:
		dst[127:0] := (a[127:0] OR b[127:0])
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		0.33
		Broadwell		1		0.33
		Haswell			1		0.33
		Ivy Bridge		1		0.33
	*/
	__m128i res;
	res.m128i_i64[0] = a.m128i_i64[0] | b.m128i_i64[0];
	res.m128i_i64[1] = a.m128i_i64[1] | b.m128i_i64[1];
	return res;
}

__m64 _mm_or_si64_(__m64 a, __m64 b)
{
	/*
	Synopsis:
		__m64 _mm_or_si64 (__m64 a, __m64 b)
		#include <mmintrin.h>
	Instruction: por mm, mm
	CPUID Flags: MMX
	Description:
		Compute the bitwise OR of 64 bits (representing integer data) in a and b, and store the result in dst.
	Operation:
		dst[63:0] := (a[63:0] OR b[63:0])
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			1		0.5
		Skylake			1		0.5
	*/
	__m64 res;
	res.m64_i64 = a.m64_i64 | b.m64_i64;
	return res;
}

__m256i _mm256_xor_si256_(__m256i a, __m256i b)
{
	/*
	Synopsis:
		__m256i _mm256_xor_si256 (__m256i a, __m256i b)
		#include <immintrin.h>
	Instruction: vpxor ymm, ymm, ymm
	CPUID Flags: AVX2
	Description:
		Compute the bitwise XOR of 256 bits (representing integer data) in a and b, and store the result in dst.
	Operation:
		dst[255:0] := (a[255:0] XOR b[255:0])
		dst[MAX:256] := 0
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			1	0.33
		Skylake			1	0.33
		Broadwell		1	0.33
		Haswell			1	0.33
	*/
	__m256i res;
	res.m256i_i64[0] = a.m256i_i64[0] ^ b.m256i_i64[0];
	res.m256i_i64[1] = a.m256i_i64[1] ^ b.m256i_i64[1];
	res.m256i_i64[2] = a.m256i_i64[2] ^ b.m256i_i64[2];
	res.m256i_i64[3] = a.m256i_i64[3] ^ b.m256i_i64[3];	
	return res;
}

__m128i _mm_hadd_epi32_(__m128i a, __m128i b)
{
	/*
	Synopsis:
		__m128i _mm_hadd_epi32 (__m128i a, __m128i b)
		#include <tmmintrin.h>
	Instruction: phaddd xmm, xmm
	CPUID Flags: SSSE3
	Description:
		Horizontally add adjacent pairs of 32-bit integers in a and b, and pack the signed 32-bit results in dst.
	Operation:
		dst[31:0] := a[63:32] + a[31:0]
		dst[63:32] := a[127:96] + a[95:64]
		dst[95:64] := b[63:32] + b[31:0]
		dst[127:96] := b[127:96] + b[95:64]
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			3		2
		Broadwell		3		2
		Haswell			3		2
		Ivy Bridge		3		1.5
	*/
	__m128i res;
	res.m128i_i32[0] = a.m128i_i32[1] + a.m128i_i32[0];
	res.m128i_i32[1] = a.m128i_i32[3] + a.m128i_i32[2];
	res.m128i_i32[2] = b.m128i_i32[1] + b.m128i_i32[0];
	res.m128i_i32[3] = b.m128i_i32[3] + b.m128i_i32[2];
	return res;
}

__m64 _mm_movepi64_pi64_(__m128i a)
{
	/*
	Synopsis:
		__m64 _mm_movepi64_pi64 (__m128i a)
		#include <emmintrin.h>
	Instruction: movdq2q mm, xmm
	CPUID Flags: SSE2
	Description:
		Copy the lower 64-bit integer in a to dst.
	Operation:
		dst[63:0] := a[63:0]
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			2		1
	*/
	__m64 res;
	res.m64_i64 = a.m128i_i64[0];
	return res;
}

__m64 _mm_packs_pi32_(__m64 a, __m64 b)
{
	/*	
	Synopsis:
		__m64 _mm_packs_pi32 (__m64 a, __m64 b)
		#include <mmintrin.h>
	Instruction: packssdw mm, mm
	CPUID Flags: MMX
	Description:
		Convert packed signed 32-bit integers from a and b to packed 16-bit integers using signed saturation, and store the results in dst.
	Operation:
		dst[15:0] := Saturate16(a[31:0])
		dst[31:16] := Saturate16(a[63:32])
		dst[47:32] := Saturate16(b[31:0])
		dst[63:48] := Saturate16(b[63:32])
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			-		2
		Skylake			3		2
	*/
	__m64 res;
	res.m64_i16[0] = Saturate16(a.m64_i32[0]);
	res.m64_i16[1] = Saturate16(a.m64_i32[1]);
	res.m64_i16[2] = Saturate16(b.m64_i32[0]);
	res.m64_i16[3] = Saturate16(b.m64_i32[1]);
	return res;
}

int _mm_cvtsi64_si32_(__m64 a)
{
	/*
	Synopsis:
		int _mm_cvtsi64_si32 (__m64 a)
		#include <mmintrin.h>
	Instruction: movd r32, mm
	CPUID Flags: MMX
	Description:
		Copy the lower 32-bit integer in a to dst.
	Operation:
		dst[31:0] := a[31:0]
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			-		1
		Skylake			2		1
	*/
	int res;
	res = a.m64_i32[0];
	return res;
}

__m256i _mm256_insert_epi16_(__m256i a, __int16 i, const int index)
{
	/*
	Synopsis:
		__m256i _mm256_insert_epi16 (__m256i a, __int16 i, const int index)
		#include <immintrin.h>
	Instruction: Sequence
	CPUID Flags: AVX
	Description:
		Copy a to dst, and insert the 16-bit integer i into dst at the location specified by index.
	Operation:
		dst[255:0] := a[255:0]
		sel := index[3:0]*16
		dst[sel+15:sel] := i[15:0]
	*/
	__m256i res;
	res = a;
	res.m256i_i16[index] = i;
	return res;
}

__m128i _mm_max_epi16_(__m128i a, __m128i b)
{
	/*
	Synopsis:
		__m128i _mm_max_epi16 (__m128i a, __m128i b)
		#include <emmintrin.h>
	Instruction: pmaxsw xmm, xmm
	CPUID Flags: SSE2
	Description:
		Compare packed signed 16-bit integers in a and b, and store packed maximum values in dst.
	Operation:
		FOR j := 0 to 7
		i := j*16
		dst[i+15:i] := MAX(a[i+15:i], b[i+15:i])
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		0.5
	*/
	__m128i res;
	for (int i = 0; i <= 7; i++)
	{
		res.m128i_i16[i] = max(a.m128i_i16[i], b.m128i_i16[i]);
	}
	return res;
}

__m128i _mm_min_epi16_(__m128i a, __m128i b)
{
	/*
	Synopsis:
		__m128i _mm_min_epi16 (__m128i a, __m128i b)
		#include <emmintrin.h>
	Instruction: pminsw xmm, xmm
	CPUID Flags: SSE2
	Description:
		Compare packed signed 16-bit integers in a and b, and store packed minimum values in dst.
	Operation:
		FOR j := 0 to 7
			i := j*16
			dst[i+15:i] := MIN(a[i+15:i], b[i+15:i])
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		0.5
	*/
	__m128i res;
	for (int i = 0; i <= 7; i++)
	{
		res.m128i_i16[i] = min(a.m128i_i16[i], b.m128i_i16[i]);
	}
	return res;
}

__m64 _mm_min_pi16_(__m64 a, __m64 b)
{
	/*
	Synopsis:
		__m64 _mm_min_pi16 (__m64 a, __m64 b)
		#include <xmmintrin.h>
	Instruction: pminsw mm, mm
	CPUID Flags: SSE
	Description:
		Compare packed signed 16-bit integers in a and b, and store packed minimum values in dst.
	Operation:
		FOR j := 0 to 3
			i := j*16
			dst[i+15:i] := MIN(a[i+15:i], b[i+15:i])
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			-		1
		Skylake			1		1
	*/
	__m64 res;
	for (int i = 0; i <= 3; i++)
	{
		res.m64_i16[i] = min(a.m64_i16[i], b.m64_i16[i]);
	}
	return res;
}

__m256i _mm256_adds_epi16_(__m256i a, __m256i b)
{
	/*
	Synopsis:
		__m256i _mm256_adds_epi16 (__m256i a, __m256i b)
		#include <immintrin.h>
	Instruction: vpaddsw ymm, ymm, ymm
	CPUID Flags: AVX2
	Description:
		Add packed 16-bit integers in a and b using saturation, and store the results in dst.
	Operation:
		FOR j := 0 to 15
			i := j*16
			dst[i+15:i] := Saturate16( a[i+15:i] + b[i+15:i] )
		ENDFOR
		dst[MAX:256] := 0
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			1		0.5
		Skylake			1		0.5
	*/
	__m256i res;
	for (int i = 0; i <= 15; i++)
	{
		res.m256i_i16[i] = Saturate16(a.m256i_i16[i] + b.m256i_i16[i]);
	}
	return res;
}

__m256i _mm256_mullo_epi16_(__m256i a, __m256i b)
{
	/*
	Synopsis:
		__m256i _mm256_mullo_epi16 (__m256i a, __m256i b)
		#include <immintrin.h>
	Instruction: vpmullw ymm, ymm, ymm
	CPUID Flags: AVX2
	Description:
		Multiply the packed signed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the low 16 bits of the intermediate integers in dst.
	Operation:
		FOR j := 0 to 15
			i := j*16
			tmp[31:0] := SignExtend32(a[i+15:i]) * SignExtend32(b[i+15:i])
			dst[i+15:i] := tmp[15:0]
		ENDFOR
		dst[MAX:256] := 0
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			-		0.5
		Skylake			5		0.5
		Broadwell		5		1
		Haswell			5		1
	*/
	__m256i res;
	for (int i = 0; i <= 15; i++)
	{
		res.m256i_i16[i] = SignExtend32(a.m256i_i16[i]) * SignExtend32(b.m256i_i16[i]);
	}
	return res;
}

__m256i _mm256_subs_epi16_(__m256i a, __m256i b)
{
	/*
	Synopsis:
		__m256i _mm256_subs_epi16 (__m256i a, __m256i b)
		#include <immintrin.h>
	Instruction: vpsubsw ymm, ymm, ymm
	CPUID Flags: AVX2
	Description:
		Subtract packed signed 16-bit integers in b from packed 16-bit integers in a using saturation, and store the results in dst.
	Operation:
		FOR j := 0 to 15
			i := j*16
			dst[i+15:i] := Saturate16(a[i+15:i] - b[i+15:i])
		ENDFOR
		dst[MAX:256] := 0
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			1		0.5
		Skylake			1		0.5

	*/
	__m256i res;
	for (int i = 0; i <= 15; i++)
	{
		res.m256i_i16[i] = Saturate16(a.m256i_i16[i] - b.m256i_i16[i]);
	}
	return res;
}

__m256i _mm256_hadds_epi16_(__m256i a, __m256i b)
{
	/*
	Synopsis:
		__m256i _mm256_hadds_epi16 (__m256i a, __m256i b)
		#include <immintrin.h>
	Instruction: vphaddsw ymm, ymm, ymm
	CPUID Flags: AVX2
	Description:
		Horizontally add adjacent pairs of signed 16-bit integers in a and b using saturation, and pack the signed 16-bit results in dst.
	Operation:
		dst[15:0] := Saturate16(a[31:16] + a[15:0])
		dst[31:16] := Saturate16(a[63:48] + a[47:32])
		dst[47:32] := Saturate16(a[95:80] + a[79:64])
		dst[63:48] := Saturate16(a[127:112] + a[111:96])
		dst[79:64] := Saturate16(b[31:16] + b[15:0])
		dst[95:80] := Saturate16(b[63:48] + b[47:32])
		dst[111:96] := Saturate16(b[95:80] + b[79:64])
		dst[127:112] := Saturate16(b[127:112] + b[111:96])
		dst[143:128] := Saturate16(a[159:144] + a[143:128])
		dst[159:144] := Saturate16(a[191:176] + a[175:160])
		dst[175:160] := Saturate16(a[223:208] + a[207:192])
		dst[191:176] := Saturate16(a[255:240] + a[239:224])
		dst[207:192] := Saturate16(b[159:144] + b[143:128])
		dst[223:208] := Saturate16(b[191:176] + b[175:160])
		dst[239:224] := Saturate16(b[223:208] + b[207:192])
		dst[255:240] := Saturate16(b[255:240] + b[239:224])
		dst[MAX:256] := 0
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			-		1
		Skylake			3		2
		Broadwell		3		2
		Haswell			3		2
	*/
	__m256i res;
	res.m256i_i16[0] = Saturate16(a.m256i_i16[1] + a.m256i_i16[0]);
	res.m256i_i16[1] = Saturate16(a.m256i_i16[3] + a.m256i_i16[2]);
	res.m256i_i16[2] = Saturate16(a.m256i_i16[5] + a.m256i_i16[4]);
	res.m256i_i16[3] = Saturate16(a.m256i_i16[7] + a.m256i_i16[6]);
	res.m256i_i16[4] = Saturate16(b.m256i_i16[1] + b.m256i_i16[0]);
	res.m256i_i16[5] = Saturate16(b.m256i_i16[3] + b.m256i_i16[2]);
	res.m256i_i16[6] = Saturate16(b.m256i_i16[5] + b.m256i_i16[4]);
	res.m256i_i16[7] = Saturate16(b.m256i_i16[7] + b.m256i_i16[6]);
	res.m256i_i16[8] = Saturate16(a.m256i_i16[9] + a.m256i_i16[8]);
	res.m256i_i16[9] = Saturate16(a.m256i_i16[11] + a.m256i_i16[10]);
	res.m256i_i16[10] = Saturate16(a.m256i_i16[13] + a.m256i_i16[12]);
	res.m256i_i16[11] = Saturate16(a.m256i_i16[15] + a.m256i_i16[14]);
	res.m256i_i16[12] = Saturate16(b.m256i_i16[9] + b.m256i_i16[8]);
	res.m256i_i16[13] = Saturate16(b.m256i_i16[11] + b.m256i_i16[10]);
	res.m256i_i16[14] = Saturate16(b.m256i_i16[13] + b.m256i_i16[12]);
	res.m256i_i16[15] = Saturate16(b.m256i_i16[15] + b.m256i_i16[14]);
	return res;
}

__m256i _mm256_packs_epi16_(__m256i a, __m256i b)
{
	/*
	Synopsis:
		__m256i _mm256_packs_epi16 (__m256i a, __m256i b)
		#include <immintrin.h>
	Instruction: vpacksswb ymm, ymm, ymm
	CPUID Flags: AVX2
	Description:
		Convert packed signed 16-bit integers from a and b to packed 8-bit integers using signed saturation, and store the results in dst.
	Operation:
		dst[7:0] := Saturate8(a[15:0])
		dst[15:8] := Saturate8(a[31:16])
		dst[23:16] := Saturate8(a[47:32])
		dst[31:24] := Saturate8(a[63:48])
		dst[39:32] := Saturate8(a[79:64])
		dst[47:40] := Saturate8(a[95:80])
		dst[55:48] := Saturate8(a[111:96])
		dst[63:56] := Saturate8(a[127:112])
		dst[71:64] := Saturate8(b[15:0])
		dst[79:72] := Saturate8(b[31:16])
		dst[87:80] := Saturate8(b[47:32])
		dst[95:88] := Saturate8(b[63:48])
		dst[103:96] := Saturate8(b[79:64])
		dst[111:104] := Saturate8(b[95:80])
		dst[119:112] := Saturate8(b[111:96])
		dst[127:120] := Saturate8(b[127:112])
		dst[135:128] := Saturate8(a[143:128])
		dst[143:136] := Saturate8(a[159:144])
		dst[151:144] := Saturate8(a[175:160])
		dst[159:152] := Saturate8(a[191:176])
		dst[167:160] := Saturate8(a[207:192])
		dst[175:168] := Saturate8(a[223:208])
		dst[183:176] := Saturate8(a[239:224])
		dst[191:184] := Saturate8(a[255:240])
		dst[199:192] := Saturate8(b[143:128])
		dst[207:200] := Saturate8(b[159:144])
		dst[215:208] := Saturate8(b[175:160])
		dst[223:216] := Saturate8(b[191:176])
		dst[231:224] := Saturate8(b[207:192])
		dst[239:232] := Saturate8(b[223:208])
		dst[247:240] := Saturate8(b[239:224])
		dst[255:248] := Saturate8(b[255:240])
		dst[MAX:256] := 0
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			3		1
		Skylake			1		1
		Broadwell		1		1
		Haswell			1		1
	*/
	__m256i res;
	res.m256i_i8[0] = Saturate8(a.m256i_i16[0]);
	res.m256i_i8[1] = Saturate8(a.m256i_i16[1]);
	res.m256i_i8[2] = Saturate8(a.m256i_i16[2]);
	res.m256i_i8[3] = Saturate8(a.m256i_i16[3]);
	res.m256i_i8[4] = Saturate8(a.m256i_i16[4]);
	res.m256i_i8[5] = Saturate8(a.m256i_i16[5]);
	res.m256i_i8[6] = Saturate8(a.m256i_i16[6]);
	res.m256i_i8[7] = Saturate8(a.m256i_i16[7]);
	res.m256i_i8[8] = Saturate8(b.m256i_i16[0]);
	res.m256i_i8[9] = Saturate8(b.m256i_i16[1]);
	res.m256i_i8[10] = Saturate8(b.m256i_i16[2]);
	res.m256i_i8[11] = Saturate8(b.m256i_i16[3]);
	res.m256i_i8[12] = Saturate8(b.m256i_i16[4]);
	res.m256i_i8[13] = Saturate8(b.m256i_i16[5]);
	res.m256i_i8[14] = Saturate8(b.m256i_i16[6]);
	res.m256i_i8[15] = Saturate8(b.m256i_i16[7]);
	res.m256i_i8[16] = Saturate8(a.m256i_i16[8]);
	res.m256i_i8[17] = Saturate8(a.m256i_i16[9]);
	res.m256i_i8[18] = Saturate8(a.m256i_i16[10]);
	res.m256i_i8[19] = Saturate8(a.m256i_i16[11]);
	res.m256i_i8[20] = Saturate8(a.m256i_i16[12]);
	res.m256i_i8[21] = Saturate8(a.m256i_i16[13]);
	res.m256i_i8[22] = Saturate8(a.m256i_i16[14]);
	res.m256i_i8[23] = Saturate8(a.m256i_i16[15]);
	res.m256i_i8[24] = Saturate8(b.m256i_i16[8]);
	res.m256i_i8[25] = Saturate8(b.m256i_i16[9]);
	res.m256i_i8[26] = Saturate8(b.m256i_i16[10]);
	res.m256i_i8[27] = Saturate8(b.m256i_i16[11]);
	res.m256i_i8[28] = Saturate8(b.m256i_i16[12]);
	res.m256i_i8[29] = Saturate8(b.m256i_i16[13]);
	res.m256i_i8[30] = Saturate8(b.m256i_i16[14]);
	res.m256i_i8[31] = Saturate8(b.m256i_i16[15]);	
	return res;
}
__m64 _mm_mullo_pi16_(__m64 a, __m64 b)
{
	/*
	Synopsis:
		__m64 _mm_mullo_pi16 (__m64 a, __m64 b)
		#include <mmintrin.h>
	Instruction: pmullw mm, mm
	CPUID Flags: MMX
	Description:
		Multiply the packed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the low 16 bits of the intermediate integers in dst.
	Operation:
		FOR j := 0 to 3
			i := j*16
			tmp[31:0] := a[i+15:i] * b[i+15:i]
			dst[i+15:i] := tmp[15:0]
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			-		1
		Skylake			5		1
	*/
	__m64 res;
	int32_t tmp; 
	for (int i = 0; i <= 3; i++)
	{
		tmp = a.m64_i16[i] * b.m64_i16[i];
		res.m64_i16[i] = (int16_t) tmp;
	}
	return res;
}

__m64 _mm_adds_pi16_(__m64 a, __m64 b)
{
	/*
	Synopsis:
		__m64 _mm_adds_pi16 (__m64 a, __m64 b)
		#include <mmintrin.h>
	Instruction: paddsw mm, mm
	CPUID Flags: MMX
	Description:
		Add packed signed 16-bit integers in a and b using saturation, and store the results in dst.
	Operation:
		FOR j := 0 to 3
			i := j*16
			dst[i+15:i] := Saturate16( a[i+15:i] + b[i+15:i] )
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			1		1
		Skylake			1		1
	*/
	__m64 res;
	for (int i = 0; i <= 3; i++)
	{
		res.m64_i16[i] = Saturate16(a.m64_i16[i] + b.m64_i16[i]);
	}
	return res;
}

__m64 _mm_subs_pi16_(__m64 a, __m64 b)
{
	/*
	Synopsis:
		__m64 _mm_subs_pi16 (__m64 a, __m64 b)
		#include <mmintrin.h>
	Instruction: psubsw mm, mm
	CPUID Flags: MMX
	Description:
		Subtract packed signed 16-bit integers in b from packed 16-bit integers in a using saturation, and store the results in dst.
	Operation:
		FOR j := 0 to 3
			i := j*16
			dst[i+15:i] := Saturate16(a[i+15:i] - b[i+15:i])
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			1		1
		Skylake			1		1
	*/
	__m64 res;
	for (int i = 0; i <= 3; i++)
	{
		res.m64_i16[i] = Saturate16(a.m64_i16[i] - b.m64_i16[i]);
	}
	return res;
}

__m64 _mm_srai_pi16_(__m64 a, int imm8)
{
	/*
	Synopsis:
		__m64 _mm_srai_pi16 (__m64 a, int imm8)
		#include <mmintrin.h>
	Instruction: psraw mm, imm8
	CPUID Flags: MMX
	Description:
		Shift packed 16-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst.
	Operation:
		FOR j := 0 to 3
			i := j*16
			IF imm8[7:0] > 15
				dst[i+15:i] := (a[i+15] ? 0xFFFF : 0x0)
			ELSE
				dst[i+15:i] := SignExtend16(a[i+15:i] >> imm8[7:0])
			FI
		ENDFOR
	Performance:
		Architecture	Latency	Throughput (CPI)
		Skylake			1		1
	*/
	__m64 res;
	for (int i = 0; i <= 7; i++)
	{
		if (imm8 > 15)
			res.m64_i16[i] = a.m64_i16[i] < 0 ? 0xFFFF : 0x0;
		else
			res.m64_i16[i] = SignExtend16(a.m64_i16[i] >> imm8);
	}
	return res;
}

__m64 _mm_hadds_pi16_(__m64 a, __m64 b)
{
	/*
	Synopsis:
		__m64 _mm_hadds_pi16 (__m64 a, __m64 b)
		#include <tmmintrin.h>
	Instruction: phaddsw mm, mm
	CPUID Flags: SSSE3
	Description:
		Horizontally add adjacent pairs of signed 16-bit integers in a and b using saturation, and pack the signed 16-bit results in dst.
	Operation:
		dst[15:0] := Saturate16(a[31:16] + a[15:0])
		dst[31:16] := Saturate16(a[63:48] + a[47:32])
		dst[47:32] := Saturate16(b[31:16] + b[15:0])
		dst[63:48] := Saturate16(b[63:48] + b[47:32])
	Performance:
		Architecture	Latency	Throughput (CPI)
		Icelake			3		2
		Skylake			3		2
		Broadwell		3		2
		Haswell			3		2
		Ivy Bridge		3		1.5
	*/
	__m64 res;
	res.m64_i16[0] = Saturate16(a.m64_i16[1] + a.m64_i16[0]);
	res.m64_i16[1] = Saturate16(a.m64_i16[3] + a.m64_i16[2]);
	res.m64_i16[2] = Saturate16(b.m64_i16[1] + b.m64_i16[0]);
	res.m64_i16[3] = Saturate16(b.m64_i16[3] + b.m64_i16[2]);
	return res;
}

__m128i _mm_tmp(__m128i a, __m128i b)
{
	/*
	*/
	__m128i res;
	for (int i = 0; i <= 7; i++)
	{
		res.m128i_i16[i] = a.m128i_i16[i];
	}
	return res;
}


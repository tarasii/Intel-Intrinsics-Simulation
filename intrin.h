#ifndef _INTRIN
#define _INTRIN

#include <stdint.h>
#include "unistd.h"

#define _MM_SHUFFLE2(x,y) (((x)<<1) | (y))
#define _MM_SHUFFLE(fp3,fp2,fp1,fp0) (((fp3) << 6) | ((fp2) << 4) | \
                                     ((fp1) << 2) | ((fp0)))


#ifndef  WIN32

#ifndef simd_q15_t
#define simd_q15_t __m128i
#endif
#ifndef simdshort_q15_t
#define simdshort_q15_t __m64
#endif

typedef union __m64
{
	uint64_t    m64_u64;
	float       m64_f32[2];
	int8_t      m64_i8[8];
	int16_t     m64_i16[4];
	int32_t     m64_i32[2];
	int64_t     m64_i64;
	uint8_t     m64_u8[8];
	uint16_t    m64_u16[4];
	uint32_t    m64_u32[2];
} __m64;

typedef union  __m128i {
	int8_t              m128i_i8[16];
	int16_t             m128i_i16[8];
	int32_t             m128i_i32[4];
	int64_t             m128i_i64[2];
	uint8_t     m128i_u8[16];
	uint16_t    m128i_u16[8];
	uint32_t    m128i_u32[4];
	uint64_t    m128i_u64[2];
} __m128i;

typedef struct  __m128d {
    double              m128d_f64[2];
} __m128d;

typedef union  __m128 {
	float               m128_f32[4];
	uint64_t    m128_u64[2];
	int8_t              m128_i8[16];
	int16_t             m128_i16[8];
	int32_t             m128_i32[4];
	int64_t             m128_i64[2];
	uint8_t     m128_u8[16];
	uint16_t    m128_u16[8];
	uint32_t    m128_u32[4];
} __m128;

typedef union __m256 {
	float m256_f32[8];
} __m256;

typedef struct __m256d {
	double m256d_f64[4];
} __m256d;

typedef union __m256i {
	int8_t              m256i_i8[32];
	int16_t             m256i_i16[16];
	int32_t             m256i_i32[8];
	int64_t             m256i_i64[4];
	uint8_t     m256i_u8[32];
	uint16_t    m256i_u16[16];
	uint32_t    m256i_u32[8];
	uint64_t    m256i_u64[4];
} __m256i;

#define _m_empty _mm_empty_

#else
#ifndef __AVX2__
typedef union __declspec(intrin_type) __declspec(align(8)) __m64
{
	unsigned __int64    m64_u64;
	float               m64_f32[2];
	__int8              m64_i8[8];
	__int16             m64_i16[4];
	__int32             m64_i32[2];
	__int64             m64_i64;
	unsigned __int8     m64_u8[8];
	unsigned __int16    m64_u16[4];
	unsigned __int32    m64_u32[2];
} __m64;

typedef union __declspec(intrin_type) __declspec(align(16)) __m128i {
    __int8              m128i_i8[16];
    __int16             m128i_i16[8];
    __int32             m128i_i32[4];
    __int64             m128i_i64[2];
    unsigned __int8     m128i_u8[16];
    unsigned __int16    m128i_u16[8];
    unsigned __int32    m128i_u32[4];
    unsigned __int64    m128i_u64[2];
} __m128i;

typedef struct __declspec(intrin_type) __declspec(align(16)) __m128d {
    double              m128d_f64[2];
} __m128d;

typedef union __declspec(intrin_type) __declspec(align(16)) __m128 {
	float               m128_f32[4];
	unsigned __int64    m128_u64[2];
	__int8              m128_i8[16];
	__int16             m128_i16[8];
	__int32             m128_i32[4];
	__int64             m128_i64[2];
	unsigned __int8     m128_u8[16];
	unsigned __int16    m128_u16[8];
	unsigned __int32    m128_u32[4];
} __m128;

typedef union __declspec(intrin_type)_CRT_ALIGN(32) __m256 {
	float m256_f32[8];
} __m256;

typedef struct __declspec(intrin_type)_CRT_ALIGN(32) __m256d {
	double m256d_f64[4];
} __m256d;

typedef union  __declspec(intrin_type)_CRT_ALIGN(32) __m256i {
	__int8              m256i_i8[32];
	__int16             m256i_i16[16];
	__int32             m256i_i32[8];
	__int64             m256i_i64[4];
	unsigned __int8     m256i_u8[32];
	unsigned __int16    m256i_u16[16];
	unsigned __int32    m256i_u32[8];
	unsigned __int64    m256i_u64[4];
} __m256i;
#else
#include <xmmintrin.h>
#include <immintrin.h>
#endif

#endif

#define __m64i  __m64

//MMX
__m64 _mm_unpackhi_pi32_(__m64 a, __m64 b);
__m64 _mm_unpacklo_pi32_(__m64 a, __m64 b);
int _mm_extract_epi16_(__m128i a, int imm8);
__m64 _mm_setzero_si64_();
__m64 _m_pmaddwd_(__m64 a, __m64 b);
__m64 _m_paddd_(__m64 a, __m64 b);
__m64 _m_paddw_(__m64 a, __m64 b);
__m64 _m_psradi_(__m64 a, int imm8);
__m64 _m_psrlqi_(__m64 a, int imm8);
__m64 _mm_xor_si64_(__m64 a, __m64 b);
__m64 _mm_packs_pi32_(__m64 a, __m64 b);
int _mm_cvtsi64_si32_(__m64 a);
int _m_to_int_(__m64 a);
__m64 _mm_mullo_pi16_(__m64 a, __m64 b);
__m64 _mm_adds_pi16_(__m64 a, __m64 b);
__m64 _mm_subs_pi16_(__m64 a, __m64 b);
__m64 _mm_srai_pi16_(__m64 a, int imm8);
__m64 _mm_hadds_pi16_(__m64 a, __m64 b);
__m64 _mm_or_si64_(__m64 a, __m64 b);
__m64 _mm_cmpeq_pi16_(__m64 a, __m64 b);
__m64 _mm_sign_pi16_(__m64 a, __m64 b);

#define _mm_unpackhi_pi32 _mm_unpackhi_pi32_
#define _mm_unpacklo_pi32 _mm_unpacklo_pi32_
#define _mm_extract_epi16 _mm_extract_epi16_
#define _mm_setzero_si64 _mm_setzero_si64_
#define _mm_packs_pi32 _mm_packs_pi32_
#define _m_pmaddwd _m_pmaddwd_
#define _m_paddd _m_paddd_
#define _m_paddw _m_paddw_
#define _m_psradi _m_psradi_
#define _m_psrlqi _m_psrlqi_
#define _m_to_int _m_to_int_
#define _mm_xor_si64 _mm_xor_si64_
#define _mm_cvtsi64_si32 _mm_cvtsi64_si32_
#define _mm_mullo_pi16 _mm_mullo_pi16_
#define _mm_adds_pi16 _mm_adds_pi16_
#define _mm_subs_pi16 _mm_subs_pi16_
#define _mm_srai_pi16 _mm_srai_pi16_
#define _mm_hadds_pi16 _mm_hadds_pi16_
#define _mm_or_si64 _mm_or_si64_
#define _mm_cmpeq_pi16 _mm_cmpeq_pi16_
#define _mm_sign_pi16 _mm_sign_pi16_

//SSE
__m128 _mm_setzero_ps_();
__m128 _mm_rcp_ps_(__m128 a);
__m128 _mm_add_ps_(__m128 a, __m128 b);
__m128 _mm_mul_ps_(__m128 a, __m128 b);
__m128 _mm_sub_ps_(__m128 a, __m128 b);
__m64 _mm_insert_pi16_(__m64 a, int i, int imm8);
__m64 _mm_min_pi16_(__m64 a, __m64 b);

#define _mm_setzero_ps _mm_setzero_ps_
#define _mm_rcp_ps _mm_rcp_ps_
#define _mm_add_ps _mm_add_ps_
#define _mm_mul_ps _mm_mul_ps_
#define _mm_sub_ps _mm_sub_ps_
#define _mm_insert_pi16 _mm_insert_pi16_
#define _mm_min_pi16 _mm_min_pi16_

//SSE2
__m128i _mm_madd_epi16_(__m128i a, __m128i b);
__m128i _mm_slli_epi16_(__m128i a, int imm8);
__m128i _mm_mullo_epi16_(__m128i a, __m128i b);
__m128i _mm_mulhi_epi16_(__m128i a, __m128i b);
__m128i _mm_setzero_si128_();
__m128i _mm_set1_epi16_(short a);
__m128i _mm_srai_epi32_(__m128i a, int imm8);
__m128i _mm_packs_epi32_(__m128i a, __m128i b);
__m128i _mm_unpacklo_epi16_(__m128i a, __m128i b);
__m128i _mm_unpackhi_epi16_(__m128i a, __m128i b);
__m128i _mm_adds_epi16_(__m128i a, __m128i b);
__m128i _mm_add_epi16_(__m128i a, __m128i b);
__m128i _mm_setr_epi16_(short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0);
__m128i _mm_set_epi16_(short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0);
__m128i _mm_shufflelo_epi16_(__m128i a, int imm8);
__m128i _mm_shufflehi_epi16_(__m128i a, int imm8);
__m128i _mm_add_epi32_(__m128i a, __m128i b);
__m128i _mm_unpacklo_epi32_(__m128i a, __m128i b);
__m128i _mm_unpackhi_epi32_(__m128i a, __m128i b);
__m128i _mm_sub_epi32_(__m128i a, __m128i b);
__m128i _mm_set_epi8_(char e15, char e14, char e13, char e12, char e11, char e10, char e9, char e8, char e7, char e6, char e5, char e4, char e3, char e2, char e1, char e0);
__m128i _mm_subs_epi16_(__m128i a, __m128i b);
__m128i _mm_unpacklo_epi64_(__m128i a, __m128i b);
__m128i _mm_unpackhi_epi64_(__m128i a, __m128i b);
__m128i _mm_srai_epi16_(__m128i a, int imm8);
__m128i _mm_set1_epi32_(int a);
__m128 _mm_cvtepi32_ps_(__m128i a);
__m128i _mm_cmpeq_epi16_(__m128i a, __m128i b);
int _mm_cvtsi128_si32_(__m128i a);
__m128i _mm_cvttps_epi32_(__m128 a);
__m128i _mm_loadu_si128_(__m128i const* mem_addr);
__m128i _mm_shuffle_epi32_(__m128i a, int imm8);
void _mm_storeu_si128_(__m128i* mem_addr, __m128i a);
__m128i _mm_sra_epi32_(__m128i a, __m128i count);
__m128i _mm_setr_epi32_(int e3, int e2, int e1, int e0);
__m128i _mm_cvtsi32_si128_(int a);
__m128i _mm_packs_epi16_(__m128i a, __m128i b);
__m128i _mm_insert_epi16_(__m128i a, int i, int imm8);
__m128i _mm_or_si128_(__m128i a, __m128i b);
__m128i _mm_xor_si128_(__m128i a, __m128i b);
__m64 _mm_movepi64_pi64_(__m128i a);
__m128i _mm_max_epi16_(__m128i a, __m128i b);
__m128i _mm_min_epi16_(__m128i a, __m128i b);
void _mm_empty_(void);

#define _mm_add_epi16 _mm_add_epi16_
#define _mm_add_epi32 _mm_add_epi32_
#define _mm_adds_epi16 _mm_adds_epi16_
#define _mm_sub_epi32 _mm_sub_epi32_
#define _mm_subs_epi16 _mm_subs_epi16_
#define _mm_mullo_epi16 _mm_mullo_epi16_
#define _mm_mulhi_epi16 _mm_mulhi_epi16_
#define _mm_madd_epi16 _mm_madd_epi16_
#define _mm_slli_epi16 _mm_slli_epi16_
#define _mm_srai_epi32 _mm_srai_epi32_
#define _mm_sra_epi32 _mm_sra_epi32_
#define _mm_srai_epi16 _mm_srai_epi16_
#define _mm_setzero_si128 _mm_setzero_si128_
#define _mm_set_epi8 _mm_set_epi8_
#define _mm_set1_epi16 _mm_set1_epi16_
#define _mm_set1_epi32 _mm_set1_epi32_
#define _mm_setr_epi16 _mm_setr_epi16_
#define _mm_set_epi16 _mm_set_epi16_
#define _mm_setr_epi32 _mm_setr_epi32_
#define _mm_packs_epi32 _mm_packs_epi32_
#define _mm_packs_epi16 _mm_packs_epi16_
#define _mm_unpacklo_epi16 _mm_unpacklo_epi16_
#define _mm_unpackhi_epi16 _mm_unpackhi_epi16_
#define _mm_unpacklo_epi32 _mm_unpacklo_epi32_
#define _mm_unpackhi_epi32 _mm_unpackhi_epi32_
#define _mm_unpacklo_epi64 _mm_unpacklo_epi64_
#define _mm_unpackhi_epi64 _mm_unpackhi_epi64_
#define _mm_shufflelo_epi16 _mm_shufflelo_epi16_
#define _mm_shufflehi_epi16 _mm_shufflehi_epi16_
#define _mm_shuffle_epi32 _mm_shuffle_epi32_
#define _mm_cmpeq_epi16 _mm_cmpeq_epi16_
#define _mm_cvtepi32_ps _mm_cvtepi32_ps_
#define _mm_cvtsi128_si32 _mm_cvtsi128_si32_ 
#define _mm_cvttps_epi32 _mm_cvttps_epi32_
#define _mm_cvtsi32_si128 _mm_cvtsi32_si128_
#define _mm_insert_epi16 _mm_insert_epi16_
#define _mm_xor_si128 _mm_xor_si128_
#define _mm_loadu_si128 _mm_loadu_si128_
#define _mm_storeu_si128 _mm_storeu_si128_
#define _mm_movepi64_pi64 _mm_movepi64_pi64_
#define _mm_max_epi16 _mm_max_epi16_
#define _mm_or_si128 _mm_or_si128_
#define _mm_min_epi16 _mm_min_epi16_
#define _mm_empty _mm_empty_

//SSSE3
__m128i _mm_sign_epi16_(__m128i a, __m128i b);
__m128i _mm_mulhrs_epi16_(__m128i a, __m128i b);
__m128i _mm_shuffle_epi8_(__m128i a, __m128i b);
__m128i _mm_abs_epi16_(__m128i a);
__m128i _mm_hadd_epi32_(__m128i a, __m128i b);
__m64 _mm_abs_pi16_(__m64 a);

#define _mm_sign_epi16 _mm_sign_epi16_
#define _mm_mulhrs_epi16 _mm_mulhrs_epi16_
#define _mm_shuffle_epi8 _mm_shuffle_epi8_
#define _mm_abs_epi16 _mm_abs_epi16_
#define _mm_hadd_epi32 _mm_hadd_epi32_
#define _mm_abs_pi16 _mm_abs_pi16_

//SSE4.1
__m128 _mm_dp_ps_(__m128 a, __m128 b, const int imm8);
__m128i _mm_cvtepi16_epi32_(__m128i a);
int _mm_extract_epi32_(__m128i a, int imm8);
__int64 _mm_extract_epi64_(__m128i a, const int imm8);

#define _mm_dp_ps _mm_dp_ps_
#define _mm_cvtepi16_epi32 _mm_cvtepi16_epi32_
#define _mm_extract_epi32 _mm_extract_epi32_
#define _mm_extract_epi64 _mm_extract_epi64_

//AVX
__m256i _mm256_set_epi64x_(__int64 e3, __int64 e2, __int64 e1, __int64 e0);
__m256i _mm256_set1_epi64x_(long long a);
#define _mm256_set_epi64x _mm256_set_epi64x_
#define _mm256_set1_epi64x _mm256_set1_epi64x_

#ifndef __AVX2__
__m256i _mm256_xor_si256_(__m256i a, __m256i b);
__m256i _mm256_setzero_si256_(void);
__m256i _mm256_and_si256_(__m256i a, __m256i b);
__m256i _mm256_andnot_si256_(__m256i a, __m256i b);
__m256i _mm256_set1_epi32_(int a);
__m256i _mm256_cmpeq_epi8_(__m256i a, __m256i b);
__m256i _mm256_shuffle_epi8_(__m256i a, __m256i b);
__m256i _mm256_or_si256_(__m256i a, __m256i b);
__m256i _mm256_set1_epi8_(char a);
__m256i _mm256_srai_epi16_(__m256i a, int imm8);
__m256i _mm256_insert_epi16_(__m256i a, __int16 i, const int index);
__m256i _mm256_adds_epi16_(__m256i a, __m256i b);
__m256i _mm256_subs_epi16_(__m256i a, __m256i b);
__m256i _mm256_mullo_epi16_(__m256i a, __m256i b);
__m256i _mm256_hadds_epi16_(__m256i a, __m256i b);

#define _mm256_xor_si256 _mm256_xor_si256_
#define _mm256_setzero_si256 _mm256_setzero_si256_
#define _mm256_and_si256 _mm256_and_si256_
#define _mm256_andnot_si256 _mm256_andnot_si256_
#define _mm256_set1_epi32 _mm256_set1_epi32_
#define _mm256_cmpeq_epi8 _mm256_cmpeq_epi8_
#define _mm256_shuffle_epi8 _mm256_shuffle_epi8_
#define _mm256_or_si256 _mm256_or_si256_
#define _mm256_set1_epi8 _mm256_set1_epi8_
#define _mm256_srai_epi16 _mm256_srai_epi16_
#define _mm256_insert_epi16 _mm256_insert_epi16_
#define _mm256_adds_epi16 _mm256_adds_epi16_
#define _mm256_mullo_epi16 _mm256_mullo_epi16_
#define _mm256_subs_epi16 _mm256_subs_epi16_
#define _mm256_hadds_epi16 _mm256_hadds_epi16_

#endif


#endif //_INTRIN


/*******************************************************************************
 * pmath-x86.h                                                                 *
 *                                                                             *
 * Copyright (c) 2013-2017 Ronny Press                                         *
 *                                                                             *
 * rpress@soprero.de                                                           *
 *                                                                             *
 * All rights reserved.                                                        *
 *******************************************************************************/

#ifndef PMATH_X86_H
#define PMATH_X86_H

#ifndef PMATH_H
#error Do not include this file directly, include pmath.h instead
#endif


//
// a few macros
//
#define OP_(op,scalar,packed,prefix,intrinop,postfix) \
static inline packed operator op(packed op1, packed op2) { return prefix##intrinop##postfix(op1, op2); } \
static inline packed operator op(scalar op1, packed op2) { return prefix##intrinop##postfix(prefix##set1##postfix(op1), op2); } \
static inline packed operator op(packed op1, scalar op2) { return prefix##intrinop##postfix(op1, prefix##set1##postfix(op2)); }
#ifdef _MSC_VER
#define ARITH_OPS_(scalar,packed,prefix,postfix) \
    OP_(+,scalar,packed,prefix,add,postfix) \
    OP_(-,scalar,packed,prefix,sub,postfix) \
    OP_(*,scalar,packed,prefix,mul,postfix) \
    OP_(/,scalar,packed,prefix,div,postfix) \
    OP_(&,scalar,packed,prefix,and,postfix) \
    OP_(|,scalar,packed,prefix,or,postfix) \
    OP_(^,scalar,packed,prefix,xor,postfix)
#define CMP_OPS_(scalar,packed,prefix,postfix) \
    OP_(<,scalar,packed,prefix,cmplt,postfix) \
    OP_(>,scalar,packed,prefix,cmpgt,postfix) \
    OP_(==,scalar,packed,prefix,cmpeq,postfix) \
    OP_(<=,scalar,packed,prefix,cmple,postfix) \
    OP_(>=,scalar,packed,prefix,cmpge,postfix) \
    OP_(!=,scalar,packed,prefix,cmpneq,postfix)
#else
// GCC has some of the operators built-in
//#define ARITH_OPS_(scalar,packed,prefix,postfix) \
//    OP_(&,scalar,packed,prefix,and,postfix) \
//    OP_(|,scalar,packed,prefix,or,postfix) \
//    OP_(^,scalar,packed,prefix,xor,postfix)
#define ARITH_OPS_(scalar,packed,prefix,postfix)
#define CMP_OPS_(scalar,packed,prefix,postfix)
#endif
#define AVX_CMP_OP_(op,oparg,scalar,packed,postfix) \
inline packed operator op(packed op1, packed op2) { return _mm256_cmp##postfix(op1, op2, _CMP_##oparg##_OQ); } \
inline packed operator op(scalar op1, packed op2) { return _mm256_cmp##postfix(_mm256_set1##postfix(op1), op2, _CMP_##oparg##_OQ); } \
inline packed operator op(packed op1, scalar op2) { return _mm256_cmp##postfix(op1, _mm256_set1##postfix(op2), _CMP_##oparg##_OQ); }
#define AVX_CMP_OPS_(scalar,packed,postfix) \
    AVX_CMP_OP_(<,LT,scalar,packed,postfix) \
    AVX_CMP_OP_(>,GT,scalar,packed,postfix) \
    AVX_CMP_OP_(==,EQ,scalar,packed,postfix) \
    AVX_CMP_OP_(<=,LE,scalar,packed,postfix) \
    AVX_CMP_OP_(>=,GE,scalar,packed,postfix) \
    AVX_CMP_OP_(!=,NEQ,scalar,packed,postfix)


//
// packed float4 arithmetic and bitwise logical
//
ARITH_OPS_(float,__m128,_mm_,_ps)

#ifdef _MSC_VER
inline __m128 operator-(__m128 op) { return _mm_sub_ps(_mm_setzero_ps(), op); }
#endif

inline __m128 notAandB_(__m128 a, __m128 b) { return _mm_andnot_ps(a, b); }


//
// packed float4 comparisons
//
CMP_OPS_(float,__m128,_mm_,_ps)



// packed double precision floating point calculations are supported starting
// with SSE2
#if defined(SSE2) || defined(AVX)
//
// packed double2 arithmetic and bitwise logical
//
ARITH_OPS_(double,__m128d,_mm_,_pd)
#ifdef _MSC_VER
inline __m128d operator-(__m128d op) { return _mm_sub_pd(_mm_setzero_pd(), op); }
inline __m128d notAandB_(__m128d a, __m128d b) { return _mm_andnot_pd(a, b); }
#endif

//
// packed double2 comparisons
//
CMP_OPS_(double,__m128d,_mm_,_pd)
#endif // SSE2 || AVX


#ifdef AVX
//
// packed double4 arithmetic and bitwise logical
//
ARITH_OPS_(double,__m256d,_mm256_,_pd)
inline __m256d operator-(__m256d op) { return _mm256_sub_pd(_mm256_setzero_pd(), op); }

//
// packed double4 comparisons
//
AVX_CMP_OPS_(double,__m256d,_pd)


//
// packed float8 arithmetic and bitwise logical
//
ARITH_OPS_(float,__m256,_mm256_,_ps)
inline __m256 operator-(__m256 op) { return _mm256_sub_ps(_mm256_setzero_ps(), op); }


//
// packed float8 comparisons
//
AVX_CMP_OPS_(float, __m256, _ps)

#endif // AVX


namespace math {
//
// shuffles/swizzles
//
// memory layout is: x,y,z,w
// register layout is the other way around: w,z,y,x
// THIS IS WRONG!
//
// TODO: add static asserts for: N != 4 for {x,y,z,w}{4}()
//                               N != 2 for {x,y}{2}()
// to implement this, add a static const unsigned with the value of N
// so it can be checked
//
// usage:
// the functions with one parameter return the respective permutation
//   of its parameter, i.e. xyzw() returns a copy and wzyx() returns
//   a value with reversed order of the elements
// the functions with two parameters return a permutation with the
// first and second letter of the function name denoting the elements
// to be taken from the first parameter and the third and fourth
// letters of the function name denoting the element to be taken from
// the second parameter, i.e. xxyy(a,b) returns {a.x,a.x,b.y,b.y}
//                            zzww(a,b) returns {a.z,a.z,b.w,b.w}

#define SWIZZLE_FLOAT_4____(a,b,c,d,a_,b_,c_,d_) \
template<typename T> T a##b##c##d(const T &); \
template<typename T> T a##b##c##d(const T &, const T &); \
template<> inline __m128 a##b##c##d(const __m128 & v) { return _mm_shuffle_ps(v, v, _MM_SHUFFLE(d_,c_,b_,a_)); } \
template<> inline __m128 a##b##c##d(const __m128 & v1, const __m128 & v2) { return _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(d_, c_, b_, a_)); }
#define SWIZZLE_FLOAT_4___(a,b,c,a_,b_,c_) \
    SWIZZLE_FLOAT_4____(a,b,c,x,a_,b_,c_,0) \
    SWIZZLE_FLOAT_4____(a,b,c,y,a_,b_,c_,1) \
    SWIZZLE_FLOAT_4____(a,b,c,z,a_,b_,c_,2) \
    SWIZZLE_FLOAT_4____(a,b,c,w,a_,b_,c_,3)
#define SWIZZLE_FLOAT_4__(a,b,a_,b_) \
    SWIZZLE_FLOAT_4___(a,b,x,a_,b_,0) \
    SWIZZLE_FLOAT_4___(a,b,y,a_,b_,1) \
    SWIZZLE_FLOAT_4___(a,b,z,a_,b_,2) \
    SWIZZLE_FLOAT_4___(a,b,w,a_,b_,3)
#define SWIZZLE_FLOAT_4_(a,a_) \
    SWIZZLE_FLOAT_4__(a,x,a_,0) \
    SWIZZLE_FLOAT_4__(a,y,a_,1) \
    SWIZZLE_FLOAT_4__(a,z,a_,2) \
    SWIZZLE_FLOAT_4__(a,w,a_,3)
#define SWIZZLE_FLOAT_4 \
    SWIZZLE_FLOAT_4_(x,0) \
    SWIZZLE_FLOAT_4_(y,1) \
    SWIZZLE_FLOAT_4_(z,2) \
    SWIZZLE_FLOAT_4_(w,3)

SWIZZLE_FLOAT_4

#undef SWIZZLE_FLOAT_4____
#undef SWIZZLE_FLOAT_4___
#undef SWIZZLE_FLOAT_4__
#undef SWIZZLE_FLOAT_4_
#undef SWIZZLE_FLOAT_4

// packed double precision floating point calculations are supported starting
// with SSE2
#if defined(SSE2) || defined(AVX)
#define SWIZZLE_DOUBLE_2__(a,b,a_,b_) \
template<typename T> T a##b(const T &); \
template<typename T> T a##b(const T &, const T &); \
template<> inline __m128d a##b(const __m128d & v) { return _mm_shuffle_pd(v, v, _MM_SHUFFLE2(b_, a_)); } \
template<> inline __m128d a##b(const __m128d & v1, const __m128d & v2) { return _mm_shuffle_pd(v1, v2, _MM_SHUFFLE2(b_, a_)); }
#define SWIZZLE_DOUBLE_2_(a,a_) \
    SWIZZLE_DOUBLE_2__(a, x, a_, 0) \
    SWIZZLE_DOUBLE_2__(a, y, a_, 1)
#define SWIZZLE_DOUBLE_2 \
    SWIZZLE_DOUBLE_2_(x, 0) \
    SWIZZLE_DOUBLE_2_(y, 1)

SWIZZLE_DOUBLE_2

#undef SWIZZLE_DOUBLE_2__
#undef SWIZZLE_DOUBLE_2_
#undef SWIZZLE_DOUBLE_2
#endif // SSE2 || AVX


#ifdef AVX

#define PERM_IMM(a,b) ((a<<4)|b)
#define SHUF_IMM(a,b,c,d) ((a<<3)|(b<<2)|(c<<1)|d)
#define SWIZZLE_DOUBLE_4____(a,b,c,d,p0,p1,p2,p3,s0,s1,s2,s3)  \
template<> inline __m256d a##b##c##d(const __m256d & v)        \
{                                                              \
    return                                                     \
        _mm256_shuffle_pd(                                     \
            _mm256_permute2f128_pd(v, v, PERM_IMM(p0,p1)),     \
            _mm256_permute2f128_pd(v, v, PERM_IMM(p2,p3)),     \
            SHUF_IMM(s0,s1,s2,s3)                              \
        );                                                     \
}
#define SWIZZLE_DOUBLE_4___(a,b,c,p1,p2,p3,s1,s2,s3)    \
    SWIZZLE_DOUBLE_4____(a,b,c,x,0,p1,p2,p3,0,s1,s2,s3) \
    SWIZZLE_DOUBLE_4____(a,b,c,y,0,p1,p2,p3,1,s1,s2,s3) \
    SWIZZLE_DOUBLE_4____(a,b,c,z,1,p1,p2,p3,0,s1,s2,s3) \
    SWIZZLE_DOUBLE_4____(a,b,c,w,1,p1,p2,p3,1,s1,s2,s3)
#define SWIZZLE_DOUBLE_4__(a,b,p1,p3,s2,s3)    \
    SWIZZLE_DOUBLE_4___(a,b,x,p1,0,p3,0,s2,s3) \
    SWIZZLE_DOUBLE_4___(a,b,y,p1,0,p3,1,s2,s3) \
    SWIZZLE_DOUBLE_4___(a,b,z,p1,1,p3,0,s2,s3) \
    SWIZZLE_DOUBLE_4___(a,b,w,p1,1,p3,1,s2,s3)
#define SWIZZLE_DOUBLE_4_(a,p3,s3)    \
    SWIZZLE_DOUBLE_4__(a,x,0,p3,0,s3) \
    SWIZZLE_DOUBLE_4__(a,y,0,p3,1,s3) \
    SWIZZLE_DOUBLE_4__(a,z,1,p3,0,s3) \
    SWIZZLE_DOUBLE_4__(a,w,1,p3,1,s3)
#define SWIZZLE_DOUBLE_4     \
    SWIZZLE_DOUBLE_4_(x,0,0) \
    SWIZZLE_DOUBLE_4_(y,0,1) \
    SWIZZLE_DOUBLE_4_(z,1,0) \
    SWIZZLE_DOUBLE_4_(w,1,1)
SWIZZLE_DOUBLE_4
#undef SWIZZLE_DOUBLE_4____


#define BLEND_IMM(a,b,c,d) ((a<<3)|(b<<2)|(c<<1)|d)
// if bit in imm8 is set, then copy elem from b, else from a:
// {B3}{B2}{B1}{B0} for the four elems in a _register_
#define SWIZZLE_DOUBLE_4____(a,b,c,d,p0,p1,p2,p3,s0,s1,s2,s3)                \
template<> inline __m256d a##b##c##d(const __m256d & v0, const __m256d & v1) \
{                                                                            \
    __m256d perm0 = _mm256_permute2f128_pd(v0, v0, PERM_IMM(p0,p1));         \
    __m256d perm1 = _mm256_permute2f128_pd(v1, v1, PERM_IMM(p2,p3));         \
    perm0 = _mm256_shuffle_pd(perm0, perm0, SHUF_IMM(0,0,s2,s3));            \
    perm1 = _mm256_shuffle_pd(perm1, perm1, SHUF_IMM(s0,s1,0,0));            \
    return _mm256_blend_pd(perm0, perm1, BLEND_IMM(1,1,0,0));                \
}
SWIZZLE_DOUBLE_4

#undef PERM_IMM
#undef SHUF_IMM
#undef BLEND_IMM
#undef SWIZZLE_DOUBLE_4____
#undef SWIZZLE_DOUBLE_4___
#undef SWIZZLE_DOUBLE_4__
#undef SWIZZLE_DOUBLE_4_
#undef SWIZZLE_DOUBLE_4

#endif // AVX


} // namespace math


#endif // !defined(PMATH_X86_H)

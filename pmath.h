/*******************************************************************************
 * pmath.h                                                                     *
 *                                                                             *
 * Copyright (c) 2013-2017 Ronny Press                                         *
 *                                                                             *
 * rpress@soprero.de                                                           *
 *                                                                             *
 * All rights reserved.                                                        *
 *******************************************************************************/

#ifndef PMATH_H
#define PMATH_H

// movehl_ps(a,b):
//   {a.x,a.y,a.z,a.w}
//   {b.x,b.y,b.z,b.w}
//-> {a.x,a.y,b.x,b.y}
// movelh_ps(a,b):
//   {a.x,a.y,a.z,a.w}
//   {b.x,b.y,b.z,b.w}
//-> {b.z,b.w,a.z,a.w}

#include <limits>
#include <cmath>
#include <stdint.h>
#include <type_traits>


#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
// x86/x64 with MSVC
#  include <intrin.h>
#elif defined(_MSC_VER) && defined(_M_ARM)
// ARM NEON with MSVC
#  include <intrin.h>
#  include <arm_neon.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
// x86/x64 with GCC / MinGW
#  include <x86intrin.h>
#elif defined(__GNUC__) && defined(__ARM_ARCH) && defined(__ARM_NEON_FP)
// ARM NEON with GCC
#  include <arm_neon.h>
#endif



//#define AVX


// use target dependent declaration file

#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
// x86/x64 with MSVC
#  include <pmath-x86.h>
#elif defined(_MSC_VER) && defined(_M_ARM)
// ARM NEON with MSVC
#  include <pmath-neon.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
// x86/x64 with GCC / MinGW
#  include <pmath-x86.h>
#elif defined(__GNUC__) && defined(__ARM_ARCH) && defined(__ARM_NEON_FP)
// ARM NEON with GCC
#  include <pmath-neon.h>
#endif



namespace math {

//
// TODO: check that the approach here actually works,
//       as inlined statics may be a problem
//       the problem is, that if the compiler decides
//       not to inline these functions it must compile
//       them into normal functions without external linkage
//       (static), so each compilation unit not only gets its
//       own copy of the functions' code but also its own
//       copy of the static variable
// if it doesn't work, better calculate the values each
// time (it may not be as expensive as you think, as
// the current versions have memory accesses which can
// be expensive, too)
//
template<typename real_t> static inline real_t pi()
{ static const real_t val = static_cast<real_t>(4.0*std::atan(1.0)); return val; }
template<typename real_t> static inline real_t two_pi()
{ static const real_t val = static_cast<real_t>(8.0*std::atan(1.0)); return val; }
template<typename real_t> static inline real_t half_pi()
{ static const real_t val = static_cast<real_t>(2.0*std::atan(1.0)); return val; }
template<typename real_t> static inline real_t inv_pi()
{ static const real_t val = static_cast<real_t>(0.25/std::atan(1.0)); return val; }
template<typename real_t> static inline real_t inv_two_pi()
{ static const real_t val = static_cast<real_t>(0.125/std::atan(1.0)); return val; }
template<typename real_t> static inline real_t deg2rad()
{ static const real_t val = static_cast<real_t>(4.0*std::atan(1.0)/180.0); return val; }
template<typename real_t> static inline real_t rad2deg()
{ static const real_t val = static_cast<real_t>(45.0/std::atan(1.0)); return val; }
template<typename real_t> static inline real_t euler()
{ static const real_t val = static_cast<real_t>(std::exp(1.0)); return val; }
template<typename real_t> static inline real_t sqrt2()
{ static const real_t val = static_cast<real_t>(std::sqrt(2.0)); return val; }
template<typename real_t> static inline real_t goldenratio()
{ static const real_t val = static_cast<real_t>((1.0 + std::sqrt(5.0)) * 0.5); return val; }


//
// math_t general template
//
template<typename real_t,typename packed_t>
class math_t
{
public:
    static_assert(
        std::is_floating_point<real_t>::value,
        "math_t: supports floating point types only"
    );

    static inline packed_t set1(real_t scalar);

    static inline packed_t pi_packed();
    static inline packed_t two_pi_packed();
    static inline packed_t half_pi_packed();
    static inline packed_t inv_pi_packed();
    static inline packed_t inv_two_pi_packed();
    static inline packed_t deg2rad_packed();
    static inline packed_t rad2deg_packed();
    static inline packed_t euler_packed();
    static inline packed_t sqrt2_packed();
    static inline packed_t goldenratio_packed();

    static inline packed_t zeroes();
    static inline packed_t ones();
    static inline packed_t halves();

    static inline packed_t signs(packed_t);
    static inline packed_t reciprocals(packed_t);
    static inline packed_t inv_sqrt_packed(packed_t);
    static inline packed_t sqrt_packed(packed_t);
    static inline real_t inv_sqrt(real_t);
    static inline real_t sqrt(real_t);
    static inline real_t min_scalar(real_t a, real_t b);
    static inline real_t min_packed(real_t a, real_t b);
    static inline packed_t min_packed(packed_t a, packed_t b);
    static inline real_t max_scalar(real_t a, real_t b);
    static inline real_t max_packed(real_t a, real_t b);
    static inline packed_t max_packed(packed_t a, packed_t b);
    
    static inline packed_t dot_packed(packed_t a, packed_t b);

    // angles : [0,pi/2], error: |e(x)| <= 1.7e-4
    static inline packed_t fast_sin_0(packed_t angles);
    // angles : [0,pi/2], error: |e(x)| <= 1.9e-8
    static inline packed_t fast_sin_1(packed_t angles);
    // angles : [0,pi/2], error: |e(x)| <= 1.2e-3
    static inline packed_t fast_cos_0(packed_t angles);
    // angles : [0,pi/2], error: |e(x)| <= 6.5e-9
    static inline packed_t fast_cos_1(packed_t angles);
    // angles : [0,pi/4], error: |e(x)| <= 8.1e-4
    static inline packed_t fast_tan_0(packed_t angles);
    // angles:  [0,pi/2], error: |e(x)| <= 1.9e-8
    static inline packed_t fast_tan_1(packed_t angles);

    // vals : [0,1], error: |e(x)| <= 6.8e-5
    static inline packed_t fast_arcsin_0(packed_t vals);
    // vals : [0,1], error: |e(x)| <= 1.4e-7
    static inline packed_t fast_arcsin_1(packed_t vals);
    // vals : [0,1], error: |e(x)| <= 6.8e-5
    static inline packed_t fast_arccos_0(packed_t vals)
    { return half_pi_packed() - fast_arcsin_0(vals); }
    // vals : [0,1], error: |e(x)| <= 1.3e-7
    static inline packed_t fast_arccos_1(packed_t vals)
    { return half_pi_packed() - fast_arcsin_1(vals); }
    // vals : [-1,1], error: |e(x)| <= 1.2e-5
    static inline packed_t fast_arctan_0(packed_t vals);
    // vals : [-1,1], error: |e(x)| <= 2.3e-8
    static inline packed_t fast_arctan_1(packed_t vals);

    static inline packed_t int32_to_packed(int32_t);
    static inline packed_t int64_to_packed(int64_t);
    
private:
    math_t();
    math_t(const math_t &);
    math_t & operator=(const math_t &);
};



//
// Horner polynome evaluation (even exponents only, no absolute term)
//
template<unsigned N, typename real_t> struct horner_even_t {
    inline real_t operator()(real_t x, const real_t * coeffs) {
        return x * x * (coeffs[N-1] + horner_even_t<N-1,real_t>()(x, coeffs));
    }
};
template<typename real_t> struct horner_even_t<1,real_t> {
    inline real_t operator()(real_t x, const real_t * coeffs) {
        return coeffs[0] * x * x;
    }
};

template<unsigned N, typename real_t, typename packed_t> struct horner_even_packed_t
{
    inline packed_t operator()(packed_t x, const real_t * coeffs) {
        return x * x * (coeffs[N-1] + horner_even_packed_t<N-1,real_t,packed_t>()(x, coeffs));
    }
};
template<typename real_t, typename packed_t> struct horner_even_packed_t<1,real_t,packed_t>
{
    inline packed_t operator()(packed_t x, const real_t * coeffs) {
        return coeffs[0] * x * x;
    }
};

//
// Horner polynome evaluation (odd exponents only, no absolute term)
//
template<unsigned N, typename real_t> struct horner_odd_t
{
    inline real_t operator()(real_t x, const real_t * coeffs)
    {
        return x * (horner_even_t<N-1,real_t>()(x, coeffs) + coeffs[N-1]);
    }
};

template<unsigned N, typename real_t, typename packed_t> struct horner_odd_packed_t
{
    inline packed_t operator()(packed_t x, const real_t * coeffs)
    {
        return x * (horner_even_packed_t<N-1,real_t,packed_t>()(x,coeffs) + coeffs[N-1]);
    }
};


//
// Horner polynome evaluation (odd and even exponents, no absolute term)
//
template<unsigned N, typename real_t> struct horner_t {
    inline real_t operator()(real_t x, const real_t * coeffs) {
        return x * (coeffs[N-1] + horner_t<N-1,real_t>()(x, coeffs));
    }
};
template<typename real_t> struct horner_t<1,real_t> {
    inline real_t operator()(real_t x, const real_t * coeffs) {
        return coeffs[0] * x;
    }
};

template<unsigned N, typename real_t, typename packed_t> struct horner_packed_t {
    inline packed_t operator()(packed_t x, const real_t * coeffs) {
        return x * (coeffs[N-1] + horner_packed_t<N-1,real_t, packed_t>()(x, coeffs));
    }
};
template <typename real_t, typename packed_t> struct horner_packed_t<1,real_t,packed_t> {
    inline packed_t operator()(packed_t x, const real_t * coeffs) {
        return coeffs[0] * x;
    }
};


// expeval_t
// returns pow(x,N)
template<unsigned N, typename real_t> struct expeval_t {
    inline real_t operator()(real_t x) {
        return x * expeval_t<N-1,real_t>()(x);
    }
};
template<typename real_t> struct expeval_t<1,real_t> {
    inline real_t operator()(real_t x) { return x; }
};
template<typename real_t> struct expeval_t<0U,real_t> {
    inline real_t operator()(real_t x) { return real_t(1.0); }
};
template<unsigned N, typename real_t, typename packed_t> struct expeval_packed_t {
    inline packed_t operator()(packed_t x) {
        return x * expeval_packed_t<N-1,real_t,packed_t>()(x);
    }
};
template<typename real_t, typename packed_t> struct expeval_packed_t<1U,real_t,packed_t> {
    inline packed_t operator()(packed_t x) { return x; }
};
template<typename real_t, typename packed_t> struct expeval_packed_t<0U,real_t,packed_t> {
    inline packed_t operator()(packed_t x) { return math_t<real_t,packed_t>::ones(); }
};

// polynomialeval_t
// returns sum[0<=i<N](coeffs[i]*pow(x,i))
template<unsigned N, typename real_t> struct polynomialeval_t
{
    inline real_t operator()(real_t x, const real_t * coeffs)
    {
        return coeffs[N-1] * expeval_t<N-1,real_t>()(x) + polynomialeval_t<N-1,real_t>()(x, coeffs);
    }
};
template<typename real_t> struct polynomialeval_t<1,real_t>
{
    inline real_t operator()(real_t x, const real_t * coeffs) { return coeffs[0]; }
};
template<unsigned N,typename real_t, typename packed_t> struct polynomialeval_packed_t
{
    inline packed_t operator()(packed_t x, const real_t * coeffs)
    {
        return coeffs[N-1] * expeval_packed_t<N-1,real_t,packed_t>()(x) + polynomialeval_packed_t<N-1,real_t,packed_t>()(x, coeffs);
    }
};
template<typename real_t, typename packed_t> struct polynomialeval_packed_t<1,real_t,packed_t>
{
    // not optimal but does the conversion independent of instruction set
    inline packed_t operator()(packed_t x, const real_t * coeffs) { return math_t<real_t,packed_t>::ones() * coeffs[0]; }
};

// polynomialeval_even_t
// returns sum[0<=i<N](coeffs[i]*x^(n*2+2))
template<unsigned N, typename real_t> struct polynomialeval_even_t
{
    inline real_t operator()(real_t x, const real_t * coeffs)
    {
        return coeffs[N-1] * expeval_t<N*2,real_t>()(x) + polynomialeval_even_t<N-1,real_t>()(x, coeffs);
    }
};
template<typename real_t> struct polynomialeval_even_t<0U,real_t> {
    inline real_t operator()(real_t x, const real_t *) { return real_t(0.0); }
};
template<unsigned N, typename real_t, typename packed_t> struct polynomialeval_even_packed_t
{
    inline packed_t operator()(packed_t x, const real_t * coeffs)
    {
        return coeffs[N-1] * expeval_packed_t<N*2,real_t,packed_t>()(x) + polynomialeval_even_packed_t<N-1,real_t,packed_t>()(x, coeffs);
    }
};
template<typename real_t, typename packed_t> struct polynomialeval_even_packed_t<0U,real_t,packed_t>
{
    inline packed_t operator()(packed_t, const real_t *) { return math_t<real_t,packed_t>::zeroes(); }
};

// polynomialeval_odd_t
// returns sum[0<=i<N](coeffs[i]*x^(n*2+1))
template<unsigned N, typename real_t> struct polynomialeval_odd_t
{
    inline real_t operator()(real_t x, const real_t * coeffs)
    {
        return coeffs[N-1] * expeval_t<N*2-1,real_t>()(x) + polynomialeval_odd_t<N-1,real_t>()(x, coeffs);
    }
};
template<typename real_t> struct polynomialeval_odd_t<0U,real_t>
{
    inline real_t operator()(real_t x, const real_t *) { return real_t(0.0); }
};
template<unsigned N, typename real_t, typename packed_t> struct polynomialeval_odd_packed_t
{
    inline packed_t operator()(packed_t x, const real_t * coeffs) {
        return coeffs[N-1] * expeval_packed_t<N*2-1,real_t,packed_t>()(x) + polynomialeval_odd_packed_t<N-1,real_t,packed_t>()(x, coeffs);
    }
};
template<typename real_t, typename packed_t> struct polynomialeval_odd_packed_t<0U,real_t,packed_t>
{
    inline packed_t operator()(packed_t x, const real_t *) { return math_t<real_t,packed_t>::zeroes(); }
};



template<typename real_t>
static inline real_t max_real() { return (std::numeric_limits<real_t>::max)(); }
    
template<typename real_t>
inline real_t epsilon() { return std::numeric_limits<real_t>::epsilon(); }

template<typename real_t>
inline bool almost_equal(real_t value, real_t ref_value)
{ return (value >= (ref_value - epsilon<real_t>())) &&
         (value <= (ref_value + epsilon<real_t>())); }
template<typename real_t>
inline bool almost_equal(real_t value, real_t ref_value, real_t range)
{ return (value >= (ref_value - range)) &&
         (value <= (ref_value + range)); }


// use target dependent template specialization file

#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
// x86/x64 with MSVC
#  include "pmath-x86-spec.h"
#elif defined(_MSC_VER) && defined(_M_ARM)
// ARM NEON with MSVC
#  include "pmath-neon-spec.h"
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
// x86/x64 with GCC / MinGW
#  include "pmath-x86-spec.h"
#elif defined(__GNUC__) && defined(__ARM_ARCH) && defined(__ARM_NEON_FP)
// ARM NEON with GCC
#  include "pmath-neon-spec.h"
#endif


} // namespace math

#endif // !defined(PMATH_H)

/*******************************************************************************
 * pmath-x86-spec.h                                                            *
 *                                                                             *
 * Copyright (c) 2013-2017 Ronny Press                                         *
 *                                                                             *
 * rpress@soprero.de                                                           *
 *                                                                             *
 * All rights reserved.                                                        *
 *******************************************************************************/

#ifndef PMATH_X86_SPEC_H
#define PMATH_X86_SPEC_H

#ifndef PMATH_H
#error Do not include this file directly, include pmath.h instead
#endif


//
// math_t specialization for float{4}/__m128
//
template<>
class math_t<float,__m128>
{
public:
    typedef float real_t;
    typedef __m128 packed_t;

    static inline __m128 set1(float scalar) { return _mm_set1_ps(scalar); }

    static inline __m128 pi_packed()          { return _mm_set_ps1(pi<float>());          }
    static inline __m128 two_pi_packed()      { return _mm_set_ps1(two_pi<float>());      }
    static inline __m128 half_pi_packed()     { return _mm_set_ps1(half_pi<float>());     }
    static inline __m128 inv_pi_packed()      { return _mm_set_ps1(inv_pi<float>());      }
    static inline __m128 inv_two_pi_packed()  { return _mm_set_ps1(inv_two_pi<float>());  }
    static inline __m128 deg2rad_packed()     { return _mm_set_ps1(deg2rad<float>());     }
    static inline __m128 rad2deg_packed()     { return _mm_set_ps1(rad2deg<float>());     }
    static inline __m128 euler_packed()       { return _mm_set_ps1(euler<float>());       }
    static inline __m128 sqrt2_packed()       { return _mm_set_ps1(sqrt2<float>());       }
    static inline __m128 goldenratio_packed() { return _mm_set_ps1(goldenratio<float>()); }

    static inline __m128 zeroes() { return _mm_setzero_ps(); }
    static inline __m128 ones() {
        static const __m128 ones_ = { 1.0f, 1.0f, 1.0f, 1.0f }; return ones_;
    }
    static inline __m128 halves() {
        static const __m128 halves_ = { 0.5f, 0.5f, 0.5f, 0.5f }; return halves_;
    }

    static inline __m128 signs(__m128 v)
    {
        // v   | cmpneq     | and1       | or       | and2
        // 0   |   0        | 0          | 1        | 0
        // >0  | 0xFFFFFFFF | 0          | 1        | 1
        // <0  | 0xFFFFFFFF | 0x80000000 | -1       | -1
        #ifdef _MSC_VER
        return ((v & -0.0f) | 1.0f) & (v != zeroes());
        #else
        // GCC does not allow operator overloads on __m128 and has the operator&
        // not implemented for __m128 (which makes sense)
        return _mm_and_ps(_mm_or_ps(_mm_and_ps(v, _mm_set1_ps(-0.0f)), ones()), _mm_cmpneq_ps(v, zeroes()));
        #endif
    }

    static inline __m128 reciprocals(__m128 x) {
        // F(y) = 1/y - x = 0
        // Newton-Raphson:
        // y[n+1] = y[n] - F(y[n]) / F'(y[n]) = y[n](2-y[n]*x)
        __m128 approx = _mm_rcp_ps(x);
        return approx * ((ones() + ones()) - (approx * x));
    }

    static inline __m128 inv_sqrt_packed(__m128 x) {
        // F(y) = (1/(y*y)) - x = 0
        // Newton-Raphson:
        // y[n+1] = y[n] - F(y[n]) / F'(y[n]) = 0.5y[n]*(3-y[n]*y[n]*x)
        static const __m128 threes = { 3.0f, 3.0f, 3.0f, 3.0f };
        __m128 approx = _mm_rsqrt_ps(x);
        return halves() * approx * (threes - (approx * approx * x));
    }
    static inline __m128 sqrt_packed(__m128 x)
    { return _mm_mul_ps(x, inv_sqrt_packed(x)); }
    static inline float inv_sqrt(float x)
    { return _mm_cvtss_f32(inv_sqrt_packed(_mm_load_ps1(&x))); }
    static inline float sqrt(float x)
    { // x * (1 / sqrt(x)) = sqrt(x)
      return x * inv_sqrt(x); }
    static inline float min_scalar(float a, float b)
    { return _mm_cvtss_f32(_mm_min_ss(_mm_set_ss(a), _mm_set_ss(b))); }
    static inline float min_packed(float a, float b)
    { return _mm_cvtss_f32(_mm_min_ps(_mm_load_ps1(&a), _mm_load_ps1(&b))); }
    static inline __m128 min_packed(__m128 a, __m128 b)
    { return _mm_min_ps(a, b); }
    static inline float max_scalar(float a, float b)
    { return _mm_cvtss_f32(_mm_max_ss(_mm_set_ss(a), _mm_set_ss(b))); }
    static inline float max_packed(float a, float b)
    { return _mm_cvtss_f32(_mm_max_ps(_mm_load_ps1(&a), _mm_load_ps1(&b))); }
    static inline __m128 max_packed(__m128 a, __m128 b)
    { return _mm_max_ps(a, b); }

    static inline __m128 dot_packed(__m128 a, __m128 b)
// TODO: dp_ps() and hadd_ps() are rumored to be slow (too complex which stalls
//       the pipeline, even though less code)
#ifdef SSE4
    { return _mm_dp_ps(a, b, 0xFF); }
#elif defined(SSE3)
    { __m128 tmp = a * b; // {ae,bf,cg,dh}
      tmp = _mm_hadd_ps(tmp, tmp); // {ae+bf,cg+dh,ae+bf,cg+dh}
      return _mm_hadd_ps(tmp, tmp); // {ae+bf+cg+dh,ae+bf+cg+dh,ae+bf+cg+dh,ae+bf+cg+dh}
    }
#else
    { // [a]     [e]
      // [b]     [f]
      // [c] dot [g] = ae+bf+cg+dh
      // [d]     [h]
      __m128 tmp = a * b; // {ae,bf,cg,dh}
      tmp = tmp + yxwz(tmp); // {ae+bf,bf+ae,cg+dh,dh+cg}
      return tmp + zwyx(tmp); // {ae+bf+cg+dh.bf+ae+dh+cg,cg+dh+bf+ae,dh+cg+ae+bf}
    }
#endif

    static inline __m128 fast_sin_0(__m128 angles)
    { static const float coeffs[] = { -0.16605f, 0.00761f };
      return angles * (
          ones() + polynomialeval_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,__m128>()(angles, coeffs)
      ); }
    static inline __m128 fast_sin_1(__m128 angles)
    { static const float coeffs[] = { -0.1666666664f, 0.0083333315f, -0.0001984090f, 0.0000027526f, -0.0000000239f };
      return angles * (
          ones() + polynomialeval_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,__m128>()(angles, coeffs)
      ); }
    static inline __m128 fast_cos_0(__m128 angles)
    { static const float coeffs[] = { -0.49670f, 0.03705f };
      return
          ones() + polynomialeval_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,__m128>()(angles, coeffs);
    }
    static inline __m128 fast_cos_1(__m128 angles)
    { //static const float coeffs[] = { -0.0000002605f, 0.0000247609f, -0.0013888397f, 0.0416666418f, -0.4999999963f };
      static const float coeffs[] = { -0.4999999963f, 0.0416666418f, -0.0013888397f, 0.0000247609f, -0.0000002605f };
      return
          //ones() + horner_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,__m128>()(angles*angles, coeffs);
          ones() + polynomialeval_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,__m128>()(angles, coeffs);
    }
    static inline __m128 fast_tan_0(__m128 angles)
    { static const float coeffs[] = {0.20330f, 0.31755f};
      return angles * (
          ones() + horner_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,__m128>()(angles*angles, coeffs)
      ); }
    static inline __m128 fast_tan_1(__m128 angles)
    { static const float coeffs[] = {
        0.0095168091f, 0.0029005250f, 0.0245650893f, 0.0533740603f, 0.1333923995f, 0.3333314036f
      };
      return angles * (
          ones() + horner_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,__m128>()(angles*angles, coeffs)
      ); }

    static inline __m128 fast_arcsin_0(__m128 vals)
    { // arcsin0(x) = pi/2 - sqrt(1-x)(1.5707288 - 0.2121144x + 0.0742610x*x - 0.0187293x*x*x)
      static const float coeffs[] = {-0.0187293f, 0.0742610f, -0.2121144f};
      __m128 v = ones() - vals;
      return half_pi_packed() -
          v * inv_sqrt_packed(v) * (
            horner_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,__m128>()(vals, coeffs)
            + _mm_set1_ps(1.5707288f)
          ); }
    static inline __m128 fast_arcsin_1(__m128 vals)
    { static const float coeffs[] = {
          -0.0012624911f, 0.0066700901f, -0.01708812556f, 0.0308918810f,
          -0.0501743046f, 0.0889789874f, -0.21459880160f
      };
      __m128 v = ones() - vals;
      return half_pi_packed() -
          v * inv_sqrt_packed(v) * (
            horner_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,__m128>()(vals, coeffs)
            + _mm_set1_ps(1.5707963050f)
          );
    }
    static inline __m128 fast_arctan_0(__m128 vals)
    { static const float coeffs[] = {
           0.0208351f, -0.0851330f, 0.1801410f,
          -0.3302995f,  0.9998660f
      };
      return horner_odd_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,__m128>()(vals, coeffs);
    }
    static inline __m128 fast_arctan_1(__m128 vals)
    { static const float coeffs[] = {
          0.0028662257f, -0.0161657367f, 0.0429096138f, -0.0752896400f,
          0.1065626393f, -0.1420889944f, 0.1999355085f, -0.3333314528f
      };
      return vals * (
          ones() + horner_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,__m128>()(vals, coeffs)
      );
    }

    static inline __m128 int32_to_packed(int32_t i)
    { return _mm_cvt_si2ss(zeroes(), i); }
#ifdef _M_X64
    static inline __m128 int64_to_packed(int64_t i)
    { return _mm_cvtsi64_ss(zeroes(), i); }
#endif
};



// packed double precision floating point calculations are supported starting
// with SSE2
#if defined(SSE2) || defined(AVX)
//
// math_t specialization for double{2}/__m128d
//
template<>
class math_t<double, __m128d>
{
public:
    typedef double real_t;
    typedef __m128d packed_t;

    static inline __m128d set1(double scalar) { return _mm_set1_pd(scalar); }

    static inline __m128d pi_packed()          { return _mm_set1_pd(pi<double>());          }
    static inline __m128d two_pi_packed()      { return _mm_set1_pd(two_pi<double>());      }
    static inline __m128d half_pi_packed()     { return _mm_set1_pd(half_pi<double>());     }
    static inline __m128d inv_pi_packed()      { return _mm_set1_pd(inv_pi<double>());      }
    static inline __m128d inv_two_pi_packed()  { return _mm_set1_pd(inv_two_pi<double>());  }
    static inline __m128d deg2rad_packed()     { return _mm_set1_pd(deg2rad<double>());     }
    static inline __m128d rad2deg_packed()     { return _mm_set1_pd(rad2deg<double>());     }
    static inline __m128d euler_packed()       { return _mm_set1_pd(euler<double>());       }
    static inline __m128d sqrt2_packed()       { return _mm_set1_pd(sqrt2<double>());       }
    static inline __m128d goldenratio_packed() { return _mm_set1_pd(goldenratio<double>()); }

    static inline __m128d zeroes() { return _mm_setzero_pd(); }
    static inline __m128d ones()
    {
        static const __m128d ones_ = {1.0, 1.0};
        return ones_;
    }
    static inline __m128d halves()
    {
        static const __m128d halves_ = {0.5, 0.5};
        return halves_;
    }

    static inline __m128d signs(__m128d v)
    {
        #ifdef _MSC_VER
        return ((v & -0.0) | 1.0) & (v != zeroes());
        #else
        // GCC does not allow operator overloads on __m128 and has the operator&
        // not implemented for __m128 (which makes sense)
        return _mm_and_pd(_mm_or_pd(_mm_and_pd(v, _mm_set1_pd(-0.0)), ones()), _mm_cmpneq_pd(v, zeroes()));
        #endif
    }

    // not recommended to be used for division (slower than
    // a direct division, and more accurate)
    static inline __m128d reciprocals(__m128d x)
    { return _mm_div_pd(ones(), x); }
    static inline __m128d inv_sqrt_packed(__m128d x)
    { // there's no _mm_rsqrt_pd() in SSE2!
      // if to be replaced by 1/_mm_sqrt_pd(): is there a N-R iteration
      // even needed here???
      // this assumes "not":
      return _mm_div_pd(ones(), sqrt_packed(x)); }
    static inline __m128d sqrt_packed(__m128d x) { return _mm_sqrt_pd(x); }
    static inline double inv_sqrt(double x)
    { return _mm_cvtsd_f64(inv_sqrt_packed(_mm_set1_pd(x))); }
    static inline double sqrt(double x)
    { return _mm_cvtsd_f64(sqrt_packed(_mm_set1_pd(x))); }
    static inline double min_scalar(double a, double b)
    { return _mm_cvtsd_f64(min_packed(_mm_set1_pd(a), _mm_set1_pd(b))); }
    static inline double min_packed(double a, double b)
    { return _mm_cvtsd_f64(min_packed(_mm_set1_pd(a), _mm_set1_pd(b))); }
    static inline __m128d min_packed(__m128d a, __m128d b)
    { return _mm_min_pd(a, b); }
    static inline double max_scalar(double a, double b)
    { return _mm_cvtsd_f64(max_packed(_mm_set1_pd(a), _mm_set1_pd(b))); }
    static inline double max_packed(double a, double b)
    { return _mm_cvtsd_f64(max_packed(_mm_set1_pd(a), _mm_set1_pd(b))); }
    static inline __m128d max_packed(__m128d a, __m128d b)
    { return _mm_max_pd(a, b); }

    static inline __m128d dot_packed(__m128d a, __m128d b)
        // TODO: dp_pd() and hadd_pd() are rumored to be slow (too complex which stalls
        //       the pipeline, even though less code)
#ifdef SSE4
    { return _mm_dp_pd(a, b, 0x33); }
#elif defined(SSE3)
    { __m128d tmp = a * b; return _mm_hadd_pd(tmp, tmp); }
#else
    { __m128d tmp = a * b; return tmp + yx(tmp); }
#endif

    static inline __m128d fast_sin_0(__m128d angles)
    { static const double coeffs[] = { 0.00761, -0.16605 };
      return angles * (
          ones() + horner_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),double,__m128d>()(angles*angles, coeffs)
      ); }
    static inline __m128d fast_sin_1(__m128d angles)
    { static const double coeffs[] = { -0.0000000239, 0.0000027526, -0.0001984090, 0.0083333315, -0.1666666664 };
      return angles * (
          ones() + horner_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),double,__m128d>()(angles*angles, coeffs)
      ); }
    static inline __m128d fast_cos_0(__m128d angles)
    { static const double coeffs[] = { 0.03705, -0.49670 };
      return
          ones() + horner_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),double,__m128d>()(angles*angles, coeffs);
    }
    static inline __m128d fast_cos_1(__m128d angles)
    { static const double coeffs[] = { -0.0000002605, 0.0000247609, -0.0013888397, 0.0416666418, -0.4999999963 };
      return
          ones() + horner_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),double,__m128d>()(angles*angles, coeffs);
    }
    static inline __m128d fast_tan_0(__m128d angles)
    { static const double coeffs[] = {0.20330, 0.31755};
      return angles * (
          ones() + horner_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),double,__m128d>()(angles*angles, coeffs)
      ); }
    static inline __m128d fast_tan_1(__m128d angles)
    { static const double coeffs[] = {
        0.0095168091, 0.0029005250, 0.0245650893, 0.0533740603, 0.1333923995, 0.3333314036
      };
      return angles * (
          ones() + horner_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),double,__m128d>()(angles*angles, coeffs)
      ); }

    static inline __m128d fast_arcsin_0(__m128d vals)
    { static const double coeffs[] = {-0.0187293, 0.0742610, -0.2121144};
      __m128d v = ones() - vals;
      return half_pi_packed() -
          v * inv_sqrt_packed(v) * (
            horner_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),double,__m128d>()(vals, coeffs)
            + _mm_set1_pd(1.5707288)
          ); }
    static inline __m128d fast_arcsin_1(__m128d vals)
    { static const double coeffs[] = {
          -0.0012624911, 0.0066700901, -0.01708812556, 0.0308918810,
          -0.0501743046, 0.0889789874, -0.21459880160
      };
      __m128d v = ones() - vals;
      return half_pi_packed() -
          v * inv_sqrt_packed(v) * (
            horner_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),double,__m128d>()(vals, coeffs)
            + _mm_set1_pd(1.5707963050)
          ); 
    }
    static inline __m128d fast_arctan_0(__m128d vals)
    { static const double coeffs[] = {
           0.0208351, -0.0851330, 0.1801410,
          -0.3302995,  0.9998660
      };
      return horner_odd_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),double,__m128d>()(vals, coeffs);
    }
    static inline __m128d fast_arctan_1(__m128d vals)
    { static const double coeffs[] = {
          0.0028662257, -0.0161657367, 0.0429096138, -0.0752896400,
          0.1065626393, -0.1420889944, 0.1999355085, -0.3333314528
      };
      return vals * (
          ones() + horner_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),double,__m128d>()(vals, coeffs)
      );
    }

    static inline __m128d int32_to_packed(int32_t i)
    { return _mm_cvtsi32_sd(zeroes(), i); }
#ifdef _M_X64
    static inline __m128d int64_to_packed(int64_t i)
    { return _mm_cvtsi64_sd(zeroes(), i); }
#endif
};
#endif // SSE2 || AVX



#ifdef AVX
//
// math_t specialization for float{8}/__m256
//
template<>
class math_t<float,__m256>
{
public:
    typedef float real_t;
    typedef __m256 packed_t;

    static inline __m256 pi_packed()          { return _mm256_set1_ps(pi<float>());          }
    static inline __m256 two_pi_packed()      { return _mm256_set1_ps(two_pi<float>());      }
    static inline __m256 half_pi_packed()     { return _mm256_set1_ps(half_pi<float>());     }
    static inline __m256 inv_pi_packed()      { return _mm256_set1_ps(inv_pi<float>());      }
    static inline __m256 inv_two_pi_packed()  { return _mm256_set1_ps(inv_two_pi<float>());  }
    static inline __m256 deg2rad_packed()     { return _mm256_set1_ps(deg2rad<float>());     }
    static inline __m256 rad2deg_packed()     { return _mm256_set1_ps(rad2deg<float>());     }
    static inline __m256 euler_packed()       { return _mm256_set1_ps(euler<float>());       }
    static inline __m256 sqrt2_packed()       { return _mm256_set1_ps(sqrt2<float>());       }
    static inline __m256 goldenratio_packed() { return _mm256_set1_ps(goldenratio<float>()); }

    static inline __m256 zeroes() { return _mm256_setzero_ps(); }
    static inline __m256 ones() {
        static const __m256 ones_ = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
        return ones_;
    }
    static inline __m256 halves() {
        static const __m256 halves_ = { 0.5f, 0.5, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f };
        return halves_;
    }

    static inline __m256 signs(__m256 v)
    { /*
      return
          _mm256_and_ps(
              _mm256_or_ps(
                  _mm256_and_ps(v, _mm256_set1_ps(-0.0f)),
                  _mm256_set1_ps(1.0f)
              ),
              _mm256_cmp_ps(v, zeroes(), _CMP_NEQ_OQ)
          );
      */
      return ((v & -0.0f) | 1.0f) & (v != zeroes());
    }

    static inline __m256 reciprocals(__m256 x) {
        // F(y) = 1/y - x = 0;
        // Newton-Raphson:
        // y[n+1] = y[n] - (F(y[n] / F'(y[n]) = y[n](2-y[n]*x)
        __m256 approx = _mm256_rcp_ps(x);
        return approx * ((ones() + ones()) - (approx * x));
    }
    
    static inline __m256 inv_sqrt_packed(__m256 x) {
        // F(y) = (1/y*y) - x = 0
        // Newton-Raphson:
        // y[n+1] = y[n] - (F(y[n] / F'(y[n]) = 0.5y[n]*(3-y[n]*y[n]*x)
        static const __m256 threes = {
            3.0f, 3.0f, 3.0f, 3.0f,
            3.0f, 3.0f, 3.0f, 3.0f
        };
        __m256 approx = _mm256_rsqrt_ps(x);
        return halves() * approx * (threes - (approx * approx * x));
    }
    static inline __m256 sqrt_packed(__m256 x)
    { return _mm256_mul_ps(x, inv_sqrt_packed(x)); }
    // probably slow
    static inline float inv_sqrt(float x)
    { return _mm_cvtss_f32(_mm256_castps256_ps128(inv_sqrt_packed(_mm256_set1_ps(x)))); }
    static inline float sqrt(float x)
    { // x * (1 / sqrt(x)) = sqrt(x)
      return x * inv_sqrt(x); }
    static inline float min_scalar(float a, float b)
    { return _mm_cvtss_f32(_mm_min_ss(_mm_set_ss(a), _mm_set_ss(b))); }
    static inline float min_packed(float a, float b)
    { return _mm_cvtss_f32(_mm_min_ps(_mm_load_ps1(&a), _mm_load_ps1(&b))); }
    static inline __m256 min_packed(__m256 a, __m256 b)
    { return _mm256_min_ps(a, b); }
    static inline float max_scalar(float a, float b)
    { return _mm_cvtss_f32(_mm_max_ss(_mm_set_ss(a), _mm_set_ss(b))); }
    static inline float max_packed(float a, float b)
    { return _mm_cvtss_f32(_mm_max_ps(_mm_load_ps1(&a), _mm_load_ps1(&b))); }
    static inline __m256 max_packed(__m256 a, __m256 b)
    { return _mm256_max_ps(a, b); }

    static inline __m256 dot_packed(__m256 a, __m256 b)
    { __m256 tmp = _mm256_dp_ps(a, b, 0xFF); // {dot1{4},dot0{4}}
      __m128 tmp2 =
          _mm256_extractf128_ps(
              _mm256_shuffle_ps(tmp, tmp, _MM_SHUFFLE(3, 1, 2, 0)), // {dot1{2},dot0{2},dot1{2},dot0{2}},
              0
          ); // {dot1{2},dot0{2}}
      tmp2 = _mm_shuffle_ps(tmp2, tmp2, _MM_SHUFFLE(3, 1, 2, 0)); // {dot1,dot0,dot1,dot0}
      tmp2 = _mm_hadd_ps(tmp2, tmp2); // {(dot1+dot0){4}}
      return
          _mm256_insertf128_ps(
              _mm256_insertf128_ps(
                  math_t::zeroes(),
                  tmp2,
                  0
              ), // {0,0,0,0,(dot1+dot0){4}}
              tmp2,
              1
          ); // {(dot1+dot0){4},dot1+dot0{4}};
    }

    static inline __m256 fast_sin_0(__m256 angles)
    { static const float coeffs[] = { 0.00761f, -0.16605f };
      return angles * (
          ones() + horner_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,__m256>()(angles*angles, coeffs)
      ); }
    static inline __m256 fast_sin_1(__m256 angles)
    { static const float coeffs[] = { -0.0000000239f, 0.0000027526f, -0.0001984090f, 0.0083333315f, -0.1666666664f };
      return angles * (
          ones() + horner_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,__m256>()(angles*angles, coeffs)
      ); }
    static inline __m256 fast_cos_0(__m256 angles)
    { static const float coeffs[] = { 0.03705f, -0.49670f };
      return
          ones() + horner_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,__m256>()(angles*angles, coeffs);
    }
    static inline __m256 fast_cos_1(__m256 angles)
    { static const float coeffs[] = { -0.0000002605f, 0.0000247609f, -0.0013888397f, 0.0416666418f, -0.4999999963f };
      return
          ones() + horner_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,__m256>()(angles*angles, coeffs);
    }
    static inline __m256 fast_tan_0(__m256 angles)
    { static const float coeffs[] = {0.20330f, 0.31755f};
      return angles * (
          ones() + horner_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,__m256>()(angles*angles, coeffs)
      ); }
    static inline __m256 fast_tan_1(__m256 angles)
    { static const float coeffs[] = {
        0.0095168091f, 0.0029005250f, 0.0245650893f, 0.0533740603f, 0.1333923995f, 0.3333314036f
      };
      return angles * (
          ones() + horner_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,__m256>()(angles*angles, coeffs)
      ); }

    static inline __m256 fast_arcsin_0(__m256 vals)
    { static const float coeffs[] = {-0.0187293f, 0.0742610f, -0.2121144f};
      __m256 v = ones() - vals;
      return half_pi_packed() -
          v * inv_sqrt_packed(v) * (
            horner_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,__m256>()(vals, coeffs)
            + _mm256_set1_ps(1.5707288f)
          ); }
    static inline __m256 fast_arcsin_1(__m256 vals)
    { static const float coeffs[] = {
          -0.0012624911f, 0.0066700901f, -0.01708812556f, 0.0308918810f,
          -0.0501743046f, 0.0889789874f, -0.21459880160f
      };
      __m256 v = ones() - vals;
      return half_pi_packed() -
          v * inv_sqrt_packed(v) * (
            horner_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,__m256>()(vals, coeffs)
            + _mm256_set1_ps(1.5707963050f)
          ); 
    }
    static inline __m256 fast_arctan_0(__m256 vals)
    { static const float coeffs[] = {
           0.0208351f, -0.0851330f, 0.1801410f,
          -0.3302995f,  0.9998660f
      };
      return horner_odd_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,__m256>()(vals, coeffs);
    }
    static inline __m256 fast_arctan_1(__m256 vals)
    { static const float coeffs[] = {
          0.0028662257f, -0.0161657367f, 0.0429096138f, -0.0752896400f,
          0.1065626393f, -0.1420889944f, 0.1999355085f, -0.3333314528f
      };
      return vals * (
          ones() + horner_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,__m256>()(vals, coeffs)
      );
    }

    // _not_ similar as math_t<float,__m128>::int32_to_packed()
    // this one returns a vector with all elements set to the
    // given value, while the other function returns zeroes in
    // the upper vector elements
    static inline __m256 int32_to_packed(int32_t i)
    { return _mm256_cvtepi32_ps(_mm256_set1_epi32(i)); }
    //static inline __m256 int64_to_packed(int64_t i)
    //{ return _mm_cvtsi64_ss(zeroes(), i); }
    //{ return _mm256_set1_epi64x(i); }
};


//
// math_t specialization for double{4}/__m256d
//
template<>
class math_t<double,__m256d>
{
public:
    typedef double real_t;
    typedef __m256d packed_t;

    static inline __m256d pi_packed()          { return _mm256_set1_pd(pi<double>());          }
    static inline __m256d two_pi_packed()      { return _mm256_set1_pd(two_pi<double>());      }
    static inline __m256d half_pi_packed()     { return _mm256_set1_pd(half_pi<double>());     }
    static inline __m256d inv_pi_packed()      { return _mm256_set1_pd(inv_pi<double>());      }
    static inline __m256d inv_two_pi_packed()  { return _mm256_set1_pd(inv_two_pi<double>());  }
    static inline __m256d deg2rad_packed()     { return _mm256_set1_pd(deg2rad<double>());     }
    static inline __m256d rad2deg_packed()     { return _mm256_set1_pd(rad2deg<double>());     }
    static inline __m256d euler_packed()       { return _mm256_set1_pd(euler<double>());       }
    static inline __m256d sqrt2_packed()       { return _mm256_set1_pd(sqrt2<double>());       }
    static inline __m256d goldenratio_packed() { return _mm256_set1_pd(goldenratio<double>()); }

    static inline __m256d zeroes() { return _mm256_setzero_pd(); }
    static inline __m256d ones() {
        static const __m256d ones_ = { 1.0, 1.0, 1.0, 1.0 };
        return ones_;
    }
    static inline __m256d halves() {
        static const __m256d halves_ = { 0.5, 0.5, 0.5, 0.5 };
        return halves_;
    }

    static inline __m256d signs(__m256d v)
    { return
          _mm256_and_pd(
              _mm256_or_pd(
                  _mm256_and_pd(v, _mm256_set1_pd(-0.0)),
                  _mm256_set1_pd(1.0f)
              ),
              _mm256_cmp_pd(v, zeroes(), _CMP_NEQ_OQ)
          );
    }

    // not recommended to be used for division (slower than
    // a direct division, and more accurate)
    static inline __m256d reciprocals(__m256d x) {
        return _mm256_div_pd(ones(), x);
    }
    static inline __m256d inv_sqrt_packed(__m256d x) {
        // there's no _mm256_rsqrt_pd()!
        // if to be replaced by 1/_mm256_sqrt_pd(): is there a N-R iteration
        // even needed here???
        // this assumes "not":
        return _mm256_div_pd(ones(), sqrt_packed(x));
    }
    static inline __m256d sqrt_packed(__m256d x)
    { return _mm256_mul_pd(x, inv_sqrt_packed(x)); }
    static inline double inv_sqrt(double x)
    { return _mm_cvtsd_f64(_mm256_castpd256_pd128(inv_sqrt_packed(_mm256_set1_pd(x)))); }
    static inline double sqrt(double x)
    { // x * (1 / sqrt(x)) = sqrt(x)
      return x * inv_sqrt(x); }
    static inline double min_scalar(double a, double b)
    { return _mm_cvtsd_f64(_mm_min_sd(_mm_set_sd(a), _mm_set_sd(b))); }
    // probably slow
    static inline double min_packed(double a, double b)
    { return min_scalar(a, b); }
    static inline __m256d min_packed(__m256d a, __m256d b)
    { return _mm256_min_pd(a, b); }
    static inline double max_scalar(double a, double b)
    { return _mm_cvtsd_f64(_mm_max_sd(_mm_set_sd(a), _mm_set_sd(b))); }
    static inline double max_packed(double a, double b)
    { return max_scalar(a, b); }
    static inline __m256d max_packed(__m256d a, __m256d b)
    { return _mm256_max_pd(a, b); }

    static inline __m256d dot_packed(__m256d a, __m256d b)
    { __m256d tmp = _mm256_mul_pd(a, b); // {w0w1,z0z1,y0y1,x0x1}
      tmp = _mm256_hadd_pd(tmp, tmp);    // {w0w1+z0z1,y0y1+x0x1,w0w1+z0z1,y0y1+x0x1}
      return _mm256_hadd_pd(tmp, tmp);   // {..,w0w1+z0z1+y0y1+x0x1}
    }

    static inline __m256d fast_sin_0(__m256d angles)
    { static const double coeffs[] = { 0.00761, -0.16605 };
      return angles * (
          ones() + horner_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),double,__m256d>()(angles*angles, coeffs)
      ); }
    static inline __m256d fast_sin_1(__m256d angles)
    { static const double coeffs[] = { -0.0000000239, 0.0000027526, -0.0001984090, 0.0083333315, -0.1666666664 };
      return angles * (
          ones() + horner_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),double,__m256d>()(angles*angles, coeffs)
      ); }
    static inline __m256d fast_cos_0(__m256d angles)
    { static const double coeffs[] = { 0.03705, -0.49670 };
      return
          ones() + horner_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),double,__m256d>()(angles*angles, coeffs);
    }
    static inline __m256d fast_cos_1(__m256d angles)
    { static const double coeffs[] = { -0.0000002605, 0.0000247609, -0.0013888397, 0.0416666418, -0.4999999963 };
      return
          ones() + horner_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),double,__m256d>()(angles*angles, coeffs);
    }
    static inline __m256d fast_tan_0(__m256d angles)
    { static const double coeffs[] = {0.20330, 0.31755};
      return angles * (
          ones() + horner_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),double,__m256d>()(angles*angles, coeffs)
      ); }
    static inline __m256d fast_tan_1(__m256d angles)
    { static const double coeffs[] = {
        0.0095168091, 0.0029005250, 0.0245650893, 0.0533740603, 0.1333923995, 0.3333314036
      };
      return angles * (
          ones() + horner_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),double,__m256d>()(angles*angles, coeffs)
      ); }

    static inline __m256d fast_arcsin_0(__m256d vals)
    { static const double coeffs[] = {-0.0187293, 0.0742610, -0.2121144};
      __m256d v = ones() - vals;
      return half_pi_packed() -
          v * inv_sqrt_packed(v) * (
            horner_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),double,__m256d>()(vals, coeffs)
            + _mm256_set1_pd(1.5707288)
          ); }
    static inline __m256d fast_arcsin_1(__m256d vals)
    { static const double coeffs[] = {
          -0.0012624911, 0.0066700901, -0.01708812556, 0.0308918810,
          -0.0501743046, 0.0889789874, -0.21459880160
      };
      __m256d v = ones() - vals;
      return half_pi_packed() -
          v * inv_sqrt_packed(v) * (
            horner_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),double,__m256d>()(vals, coeffs)
            + _mm256_set1_pd(1.5707963050)
          ); 
    }
    static inline __m256d fast_arctan_0(__m256d vals)
    { static const double coeffs[] = {
           0.0208351f, -0.0851330f, 0.1801410f,
          -0.3302995f,  0.9998660f
      };
      return horner_odd_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),double,__m256d>()(vals, coeffs);
    }
    static inline __m256d fast_arctan_1(__m256d vals)
    { static const double coeffs[] = {
          0.0028662257, -0.0161657367, 0.0429096138, -0.0752896400,
          0.1065626393, -0.1420889944, 0.1999355085, -0.3333314528
      };
      return vals * (
          ones() + horner_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),double,__m256d>()(vals, coeffs)
      );
    }

    // _not_ similar to math_t<float,__m128>::int32_to_packed()
    // this one returns a vector with all elements set to the
    // given value, while the other function returns zeroes in
    // the upper vector elements
    static inline __m256d int32_to_packed(int32_t i)
    { return _mm256_cvtepi32_pd(_mm_set1_epi32(i)); }
    static inline __m256d int64_to_packed(int64_t i)
    { return _mm256_cvtepi32_pd(_mm_cvtsi64_si128(i)); }
};
#endif // AVX



//#ifdef _M_X64
//inline __m128 dot_packed_4(__m128 v00, __m128 v01,
//                           __m128 v10, __m128 v11,
//                           __m128 v20, __m128 v21,
//                           __m128 v30, __m128 v31)
inline __m128 dot_packed_4(const __m128 & v00, const __m128 & v01,
                           const __m128 & v10, const __m128 & v11,
                           const __m128 & v20, const __m128 & v21,
                           const __m128 & v30, const __m128 & v31)
{ __m128 tmp0 = math_t<float,__m128>::dot_packed(v00, v01); // {..,dot0}
  __m128 tmp1 = math_t<float,__m128>::dot_packed(v10, v11); // {..,dot1}
  __m128 tmp2 = math_t<float,__m128>::dot_packed(v20, v21); // {..,dot2}
  __m128 tmp3 = math_t<float,__m128>::dot_packed(v30, v31); // {..,dot3}
  return
      xzxz(
          xxxx(tmp0, tmp1), // {dot0,dot0,dot1,dot1}
          xxxx(tmp2, tmp3)  // {dot2,dot2,dot3,dot3}
      ); // {dot0,dot1,dot2,dot3}
}
//#endif



#endif // !defined(PMATH_X86_SPEC_H)


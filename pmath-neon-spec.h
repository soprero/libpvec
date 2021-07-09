/*******************************************************************************
 * pmath-neon-spec.h                                                           *
 *                                                                             *
 * Copyright (c) 2015-2017 Ronny Press                                         *
 *                                                                             *
 * rpress@soprero.de                                                           *
 *                                                                             *
 * All rights reserved.                                                        *
 *******************************************************************************/

#ifndef PMATH_NEON_SPEC_H
#define PMATH_NEON_SPEC_H

#ifndef PMATH_H
#error Do not include this file directly, include pmath.h instead
#endif


#if defined(__GNUC__)
// this is not the right way: __n128 includes all element types
//typedef float32x4_t __n128;
/*
typedef union __attribute__ ((aligned(8))) __n128
{
    float32x4_t f32x4;
    int8x16_t   i8x16;
    int16x8_t   i16x8;
    int32x4_t   i32x4;
    int64x2_t   i64x2;
    uint8x16_t  u8x16;
    uint16x8_t  u16x8;
    uint32x4_t  u32x4;
    uint64x2_t  u64x2;

    __n128() { f32x4 = vdupq_n_f32(0.0f); }
    __n128(float32x4_t v) : f32x4(v) {}
    
    __n128(int8x16_t v)  : i8x16(v) {}
    __n128(int16x8_t v)  : i16x8(v) {}
    __n128(int32x4_t v)  : i32x4(v) {}
    __n128(int64x2_t v)  : i64x2(v) {}

    __n128(uint8x16_t v) : u8x16(v) {}
    __n128(uint16x8_t v) : u16x8(v) {}
    __n128(uint32x4_t v) : u32x4(v) {}
    __n128(uint64x2_t v) : u64x2(v) {}

    operator float32x4_t() { return f32x4; }
} __n128;
*/
#endif


//
// math_t specialization for float{4}/__n128
//
#ifdef _MSC_VER
#define vec __n128
#else
#define vec float32x4_t
#endif

template<>
class math_t<float,vec>
{
public:
    typedef float real_t;
    typedef vec packed_t;

    static inline vec set1(float scalar) { return vdupq_n_f32(scalar); }

    static inline vec pi_packed()          { return vdupq_n_f32(pi<float>());          }
    static inline vec two_pi_packed()      { return vdupq_n_f32(two_pi<float>());      }
    static inline vec half_pi_packed()     { return vdupq_n_f32(half_pi<float>());     }
    static inline vec inv_pi_packed()      { return vdupq_n_f32(inv_pi<float>());      }
    static inline vec inv_two_pi_packed()  { return vdupq_n_f32(inv_two_pi<float>());  }
    static inline vec deg2rad_packed()     { return vdupq_n_f32(deg2rad<float>());     }
    static inline vec rad2deg_packed()     { return vdupq_n_f32(rad2deg<float>());     }
    static inline vec euler_packed()       { return vdupq_n_f32(euler<float>());       }
    static inline vec sqrt2_packed()       { return vdupq_n_f32(sqrt2<float>());       }
    static inline vec goldenratio_packed() { return vdupq_n_f32(goldenratio<float>()); }

    static inline vec zeroes() { return vdupq_n_f32(0.0f); /* I'd use VEOR here, but need an operand */ }
    static inline vec ones() { return vdupq_n_f32(1.0f); }
    static inline vec halves() { return vdupq_n_f32(0.5f); }

    static inline vec signs(vec v)
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
        //return _mm_and_ps(_mm_or_ps(_mm_and_ps(v, _mm_set1_ps(-0.0f)), ones()), _mm_cmpneq_ps(v, zeroes()));
        return
            vreinterpretq_f32_u32(
                vandq_u32(
                    vorrq_u32(
                        vandq_u32(
                            vreinterpretq_u32_f32(v),
                            vreinterpretq_u32_f32(vdupq_n_f32(-0.0f))
                        ),
                        vreinterpretq_u32_f32(ones())
                    ),
                    vmvnq_u32(vceqq_f32(v, zeroes()))
                )
            );
        #endif
    }

    static inline vec reciprocals(vec x)
    {
        // F(y) = 1/y - x = 0
        // Newton-Raphson:
        // y[n+1] = y[n] - F(y[n]) / F'(y[n]) = y[n](2-y[n]*x)
        vec approx = vrecpeq_f32(x);
        // NOTE: two NR iterations per hardware instruction
        approx = vmulq_f32(vrecpsq_f32(x, approx), approx);
        return   vmulq_f32(vrecpsq_f32(x, approx), approx);
    }

    static inline vec inv_sqrt_packed(vec x)
    {
        // F(y) = (1/(y*y)) - x = 0
        // Newton-Raphson:
        // y[n+1] = y[n] - F(y[n]) / F'(y[n]) = 0.5y[n]*(3-y[n]*y[n]*x)
        vec approx = vrsqrteq_f32(x);
        // NOTE: two NR iterations per hardware instruction
        approx = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, approx), approx), approx);
        //return   vmulq_f32(vrsqrtsq_f32(x, approx), approx);
        return vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, approx), approx), approx);
    }

    static inline vec sqrt_packed(vec x)
    { return vmulq_f32(x, inv_sqrt_packed(x)); }
    static inline float inv_sqrt(float x)
    { return vgetq_lane_f32(inv_sqrt_packed(vdupq_n_f32(x)), 0); }
    static inline float sqrt(float x)
    { // x * (1 / sqrt(x)) = sqrt(x)
      return x * inv_sqrt(x); }
    static inline float min_scalar(float a, float b)
    { return vget_lane_f32(vmin_f32(vdup_n_f32(a), vdup_n_f32(b)), 0); }
    // no difference (NEON doesn't have scalar support as SSE)
    static inline float min_packed(float a, float b)
    { return vget_lane_f32(vmin_f32(vdup_n_f32(a), vdup_n_f32(b)), 0); }
    static inline vec min_packed(vec a, vec b)
    { return vminq_f32(a, b); }
    static inline float max_scalar(float a, float b)
    { return vget_lane_f32(vmax_f32(vdup_n_f32(a), vdup_n_f32(b)), 0); }
    // no difference (NEON doesn't have scalar support as SSE)
    static inline float max_packed(float a, float b)
    { return vget_lane_f32(vmax_f32(vdup_n_f32(a), vdup_n_f32(b)), 0); }
    static inline vec max_packed(vec a, vec b)
    { return vmaxq_f32(a, b); }

    static inline vec dot_packed(vec a, vec b)
    { // [a]     [e]
      // [b]     [f]
      // [c] dot [g] = ae+bf+cg+dh
      // [d]     [h]
      vec tmp = a * b; // {ae,bf,cg,dh}
      tmp = tmp + yxwz(tmp); // {ae+bf,bf+ae,cg+dh,dh+cg}
      return tmp + zwyx(tmp); // {ae+bf+cg+dh.bf+ae+dh+cg,cg+dh+bf+ae,dh+cg+ae+bf}
    }

    static inline vec fast_sin_0(vec angles)
    { static const float coeffs[] = { -0.16605f, 0.00761f };
      return angles * (
          ones() + polynomialeval_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,vec>()(angles, coeffs)
      ); }
    static inline vec fast_sin_1(vec angles)
    { static const float coeffs[] = { -0.1666666664f, 0.0083333315f, -0.0001984090f, 0.0000027526f, -0.0000000239f };
      return angles * (
          ones() + polynomialeval_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,vec>()(angles, coeffs)
      ); }
    static inline vec fast_cos_0(vec angles)
    { static const float coeffs[] = { -0.49670f, 0.03705f };
      return
          ones() + polynomialeval_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,vec>()(angles, coeffs);
    }
    static inline vec fast_cos_1(vec angles)
    { //static const float coeffs[] = { -0.0000002605f, 0.0000247609f, -0.0013888397f, 0.0416666418f, -0.4999999963f };
      static const float coeffs[] = { -0.4999999963f, 0.0416666418f, -0.0013888397f, 0.0000247609f, -0.0000002605f };
      return
          ones() + polynomialeval_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,vec>()(angles, coeffs);
    }
    static inline vec fast_tan_0(vec angles)
    { static const float coeffs[] = {0.20330f, 0.31755f};
      return angles * (
          ones() + horner_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,vec>()(angles*angles, coeffs)
      ); }
    static inline vec fast_tan_1(vec angles)
    { static const float coeffs[] = {
        0.0095168091f, 0.0029005250f, 0.0245650893f, 0.0533740603f, 0.1333923995f, 0.3333314036f
      };
      return angles * (
          ones() + horner_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,vec>()(angles*angles, coeffs)
      ); }

    static inline vec fast_arcsin_0(vec vals)
    { // arcsin0(x) = pi/2 - sqrt(1-x)(1.5707288 - 0.2121144x + 0.0742610x*x - 0.0187293x*x*x)
      static const float coeffs[] = {-0.0187293f, 0.0742610f, -0.2121144f};
      vec v = ones() - vals;
      return half_pi_packed() -
          v * inv_sqrt_packed(v) * (
            horner_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,vec>()(vals, coeffs)
            + vdupq_n_f32(1.5707288f)
          ); }
    static inline vec fast_arcsin_1(vec vals)
    { static const float coeffs[] = {
          -0.0012624911f, 0.0066700901f, -0.01708812556f, 0.0308918810f,
          -0.0501743046f, 0.0889789874f, -0.21459880160f
      };
      vec v = ones() - vals;
      return half_pi_packed() -
          v * inv_sqrt_packed(v) * (
            horner_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,vec>()(vals, coeffs)
            + vdupq_n_f32(1.5707963050f)
          );
    }
    static inline vec fast_arctan_0(vec vals)
    { static const float coeffs[] = {
           0.0208351f, -0.0851330f, 0.1801410f,
          -0.3302995f,  0.9998660f
      };
      return horner_odd_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,vec>()(vals, coeffs);
    }
    static inline vec fast_arctan_1(vec vals)
    { static const float coeffs[] = {
          0.0028662257f, -0.0161657367f, 0.0429096138f, -0.0752896400f,
          0.1065626393f, -0.1420889944f, 0.1999355085f, -0.3333314528f
      };
      return vals * (
          ones() + horner_even_packed_t<sizeof(coeffs)/sizeof(coeffs[0]),float,vec>()(vals, coeffs)
      );
    }

#if 0
    static inline __n128 int32_to_packed(int32_t i)
    { return _mm_cvt_si2ss(zeroes(), i); }
#ifdef _M_X64
    static inline __m128 int64_to_packed(int64_t i)
    { return _mm_cvtsi64_ss(zeroes(), i); }
#endif
#endif
};


// packed double precision floating point calculations are not currently supported
// by NEON (at least not in the older ARM arch versions)


//
// no 256 bit wide register support in NEON as I see it currently
//


inline vec dot_packed_4(const vec & v00, const vec & v01,
                        const vec & v10, const vec & v11,
                        const vec & v20, const vec & v21,
                        const vec & v30, const vec & v31)
{ vec tmp0 = math_t<float,vec>::dot_packed(v00, v01); // {..,dot0}
  vec tmp1 = math_t<float,vec>::dot_packed(v10, v11); // {..,dot1}
  vec tmp2 = math_t<float,vec>::dot_packed(v20, v21); // {..,dot2}
  vec tmp3 = math_t<float,vec>::dot_packed(v30, v31); // {..,dot3}
  return
      xzxz(
          xxxx(tmp0, tmp1), // {dot0,dot0,dot1,dot1}
          xxxx(tmp2, tmp3)  // {dot2,dot2,dot3,dot3}
      ); // {dot0,dot1,dot2,dot3}
}

#undef vec

#endif // !defined(PMATH_NEON_SPEC_H)


/*******************************************************************************
 * pmath-neon.h                                                                *
 *                                                                             *
 * Copyright (c) 2015-2017 Ronny Press                                              *
 *                                                                             *
 * rpress@soprero.de                                                           *
 *                                                                             *
 * All rights reserved.                                                        *
 *******************************************************************************/

#ifndef PMATH_NEON_H
#define PMATH_NEON_H

#ifndef PMATH_H
#error Do not include this file directly, include pmath.h instead
#endif




//
// a few macros
//
#if 0
#define OP_(op,scalar,packed,prefix,intrinop,postfix) \
static inline packed operator op(packed op1, packed op2) { return prefix##intrinop##postfix(op1, op2); } \
static inline packed operator op(scalar op1, packed op2) { return prefix##intrinop##postfix(prefix##set1##postfix(op1), op2); } \
static inline packed operator op(packed op1, scalar op2) { return prefix##intrinop##postfix(op1, prefix##set1##postfix(op2)); }
#ifdef _MSC_VER
#define CMP_OPS_(scalar,packed,prefix,postfix) \
    OP_(<,scalar,packed,prefix,cmplt,postfix) \
    OP_(>,scalar,packed,prefix,cmpgt,postfix) \
    OP_(==,scalar,packed,prefix,cmpeq,postfix) \
    OP_(<=,scalar,packed,prefix,cmple,postfix) \
    OP_(>=,scalar,packed,prefix,cmpge,postfix) \
    OP_(!=,scalar,packed,prefix,cmpneq,postfix)
#else
#define ARITH_OPS_(scalar,packed,prefix,postfix) \
    OP_(&,scalar,packed,prefix,and,postfix) \
    OP_(|,scalar,packed,prefix,or,postfix) \
    OP_(^,scalar,packed,prefix,xor,postfix)
#define CMP_OPS_(scalar,packed,prefix,postfix)
#endif
#endif


//
// packed float4 arithmetic and bitwise logical
//

#ifdef _MSC_VER
// for the free-standing operators: I remember GCC beginning at a certain version
// having these built-in, so these could be needed for MSVC only (don't know about
// clang yet)
inline float32x4_t operator +(float32x4_t op1, float32x4_t op2) { return vaddq_f32(op1, op2); }
inline float32x4_t operator +(float op1, float32x4_t op2) { return vaddq_f32(vdupq_n_f32(op1), op2); }
inline float32x4_t operator +(float32x4_t op1, float op2) { return vaddq_f32(op1, vdupq_n_f32(op2)); }

inline float32x4_t operator -(float32x4_t op1, float32x4_t op2) { return vsubq_f32(op1, op2); }
inline float32x4_t operator -(float op1, float32x4_t op2) { return vsubq_f32(vdupq_n_f32(op1), op2); }
inline float32x4_t operator -(float32x4_t op1, float op2) { return vsubq_f32(op1, vdupq_n_f32(op2)); }

inline float32x4_t operator *(float32x4_t op1, float32x4_t op2) { return vmulq_f32(op1, op2); }
inline float32x4_t operator *(float op1, float32x4_t op2) { return vmulq_n_f32(op2, op1); }
inline float32x4_t operator *(float32x4_t op1, float op2) { return vmulq_n_f32(op1, op2); }
// no divs in NEON
namespace priv {
    inline __n128 reciprocals(__n128 x)
    {
        // F(y) = 1/y - x = 0
        // Newton-Raphson:
        // y[n+1] = y[n] - F(y[n]) / F'(y[n]) = y[n](2-y[n]*x)
        __n128 approx = vrecpeq_f32(x);
        // docs are not clear for VRECPS / vrecpsq_f32(): the final multiplication
        // is not done according to the text but it is part of the explaining
        // formulas, so have to clear this up first; meanwhile the explicit
        // refinement step is done here
        return approx * (vdupq_n_f32(2.0f) - (approx * x));
    }
}
inline float32x4_t operator /(float32x4_t op1, float32x4_t op2) { return vmulq_f32(op1, priv::reciprocals(op2)); }
inline float32x4_t operator /(float op1, float32x4_t op2) { return vmulq_f32(vdupq_n_f32(op1), priv::reciprocals(op2)); }
inline float32x4_t operator /(float32x4_t op1, float op2) { return vmulq_f32(op1, priv::reciprocals(vdupq_n_f32(op2))); }



inline float32x4_t operator &(float32x4_t op1, float32x4_t op2) { return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(op1), vreinterpretq_u32_f32(op2))); }
inline float32x4_t operator &(float op1, float32x4_t op2) { return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(vdupq_n_f32(op1)), vreinterpretq_u32_f32(op2))); }
inline float32x4_t operator &(float32x4_t op1, float op2) { return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(op1), vreinterpretq_u32_f32(vdupq_n_f32(op2)))); }

inline float32x4_t operator |(float32x4_t op1, float32x4_t op2) { return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(op1), vreinterpretq_u32_f32(op2))); }
inline float32x4_t operator |(float op1, float32x4_t op2) { return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(vdupq_n_f32(op1)), vreinterpretq_u32_f32(op2))); }
inline float32x4_t operator |(float32x4_t op1, float op2) { return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(op1), vreinterpretq_u32_f32(vdupq_n_f32(op2)))); }

inline float32x4_t operator ^(float32x4_t op1, float32x4_t op2) { return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(op1), vreinterpretq_u32_f32(op2))); }
inline float32x4_t operator ^(float op1, float32x4_t op2) { return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(vdupq_n_f32(op1)), vreinterpretq_u32_f32(op2))); }
inline float32x4_t operator ^(float32x4_t op1, float op2) { return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(op1), vreinterpretq_u32_f32(vdupq_n_f32(op2)))); }

inline float32x4_t operator-(float32x4_t op) { return vnegq_f32(op); }
#endif // defined(_MSC_VER)

inline float32x4_t notAandB_(float32x4_t a, float32x4_t b) {
    return vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(b), vreinterpretq_u32_f32(a)));
}


//
// packed float4 comparisons
//

#ifdef _MSC_VER
// the intrinsics used here return uint32x4_t because all bits of each element are either set or cleared (element values: 0 or -1 when interpreted signed)

// GCC does not allow to overload these, either (TODO: need to find out the types returned by the
// GCC builtin operators)

inline uint32x4_t operator ==(float32x4_t op1, float32x4_t op2) { return vceqq_f32(op1, op2); }
inline uint32x4_t operator ==(float op1, float32x4_t op2) { return vceqq_f32(vdupq_n_f32(op1), op2); }
inline uint32x4_t operator ==(float32x4_t op1, float op2) { return vceqq_f32(op1, vdupq_n_f32(op2)); }

inline uint32x4_t operator >=(float32x4_t op1, float32x4_t op2) { return vcgeq_f32(op1, op2); }
inline uint32x4_t operator >=(float op1, float32x4_t op2) { return vcgeq_f32(vdupq_n_f32(op1), op2); }
inline uint32x4_t operator >=(float32x4_t op1, float op2) { return vcgeq_f32(op1, vdupq_n_f32(op2)); }

inline uint32x4_t operator <=(float32x4_t op1, float32x4_t op2) { return vcleq_f32(op1, op2); }
inline uint32x4_t operator <=(float op1, float32x4_t op2) { return vcleq_f32(vdupq_n_f32(op1), op2); }
inline uint32x4_t operator <=(float32x4_t op1, float op2) { return vcleq_f32(op1, vdupq_n_f32(op2)); }

inline uint32x4_t operator >(float32x4_t op1, float32x4_t op2) { return vcgtq_f32(op1, op2); }
inline uint32x4_t operator >(float op1, float32x4_t op2) { return vcgtq_f32(vdupq_n_f32(op1), op2); }
inline uint32x4_t operator >(float32x4_t op1, float op2) { return vcgtq_f32(op1, vdupq_n_f32(op2)); }

inline uint32x4_t operator <(float32x4_t op1, float32x4_t op2) { return vcltq_f32(op1, op2); }
inline uint32x4_t operator <(float op1, float32x4_t op2) { return vcltq_f32(vdupq_n_f32(op1), op2); }
inline uint32x4_t operator <(float32x4_t op1, float op2) { return vcltq_f32(op1, vdupq_n_f32(op2)); }

// this uses vmvnq_u32() to NOT all bits of the results of the comparison operation
inline uint32x4_t operator !=(float32x4_t op1, float32x4_t op2) { return vmvnq_u32(vceqq_f32(op1, op2)); }
inline uint32x4_t operator !=(float op1, float32x4_t op2) { return vmvnq_u32(vceqq_f32(vdupq_n_f32(op1), op2)); }
inline uint32x4_t operator !=(float32x4_t op1, float op2) { return vmvnq_u32(vceqq_f32(op1, vdupq_n_f32(op2))); }

#endif


// packed double precision floating point calculations are not currently supported
// by NEON (at least not in the older ARM arch versions)


//
// no 256 bit wide register support in NEON as I see it currently
//


namespace math {

namespace priv {
    // get_lane<>() for f32
    template<unsigned t_lane> float32x2_t get_lane_f32x4(float32x4_t v);
    //{ static_assert(false, "template instantiation not meant to be used"); };
    template<> inline float32x2_t get_lane_f32x4<0>(float32x4_t v) { return vget_low_f32(v); }
    template<> inline float32x2_t get_lane_f32x4<1>(float32x4_t v) { return vget_high_f32(v); }

    // get_lane<>() for s32
    template<unsigned t_lane> int32x2_t get_lane_s32x4(int32x4_t v);
    //{ static_assert(false, "template instantiation not meant to be used"); };
    template<> inline int32x2_t get_lane_s32x4<0>(int32x4_t v) { return vget_low_s32(v); }
    template<> inline int32x2_t get_lane_s32x4<1>(int32x4_t v) { return vget_high_s32(v); }

    // get_lane<>() for u32
    template<unsigned t_lane> uint32x2_t get_lane_u32x4(uint32x4_t v);
    //{ static_assert(false, "template instantiation not meant to be used"); };
    template<> inline uint32x2_t get_lane_u32x4<0>(uint32x4_t v) { return vget_low_u32(v); }
    template<> inline uint32x2_t get_lane_u32x4<1>(uint32x4_t v) { return vget_high_u32(v); }


    // shuffle<>() for f32
    template<unsigned t_lo, unsigned t_hi> float32x2_t shuffle_f32x2(float32x2_t v);
    //{ static_assert(false, "template instantiation not meant to be used"); };
    template<> inline float32x2_t shuffle_f32x2<0,0>(float32x2_t v) { return vdup_lane_f32(v, 0); }
    template<> inline float32x2_t shuffle_f32x2<0,1>(float32x2_t v) { return v; }
    template<> inline float32x2_t shuffle_f32x2<1,0>(float32x2_t v) { return vrev64_f32(v); }
    template<> inline float32x2_t shuffle_f32x2<1,1>(float32x2_t v) { return vdup_lane_f32(v, 1); }
    
    // shuffle<>() for s32
    template<unsigned t_lo, unsigned t_hi> int32x2_t shuffle_s32x2(int32x2_t v);
    //{ static_assert(false, "template instantiation not meant to be used"); };
    template<> inline int32x2_t shuffle_s32x2<0,0>(int32x2_t v) { return vdup_lane_s32(v, 0); }
    template<> inline int32x2_t shuffle_s32x2<0,1>(int32x2_t v) { return v; }
    template<> inline int32x2_t shuffle_s32x2<1,0>(int32x2_t v) { return vrev64_s32(v); }
    template<> inline int32x2_t shuffle_s32x2<1,1>(int32x2_t v) { return vdup_lane_s32(v, 1); }

    // shuffle<>() for s32
    template<unsigned t_lo, unsigned t_hi> uint32x2_t shuffle_u32x2(uint32x2_t v);
    //{ static_assert(false, "template instantiation not meant to be used"); };
    template<> inline uint32x2_t shuffle_u32x2<0, 0>(uint32x2_t v) { return vdup_lane_u32(v, 0); }
    template<> inline uint32x2_t shuffle_u32x2<0, 1>(uint32x2_t v) { return v; }
    template<> inline uint32x2_t shuffle_u32x2<1, 0>(uint32x2_t v) { return vrev64_u32(v); }
    template<> inline uint32x2_t shuffle_u32x2<1, 1>(uint32x2_t v) { return vdup_lane_u32(v, 1); }


    inline float32x2_t get_lane_f32x4_dyn(float32x4_t v, int lane)
    { return lane == 0 ? vget_low_f32(v) : vget_high_f32(v); }
    inline float32x2_t shuffle_f32x2_dyn(float32x2_t v, int lo, int hi)
    {
        float32x2_t ret;
        switch((lo << 1) | hi) {
        case 0:
            ret = vdup_lane_f32(v, 0);
            break;
        case 1:
            ret = v;
            break;
        case 2:
            ret = vrev64_f32(v);
            break;
        case 3:
            ret = vdup_lane_f32(v, 1);
            break;
        }
        return ret;
    }


    template<unsigned t_lo, unsigned t_hi> inline float32x4_t combine_f32x2x2(float32x2x2_t v)
    { return vcombine_f32(v.val[t_lo], v.val[t_hi]); }
    
    template<unsigned t_idx> inline float32x4_t combine_f32x2_f32x2x2(float32x2_t v1, float32x2x2_t v2)
    { return vcombine_f32(v1, v2.val[t_idx]); }

    template<unsigned t_idx> inline float32x4_t combine_f32x2x2_f32x2(float32x2x2_t v1, float32x2_t v2)
    { return vcombine_f32(v1.val[t_idx], v2); }


}


//
// shuffles for f32x4
//
#define ZIP____(a,b,c,d,lane1,l1,lane2,l2,lane3,l3,lane4,l4)    \
template<typename T> T a##b##c##d(const T &);                   \
template<> inline float32x4_t a##b##c##d(const float32x4_t & v) \
{                                                               \
    return                                                      \
        vcombine_f32(                                           \
            vzip_f32(                                           \
                priv::shuffle_f32x2<l1,l1>(                     \
                    priv::get_lane_f32x4<lane1>(v)              \
                ),                                              \
                priv::shuffle_f32x2<l2,l2>(                     \
                    priv::get_lane_f32x4<lane2>(v)              \
                )                                               \
            ).val[0],                                           \
            vzip_f32(                                           \
                priv::shuffle_f32x2<l3,l3>(                     \
                    priv::get_lane_f32x4<lane3>(v)              \
                ),                                              \
                priv::shuffle_f32x2<l4,l4>(                     \
                    priv::get_lane_f32x4<lane4>(v)              \
                )                                               \
            ).val[0]                                            \
        );                                                      \
}

#define ZIP___(a,b,c,lane1,l1,lane2,l2,lane3,l3) \
    ZIP____(a,b,c,x,lane1,l1,lane2,l2,lane3,l3,0,0) \
    ZIP____(a,b,c,y,lane1,l1,lane2,l2,lane3,l3,0,1) \
    ZIP____(a,b,c,z,lane1,l1,lane2,l2,lane3,l3,1,0) \
    ZIP____(a,b,c,w,lane1,l1,lane2,l2,lane3,l3,1,1)
#define ZIP__(a,b,lane1,l1,lane2,l2) \
    ZIP___(a,b,x,lane1,l1,lane2,l2,0,0) \
    ZIP___(a,b,y,lane1,l1,lane2,l2,0,1) \
    ZIP___(a,b,z,lane1,l1,lane2,l2,1,0) \
    ZIP___(a,b,w,lane1,l1,lane2,l2,1,1)
#define ZIP_(a,lane1,l1) \
    ZIP__(a,x,lane1,l1,0,0) \
    ZIP__(a,y,lane1,l1,0,1) \
    ZIP__(a,z,lane1,l1,1,0) \
    ZIP__(a,w,lane1,l1,1,1)
#define ZIP \
    ZIP_(x,0,0) \
    ZIP_(y,0,1) \
    ZIP_(z,1,0) \
    ZIP_(w,1,1)

ZIP

#undef ZIP____
#undef ZIP___
#undef ZIP__
#undef ZIP_
#undef ZIP


#define ZIP2____(a,b,c,d,lane1,l1,lane2,l2,lane3,l3,lane4,l4)   \
template<typename T> T a##b##c##d(const T &, const T &);        \
template<> inline float32x4_t a##b##c##d(                       \
        const float32x4_t & v1, const float32x4_t & v2          \
)                                                               \
{                                                               \
    return                                                      \
        vcombine_f32(                                           \
            vzip_f32(                                           \
                priv::shuffle_f32x2<l1,l1>(                     \
                    priv::get_lane_f32x4<lane1>(v1)             \
                ),                                              \
                priv::shuffle_f32x2<l2,l2>(                     \
                    priv::get_lane_f32x4<lane2>(v1)             \
                )                                               \
            ).val[0],                                           \
            vzip_f32(                                           \
                priv::shuffle_f32x2<l3,l3>(                     \
                    priv::get_lane_f32x4<lane3>(v2)             \
                ),                                              \
                priv::shuffle_f32x2<l4,l4>(                     \
                    priv::get_lane_f32x4<lane4>(v2)             \
                )                                               \
            ).val[0]                                            \
        );                                                      \
}

#define ZIP2___(a,b,c,lane1,l1,lane2,l2,lane3,l3) \
    ZIP2____(a,b,c,x,lane1,l1,lane2,l2,lane3,l3,0,0) \
    ZIP2____(a,b,c,y,lane1,l1,lane2,l2,lane3,l3,0,1) \
    ZIP2____(a,b,c,z,lane1,l1,lane2,l2,lane3,l3,1,0) \
    ZIP2____(a,b,c,w,lane1,l1,lane2,l2,lane3,l3,1,1)
#define ZIP2__(a,b,lane1,l1,lane2,l2) \
    ZIP2___(a,b,x,lane1,l1,lane2,l2,0,0) \
    ZIP2___(a,b,y,lane1,l1,lane2,l2,0,1) \
    ZIP2___(a,b,z,lane1,l1,lane2,l2,1,0) \
    ZIP2___(a,b,w,lane1,l1,lane2,l2,1,1)
#define ZIP2_(a,lane1,l1) \
    ZIP2__(a,x,lane1,l1,0,0) \
    ZIP2__(a,y,lane1,l1,0,1) \
    ZIP2__(a,z,lane1,l1,1,0) \
    ZIP2__(a,w,lane1,l1,1,1)
#define ZIP2 \
    ZIP2_(x,0,0) \
    ZIP2_(y,0,1) \
    ZIP2_(z,1,0) \
    ZIP2_(w,1,1)

ZIP2

#undef ZIP2____
#undef ZIP2___
#undef ZIP2__
#undef ZIP2_
#undef ZIP2


//
// shuffles for int32x4
//
#define ZIP____(a,b,c,d,lane1,l1,lane2,l2,lane3,l3,lane4,l4)    \
template<typename T> T a##b##c##d(const T &);                   \
template<> inline int32x4_t a##b##c##d(const int32x4_t & v)     \
{                                                               \
    return                                                      \
        vcombine_s32(                                           \
            vzip_s32(                                           \
                priv::shuffle_s32x2<l1,l1>(                     \
                    priv::get_lane_s32x4<lane1>(v)              \
                ),                                              \
                priv::shuffle_s32x2<l2,l2>(                     \
                    priv::get_lane_s32x4<lane2>(v)              \
                )                                               \
            ).val[0],                                           \
            vzip_s32(                                           \
                priv::shuffle_s32x2<l3,l3>(                     \
                    priv::get_lane_s32x4<lane3>(v)              \
                ),                                              \
                priv::shuffle_s32x2<l4,l4>(                     \
                    priv::get_lane_s32x4<lane4>(v)              \
                )                                               \
            ).val[0]                                            \
        );                                                      \
}

#define ZIP___(a,b,c,lane1,l1,lane2,l2,lane3,l3) \
    ZIP____(a,b,c,x,lane1,l1,lane2,l2,lane3,l3,0,0) \
    ZIP____(a,b,c,y,lane1,l1,lane2,l2,lane3,l3,0,1) \
    ZIP____(a,b,c,z,lane1,l1,lane2,l2,lane3,l3,1,0) \
    ZIP____(a,b,c,w,lane1,l1,lane2,l2,lane3,l3,1,1)
#define ZIP__(a,b,lane1,l1,lane2,l2) \
    ZIP___(a,b,x,lane1,l1,lane2,l2,0,0) \
    ZIP___(a,b,y,lane1,l1,lane2,l2,0,1) \
    ZIP___(a,b,z,lane1,l1,lane2,l2,1,0) \
    ZIP___(a,b,w,lane1,l1,lane2,l2,1,1)
#define ZIP_(a,lane1,l1) \
    ZIP__(a,x,lane1,l1,0,0) \
    ZIP__(a,y,lane1,l1,0,1) \
    ZIP__(a,z,lane1,l1,1,0) \
    ZIP__(a,w,lane1,l1,1,1)
#define ZIP \
    ZIP_(x,0,0) \
    ZIP_(y,0,1) \
    ZIP_(z,1,0) \
    ZIP_(w,1,1)

ZIP

#undef ZIP____
#undef ZIP___
#undef ZIP__
#undef ZIP_
#undef ZIP


//
// shuffles for uint32x4
//
#define ZIP____(a,b,c,d,lane1,l1,lane2,l2,lane3,l3,lane4,l4)    \
template<typename T> T a##b##c##d(const T &);                   \
template<> inline uint32x4_t a##b##c##d(const uint32x4_t & v)   \
{                                                               \
    return                                                      \
        vcombine_u32(                                           \
            vzip_u32(                                           \
                priv::shuffle_u32x2<l1,l1>(                     \
                    priv::get_lane_u32x4<lane1>(v)              \
                ),                                              \
                priv::shuffle_u32x2<l2,l2>(                     \
                    priv::get_lane_u32x4<lane2>(v)              \
                )                                               \
            ).val[0],                                           \
            vzip_u32(                                           \
                priv::shuffle_u32x2<l3,l3>(                     \
                    priv::get_lane_u32x4<lane3>(v)              \
                ),                                              \
                priv::shuffle_u32x2<l4,l4>(                     \
                    priv::get_lane_u32x4<lane4>(v)              \
                )                                               \
            ).val[0]                                            \
        );                                                      \
}

#define ZIP___(a,b,c,lane1,l1,lane2,l2,lane3,l3) \
    ZIP____(a,b,c,x,lane1,l1,lane2,l2,lane3,l3,0,0) \
    ZIP____(a,b,c,y,lane1,l1,lane2,l2,lane3,l3,0,1) \
    ZIP____(a,b,c,z,lane1,l1,lane2,l2,lane3,l3,1,0) \
    ZIP____(a,b,c,w,lane1,l1,lane2,l2,lane3,l3,1,1)
#define ZIP__(a,b,lane1,l1,lane2,l2) \
    ZIP___(a,b,x,lane1,l1,lane2,l2,0,0) \
    ZIP___(a,b,y,lane1,l1,lane2,l2,0,1) \
    ZIP___(a,b,z,lane1,l1,lane2,l2,1,0) \
    ZIP___(a,b,w,lane1,l1,lane2,l2,1,1)
#define ZIP_(a,lane1,l1) \
    ZIP__(a,x,lane1,l1,0,0) \
    ZIP__(a,y,lane1,l1,0,1) \
    ZIP__(a,z,lane1,l1,1,0) \
    ZIP__(a,w,lane1,l1,1,1)
#define ZIP \
    ZIP_(x,0,0) \
    ZIP_(y,0,1) \
    ZIP_(z,1,0) \
    ZIP_(w,1,1)

ZIP

#undef ZIP____
#undef ZIP___
#undef ZIP__
#undef ZIP_
#undef ZIP


// packed double precision floating point calculations are not currently supported
// by NEON (at least not in the older ARM arch versions)

//
// no 256 bit wide register support in NEON as I see it currently
//


} // namespace math


#endif // !defined(PMATH_NEON_H)

/*******************************************************************************
 * pvecf.h                                                                     *
 *                                                                             *
 * Copyright (c) 2013-2017 Ronny Press                                         *
 *                                                                             *
 * rpress@soprero.de                                                           *
 *                                                                             *
 * All rights reserved.                                                        *
 *******************************************************************************/

#ifndef PVECF_H
#define PVECF_H


#include <initializer_list>

#include <pmath.h>

namespace math {

// preprocessor shortcuts

#if (defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
#define PVECF_INTEL
#elif (defined(_MSC_VER) && defined(_M_ARM)) || (defined(__GNUC__) && defined(__ARM_ARCH) && defined(__ARM_NEON_FP))
#define PVECF_ARM
#endif


// TODO: check: is it an advantage to use _vectorcall calling convention
//              added in VC++12 (VS2013)?
//              - here in the lib itself
//              - in function calls (not inlined) of apps if the signature
//                includes the vector types defined in this file

//vec4 cross(const vec4 &a, const vec4 &b)
//{
//    return a.yzxw * b.zxyw - a.zxyw * b.yzxw;
//}
// should generate: 4 shuffles, 2 muls and a sub

namespace fpriv {
    template<bool cond, typename T = void> struct enable_if {};
    template<typename T> struct enable_if<true,T> { typedef T type; };
} // namespace fpriv

//
// vecf_t general template
//
template<typename t_real, unsigned t_n, typename t_packed>
class vecf_t
{
public:
    static const unsigned N = t_n;
    static_assert(
        sizeof(t_packed) == (sizeof(t_real) * N),
        "vecf_t: wrong combo of real_t, N and packed_t template parameters used"
    );
    static_assert(
        std::is_floating_point<t_real>::value,
        "vecf_t: supports floating point types only"
    );

    typedef t_real real_t;
    typedef t_packed packed_t;
    typedef typename math::math_t<real_t,packed_t> math_t;
    
    inline vecf_t();
    inline vecf_t(real_t v0);
    inline vecf_t(real_t v0, real_t v1);
    template<typename T = real_t, unsigned N = t_n, typename = typename fpriv::enable_if<(N>2)>::type>
    inline vecf_t(real_t v0, real_t v1, real_t v2);
    template<typename T = real_t, unsigned N = t_n, typename = typename fpriv::enable_if<(N>2)>::type>
    inline vecf_t(real_t v0, real_t v1, real_t v2, real_t v3);
    template<typename T = real_t, unsigned N = t_n, typename = typename fpriv::enable_if<(N>4)>::type>
    inline vecf_t(real_t v0, real_t v1, real_t v2, real_t v3, real_t v4);
    template<typename T = real_t, unsigned N = t_n, typename = typename fpriv::enable_if<(N>4)>::type>
    inline vecf_t(real_t v0, real_t v1, real_t v2, real_t v3, real_t v4, real_t v5);
    template<typename T = real_t, unsigned N = t_n, typename = typename fpriv::enable_if<(N>4)>::type>
    inline vecf_t(real_t v0, real_t v1, real_t v2, real_t v3, real_t v4, real_t v5, real_t v6);
    template<typename T = t_real, unsigned N = t_n, typename = typename fpriv::enable_if<(N>4)>::type>
    inline vecf_t(real_t v0, real_t v1, real_t v2, real_t v3, real_t v4, real_t v5, real_t v6, real_t v7);

    inline vecf_t(const real_t * p) { for(unsigned i = 0; i < N; ++i) v[i] = p[i]; }
    inline explicit vecf_t(packed_t v) { p = v; }

    inline vecf_t(std::initializer_list<real_t> l);

    inline vecf_t & operator=(real_t v);
    inline vecf_t & operator=(packed_t v) { p = v; return *this; }

    inline real_t & operator[](int idx) { return v[idx]; }
    inline real_t operator[](int idx) const { return v[idx]; }

    inline operator const real_t *() const { return v; }
    inline operator real_t *() const { return v; }

    operator t_packed() const { return p; }

    // vector addition, subtraction and
    // componentwise multiplication and division
    vecf_t & operator+=(const vecf_t & v2);
    vecf_t & operator-=(const vecf_t & v2);
    vecf_t & operator*=(const vecf_t & v2);
    vecf_t & operator/=(const vecf_t & v2);
    vecf_t & operator&=(const vecf_t & v2);
    vecf_t & operator|=(const vecf_t & v2);
    vecf_t & operator^=(const vecf_t & v2);

    // packed vector operations
    vecf_t & operator+=(packed_t v);
    vecf_t & operator-=(packed_t v);
    vecf_t & operator*=(packed_t v);
    vecf_t & operator/=(packed_t v);
    vecf_t & operator&=(packed_t v);
    vecf_t & operator|=(packed_t v);
    vecf_t & operator^=(packed_t v);

    // scalar -> vector -> operation
    vecf_t & operator+=(real_t v);
    vecf_t & operator-=(real_t v);
    // scalar multiplication
    vecf_t & operator*=(real_t v);
    // scalar division
    vecf_t & operator/=(real_t v);

    // vector comparisons
    bool operator==(const vecf_t &) const;
    bool operator!=(const vecf_t &) const;
    bool operator<(const vecf_t &) const;
    bool operator<=(const vecf_t &) const;
    bool operator>(const vecf_t &) const;
    bool operator>=(const vecf_t &) const;
    // returns true if at least one value pair is unequal
    bool neq_one(const vecf_t &) const;

    // packed vector comparisons
    bool operator==(packed_t) const;
    bool operator!=(packed_t) const;
    bool operator<(packed_t) const;
    bool operator<=(packed_t) const;
    bool operator>(packed_t) const;
    bool operator>=(packed_t) const;
    // returns true if at least one value pair is unequal
    bool neq_one(packed_t) const;

    // for 4D points (homogenous), you must subtract 1
    // to compensate for w component
    real_t dot(const vecf_t &) const;
    packed_t dot_packed(const vecf_t &) const;
    // returns the squared length, i.e. dot product
    // see above for w component compensation
    real_t sqlen() const;
    packed_t sqlen_packed() const;

    real_t len() const;
    packed_t len_packed() const;

    void normalize();
    vecf_t & normalize_packed();

    void clamp_0_1();

    // TODO: check/look at code/measure whether these actually need to take
    //       references or copies would be faster
    //       (for the copies the compiler just needs to generate a copy
    //        from XMMreg to another XMMreg, for the references it may have
    //        to store into mem first to be able to take the address and
    //        use that address as the argument value; if the functions are
    //        inlined -- in the first case -- it may not even be necessary
    //        to copy into another register because the inlined code can
    //        directly use the source register as its source register)
    vecf_t cross(const vecf_t &) const;
    packed_t cross_packed(const vecf_t &) const;
    vecf_t unit_cross(const vecf_t &) const;
    packed_t unit_cross_packed(const vecf_t &) const;

    static vecf_t min_(const vecf_t &, const vecf_t &);
    static vecf_t max_(const vecf_t &, const vecf_t &);

    vecf_t abs_() const;
    void abs_();

    // returns a vector with the IDXth element being the
    // absolute value of the input value; all other elements
    // are left unchanged
    template<unsigned IDX>
    vecf_t elemAbs() const;
    template<unsigned IDX>
    void elemAbs();


    void trunc();
    vecf_t trunc() const;

    void floor();
    vecf_t floor() const;

    void ceil();
    vecf_t ceil() const;

    void frac();
    vecf_t frac() const;

    //void round_even();
    //vec4f_t round_even() const;

    vecf_t isnan() const;
    bool isnan_all() const;

    // load aligned
    inline void loada(const t_real * p);
    // load unaligned
    inline void loadu(const t_real * p);

    // store aligned
    inline void storea(t_real * p);
    // store unaligned
    inline void storeu(t_real * p);


    union {
        real_t v[N];
        packed_t p;
    };
};



//
// a few macros
//
#define MEMBER_ARITH_OP_(op,scalar,N,packed,prefix,intrinop,postfix) \
template<> inline vecf_t<scalar,N,packed> & vecf_t<scalar,N,packed>::operator op(const vecf_t<scalar,N,packed> & v2) { \
    p = prefix##intrinop##postfix(p, v2.p); return *this; } \
template<> inline vecf_t<scalar,N,packed> & vecf_t<scalar,N,packed>::operator op(packed val) { \
    p = prefix##intrinop##postfix(p, val); return *this; }
#define MEMBER_ARITH_OPS_(scalar,N,packed,prefix,postfix) \
    MEMBER_ARITH_OP_(+=,scalar,N,packed,prefix,add,postfix) \
    MEMBER_ARITH_OP_(-=,scalar,N,packed,prefix,sub,postfix) \
    MEMBER_ARITH_OP_(*=,scalar,N,packed,prefix,mul,postfix) \
    MEMBER_ARITH_OP_(/=,scalar,N,packed,prefix,div,postfix) \
    MEMBER_ARITH_OP_(&=,scalar,N,packed,prefix,and,postfix) \
    MEMBER_ARITH_OP_(|=,scalar,N,packed,prefix,or ,postfix) \
    MEMBER_ARITH_OP_(^=,scalar,N,packed,prefix,xor,postfix)
#ifdef PVECF_INTEL
#define MEMBER_CMP_OP_(op,scalar,N,packed,prefix,postfix,bitmask) \
template<> inline bool vecf_t<scalar,N,packed>::operator op(const vecf_t<scalar,N,packed> & v2) const { \
    return prefix##movemask##postfix(p op v2.p) == bitmask; } \
template<> inline bool vecf_t<scalar,N,packed>::operator op(packed v2) const { \
    return prefix##movemask##postfix(p op v2) == bitmask; }
#elif defined(PVECF_ARM)
#define MEMBER_CMP_OP_(op,scalar,N,packed,prefix,postfix,bitmask) \
template<> inline bool vecf_t<scalar, N, packed>::operator op(const vecf_t<scalar, N, packed> & v2) const { \
    return prefix##movemask##postfix(p op v2.p) == bitmask;} \
template<> inline bool vecf_t<scalar,N,packed>::operator op(packed v2) const { \
    return prefix##movemask##postfix(p op v2) == bitmask; }
#endif
#define MEMBER_CMP_OPS_(scalar,N,packed,prefix,postfix,bitmask) \
    MEMBER_CMP_OP_(< ,scalar,N,packed,prefix,postfix,bitmask)   \
    MEMBER_CMP_OP_(> ,scalar,N,packed,prefix,postfix,bitmask)   \
    MEMBER_CMP_OP_(==,scalar,N,packed,prefix,postfix,bitmask)   \
    MEMBER_CMP_OP_(<=,scalar,N,packed,prefix,postfix,bitmask)   \
    MEMBER_CMP_OP_(>=,scalar,N,packed,prefix,postfix,bitmask)   \
    MEMBER_CMP_OP_(!=,scalar,N,packed,prefix,postfix,bitmask)
#define FREE_ARITH_OP_(op,scalar,N,packed,prefix,intrinop,postfix) \
inline vecf_t<scalar,N,packed> operator op(const vecf_t<scalar,N,packed> & v1, const vecf_t<scalar,N,packed> & v2) { \
    return vecf_t<scalar,N,packed>(prefix##intrinop##postfix(v1.p, v2.p)); } \
inline vecf_t<scalar,N,packed> operator op(const vecf_t<scalar,N,packed> & v, scalar s) { \
    return vecf_t<scalar,N,packed>(prefix##intrinop##postfix(v.p, prefix##set1##postfix(s))); } \
inline vecf_t<scalar,N,packed> operator op(scalar s, const vecf_t<scalar,N,packed> & v) { \
    return vecf_t<scalar,N,packed>(prefix##intrinop##postfix(prefix##set1##postfix(s), v.p)); } \
inline vecf_t<scalar,N,packed> operator op(const vecf_t<scalar,N,packed> & v, packed s) { \
    return vecf_t<scalar,N,packed>(prefix##intrinop##postfix(v.p, s)); } \
inline vecf_t<scalar,N,packed> operator op(packed s, const vecf_t<scalar,N,packed> & v) { \
    return vecf_t<scalar,N,packed>(prefix##intrinop##postfix(s, v.p)); }
#define FREE_ARITH_OPS_(scalar,N,packed,prefix,postfix)  \
    FREE_ARITH_OP_(+,scalar,N,packed,prefix,add,postfix) \
    FREE_ARITH_OP_(-,scalar,N,packed,prefix,sub,postfix) \
    FREE_ARITH_OP_(*,scalar,N,packed,prefix,mul,postfix) \
    FREE_ARITH_OP_(/,scalar,N,packed,prefix,div,postfix)
#define FREE_BIT_OP_(op,scalar,N,packed,prefix,intrinop,postfix) \
inline vecf_t<scalar,N,packed> operator op(const vecf_t<scalar,N,packed> & v1, const vecf_t<scalar,N,packed> & v2) { \
    return vecf_t<scalar,N,packed>(prefix##intrinop##postfix(v1.p, v2.p)); } \
inline vecf_t<scalar,N,packed> operator op(const vecf_t<scalar,N,packed> & v1, packed v2) { \
    return vecf_t<scalar,N,packed>(prefix##intrinop##postfix(v1.p, v2)); } \
inline vecf_t<scalar,N,packed> operator op(packed v1, const vecf_t<scalar,N,packed> & v2) { \
    return vecf_t<scalar,N,packed>(prefix##intrinop##postfix(v1, v2.p)); }
#define FREE_BIT_OPS_(scalar,N,packed,prefix,postfix)  \
    FREE_BIT_OP_(&,scalar,N,packed,prefix,and,postfix) \
    FREE_BIT_OP_(|,scalar,N,packed,prefix,or ,postfix) \
    FREE_BIT_OP_(^,scalar,N,packed,prefix,xor,postfix)

#define FREE_ARITHBIT_OP_NAMED_(opname,scalar,N,packed,prefix,intrinop,postfix) \
inline vecf_t<scalar,N,packed> opname(const vecf_t<scalar,N,packed> & v1, const vecf_t<scalar,N,packed> & v2) { \
    return vecf_t<scalar,N,packed>(prefix##intrinop##postfix(v1.p, v2.p)); } \
inline vecf_t<scalar,N,packed> opname(const vecf_t<scalar,N,packed> & v, packed s) { \
    return vecf_t<scalar,N,packed>(prefix##intrinop##postfix(v.p, s)); } \
inline vecf_t<scalar,N,packed> opname(packed s, const vecf_t<scalar,N,packed> & v) { \
    return vecf_t<scalar,N,packed>(prefix##intrinop##postfix(s, v.p)); } \
inline vecf_t<scalar,N,packed> opname(const vecf_t<scalar,N,packed> & v, scalar s) { \
    return vecf_t<scalar,N,packed>(prefix##intrinop##postfix(v.p, prefix##set1##postfix(s))); } \
inline vecf_t<scalar,N,packed> opname(scalar s, const vecf_t<scalar,N,packed> & v) { \
    return vecf_t<scalar,N,packed>(prefix##intrinop##postfix(prefix##set1##postfix(s), v.p)); }

#define FREE_ARITHBIT_OPS_NAMED_(scalar,N,packed,prefix,postfix) \
    FREE_ARITHBIT_OP_NAMED_(add,scalar,N,packed,prefix,add,postfix) \
    FREE_ARITHBIT_OP_NAMED_(sub,scalar,N,packed,prefix,sub,postfix) \
    FREE_ARITHBIT_OP_NAMED_(mul,scalar,N,packed,prefix,mul,postfix) \
    FREE_ARITHBIT_OP_NAMED_(div,scalar,N,packed,prefix,div,postfix) \
    FREE_ARITHBIT_OP_NAMED_(and_,scalar,N,packed,prefix,and,postfix) \
    FREE_ARITHBIT_OP_NAMED_(or_,scalar,N,packed,prefix,or,postfix) \
    FREE_ARITHBIT_OP_NAMED_(xor_,scalar,N,packed,prefix,xor,postfix)


//
// vec4f_t implementation
//
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
// x86/x64 with MSVC
#  define vec4f_t vecf_t<float,4,__m128>
#elif defined(_MSC_VER) && defined(_M_ARM)
// ARM NEON with MSVC
#  define vec4f_t vecf_t<float,4,__n128>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
// x86/x64 with GCC / MinGW
#  define vec4f_t vecf_t<float,4,__m128>
#elif defined(__GNUC__) && defined(__ARM_ARCH) && defined(__ARM_NEON_FP)
// ARM NEON with GCC
typedef float32x4_t __n128;
#  define vec4f_t vecf_t<float,4,float32x4_t>
#endif




//
// swizzle ops for vec4f_t
//
#define SWIZZLE_FLOAT_4____(a,b,c,d,a_,b_,c_,d_) \
inline vec4f_t a##b##c##d(const vec4f_t & v) { return vec4f_t(a##b##c##d(v.p)); } \
inline vec4f_t a##b##c##d(const vec4f_t & v1, const vec4f_t & v2) { return vec4f_t(a##b##c##d(v1.p, v2.p)); }
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



template<> inline vec4f_t::vecf_t() { p = math_t::zeroes(); }
template<> inline vec4f_t::vecf_t(float v0) { v[0] = v0; v[1] = v[2] = v[3] = 0.0f; }
template<> inline vec4f_t::vecf_t(float v0, float v1) { v[0] = v0; v[1] = v1; v[2] = v[3] = 0.0f; }
template<> template<> inline vec4f_t::vecf_t(float v0, float v1, float v2) { v[0] = v0; v[1] = v1; v[2] = v2; v[3] = 0.0f; }
template<> template<> inline vec4f_t::vecf_t(float v0, float v1, float v2, float v3) { v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3; }
template<> inline vec4f_t::vecf_t(std::initializer_list<real_t> l)
{
    p = math_t::zeroes();
    int count = int(l.size() <= 4 ? l.size() : 4);
    int i = 0; for(auto it = l.begin(); i < count; ++i, ++it) v[i] = *it;
}

template<> inline vec4f_t & vec4f_t::operator=(float v) { p = math_t::set1(v); return *this; }


#if defined(PVECF_INTEL)

MEMBER_ARITH_OPS_(float,4,__m128,_mm_,_ps)

#elif defined(PVECF_ARM)

template<> inline vec4f_t & vec4f_t::operator +=(const vec4f_t & v2) { p = vaddq_f32(p, v2.p); return *this; }
template<> inline vec4f_t & vec4f_t::operator +=(__n128 val) { p = vaddq_f32(p, val); return *this; }

template<> inline vec4f_t & vec4f_t::operator -=(const vec4f_t & v2) { p = vsubq_f32(p, v2.p); return *this; }
template<> inline vec4f_t & vec4f_t::operator -=(__n128 val) { p = vsubq_f32(p, val); return *this; }

template<> inline vec4f_t & vec4f_t::operator *=(const vec4f_t & v2) { p = vmulq_f32(p, v2.p); return *this; }
template<> inline vec4f_t & vec4f_t::operator *=(__n128 val) { p = vmulq_f32(p, val); return *this; }

template<> inline vec4f_t & vec4f_t::operator /=(const vec4f_t & v2) { p = p / v2.p; return *this; }
template<> inline vec4f_t & vec4f_t::operator /=(__n128 val) { p = p / val; return *this; }

template<> inline vec4f_t & vec4f_t::operator &=(const vec4f_t & v2) { p = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(p), vreinterpretq_u32_f32(v2.p))); return *this; }
template<> inline vec4f_t & vec4f_t::operator &=(__n128 val) { p = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(p), vreinterpretq_u32_f32(val))); return *this; }

template<> inline vec4f_t & vec4f_t::operator |=(const vec4f_t & v2) { p = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(p), vreinterpretq_u32_f32(v2.p))); return *this; }
template<> inline vec4f_t & vec4f_t::operator |=(__n128 val) { p = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(p), vreinterpretq_u32_f32(val))); return *this; }

template<> inline vec4f_t & vec4f_t::operator ^=(const vec4f_t & v2) { p = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(p), vreinterpretq_u32_f32(v2.p))); return *this; }
template<> inline vec4f_t & vec4f_t::operator ^=(__n128 val) { p = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(p), vreinterpretq_u32_f32(val))); return *this; }

#endif


#if defined(PVECF_INTEL)

template<> inline vec4f_t & vec4f_t::operator+=(float v) { p = _mm_add_ps(p, math_t::set1(v)); return *this; }
template<> inline vec4f_t & vec4f_t::operator-=(float v) { p = _mm_sub_ps(p, math_t::set1(v)); return *this; }
// scalar multiplication
template<> inline vec4f_t & vec4f_t::operator*=(float v) { p = _mm_mul_ps(p, math_t::set1(v)); return *this; }
// scalar division
template<> inline vec4f_t & vec4f_t::operator/=(float v) { p = _mm_div_ps(p, math_t::set1(v)); return *this; }

#elif defined(PVECF_ARM)

template<> inline vec4f_t & vec4f_t::operator+=(float v) { p = vaddq_f32(p, vdupq_n_f32(v)); return *this; }
template<> inline vec4f_t & vec4f_t::operator-=(float v) { p = vsubq_f32(p, vdupq_n_f32(v)); return *this; }
// scalar multiplication
template<> inline vec4f_t & vec4f_t::operator*=(float v) { p = vmulq_f32(p, vdupq_n_f32(v)); return *this; }
// scalar division
template<> inline vec4f_t & vec4f_t::operator/=(float v) { p = p / v; return *this; }

#endif


#if defined(PVECF_INTEL)

// TODO: major flaw here: the 0xF checks whether _all_ elements satisfy the
//       condition; for the operator!= it makes a lot more sense if it returned
//       whether at least one of the components are not equal;
//       for the other operators this might not be suitable
MEMBER_CMP_OPS_(float,4,__m128,_mm_,_ps,0xF)
template<> inline bool vec4f_t::neq_one(const vec4f_t & v2) const {
    return _mm_movemask_ps(p == v2.p) != 0;
}
template<> inline bool vec4f_t::neq_one(packed_t v2) const {
    return _mm_movemask_ps(p == v2) != 0;
}

#elif defined(PVECF_ARM)

// TODO: "conversion" to bool for the comparison operators not implemented yet

#endif


template<> inline vec4f_t::packed_t vec4f_t::dot_packed(const vec4f_t & v2) const
{ return math_t::dot_packed(p, v2.p); }
template<> inline vec4f_t::packed_t vec4f_t::cross_packed(const vec4f_t & v2) const
{ // this = {w0,z0,y0,x0}, v2 = {w1,z1,y1,x1}
  // this x v2 = {0,x0y1-x1y0,x1z0-x0z1,y0z1-y1z0}
  //return wxzy(p) * wyxz(v2.p) - wyxz(p) * wxzy(v2.p);
  // this = {x0,y0,z0,w0}, v2 = {x1,y1,z1,w1}
  // this x v2 = {y0z1-y1z0,x1z0-x0z1,x0y1-x1y0,0}
  return yzxw(p) * zxyw(v2.p) - zxyw(p) * yzxw(v2.p);
}
template<> inline vec4f_t::packed_t vec4f_t::unit_cross_packed(const vec4f_t & v2) const
{ vec4f_t::packed_t cross = cross_packed(v2);
  return cross * math_t::inv_sqrt_packed(math_t::dot_packed(cross, cross)); }

template<> inline float vec4f_t::dot(const vec4f_t & v2) const
{
#if defined(PVECF_INTEL)
    return _mm_cvtss_f32(dot_packed(v2));
#elif defined(PVECF_ARM)
    return vget_lane_f32(vget_low_f32(dot_packed(v2)), 0);
#endif
}
template<> inline float vec4f_t::sqlen() const { return dot(*this); }
template<> inline vec4f_t::packed_t vec4f_t::sqlen_packed() const { return dot_packed(*this); }
template<> inline float vec4f_t::len() const { return math_t::sqrt(dot(*this)); }
template<> inline vec4f_t::packed_t vec4f_t::len_packed() const { return math_t::sqrt_packed(dot_packed(*this)); }
template<> inline void vec4f_t::normalize()
{ p = p * math_t::inv_sqrt_packed(math_t::set1(sqlen())); }
template<> inline vec4f_t & vec4f_t::normalize_packed()
{ p = p * math_t::inv_sqrt_packed(sqlen_packed()); return *this; }
template<> inline void vec4f_t::clamp_0_1()
{
#if defined(PVECF_INTEL)
    p = _mm_max_ps(math_t::zeroes(), _mm_min_ps(p, math_t::ones()));
#elif defined(PVECF_ARM)
    p = vmaxq_f32(math_t::zeroes(), vminq_f32(p, math_t::ones()));
#endif
}
template<> inline vec4f_t vec4f_t::cross(const vec4f_t & v2) const
{ return vec4f_t(cross_packed(v2)); }
template<> inline vec4f_t vec4f_t::unit_cross(const vec4f_t & v2) const
{ return vec4f_t(unit_cross_packed(v2)); }


#if defined(PVECF_INTEL)

FREE_ARITH_OPS_(float,4,__m128,_mm_,_ps)

FREE_BIT_OPS_(float,4,__m128,_mm_,_ps)

FREE_ARITHBIT_OPS_NAMED_(float,4,__m128,_mm_,_ps)


// flip all bits; result may not be floating points any more, primary use
// is for further processing of comparison results
inline vec4f_t operator~(const vec4f_t & v)
{ return vec4f_t(_mm_xor_ps(v.p, _mm_cmpeq_ps(vec4f_t::math_t::zeroes(), vec4f_t::math_t::zeroes()))); }

inline vec4f_t not(const vec4f_t & v)
{ return vec4f_t(_mm_xor_ps(v.p, _mm_cmpeq_ps(vec4f_t::math_t::zeroes(), vec4f_t::math_t::zeroes()))); }


#elif defined(PVECF_ARM)

inline vec4f_t operator +(const vec4f_t & v1, const vec4f_t & v2) { return vec4f_t(vaddq_f32(v1.p, v2.p)); }
inline vec4f_t operator +(const vec4f_t & v, float s) { return vec4f_t(vaddq_f32(v.p, vdupq_n_f32(s))); }
inline vec4f_t operator +(float s, const vec4f_t & v) { return vec4f_t(vaddq_f32(vdupq_n_f32(s), v.p)); }
inline vec4f_t operator +(const vec4f_t & v, __n128 s) { return vec4f_t(vaddq_f32(v.p, s)); }
inline vec4f_t operator +(__n128 s, const vec4f_t & v) { return vec4f_t(vaddq_f32(s, v.p)); }

inline vec4f_t operator -(const vec4f_t & v1, const vec4f_t & v2) { return vec4f_t(vsubq_f32(v1.p, v2.p)); }
inline vec4f_t operator -(const vec4f_t & v, float s) { return vec4f_t(vsubq_f32(v.p, vdupq_n_f32(s))); }
inline vec4f_t operator -(float s, const vec4f_t & v) { return vec4f_t(vsubq_f32(vdupq_n_f32(s), v.p)); }
inline vec4f_t operator -(const vec4f_t & v, __n128 s) { return vec4f_t(vsubq_f32(v.p, s)); }
inline vec4f_t operator -(__n128 s, const vec4f_t & v) { return vec4f_t(vsubq_f32(s, v.p)); }

inline vec4f_t operator *(const vec4f_t & v1, const vec4f_t & v2) { return vec4f_t(vmulq_f32(v1.p, v2.p)); }
inline vec4f_t operator *(const vec4f_t & v, float s) { return vec4f_t(vmulq_f32(v.p, vdupq_n_f32(s))); }
inline vec4f_t operator *(float s, const vec4f_t & v) { return vec4f_t(vmulq_f32(vdupq_n_f32(s), v.p)); }
inline vec4f_t operator *(const vec4f_t & v, __n128 s) { return vec4f_t(vmulq_f32(v.p, s)); }
inline vec4f_t operator *(__n128 s, const vec4f_t & v) { return vec4f_t(vmulq_f32(s, v.p)); }

// no divs in NEON
inline vec4f_t operator /(const vec4f_t & v1, const vec4f_t & v2) { return vec4f_t(v1.p / v2.p); }
inline vec4f_t operator /(const vec4f_t & v, float s) { return vec4f_t(v.p / s); }
inline vec4f_t operator /(float s, const vec4f_t & v) { return vec4f_t(s / v.p); }
inline vec4f_t operator /(const vec4f_t & v, __n128 s) { return vec4f_t(v.p / s); }
inline vec4f_t operator /(__n128 s, const vec4f_t & v) { return vec4f_t(s / v.p); }


inline vec4f_t operator &(vec4f_t op1, vec4f_t op2) {
    return vec4f_t(vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(op1.p), vreinterpretq_u32_f32(op2.p))));
}
inline vec4f_t operator &(__n128 op1, vec4f_t op2) {
    return vec4f_t(vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(op1), vreinterpretq_u32_f32(op2.p))));
}
inline vec4f_t operator &(vec4f_t op1, __n128 op2) {
    return vec4f_t(vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(op1.p), vreinterpretq_u32_f32(op2))));
}

inline vec4f_t operator |(vec4f_t op1, vec4f_t op2) {
    return vec4f_t(vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(op1.p), vreinterpretq_u32_f32(op2.p))));
}
inline vec4f_t operator |(__n128 op1, vec4f_t op2) {
    return vec4f_t(vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(op1), vreinterpretq_u32_f32(op2.p))));
}
inline vec4f_t operator |(vec4f_t op1, __n128 op2) {
    return vec4f_t(vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(op1.p), vreinterpretq_u32_f32(op2))));
}

inline vec4f_t operator ^(vec4f_t op1, vec4f_t op2) {
    return vec4f_t(vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(op1.p), vreinterpretq_u32_f32(op2.p))));
}
inline vec4f_t operator ^(__n128 op1, vec4f_t op2) {
    return vec4f_t(vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(op1), vreinterpretq_u32_f32(op2.p))));
}
inline vec4f_t operator ^(vec4f_t op1, __n128 op2) {
    return vec4f_t(vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(op1.p), vreinterpretq_u32_f32(op2))));
}

#if 0
#define FREE_ARITHBIT_OP_NAMED_(opname,scalar,N,packed,prefix,intrinop,postfix) \
inline vecf_t<scalar,N,packed> opname(const vecf_t<scalar,N,packed> & v1, const vecf_t<scalar,N,packed> & v2) { \
    return vecf_t<scalar,N,packed>(prefix##intrinop##postfix(v1.p, v2.p)); } \
inline vecf_t<scalar,N,packed> opname(const vecf_t<scalar,N,packed> & v, packed s) { \
    return vecf_t<scalar,N,packed>(prefix##intrinop##postfix(v.p, s)); } \
inline vecf_t<scalar,N,packed> opname(packed s, const vecf_t<scalar,N,packed> & v) { \
    return vecf_t<scalar,N,packed>(prefix##intrinop##postfix(s, v.p)); } \
inline vecf_t<scalar,N,packed> opname(const vecf_t<scalar,N,packed> & v, scalar s) { \
    return vecf_t<scalar,N,packed>(prefix##intrinop##postfix(v.p, prefix##set1##postfix(s))); } \
inline vecf_t<scalar,N,packed> opname(scalar s, const vecf_t<scalar,N,packed> & v) { \
    return vecf_t<scalar,N,packed>(prefix##intrinop##postfix(prefix##set1##postfix(s), v.p)); }

#define FREE_ARITHBIT_OPS_NAMED_(scalar,N,packed,prefix,postfix) \
    FREE_ARITHBIT_OP_NAMED_(add,scalar,N,packed,prefix,add,postfix) \
    FREE_ARITHBIT_OP_NAMED_(sub,scalar,N,packed,prefix,sub,postfix) \
    FREE_ARITHBIT_OP_NAMED_(mul,scalar,N,packed,prefix,mul,postfix) \
    FREE_ARITHBIT_OP_NAMED_(div,scalar,N,packed,prefix,div,postfix) \
    FREE_ARITHBIT_OP_NAMED_(and,scalar,N,packed,prefix,and,postfix) \
    FREE_ARITHBIT_OP_NAMED_(or,scalar,N,packed,prefix,or,postfix) \
    FREE_ARITHBIT_OP_NAMED_(xor,scalar,N,packed,prefix,xor,postfix)
#endif

inline vec4f_t add(const vec4f_t & v1, const vec4f_t & v2) { return vec4f_t(vaddq_f32(v1.p, v2.p)); }
inline vec4f_t add(const vec4f_t & v, __n128 s) { return vec4f_t(vaddq_f32(v.p, s)); }
inline vec4f_t add(__n128 s, const vec4f_t & v) { return vec4f_t(vaddq_f32(s, v.p)); }
inline vec4f_t add(const vec4f_t & v, float s) { return vec4f_t(vaddq_f32(v.p, vdupq_n_f32(s))); }
inline vec4f_t add(float s, const vec4f_t & v) { return vec4f_t(vaddq_f32(vdupq_n_f32(s), v.p)); }

inline vec4f_t sub(const vec4f_t & v1, const vec4f_t & v2) { return vec4f_t(vsubq_f32(v1.p, v2.p)); }
inline vec4f_t sub(const vec4f_t & v, __n128 s) { return vec4f_t(vsubq_f32(v.p, s)); }
inline vec4f_t sub(__n128 s, const vec4f_t & v) { return vec4f_t(vsubq_f32(s, v.p)); }
inline vec4f_t sub(const vec4f_t & v, float s) { return vec4f_t(vsubq_f32(v.p, vdupq_n_f32(s))); }
inline vec4f_t sub(float s, const vec4f_t & v) { return vec4f_t(vsubq_f32(vdupq_n_f32(s), v.p)); }

inline vec4f_t mul(const vec4f_t & v1, const vec4f_t & v2) { return vec4f_t(vmulq_f32(v1.p, v2.p)); }
inline vec4f_t mul(const vec4f_t & v, __n128 s) { return vec4f_t(vmulq_f32(v.p, s)); }
inline vec4f_t mul(__n128 s, const vec4f_t & v) { return vec4f_t(vmulq_f32(s, v.p)); }
inline vec4f_t mul(const vec4f_t & v, float s) { return vec4f_t(vmulq_f32(v.p, vdupq_n_f32(s))); }
inline vec4f_t mul(float s, const vec4f_t & v) { return vec4f_t(vmulq_f32(vdupq_n_f32(s), v.p)); }

// no divs in NEON
inline vec4f_t div(const vec4f_t & v1, const vec4f_t & v2) { return vec4f_t(v1.p / v2.p); }
inline vec4f_t div(const vec4f_t & v, __n128 s) { return vec4f_t(v.p / s); }
inline vec4f_t div(__n128 s, const vec4f_t & v) { return vec4f_t(s / v.p); }
inline vec4f_t div(const vec4f_t & v, float s) { return vec4f_t(v.p / s); }
inline vec4f_t div(float s, const vec4f_t & v) { return vec4f_t(s / v.p); }

inline vec4f_t and_(const vec4f_t & v1, const vec4f_t & v2) { return vec4f_t(vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(v1.p), vreinterpretq_u32_f32(v2.p)))); }
inline vec4f_t and_(const vec4f_t & v, __n128 s) { return vec4f_t(vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(v.p), vreinterpretq_u32_f32(s)))); }
inline vec4f_t and_(__n128 s, const vec4f_t & v) { return vec4f_t(vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(s), vreinterpretq_u32_f32(v.p)))); }
inline vec4f_t and_(const vec4f_t & v, float s) { return vec4f_t(vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(v.p), vreinterpretq_u32_f32(vdupq_n_f32(s))))); }
inline vec4f_t and_(float s, const vec4f_t & v) { return vec4f_t(vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(vdupq_n_f32(s)), vreinterpretq_u32_f32(v.p)))); }

inline vec4f_t or_(const vec4f_t & v1, const vec4f_t & v2) { return vec4f_t(vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(v1.p), vreinterpretq_u32_f32(v2.p)))); }
inline vec4f_t or_(const vec4f_t & v, __n128 s) { return vec4f_t(vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(v.p), vreinterpretq_u32_f32(s)))); }
inline vec4f_t or_(__n128 s, const vec4f_t & v) { return vec4f_t(vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(s), vreinterpretq_u32_f32(v.p)))); }
inline vec4f_t or_(const vec4f_t & v, float s) { return vec4f_t(vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(v.p), vreinterpretq_u32_f32(vdupq_n_f32(s))))); }
inline vec4f_t or_(float s, const vec4f_t & v) { return vec4f_t(vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(vdupq_n_f32(s)), vreinterpretq_u32_f32(v.p)))); }

inline vec4f_t xor_(const vec4f_t & v1, const vec4f_t & v2) { return vec4f_t(vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v1.p), vreinterpretq_u32_f32(v2.p)))); }
inline vec4f_t xor_(const vec4f_t & v, __n128 s) { return vec4f_t(vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v.p), vreinterpretq_u32_f32(s)))); }
inline vec4f_t xor_(__n128 s, const vec4f_t & v) { return vec4f_t(vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(s), vreinterpretq_u32_f32(v.p)))); }
inline vec4f_t xor_(const vec4f_t & v, float s) { return vec4f_t(vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v.p), vreinterpretq_u32_f32(vdupq_n_f32(s))))); }
inline vec4f_t xor_(float s, const vec4f_t & v) { return vec4f_t(vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(vdupq_n_f32(s)), vreinterpretq_u32_f32(v.p)))); }


// flip all bits; result may not be floating points any more, primary use
// is for further processing of comparison results
inline vec4f_t operator~(const vec4f_t & v)
{
    return
        vec4f_t(
            vreinterpretq_f32_u32(
                veorq_u32(
                    vreinterpretq_u32_f32(v.p),
                    vceqq_f32(
                        vec4f_t::math_t::zeroes(),
                        vec4f_t::math_t::zeroes()
                    )
                )
            )
        );
}

#endif


inline vec4f_t notAandB(const vec4f_t & a, const vec4f_t & b) { return vec4f_t(notAandB_(a.p, b.p)); }

inline vec4f_t cmp_eq(const vec4f_t & a, const vec4f_t & b){
#if defined(PVECF_INTEL)
    return vec4f_t(_mm_cmpeq_ps(a.p, b.p));
#elif defined(PVECF_ARM)
    return vec4f_t(vreinterpretq_f32_u32(vceqq_f32(a.p, b.p)));
#endif
}

inline vec4f_t cmp_gte(const vec4f_t & a, const vec4f_t & b) {
#if defined(PVECF_INTEL)
    return vec4f_t(_mm_cmpge_ps(a.p, b.p));
#elif defined(PVECF_ARM)
    return vec4f_t(vreinterpretq_f32_u32(vcgeq_f32(a.p, b.p)));
#endif
}

inline vec4f_t cmp_lte(const vec4f_t & a, const vec4f_t & b)
{
#if defined(PVECF_INTEL)
    return vec4f_t(_mm_cmple_ps(a.p, b.p));
#elif defined(PVECF_ARM)
    return vec4f_t(vreinterpretq_f32_u32(vcleq_f32(a.p, b.p)));
#endif
}

inline vec4f_t cmp_gt(const vec4f_t & a, const vec4f_t & b)
{
#if defined(PVECF_INTEL)
    return vec4f_t(_mm_cmpgt_ps(a.p, b.p));
#elif defined(PVECF_ARM)
    return vec4f_t(vreinterpretq_f32_u32(vcgtq_f32(a.p, b.p)));
#endif
}

inline vec4f_t cmp_lt(const vec4f_t & a, const vec4f_t & b)
{
#if defined(PVECF_INTEL)
    return vec4f_t(_mm_cmplt_ps(a.p, b.p));
#elif defined(PVECF_ARM)
    return vec4f_t(vreinterpretq_f32_u32(vcltq_f32(a.p, b.p)));
#endif
}

inline vec4f_t cmp_neq(const vec4f_t & a, const vec4f_t & b)
{
#if defined(PVECF_INTEL)
    return vec4f_t(_mm_cmpneq_ps(a.p, b.p));
#elif defined(PVECF_ARM)
    return vec4f_t(
        vreinterpretq_f32_u32(
            veorq_u32(
                vceqq_f32(a.p, b.p),
                vceqq_f32(
                    vec4f_t::math_t::zeroes(),
                    vec4f_t::math_t::zeroes()
                )
            )
        )
    );
#endif
}


inline vec4f_t select(const vec4f_t & masks, const vec4f_t & a, const vec4f_t & b)
{
#if defined(PVECF_INTEL)
    return vec4f_t(_mm_or_ps(_mm_and_ps(a.p, masks.p), _mm_andnot_ps(masks.p, b.p)));
#elif defined(PVECF_ARM)
    return vec4f_t(
        vreinterpretq_f32_u32(
            vorrq_u32(
                vandq_u32(vreinterpretq_u32_f32(a.p), vreinterpretq_u32_f32(masks.p)),
                vbicq_u32(vreinterpretq_u32_f32(masks.p), vreinterpretq_u32_f32(b.p))
            )
        )
    );
#endif
}


inline vec4f_t select_eq(const vec4f_t & a, const vec4f_t & b, const vec4f_t & valsT, const vec4f_t & valsF)
{
#if defined(PVECF_INTEL)
    __m128 masks = _mm_cmpeq_ps(a.p, b.p);
    return vec4f_t(_mm_or_ps(_mm_and_ps(valsT.p, masks), _mm_andnot_ps(masks, valsF.p)));
#elif defined(PVECF_ARM)
    uint32x4_t masks = vceqq_f32(a.p, b.p);
    return vec4f_t(
        vreinterpretq_f32_u32(
            vorrq_u32(
                vandq_u32(vreinterpretq_u32_f32(valsT.p), masks),
                vbicq_u32(masks, vreinterpretq_u32_f32(valsF.p))
            )
        )
    );
#endif
}

inline vec4f_t select_neq(const vec4f_t & a, const vec4f_t & b, const vec4f_t & valsT, const vec4f_t & valsF)
{
#if defined(PVECF_INTEL)
    __m128 masks = _mm_cmpneq_ps(a.p, b.p);
    return vec4f_t(_mm_or_ps(_mm_and_ps(valsT.p, masks), _mm_andnot_ps(masks, valsF.p)));
#elif defined(PVECF_ARM)
    uint32x4_t masks =
        veorq_u32(
            vceqq_f32(a.p, b.p),
            vceqq_f32(vec4f_t::math_t::zeroes(), vec4f_t::math_t::zeroes())
        );
    return vec4f_t(
        vreinterpretq_f32_u32(
            vorrq_u32(
                vandq_u32(vreinterpretq_u32_f32(valsT.p), masks),
                vbicq_u32(masks, vreinterpretq_u32_f32(valsF.p))
            )
        )
    );
#endif
}

inline vec4f_t select_lt(const vec4f_t & a, const vec4f_t & b, const vec4f_t & valsT, const vec4f_t & valsF)
{
#if defined(PVECF_INTEL)
    __m128 masks = _mm_cmplt_ps(a.p, b.p);
    return vec4f_t(_mm_or_ps(_mm_and_ps(valsT.p, masks), _mm_andnot_ps(masks, valsF.p)));
#elif defined(PVECF_ARM)
    uint32x4_t masks = vcltq_f32(a.p, b.p);
    return vec4f_t(
        vreinterpretq_f32_u32(
            vorrq_u32(
                vandq_u32(vreinterpretq_u32_f32(valsT.p), masks),
                vbicq_u32(masks, vreinterpretq_u32_f32(valsF.p))
            )
        )
    );
#endif
}

inline vec4f_t select_lte(const vec4f_t & a, const vec4f_t & b, const vec4f_t & valsT, const vec4f_t & valsF)
{
#if defined(PVECF_INTEL)
    __m128 masks = _mm_cmple_ps(a.p, b.p);
    return vec4f_t(_mm_or_ps(_mm_and_ps(valsT.p, masks), _mm_andnot_ps(masks, valsF.p)));
#elif defined(PVECF_ARM)
    uint32x4_t masks = vcleq_f32(a.p, b.p);
    return vec4f_t(
        vreinterpretq_f32_u32(
            vorrq_u32(
                vandq_u32(vreinterpretq_u32_f32(valsT.p), masks),
                vbicq_u32(masks, vreinterpretq_u32_f32(valsF.p))
            )
        )
    );
#endif
}

inline vec4f_t select_gt(const vec4f_t & a, const vec4f_t & b, const vec4f_t & valsT, const vec4f_t & valsF)
{
#if defined(PVECF_INTEL)
    __m128 masks = _mm_cmpgt_ps(a.p, b.p);
    return vec4f_t(_mm_or_ps(_mm_and_ps(valsT.p, masks), _mm_andnot_ps(masks, valsF.p)));
#elif defined(PVECF_ARM)
    uint32x4_t masks = vcgtq_f32(a.p, b.p);
    return vec4f_t(
        vreinterpretq_f32_u32(
            vorrq_u32(
                vandq_u32(vreinterpretq_u32_f32(valsT.p), masks),
                vbicq_u32(masks, vreinterpretq_u32_f32(valsF.p))
            )
        )
    );
#endif
}

inline vec4f_t select_gte(const vec4f_t & a, const vec4f_t & b, const vec4f_t & valsT, const vec4f_t & valsF)
{
#if defined(PVECF_INTEL)
    __m128 masks = _mm_cmpge_ps(a.p, b.p);
    return vec4f_t(_mm_or_ps(_mm_and_ps(valsT.p, masks), _mm_andnot_ps(masks, valsF.p)));
#elif defined(PVECF_ARM)
    uint32x4_t masks = vcgeq_f32(a.p, b.p);
    return vec4f_t(
        vreinterpretq_f32_u32(
            vorrq_u32(
                vandq_u32(vreinterpretq_u32_f32(valsT.p), masks),
                vbicq_u32(masks, vreinterpretq_u32_f32(valsF.p))
            )
        )
    );
#endif
}


// unary minus
inline vec4f_t operator-(const vec4f_t & v)
//{ return vec4f_t(_mm_sub_ps(math_t<float,__m128>::zeroes(), v.p)); }
{ return vec4f_t(vec4f_t::math_t::zeroes() - v.p); }

inline vec4f_t::packed_t dot_packed(const vec4f_t & v0, const vec4f_t & v1)
{ return vec4f_t::math_t::dot_packed(v0.p, v1.p); }
inline float dot(const vec4f_t & v0, const vec4f_t & v1)
{
#if defined(PVECF_INTEL)
    return _mm_cvtss_f32(dot_packed(v0, v1));
#elif defined(PVECF_ARM)
    return vget_lane_f32(vget_low_f32(dot_packed(v0, v1)), 0);
#endif
}
inline vec4f_t::packed_t dot_packed_4(const vec4f_t & v00, const vec4f_t & v01,
                                      const vec4f_t & v10, const vec4f_t & v11,
                                      const vec4f_t & v20, const vec4f_t & v21,
                                      const vec4f_t & v30, const vec4f_t & v31)
{ vec4f_t::packed_t tmp0 = dot_packed(v00, v01); // {..,dot0}
  vec4f_t::packed_t tmp1 = dot_packed(v10, v11); // {..,dot1}
  vec4f_t::packed_t tmp2 = dot_packed(v20, v21); // {..,dot2}
  vec4f_t::packed_t tmp3 = dot_packed(v30, v31); // {..,dot3}
  return
      xzxz(
          xxxx(tmp0, tmp1), // {dot0,dot0,dot1,dot1}
          xxxx(tmp2, tmp3)  // {dot2,dot2,dot3,dot3}
      ); // {dot0,dot1,dot2,dot3}
}

// result = origin + t * dir
inline vec4f_t eval_line(const vec4f_t & origin, float t, const vec4f_t & dir)
{ return vec4f_t(origin.p + (vec4f_t::math_t::set1(t) * dir.p)); }
inline vec4f_t eval_line_packed(const vec4f_t & origin, vec4f_t::packed_t t, const vec4f_t & dir)
{ return vec4f_t(origin + (t * dir)); }

template<> inline vec4f_t vec4f_t::min_(const vec4f_t & v0, const vec4f_t & v1)
{
#if defined(PVECF_INTEL)
    return vec4f_t(_mm_min_ps(v0.p, v1.p));
#elif defined(PVECF_ARM)
    return vec4f_t(vminq_f32(v0.p, v1.p));
#endif
}
template<> inline vec4f_t vec4f_t::max_(const vec4f_t & v0, const vec4f_t & v1)
{
#if defined(PVECF_INTEL)
    return vec4f_t(_mm_max_ps(v0.p, v1.p));
#elif defined(PVECF_ARM)
    return vec4f_t(vmaxq_f32(v0.p, v1.p));
#endif
}

template<> inline void vec4f_t::abs_()
{ p = notAandB_(math_t::set1(-0.0f), p); }
template<> inline vec4f_t vec4f_t::abs_() const
{ vec4f_t ret(p); ret.abs_(); return ret; }



template<> template<unsigned IDX> inline vec4f_t vec4f_t::elemAbs() const
{ vec4f_t ret(p); ret.elemAbs<IDX>(); return ret; }

template<> template<unsigned IDX> inline void vec4f_t::elemAbs()
{
    static_assert(IDX <= 3, "IDX out of range");
    packed_t masks(math_t::set1(-0.0f));
    packed_t zeroes(math_t::zeroes());
    packed_t val = xxxx(masks, zeroes); // {-0,-0,0,0}
    switch(IDX) {
    case 0:
        val = xzzz(val, zeroes);   // {-0,0,0,0}
        break;
    case 1:
        val = zxzz(val, zeroes);   // {0,-0,0,0}
        break;
    case 2:
        val = xzzz(val, zeroes);   // {-0,0,0,0}
        val = zzxz(zeroes, val);   // {0,0,-0,0}
        break;
    case 3:
        val = xzzz(val, zeroes);   // {-0,0,0,0}
        val = zzzx(zeroes, val);   // {0,0,0,-0}
        break;
    }
    //p = p ^ val;
    //p ^= val;
    *this ^= val;
}




template<> inline void vec4f_t::trunc()
{
#if defined(PVECF_INTEL)
    // vector conversion to integers and back is supported starting with SSE2;
    // for SSE have to go through GP registers and in a scalar way
    // TODO: add this restriction to the documentation (performance warning)
#  if !defined(SSE2) && !defined(SSE3) && !defined(SSSE3) && !defined(SSE4) && !defined(SSE4_2) && !defined(AVX)
    int i[4] = {
        _mm_cvtt_ss2si(xxxx(p)), _mm_cvtt_ss2si(yyyy(p)),
        _mm_cvtt_ss2si(zzzz(p)), _mm_cvtt_ss2si(wwww(p))
    };
    __m128 x, y, z, w;
    x = _mm_cvt_si2ss(x, i[0]);
    y = _mm_cvt_si2ss(y, i[1]);
    z = _mm_cvt_si2ss(z, i[2]);
    w = _mm_cvt_si2ss(w, i[3]);
    p = xzxz(xxxx(x, y), xxxx(z, w));
#  elif defined(SSE4)
    p = _mm_round_ps(p, _MM_FROUND_TRUNC);
#  else
    p = _mm_cvtepi32_ps(_mm_cvttps_epi32(p));
#  endif
#elif defined(PVECF_ARM)
    p = vcvtq_f32_s32(vcvtq_s32_f32(p));
#endif
}
template<> inline vec4f_t vec4f_t::trunc() const
{ vec4f_t ret(p); ret.trunc(); return ret; }


template<> inline void vec4f_t::floor()
{
#if defined(PVECF_INTEL)
    // SSE4 (aka SSE4.1) has rounding instruction
#  if defined(SSE4)
    p = _mm_floor_ps(p);
    return;
#  else
    __m128 orig = p;
    trunc();
    __m128 tmp = orig - p;
    __m128 submask = _mm_cmplt_ps(tmp, math_t::zeroes());
    p = _mm_or_ps(notAandB_(submask, p), _mm_and_ps(p-math_t::ones(), submask));
#  endif
#elif defined(PVECF_ARM)
    __n128 orig = p;
    trunc();
    __n128 tmp = orig - p;
    __n128 submask = vreinterpretq_f32_u32(vcltq_f32(tmp, math_t::zeroes()));
    p =
        vreinterpretq_f32_u32(
            vorrq_u32(
                vreinterpretq_u32_f32(notAandB_(submask, p)),
                vandq_u32(
                    vreinterpretq_u32_f32(vsubq_f32(p, math_t::ones())),
                    vreinterpretq_u32_f32(submask)
                )
            )
        );
#endif
}
template<> inline vec4f_t vec4f_t::floor() const
{ vec4f_t ret(p); ret.floor(); return ret; }


template<> inline void vec4f_t::ceil()
{
#if defined(PVECF_INTEL)
    // SSE4 (aka SSE4.1) has rounding instruction
#  if defined(SSE4)
    p = _mm_ceil_ps(p);
    return;
#  else
    __m128 orig = p;
    trunc();
    __m128 tmp = orig - p;
    __m128 addmask = _mm_cmpgt_ps(tmp, math_t::zeroes());
    p = _mm_or_ps(notAandB_(addmask, p), _mm_and_ps(p+math_t::ones(), addmask));
#  endif
#elif defined(PVECF_ARM)
    __n128 orig = p;
    trunc();
    __n128 tmp = orig - p;
    __n128 addmask = vreinterpretq_f32_u32(vcgtq_f32(tmp, math_t::zeroes()));
    p =
        vreinterpretq_f32_u32(
            vorrq_u32(
                vreinterpretq_u32_f32(notAandB_(addmask, p)),
                vandq_u32(
                    vreinterpretq_u32_f32(vaddq_f32(p, math_t::ones())),
                    vreinterpretq_u32_f32(addmask)
                )
            )
        );
#endif
}
template<> inline vec4f_t vec4f_t::ceil() const
{ vec4f_t ret(p); ret.ceil(); return ret; }


template<> inline void vec4f_t::frac()
{ packed_t orig = p; floor(); p = orig - p; }
template<> inline vec4f_t vec4f_t::frac() const
{ vec4f_t ret(p); ret.frac(); return ret; }

/*
template<> inline void vec4f_t::round_even()
{
#if defined(PVECF_INTEL)
    // SSE4 (aka SSE4.1) has rounding instruction
#  if defined(SSE4)
    p = _mm_round_ps(p, _MM_FROUND_NINT);
    return;
#  else
    __m128 orig = p;
    trunc();
    __m128 tmp = orig - p;
    __m128 addmask = _mm_cmpgt_ps(tmp, math_t::zeroes());
#  endif
#elif defined(PVECF_ARM)
    __n128 orig = p;
    trunc();
    __n128 tmp = orig - p;
    __n128 addmask = vcgtq_f32(tmp, math_t::zeroes());
#endif
    p = notAandB_(addmask, p) | ((p+math_t::ones()) & addmask);
}
template<> inline vec4f_t vec4f_t::round_even() const
{ vec4f_t ret(p); ret.round_even(); return ret; }
*/

template<> inline vec4f_t vec4f_t::isnan() const
{
#if defined(PVECF_INTEL)
    vec4f_t v(_mm_cmpeq_ps(p, p));
#elif defined(PVECF_ARM)
    vec4f_t v(vreinterpretq_f32_u32(vceqq_f32(p, p)));
#endif
    return ~v;
}

template<> inline bool vec4f_t::isnan_all() const
{
#if defined(PVECF_INTEL)
    return _mm_movemask_ps(_mm_cmpeq_ps(_mm_cmpeq_ps(p, p), vec4f_t::math_t::zeroes())) == 0xF;
#elif defined(PVECF_ARM)
    // TODO: movemask() in NEON?
#endif
}

// load aligned
template<> inline void vec4f_t::loada(const float * ptr)
{
    p = *reinterpret_cast<const packed_t *>(ptr);
}

// load unaligned
template<> inline void vec4f_t::loadu(const float * ptr)
{
#if defined(PVECF_INTEL)
    p = _mm_loadu_ps(ptr);
    //p = *reinterpret_cast<const packed_t *>(ptr);
#elif defined(PVECF_ARM)
    // this may not even be supported (or maybe it is supported per VLDx)
    p = *reinterpret_cast<const packed_t *>(ptr);
#endif
}

// store aligned
template<> inline void vec4f_t::storea(float * ptr)
{
    *(reinterpret_cast<packed_t *>(ptr)) = p;
}
// store unaligned
template<> inline void vec4f_t::storeu(float * ptr)
{
    
#if defined(PVECF_INTEL)
    _mm_storeu_ps(ptr, p);
#elif defined(PVECF_ARM)
    // this may not even be supported (or maybe it is supported per VSTx)
    p = *reinterpret_cast<const packed_t *>(ptr);
#endif
}

#undef vec4f_t


#if defined(PVECF_INTEL)

// packed double precision floating point calculations are supported starting
// with SSE2
#if defined(SSE2) || defined(AVX)
//
// vec2d_t implementation
//
#define vec2d_t vecf_t<double,2,__m128d>

template<> inline vec2d_t::vecf_t() { p = math_t::zeroes(); }
template<> inline vec2d_t::vecf_t(double v0) { v[0] = v0; v[1] = 0.0; }
template<> inline vec2d_t::vecf_t(double v0, double v1) { v[0] = v0; v[1] = v1; }
template<> inline vec2d_t::vecf_t(std::initializer_list<real_t> l)
{
    p = math_t::zeroes();
    int count = int(l.size() <= 2 ? l.size() : 2);
    int i = 0; for(auto it = l.begin(); i < count; ++i, ++it) v[i] = *it;
}

template<> inline vec2d_t & vec2d_t::operator=(double v) { p = _mm_set1_pd(v); return *this; }

MEMBER_ARITH_OPS_(double,2,__m128d,_mm_,_pd)

template<> inline vec2d_t & vec2d_t::operator+=(double v)
{ p = _mm_add_pd(p, _mm_set1_pd(v)); return *this; }
template<> inline vec2d_t & vec2d_t::operator-=(double v)
{ p = _mm_sub_pd(p, _mm_set1_pd(v)); return *this; }
// scalar multiplication
template<> inline vec2d_t & vec2d_t::operator*=(double v)
{ p = _mm_mul_pd(p, _mm_set1_pd(v)); return *this; }
// scalar division
template<> inline vec2d_t & vec2d_t::operator/=(double v)
{ p = _mm_div_pd(p, _mm_set1_pd(v)); return *this; }

MEMBER_CMP_OPS_(double,2,__m128d,_mm_,_pd,0x3)

template<> inline __m128d vec2d_t::dot_packed(const vec2d_t & op2) const
{ return math_t::dot_packed(p, op2.p); }
template<> inline __m128d vec2d_t::cross_packed(const vec2d_t &) const
{ // this = {y,x}
  // perp(this) = {-x, y}
  return yx(math_t::zeroes() - p/*{-y,-x}*/, p); // {-x,y}
}
template<> inline __m128d vec2d_t::unit_cross_packed(const vec2d_t & v2) const
{ __m128d cross = cross_packed(v2);
  return _mm_mul_pd(cross, math_t::inv_sqrt_packed(math_t::dot_packed(cross, cross))); }

template<> inline double vec2d_t::dot(const vec2d_t & op2) const
{ return _mm_cvtsd_f64(dot_packed(op2)); }
template<> inline double vec2d_t::sqlen() const { return dot(*this); }
template<> inline __m128d vec2d_t::sqlen_packed() const { return dot_packed(*this); }
template<> inline double vec2d_t::len() const { return math_t::sqrt(dot(*this)); }
template<> inline __m128d vec2d_t::len_packed() const { return math_t::sqrt_packed(dot_packed(*this)); }
template<> inline void vec2d_t::normalize()
{ double sqlen_ = sqlen(); __m128d x = _mm_set1_pd(sqlen_);
  p = _mm_mul_pd(p, math_t::inv_sqrt_packed(x)); }
template<> inline vec2d_t & vec2d_t::normalize_packed()
{ p = _mm_mul_pd(p, math_t::inv_sqrt_packed(sqlen_packed())); }
template<> inline void vec2d_t::clamp_0_1()
{ p = _mm_max_pd(math_t::zeroes(), _mm_min_pd(p, math_t::ones())); }
// these ignore the second vector
template<> inline vec2d_t vec2d_t::cross(const vec2d_t & v2) const
{ return vec2d_t(cross_packed(v2)); }
template<> inline vec2d_t vec2d_t::unit_cross(const vec2d_t & v2) const
{ return vec2d_t(unit_cross_packed(v2)); }

FREE_ARITH_OPS_(double,2,__m128d,_mm_,_pd)

FREE_BIT_OPS_(double,2,__m128d,_mm_,_pd)

FREE_ARITHBIT_OPS_NAMED_(double, 2, __m128d, _mm_, _pd)


// flip all bits; result may not be floating points any more, primary use
// is for further processing of comparison results
inline vec2d_t operator~(const vec2d_t & v)
{ return vec2d_t(_mm_xor_pd(v.p, _mm_cmpeq_pd(vec2d_t::math_t::zeroes(), vec2d_t::math_t::zeroes()))); }

// unary minus
inline vec2d_t operator-(const vec2d_t & v)
{ return vec2d_t(_mm_sub_pd(math_t<double,__m128d>::zeroes(), v.p)); }

inline __m128d dot_packed(const vec2d_t & v1, const vec2d_t & v2)
{ return math_t<double,__m128d>::dot_packed(v1.p, v2.p); }
inline double dot(const vec2d_t & v1, const vec2d_t & v2)
{ return _mm_cvtsd_f64(dot_packed(v1, v2)); }
inline __m128d dot_packed_2(const vec2d_t & v00, const vec2d_t & v01,
                            const vec2d_t & v10, const vec2d_t & v11)
{ return xx(
      dot_packed(v00, v01), // {dot0,dot0}
      dot_packed(v10, v11)  // {dot1,dot1}
  ); // {dot0,dot1}
}

// result = origin + t * dir
inline vec2d_t eval_line(const vec2d_t & origin, double t, const vec2d_t & dir)
{ return vec2d_t(origin.p + (_mm_set1_pd(t) * dir.p)); }
inline vec2d_t eval_line_packed(const vec2d_t & origin, __m128d t, const vec2d_t & dir)
{ return vec2d_t(origin + (t * dir)); }


template<> inline vec2d_t vec2d_t::min_(const vec2d_t & v0, const vec2d_t & v1)
{ return vec2d_t(_mm_min_pd(v0.p, v1.p)); }
template<> inline vec2d_t vec2d_t::max_(const vec2d_t & v0, const vec2d_t & v1)
{ return vec2d_t(_mm_max_pd(v0.p, v1.p)); }

template<> inline void vec2d_t::abs_()
{ p = notAandB_(math_t::set1(-0.0), p); }
template<> inline vec2d_t vec2d_t::abs_() const
{ vec2d_t ret(p); ret.abs_(); return ret; }


template<> template<unsigned IDX> inline vec2d_t vec2d_t::elemAbs() const
{ vec2d_t ret(p); ret.elemAbs<IDX>(); return ret; }

template<> template<unsigned IDX> inline void vec2d_t::elemAbs()
{
    static_assert(IDX <= 1, "IDX out of range");
    packed_t masks(math_t::set1(-0.0));
    packed_t zeroes(math_t::zeroes());
    packed_t val =
        IDX == 0 ? xx(masks, zeroes)/*{-0,0}*/ : xx(zeroes, masks)/*{0,-0}*/;
    p = p ^ val;
}







template<> inline vec2d_t vec2d_t::isnan() const
{ vec2d_t v(_mm_cmpeq_pd(p, p)); return ~v; }

template<> inline bool vec2d_t::isnan_all() const
{ return _mm_movemask_pd(_mm_cmpeq_pd(_mm_cmpeq_pd(p, p), vec2d_t::math_t::zeroes())) == 0x3; }


#undef vec2d_t

#endif // SSE2 || AVX


#ifdef AVX
//
// vec4d_t implementation
//
template<> inline vec4d_t::vecf_t() { p = math_t::zeroes(); }
template<> inline vec4d_t::vecf_t(double v0) { v[0] = v0; v[1] = v[2] = v[3] = 0.0; }
template<> inline vec4d_t::vecf_t(double v0, double v1) { v[0] = v0; v[1] = v1; v[2] = v[3] = 0.0; }
template<> template<> inline vec4d_t::vecf_t(double v0, double v1, double v2) { v[0] = v0; v[1] = v1; v[2] = v2; v[3] = 0.0; }
template<> template<> inline vec4d_t::vecf_t(double v0, double v1, double v2, double v3) { v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3; }

template<> inline vec4d_t & vec4d_t::operator=(double v) { p = _mm256_set1_pd(v); return *this; }

MEMBER_ARITH_OPS_(vec4d_t,__m256d,_mm256_,_pd)

template<> inline vec4d_t & vec4d_t::operator+=(double v)
{ p = _mm256_add_pd(p, _mm256_set1_pd(v)); return *this; }
template<> inline vec4d_t & vec4d_t::operator-=(double v)
{ p = _mm256_sub_pd(p, _mm256_set1_pd(v)); return *this; }
// scalar multiplication
template<> inline vec4d_t & vec4d_t::operator*=(double v)
{ p = _mm256_mul_pd(p, _mm256_set1_pd(v)); return *this; }
// scalar division
template<> inline vec4d_t & vec4d_t::operator/=(double s)
{ p = _mm256_div_pd(p, _mm256_set1_pd(s)); return *this; }

MEMBER_CMP_OPS_(vec4d_t,__m256d,_mm256_,_pd,0xF)

template<> inline double vec4d_t::dot(const vec4d_t & op2) const
{ return _mm_cvtsd_f64(_mm256_castpd256_pd128(dot_packed(op2))); }
template<> inline __m256d vec4d_t::dot_packed(const vec4d_t & op2) const
{ return math_t::dot_packed(p, op2.p); }
template<> inline double vec4d_t::sqlen() const { return dot(*this); }
template<> inline __m256d vec4d_t::sqlen_packed() const { return dot_packed(*this); }
template<> inline double vec4d_t::len() const { return math_t::sqrt(dot(*this)); }
template<> inline __m256d vec4d_t::len_packed() const { return math_t::sqrt_packed(dot_packed(*this)); }
template<> inline void vec4d_t::normalize() {
    double sqlen_ = sqlen(); __m256d x = _mm256_set1_pd(sqlen_);
    p = _mm256_mul_pd(p, math_t::inv_sqrt_packed(x));
}
template<> inline void vec4d_t::normalize_packed()
{ p = _mm256_mul_pd(p, math_t::inv_sqrt_packed(sqlen_packed())); }
template<> inline void vec4d_t::clamp_0_1()
{ p = _mm256_max_pd(math_t::zeroes(), _mm256_min_pd(p, math_t::ones())); }
template<> inline vec4d_t vec4d_t::cross(const vec4d_t & v2) const
{ return vec4d_t(cross_packed(v2)); }
template<> inline __m256d vec4d_t::cross_packed(const vec4d_t & v2) const
{ return wxzy(p) * wyxz(v2.p) - wyxz(p) * wxzy(v2.p); }
/*
template<> inline __m256d vec4d_t::cross_packed(const vec4d_t & v2) const
{ // this = {w0,z0,y0,x0}, v2 = {w1,z1,y1,x1}
  // this x v2 = {0,x0y1-x1y0,x1z0-x0z1,y0z1-y1z0}
  return
      _mm256_sub_pd(
          _mm256_mul_pd(
              _mm256_shuffle_pd(   p,    p, _MM_SHUFFLE(3, 0, 2, 1)), // (0) = this.3021
              _mm256_shuffle_pd(v2.p, v2.p, _MM_SHUFFLE(3, 1, 0, 2))  // (1) = v2  .3102
          ), //   (0) {w0,x0,z0,y0}
             // * (1) {w1,y1,x1,z1} = {w0w1,x0y1,x1z0,y0z1} (a)
          _mm256_mul_pd(
              _mm256_shuffle_pd(   p,    p, _MM_SHUFFLE(3, 1, 0, 2)), // (2) = this.3102
              _mm256_shuffle_pd(v2.p, v2.p, _MM_SHUFFLE(3, 0, 2, 1))  // (3) = v2  .3021
          )  //   (2) {w0,y0,x0,z0}
             // * (3) {w1,x1,z1,y1} = {w0w1,x1y0,x0z1,y1z0} (b)
      ); // (a) - (b) = {0,x0y1-x1y0,x1z0-x0z1,y0z1-y1z0}
}
*/
template<> inline vec4d_t vec4d_t::unit_cross(const vec4d_t & v2) const
{ return vec4d_t(unit_cross_packed(v2)); }
template<> inline __m256d vec4d_t::unit_cross_packed(const vec4d_t & v2) const
{ __m256d cross = cross_packed(v2);
  return _mm256_mul_pd(cross, math_t::inv_sqrt_packed(math_t::dot_packed(cross, cross))); }

FREE_ARITH_OPS_(vec4d_t,__m256d,float,_mm256_,_pd)

FREE_BIT_OPS_(vec4d_t,__m256d,_mm256_,_pd)

// unary minus
inline vec4d_t operator-(const vec4d_t & v)
{ return vec4d_t(_mm256_sub_pd(math_t<double,__m256d>::zeroes(), v.p)); }

inline __m256d dot_packed(const vec4d_t & v1, const vec4d_t & v2)
{ return math_t<double,__m256d>::dot_packed(v1.p, v2.p); }
inline double dot(const vec4d_t & v1, const vec4d_t & v2)
{ return _mm_cvtsd_f64(_mm256_castpd256_pd128(dot_packed(v1, v2))); }
inline __m256d dot_packed_4(const vec4d_t & v00, const vec4d_t & v01,
                            const vec4d_t & v10, const vec4d_t & v11,
                            const vec4d_t & v20, const vec4d_t & v21,
                            const vec4d_t & v30, const vec4d_t & v31)
{
    __m256d dot0 = dot_packed(v00, v01); // {..,dot0}
    __m256d dot1 = dot_packed(v10, v11); // {..,dot1}
    __m256d dot2 = dot_packed(v20, v21); // {..,dot2}
    __m256d dot3 = dot_packed(v30, v31); // {..,dot3}

    return xzxz(
        xxxx(dot0, dot1), // {dot0,dot0,dot1,dot1}
        xxxx(dot2, dot3)  // {dot2,dot2,dot3,dot3}
    ); // {dot0,dot1,dot2,dot3}
    /*
    return _mm256_shuffle_pd(
        _mm256_shuffle_pd(dot0, dot1, _MM_SHUFFLE2(0,0)), // {dot1,dot1,dot0,dot0}
        _mm256_shuffle_pd(dot2, dot3, _MM_SHUFFLE2(0,0)), // {dot3,dot3,dot2,dot2}
        _MM_SHUFFLE(2, 0, 2, 0)
    ); // {dot3,dot2,dot1,dot0}
    */
}

// result = origin + t * dir
inline vec4d_t eval_line(const vec4d_t & origin, double t, const vec4d_t & dir)
{ return vec4d_t(origin.p + (_mm256_set1_pd(t) * dir.p)); }
inline vec4d_t eval_line_packed(const vec4d_t & origin, __m256d t, const vec4d_t & dir)
{ return vec4d_t(origin + (t * dir)); }


//
// vec8f_t implementation
//
template<> inline vec8f_t::vecf_t() { p = math_t::zeroes(); }
template<> inline vec8f_t::vecf_t(float v0) { v[0] = v0; v[1] = v[2] = v[3] = 0.0f; }
template<> inline vec8f_t::vecf_t(float v0, float v1) { v[0] = v0; v[1] = v1; v[2] = v[3] = 0.0f; }
template<> template<> inline vec8f_t::vecf_t(float v0, float v1, float v2) { v[0] = v0; v[1] = v1; v[2] = v2; v[3] = 0.0f; }
template<> template<> inline vec8f_t::vecf_t(float v0, float v1, float v2, float v3) { v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3; }
template<> template<> inline vec8f_t::vecf_t(float v0, float v1, float v2, float v3, float v4) { v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3; v[4] = v4; }
template<> template<> inline vec8f_t::vecf_t(float v0, float v1, float v2, float v3, float v4, float v5) { v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3; v[4] = v4; v[5] = v5; }
template<> template<> inline vec8f_t::vecf_t(float v0, float v1, float v2, float v3, float v4, float v5, float v6) { v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3; v[4] = v4; v[5] = v5; v[6] = v6; }
template<> template<> inline vec8f_t::vecf_t(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7) { v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3; v[4] = v4; v[5] = v5; v[6] = v6; v[7] = v7; }

template<> inline vec8f_t & vec8f_t::operator=(float v) { p = _mm256_set1_ps(v); return *this; }

MEMBER_ARITH_OPS_(vec8f_t, __m256, _mm256_, _ps)

template<> inline vec8f_t & vec8f_t::operator+=(float v)
{ p = _mm256_add_ps(p, _mm256_set1_ps(v)); return *this; }
template<> inline vec8f_t & vec8f_t::operator-=(float v)
{ p = _mm256_sub_ps(p, _mm256_set1_ps(v)); return *this; }
// scalar multiplication
template<> inline vec8f_t & vec8f_t::operator*=(float v)
{ p = _mm256_mul_ps(p, _mm256_set1_ps(v)); return *this; }
// scalar division
template<> inline vec8f_t & vec8f_t::operator/=(float v)
{ p = _mm256_div_ps(p, _mm256_set1_ps(v)); return *this; }

MEMBER_CMP_OPS_(vec8f_t, __m256, _mm256_, _ps, 0xFF)

template<> inline float vec8f_t::dot(const vec8f_t & op2) const
{ return _mm_cvtss_f32(_mm256_castps256_ps128(dot_packed(op2))); }
template<> inline __m256 vec8f_t::dot_packed(const vec8f_t & op2) const
{ return math_t::dot_packed(p, op2.p); }
template<> inline float vec8f_t::sqlen() const { return dot(*this); }
template<> inline __m256 vec8f_t::sqlen_packed() const { return dot_packed(*this); }
template<> inline float vec8f_t::len() const { return math_t::sqrt(dot(*this)); }
template<> inline __m256 vec8f_t::len_packed() const { return math_t::sqrt_packed(dot_packed(*this)); }
template<> inline void vec8f_t::normalize() {
    float sqlen_ = sqlen(); __m256 x = _mm256_set1_ps(sqlen_);
    p = _mm256_mul_ps(p, math_t::inv_sqrt_packed(x));
}
template<> inline void vec8f_t::normalize_packed()
{ p = _mm256_mul_ps(p, math_t::inv_sqrt_packed(sqlen_packed())); }
template<> inline void vec8f_t::clamp_0_1()
{ p = _mm256_max_ps(math_t::zeroes(), _mm256_min_ps(p, math_t::ones())); }
/*
template<> inline vec8f_t vec8f_t::cross(const vec8f_t &) const;
template<> inline __m256 vec8f_t::cross_packed(const vec8f_t &) const;
template<> inline vec8f_t vec8f_t::unit_cross(const vec8f_t &) const;
template<> inline __m256 vec8f_t::unit_cross_packed(const vec8f_t &) const;
*/

FREE_ARITH_OPS_(vec8f_t,__m256,float,_mm256_,_ps)

FREE_BIT_OPS_(vec8f_t,__m256,_mm256_,_ps)

// unary minus
inline vec8f_t operator-(const vec8f_t & v)
{ return vec8f_t(_mm256_sub_ps(math_t<float,__m256>::zeroes(), v.p)); }

inline __m256 dot_packed(const vec8f_t & v1, const vec8f_t & v2)
{ return math_t<float,__m256>::dot_packed(v1.p, v2.p); }
inline float dot(const vec8f_t & v1, const vec8f_t & v2)
{ return _mm_cvtss_f32(_mm256_castps256_ps128(dot_packed(v1, v2))); }
inline __m256 dot_packed_4(const vec8f_t & v00, const vec8f_t & v01,
                           const vec8f_t & v10, const vec8f_t & v11,
                           const vec8f_t & v20, const vec8f_t & v21,
                           const vec8f_t & v30, const vec8f_t & v31)
{
    __m256 tmp0 = v00.dot_packed(v01); // {d0{8}}
    __m256 tmp1 = v10.dot_packed(v11); // {d1{8}}
    __m256 tmp2 = v20.dot_packed(v21); // {d2{8}}
    __m256 tmp3 = v30.dot_packed(v31); // {d3{8}}

    __m256 tmp4 =
        _mm256_shuffle_ps(
            _mm256_shuffle_ps(tmp2, tmp3, _MM_SHUFFLE(0, 0, 0, 0)), // {d3{2},d3{2},d2{2},d2{2}}
            _mm256_shuffle_ps(tmp0, tmp1, _MM_SHUFFLE(0, 0, 0, 0)), // {d1{2},d1{2},d0{2},d0{2}}
            _MM_SHUFFLE(2, 0, 2, 0)
        ); // {d3{2},d2{2},d1{2},d0{2}}
    __m128 tmp0_128 = _mm256_extractf128_ps(tmp4, 1); // {d3{2},d2{2}}
    __m128 tmp1_128 = _mm256_extractf128_ps(tmp4, 0); // {d1{2},d0{2}}
    tmp0_128 =
        _mm_shuffle_ps(
            tmp0_128,
            tmp1_128,
            _MM_SHUFFLE(2, 0, 2, 0)
        ); // {d3,d2,d1,d0}
    return
        _mm256_insertf128_ps(
            _mm256_insertf128_ps(
                math_t<float,__m256>::zeroes(),
                tmp0_128,
                0
            ), // {0,0,0,0,d3,d2,d1,d0}
            tmp0_128,
            1
        ); // {d3,d2,d1,d0,d3,d2,d1,d0}
}
inline __m256 dot_packed_8(const vec8f_t & v00, const vec8f_t & v01,
                           const vec8f_t & v10, const vec8f_t & v11,
                           const vec8f_t & v20, const vec8f_t & v21,
                           const vec8f_t & v30, const vec8f_t & v31,
                           const vec8f_t & v40, const vec8f_t & v41,
                           const vec8f_t & v50, const vec8f_t & v51,
                           const vec8f_t & v60, const vec8f_t & v61,
                           const vec8f_t & v70, const vec8f_t & v71)
{
    __m256 tmp0 =
        _mm256_shuffle_ps(
            _mm256_shuffle_ps(
                v00.dot_packed(v01), // {d0{8}}
                v10.dot_packed(v11), // {d1{8}}
                _MM_SHUFFLE(0, 0, 0, 0)
            ), // {d1{2},d1{2},d0{2},d0{2}}
            _mm256_shuffle_ps(
                v20.dot_packed(v21), // {d2{8}}
                v30.dot_packed(v31), // {d3{8}}
                _MM_SHUFFLE(0, 0, 0, 0)
            ), // {d3{2},d3{2},d2{2},d2{2}}
            _MM_SHUFFLE(2, 0, 2, 0)
        ); // {d3{2},d2{2},d1{2},d0{2}}
    __m256 tmp1 =
        _mm256_shuffle_ps(
            _mm256_shuffle_ps(
                v40.dot_packed(v41), // {d4{8}}
                v50.dot_packed(v51), // {d5{8}}
                _MM_SHUFFLE(0, 0, 0, 0)
            ), // {d5{2},d5{2},d4{2},d4{2}}
            _mm256_shuffle_ps(
                v60.dot_packed(v61), // {d6{8}}
                v70.dot_packed(v71), // {d7{8}}
                _MM_SHUFFLE(0, 0, 0, 0)
            ), // {d7{2},d7{2},d6{2},d6{2}}
            _MM_SHUFFLE(2, 0, 2, 0)
        ); // {d7{2},d6{2},d5{2},d4{2}}
    return
        _mm256_insertf128_ps(
            _mm256_insertf128_ps(
                math_t<float,__m256>::zeroes(),
                _mm_shuffle_ps(
                    _mm256_extractf128_ps(tmp0, 0), // {d1{2},d0{2}}
                    _mm256_extractf128_ps(tmp0, 1), // {d3{2},d2{2}}
                    _MM_SHUFFLE(2, 0, 2, 0)
                ), // {d3,d2,d1,d0}
                0
            ), // {0,0,0,0,d3,d2,d1,d0}
            _mm_shuffle_ps(
                _mm256_extractf128_ps(tmp1, 0), // {d5{2},d4{2}}
                _mm256_extractf128_ps(tmp1, 1), // {d7{2},d6{2}}
                _MM_SHUFFLE(2, 0, 2, 0)
            ), // {d7,d6,d5,d4}
            1
        ); // {d7,d6,d5,d4,d3,d2,d1,d0}
}

// result = origin + t * dir
inline vec8f_t eval_line(const vec8f_t & origin, float t, const vec8f_t & dir)
{ return vec8f_t(origin.p + (_mm256_set1_ps(t) * dir.p)); }
inline vec8f_t eval_line_packed(const vec8f_t & origin, __m256 t, const vec8f_t & dir)
{ return vec8f_t(origin + (t * dir)); }

#endif // AVX

#endif // defined(PVECF_INTEL)


#if defined(_MSC_VER)
#define PVECF_ALIGN(x) __declspec(align(x))
#elif defined(__GNUC__)
#define PVECF_ALIGN(x) __attribute__((aligned (x)))
#endif

//typedef vecf_t<float,2,__m128> __declspec(align(16)) vec2f_t;
//typedef vecf_t<float,3,__m128> __declspec(align(16)) vec3f_t;
#if defined(PVECF_INTEL)
typedef vecf_t<float,4,__m128> PVECF_ALIGN(16) vec4f_t;
#elif defined(PVECF_ARM)
typedef vecf_t<float,4,__n128> PVECF_ALIGN(8) vec4f_t;
#endif


#if defined(PVECF_INTEL)
// packed double precision floating point calculations are supported starting
// with SSE2
#if defined(SSE2) || defined(AVX)
typedef vecf_t<double,2,__m128d> __declspec(align(16)) vec2d_t;
#endif

////typedef vecf_t<double,3,__m128d> vec3d_t;
// TODO: add a complex_2d_t class with operators defined otherwise
//       (elementwise-add, -sub, multiplication, division)
// then: do the same for float, use __m128 there but just ignore the upper
//       two floats in all operations
// this needs to be coupled with a nicer interface specifically designed for
// complex numbers (e.g. a conj() operation, etc.)
#ifdef AVX
typedef vecf_t<float,8,__m256> __declspec(align(32)) vec8f_t;
typedef vecf_t<double,4,__m256d> __declspec(align(32)) vec4d_t;
#endif // AVX


// packed double precision floating point calculations are supported starting
// with SSE2
#if defined(SSE2) || defined(AVX)
#define SWIZZLE_DOUBLE_2_(a,b) \
template<> inline vec2d_t a##b(const vec2d_t & v) { return vec2d_t(a##b(v.p)); } \
template<> inline vec2d_t a##b(const vec2d_t & v1, const vec2d_t & v2) { return vec2d_t(a##b(v1.p, v2.p)); }
#define SWIZZLE_DOUBLE_2    \
    SWIZZLE_DOUBLE_2_(x, x) \
    SWIZZLE_DOUBLE_2_(x, y) \
    SWIZZLE_DOUBLE_2_(y, x) \
    SWIZZLE_DOUBLE_2_(y, y)
SWIZZLE_DOUBLE_2
#undef SWIZZLE_DOUBLE_2_
#undef SWIZZLE_DOUBLE_2
#endif // SSE2 || AVX


#ifdef AVX

#define SWIZZLE_DOUBLE_4____(a,b,c,d) \
template<> inline vec4d_t a##b##c##d(const vec4d_t & v) { return vec4d_t(a##b##c##d(v.p)); } \
template<> inline vec4d_t a##b##c##d(const vec4d_t & v0, const vec4d_t & v1) { return vec4d_t(a##b##c##d(v0.p, v1.p)); }
#define SWIZZLE_DOUBLE_4___(a,b,c) \
    SWIZZLE_DOUBLE_4____(a,b,c,x)  \
    SWIZZLE_DOUBLE_4____(a,b,c,y)  \
    SWIZZLE_DOUBLE_4____(a,b,c,z)  \
    SWIZZLE_DOUBLE_4____(a,b,c,w)
#define SWIZZLE_DOUBLE_4__(a,b) \
    SWIZZLE_DOUBLE_4___(a,b,x)  \
    SWIZZLE_DOUBLE_4___(a,b,y)  \
    SWIZZLE_DOUBLE_4___(a,b,z)  \
    SWIZZLE_DOUBLE_4___(a,b,w)
#define SWIZZLE_DOUBLE_4_(a) \
    SWIZZLE_DOUBLE_4__(a,x)  \
    SWIZZLE_DOUBLE_4__(a,y)  \
    SWIZZLE_DOUBLE_4__(a,z)  \
    SWIZZLE_DOUBLE_4__(a,w)
#define SWIZZLE_DOUBLE_4 \
    SWIZZLE_DOUBLE_4_(x) \
    SWIZZLE_DOUBLE_4_(y) \
    SWIZZLE_DOUBLE_4_(z) \
    SWIZZLE_DOUBLE_4_(w)
SWIZZLE_DOUBLE_4
#undef SWIZZLE_DOUBLE_4____
#undef SWIZZLE_DOUBLE_4___
#undef SWIZZLE_DOUBLE_4__
#undef SWIZZLE_DOUBLE_4_
#undef SWIZZLE_DOUBLE_4

#endif // AVX

#endif // defined(PVECF_INTEL)


//#define HAS_LOOP_UNROLLING



// temporarily disabled because the shuffle operations for 2 operands are not implemented yet for ARM NEON
//#if defined(PVECF_INTEL)

//
// mat_t general template (NxN matrix)
//
template<typename t_real, unsigned N, typename t_packed>
class mat_t
{
public:
    typedef t_real real_t;
    typedef t_packed packed_t;

    typedef typename math::math_t<t_real,t_packed> math_t;
    typedef typename math::vecf_t<t_real,N,t_packed> vecf_t;

    mat_t();
    mat_t(const real_t * p)
    { for(int row = 0; row < N; ++row)
          for(int col = 0; col < N; ++col)
              v[row*N+col] = p[row*N+col]; }
    explicit mat_t(const packed_t * v)
    { for(int row = 0; row < N; ++row) m[row] = v[row]; }
    explicit mat_t(const vecf_t * v)
    { for(int row = 0; row < N; ++row) m[row] = v[row].p; }

    static inline mat_t identity();
    static inline mat_t zero() { return mat_t(); }

    inline const real_t * operator[](size_t row) const { return v[row*N]; }
    inline       real_t * operator[](size_t row)       { return v[row*N]; }
    
    inline real_t   operator()(size_t row, size_t col) const { return v[row*N+col]; }
    inline real_t & operator()(size_t row, size_t col) { return v[row*N+col]; }
    
    inline real_t   elem(size_t row, size_t col) const { return v[row*N+col]; }
    inline real_t & elem(size_t row, size_t col) { return v[row*N+col]; }
    
    inline operator const real_t *() const { return v; }
    inline operator real_t *() const { return v; }

    inline void set_identity();
    inline void set_zero() { for(int i = 0; i < N; ++i) m[i] = math_t::zeroes(); }

    inline void set_row(size_t row, const vecf_t & vec)
    { if(row >= N) return;
      m[row] = vec.p; }
    inline void set_col(size_t col, const vecf_t & vec)
    { if(col >= 4) return;
      for(int i = 0; i < N; ++i) v[i*N+col] = vec.v[i]; }

    inline vecf_t row(size_t row) const;
    inline vecf_t col(size_t col) const;

    inline bool operator==(const mat_t &) const;
    inline bool operator!=(const mat_t &) const;
    inline bool operator<(const mat_t &) const;
    inline bool operator<=(const mat_t &) const;
    inline bool operator>(const mat_t &) const;
    inline bool operator>=(const mat_t &) const;

#ifdef HAS_LOOP_UNROLLING
    inline mat_t & operator+=(const mat_t & mat)
    { for(int i = 0; i < N; ++i) m[i] = m[i] + mat.m[i];
      return *this; }
    inline mat_t & operator-=(const mat_t & mat)
    { for(int i = 0; i < N; ++i) m[i] = m[i] - mat.m[i];
      return *this; }
    // pre-multiplication (as matrices are stored in row major order)
    inline mat_t & mat_t::operator*=(const mat_t & mat)
    { mat_t tmp_mat;
      vecf_t tmp_vec;
      for(int i = 0; i < N; ++i) {
          for(int j = 0; j < N; ++j)
              tmp_vec[j] = v[j*N+i];
            
          for(int j = 0; j < N; ++j)
              tmp_mat.v[j*N+i] = vec4f_t(mat.m[j]).dot(tmp_vec);
      }
      for(int i = 0; i < N; ++i)
          m[i] = tmp_mat.m[i];
      return *this; }
    inline mat_t & operator*=(real_t s)
    { for(int i = 0; i < N; ++i) m[i] = m[i] * s; return *this; }
    inline mat_t & operator/=(real_t s)
    { for(int i = 0; i < N; ++i) m[i] = m[i] / s; return *this; }
    inline mat_t transpose() const
    { mat_t ret;
      for(int i = 0; i < N; ++i) ret.set_col(i, row(i));
      return ret; }
    inline void transpose()
    { mat_t tmp;
      for(int i = 0; i < N; ++i) ret.set_col(i, row(i));
      for(int i = 0; i < N; ++i) m[i] = tmp.m[i]; }
#else
    // manually unrolled in specializations
    inline mat_t & operator+=(const mat_t &);
    inline mat_t & operator-=(const mat_t &);
    // pre-multiplication (as matrices are stored in row major order)
    inline mat_t & operator*=(const mat_t &);
    inline mat_t & operator*=(real_t s);
    inline mat_t & operator/=(real_t s);
    inline mat_t transpose() const;
    inline void transpose();
    inline vecf_t operator*(const vecf_t & vec) const;
#endif // !HAS_LOOP_UNROLLING

    inline mat_t transpose_times(const mat_t & mat) const
    { mat_t ret(transpose()); ret *= mat; return ret;  }
    inline mat_t times_transpose(const mat_t & mat) const
    { mat_t ret(*this); ret *= mat.transpose(); return ret; }
    inline real_t det() const;
    inline mat_t adjugate() const;
    inline void adjugate();
    inline mat_t inverse() const;
    inline void inverse();

    inline void transform(vecf_t & vec) const;
    inline vecf_t transform(const vecf_t & vec) const;
    inline void transform_many(vecf_t * vecs, size_t count) const
    {
        for(unsigned i = 0; i < count; ++i) {
            transform(vecs[i]);
        }
    }


    union {
        real_t v[N*N];
        packed_t m[N];
    };
};

#if defined(PVECF_INTEL)
typedef mat_t<float,4,__m128> mat4f_t;
#elif defined(PVECF_ARM)
typedef mat_t<float,4,__n128> mat4f_t;
#endif


#if defined(PVECF_INTEL)

// packed double precision floating point calculations are supported starting
// with SSE2
#if defined(SSE2) || defined(AVX)
typedef mat_t<double,2,__m128d> mat2d_t;
#endif

#ifdef AVX
typedef mat_t<float,8,__m256> mat8f_t;
typedef mat_t<double,4,__m256d> mat4d_t;
#endif // AVX

#endif // defined(PVECF_INTEL)


//
// mat4f_t implementation
//
template<> inline mat4f_t::mat_t()
{ m[0] = m[1] = m[2] = m[3] = math_t::zeroes(); }
template<> inline vec4f_t mat4f_t::row(size_t row) const
{ return vec4f_t(m[row]); }
template<> inline void mat4f_t::set_identity()
{
    m[0] = m[1] = m[2] = m[3] = math_t::zeroes();
    vec4f_t row_(row(0));
    row_[0] = 1.0f;
    set_row(0, row_);

    row_ = row(1);
    row_[1] = 1.0f;
    set_row(1, row_);

    row_ = row(2);
    row_[2] = 1.0f;
    set_row(2, row_);

    row_ = row(3);
    row_[3] = 1.0f;
    set_row(3, row_);
}
template<> inline mat4f_t mat4f_t::identity()
{
    mat4f_t ret;
    ret.set_identity();
    return ret;
}
template<> inline vec4f_t mat4f_t::col(size_t col) const
{
    switch(col) {
    case 0:
        return vec4f_t(
            xzxz(
                xxxx(
                    xxxx(m[0], m[0]), // {x0,x0,x0,x0}
                    xxxx(m[1], m[1])  // {x1,x1,x1,x1}
                ), // {x0,x0,x1,x1}
                xxxx(
                    xxxx(m[2], m[2]), // {x2,x2,x2,x2}
                    xxxx(m[3], m[3])  // {x3,x3,x3,x3}
                )  // {x2,x2,x3,x3}
            ) // {x0,x1,x2,x3}
        );
    case 1:
        return vec4f_t(
            xzxz(
                xxxx(
                    yyyy(m[0], m[0]), // {y0,y0,y0,y0}
                    yyyy(m[1], m[1])  // {y1,y1,y1,y1}
                ), // {y0,y0,y1,y1}
                xxxx(
                    yyyy(m[2], m[2]), // {y2,y2,y2,y2}
                    yyyy(m[3], m[3])  // {y3,y3,y3,y3}
                )  // {y2,y2,y3,y3}
            ) // {y0,y1,y2,t3}
        );
    case 2:
        return vec4f_t(
            xzxz(
                xxxx(
                    zzzz(m[0], m[0]), // {z0,z0,z0,z0}
                    zzzz(m[1], m[1])  // {z1,z1,z1,z1}
                ), // {z0,z0,z1,z1}
                xxxx(
                    zzzz(m[2], m[2]), // {z2,z2,z2,z2}
                    zzzz(m[3], m[3])  // {z3,z3,z3,z3}
                )  // {z2,z2,z3,Z3}
            ) // {z0,z1,z2,z3}
        );
    case 3:
        return vec4f_t(
            xzxz(
                xxxx(
                    wwww(m[0], m[0]), // {w0,w0,w0,w0}
                    wwww(m[1], m[1])  // {w1,w1,w1,w1}
                ), // {w0,w0,w1,w1}
                xxxx(
                    wwww(m[2], m[2]), // {w2,w2,w2,w2}
                    wwww(m[3], m[3])  // {w3,w3,w3,w3}
                )  // {w2,w2,w3,w3}
            ) // {w0,w1,w2,w3}
        );
    }
    return vec4f_t(); // zero vector
}
#define MAT4F_CMP_OP(op,mat,vec) \
template<> inline bool mat::operator op(const mat & mat_) const \
{ return (vec(m[0]) op mat_.m[0]) && (vec(m[1]) op mat_.m[1]) && \
         (vec(m[2]) op mat_.m[2]) && (vec(m[3]) op mat_.m[3]); }
MAT4F_CMP_OP(==,mat4f_t,vec4f_t)
MAT4F_CMP_OP(!=,mat4f_t,vec4f_t)
MAT4F_CMP_OP(< ,mat4f_t,vec4f_t)
MAT4F_CMP_OP(<=,mat4f_t,vec4f_t)
MAT4F_CMP_OP(> ,mat4f_t,vec4f_t)
MAT4F_CMP_OP(>=,mat4f_t,vec4f_t)
#undef MAT4F_CMP_OP

#ifndef HAS_LOOP_UNROLLING
template<> inline mat4f_t & mat4f_t::operator+=(const mat4f_t & mat)
{ m[0] = m[0] + mat.m[0]; m[1] = m[1] + mat.m[1];
  m[2] = m[2] + mat.m[2]; m[3] = m[3] + mat.m[3];
  return *this; }
template<> inline mat4f_t & mat4f_t::operator-=(const mat4f_t & mat)
{ m[0] = m[0] - mat.m[0]; m[1] = m[1] - mat.m[1];
  m[2] = m[2] - mat.m[2]; m[3] = m[3] - mat.m[3];
  return *this; }

// temporarily disabled because the shuffle operations for 2 operands are not implemented yet for ARM NEON
//#if defined(PVECF_INTEL)
template<> inline mat4f_t mat4f_t::transpose() const
{ mat4f_t tmp_mat;
  mat4f_t::packed_t tmp0 = xyxy(m[0], m[1]); mat4f_t::packed_t tmp1 = xyxy(m[2], m[3]);
  mat4f_t::packed_t tmp2 = zwzw(m[0], m[1]); mat4f_t::packed_t tmp3 = zwzw(m[2], m[3]);
  tmp_mat.m[0] = xzxz(tmp0, tmp1); tmp_mat.m[1] = ywyw(tmp0, tmp1);
  tmp_mat.m[2] = xzxz(tmp2, tmp3); tmp_mat.m[3] = ywyw(tmp2, tmp3);
  return tmp_mat; }
template<> inline void mat4f_t::transpose()
{ // [a b c d]    [a e i m]
  // [e f g h]    [b f j n]
  // [i j k l] -> [c g k o]
  // [m n o p]    [d h l p]
  mat4f_t::packed_t tmp0 = xyxy(m[0], m[1]); // {a,b,e,f}
  mat4f_t::packed_t tmp1 = xyxy(m[2], m[3]); // {i,j,m,n}
  mat4f_t::packed_t tmp2 = zwzw(m[0], m[1]); // {c,d,g,h}
  mat4f_t::packed_t tmp3 = zwzw(m[2], m[3]); // {k,l,o,p}
  m[0] = xzxz(tmp0, tmp1); // {a,e,i,m}
  m[1] = ywyw(tmp0, tmp1); // {b,f,j,n}
  m[2] = xzxz(tmp2, tmp3); // {c,g,k,o}
  m[3] = ywyw(tmp2, tmp3); // {d,h,l,p}
}
//#endif

template<>
/*
#if defined(_MSC_VER)
__forceinline
#elif defined(__GNUC__)
__attribute__((always_inline))
#endif
*/
inline
mat4f_t & mat4f_t::operator*=(const mat4f_t & mat)
/*
{ mat4f_t tmp_mat;
  vec4f_t tmp_vec;
  tmp_vec[0] = mat.v[ 0]; tmp_vec[1] = mat.v[ 4];
  tmp_vec[2] = mat.v[ 8]; tmp_vec[3] = mat.v[12];
  tmp_mat.v[ 0] = vec4f_t(m[0]).dot(tmp_vec);
  tmp_mat.v[ 4] = vec4f_t(m[1]).dot(tmp_vec);
  tmp_mat.v[ 8] = vec4f_t(m[2]).dot(tmp_vec);
  tmp_mat.v[12] = vec4f_t(m[3]).dot(tmp_vec);
  
  tmp_vec[0] = mat.v[ 1]; tmp_vec[1] = mat.v[ 5];
  tmp_vec[2] = mat.v[ 9]; tmp_vec[3] = mat.v[13];
  tmp_mat.v[ 1] = vec4f_t(m[0]).dot(tmp_vec);
  tmp_mat.v[ 5] = vec4f_t(m[1]).dot(tmp_vec);
  tmp_mat.v[ 9] = vec4f_t(m[2]).dot(tmp_vec);
  tmp_mat.v[13] = vec4f_t(m[3]).dot(tmp_vec);
  
  tmp_vec[0] = mat.v[ 2]; tmp_vec[1] = mat.v[ 6];
  tmp_vec[2] = mat.v[10]; tmp_vec[3] = mat.v[14];
  tmp_mat.v[ 2] = vec4f_t(m[0]).dot(tmp_vec);
  tmp_mat.v[ 6] = vec4f_t(m[1]).dot(tmp_vec);
  tmp_mat.v[10] = vec4f_t(m[2]).dot(tmp_vec);
  tmp_mat.v[14] = vec4f_t(m[3]).dot(tmp_vec);
  
  tmp_vec[0] = mat.v[ 3]; tmp_vec[1] = mat.v[ 7];
  tmp_vec[2] = mat.v[11]; tmp_vec[3] = mat.v[15];
  tmp_mat.v[ 3] = vec4f_t(m[0]).dot(tmp_vec);
  tmp_mat.v[ 7] = vec4f_t(m[1]).dot(tmp_vec);
  tmp_mat.v[11] = vec4f_t(m[2]).dot(tmp_vec);
  tmp_mat.v[15] = vec4f_t(m[3]).dot(tmp_vec);
  m[0] = tmp_mat.m[0]; m[1] = tmp_mat.m[1];
  m[2] = tmp_mat.m[2]; m[3] = tmp_mat.m[3];
  return *this; }
*/
{
    /*
    mat4f_t mat_transposed(mat);
    mat_transposed.transpose();
    mat4f_t tmp_mat;

    tmp_mat.m[0] =
        dot_packed_4(
            m[0], mat_transposed.m[0],
            m[0], mat_transposed.m[1],
            m[0], mat_transposed.m[2],
            m[0], mat_transposed.m[3]
        );
    tmp_mat.m[1] =
        dot_packed_4(
            m[1], mat_transposed.m[0],
            m[1], mat_transposed.m[1],
            m[1], mat_transposed.m[2],
            m[1], mat_transposed.m[3]
        );
    tmp_mat.m[2] =
        dot_packed_4(
            m[2], mat_transposed.m[0],
            m[2], mat_transposed.m[1],
            m[2], mat_transposed.m[2],
            m[2], mat_transposed.m[3]
        );
    tmp_mat.m[3] =
        dot_packed_4(
            m[3], mat_transposed.m[0],
            m[3], mat_transposed.m[1],
            m[3], mat_transposed.m[2],
            m[3], mat_transposed.m[3]
        );
    // {dot0,dot1,dot2,dot3}
    m[0] = tmp_mat.m[0]; m[1] = tmp_mat.m[1];
    m[2] = tmp_mat.m[2]; m[3] = tmp_mat.m[3];
    return *this;
    */

#if 1
    mat4f_t mat_transposed(*this);
    mat_transposed.transpose();
    mat4f_t tmp_mat;
    
    /*
    dot_packed(a,b):
        { __m128 tmp = a * b; // {ae,bf,cg,dh}
          tmp = tmp + yxwz(tmp); // {ae+bf,bf+ae,cg+dh,dh+cg}
          return tmp + zwyx(tmp); // {ae+bf+cg+dh.bf+ae+dh+cg,cg+dh+bf+ae,dh+cg+ae+bf}
        }
    */
    tmp_mat.m[0] =
        xzxz(
            xxxx(
               math_t::dot_packed(mat.m[0], mat_transposed.m[0]), // {..,dot0}
               math_t::dot_packed(mat.m[0], mat_transposed.m[1])  // {..,dot1}
            ), // {dot0,dot0,dot1,dot1}
            xxxx(
               math_t::dot_packed(mat.m[0], mat_transposed.m[2]), // {..,dot2}
               math_t::dot_packed(mat.m[0], mat_transposed.m[3])  // {..,dot3}
            )  // {dot2,dot2,dot3,dot3}
        ); // {dot0,dot1,dot2,dot3}
    tmp_mat.m[1] =
        xzxz(
            xxxx(
               math_t::dot_packed(mat.m[1], mat_transposed.m[0]),
               math_t::dot_packed(mat.m[1], mat_transposed.m[1])
            ),
            xxxx(
               math_t::dot_packed(mat.m[1], mat_transposed.m[2]),
               math_t::dot_packed(mat.m[1], mat_transposed.m[3])
            )
        );
    tmp_mat.m[2] =
        xzxz(
            xxxx(
               math_t::dot_packed(mat.m[2], mat_transposed.m[0]),
               math_t::dot_packed(mat.m[2], mat_transposed.m[1])
            ),
            xxxx(
               math_t::dot_packed(mat.m[2], mat_transposed.m[2]),
               math_t::dot_packed(mat.m[2], mat_transposed.m[3])
            )
        );
    tmp_mat.m[3] =
        xzxz(
            xxxx(
               math_t::dot_packed(mat.m[3], mat_transposed.m[0]),
               math_t::dot_packed(mat.m[3], mat_transposed.m[1])
            ),
            xxxx(
               math_t::dot_packed(mat.m[3], mat_transposed.m[2]),
               math_t::dot_packed(mat.m[3], mat_transposed.m[3])
            )
        );
    //_mm_stream_ps(reinterpret_cast<float *>(&m[0]), tmp_mat.m[0]);
    //_mm_stream_ps(reinterpret_cast<float *>(&m[1]), tmp_mat.m[1]);
    //_mm_stream_ps(reinterpret_cast<float *>(&m[2]), tmp_mat.m[2]);
    //_mm_stream_ps(reinterpret_cast<float *>(&m[3]), tmp_mat.m[3]);
    m[0] = tmp_mat.m[0]; m[1] = tmp_mat.m[1];
    m[2] = tmp_mat.m[2]; m[3] = tmp_mat.m[3];
#endif
#if 0 // slower than the above (and not updated for pre-multiplication!)
    __m128 rowa[4] = { m[0], m[1], m[2], m[3] };
    __m128 tmp0 = xyxy(mat.m[0], mat.m[1]); // {q,r,u,v}
    __m128 tmp1 = xyxy(mat.m[2], mat.m[3]); // {y,z,C,D}
    __m128 tmp2 = zwzw(mat.m[0], mat.m[1]); // {s,t,w,x}
    __m128 tmp3 = zwzw(mat.m[2], mat.m[3]); // {A,B,E,F}
    __m128 colb[4] = {
        xzxz(tmp0, tmp1), // {q,u,y,C}
        ywyw(tmp0, tmp1), // {r,v,z,D}
        xzxz(tmp2, tmp3), // {s,w,A,E}
        ywyw(tmp2, tmp3)  // {t,x,B,F}
    };
    
    //tmp = tmp + yxwz(tmp); // {ae+bf,bf+ae,cg+dh,dh+cg}
    //return tmp + zwyx(tmp); // {ae+bf+cg+dh.bf+ae+dh+cg,cg+dh+bf+ae,dh+cg+ae+bf}
    
    for(unsigned i = 0; i < 4; ++i) {
        tmp0 = rowa[0] * colb[i];                   // {aq,bu,cy,dC}
        tmp0 = tmp0 + yxwz(tmp0);               // {aq+bu,bu+aq,cy+dC,dC+cy}
        _mm_store_ss(&v[ 0+i], tmp0 + zwyx(tmp0)); // {aq+bu+cy+dC,..}

        tmp1 = rowa[1] * colb[i];                   // {eq,fu,gy,hC}
        tmp1 = tmp1 + yxwz(tmp1);               // {eq+fu,fu+eq,gy+hC,hC+gy}
        _mm_store_ss(&v[ 4+i], tmp1 + zwyx(tmp1)); // {eq+fu+gy+hC,..}

        tmp2 = rowa[2] * colb[i];                   // {eq,fu,gy,hC}
        tmp2 = tmp2 + yxwz(tmp2);               // {eq+fu,fu+eq,gy+hC,hC+gy}
        _mm_store_ss(&v[ 8+i], tmp2 + zwyx(tmp2)); // {eq+fu+gy+hC,..}

        tmp3 = rowa[3] * colb[i];                   // {eq,fu,gy,hC}
        tmp3 = tmp3 + yxwz(tmp3);               // {eq+fu,fu+eq,gy+hC,hC+gy}
        _mm_store_ss(&v[12+i], tmp3 + zwyx(tmp3)); // {eq+fu+gy+hC,..}
    }

    // memory layout
    // m[]     = [a,b,c,d][e,f,g,h][i,j,k,l][m,n,o,p]
    // mat.m[] = [q,r,s,t][u,v,w,x][y,z,A,B][C,D,E,F]
#endif
    return *this;
}
template<> inline mat4f_t & mat4f_t::operator*=(float s)
{ m[0] = m[0] * s; m[1] = m[1] * s; m[2] = m[2] * s; m[3] = m[3] * s; return *this; }
template<> inline mat4f_t & mat4f_t::operator/=(float s)
{ m[0] = m[0] / s; m[1] = m[1] / s; m[2] = m[2] / s; m[3] = m[3] / s; return *this; }
#endif // HAS_LOOP_UNROLLING

template<> inline float mat4f_t::det() const
{
    // [a b c d]
    // [e f g h]
    // [i j k l]
    // [m n o p]
    
    // af-be, kp-lo, ag-ce, jp-ln, ah-de, jo-kn, bg-cf, ip-lm, bh-df, io-km, ch-dg, in-jm
    // x  x   x  x   x  x   x  x   x  x   x  x   x  x   x  x   x  x   x  x   x  x   x  x
    // (af-be)*(kp-lo) - (ag-ce)*(jp-ln) + (ah-de)*(jo-kn) + (bg-cf)*(ip-lm) - (bh-df)*(io-km) + (ch-dg)*(in-jm)
    /*
    __m128 af = xxxx(m[0]) * yyyy(m[1]); // {a,a,a,a}*{f,f,f,f} = {af,af,af,af}
    __m128 be = yyyy(m[0]) * xxxx(m[1]); // {b,b,b,b}*{e,e,e,e} = {be,be,be,be}
    __m128 kp = zzzz(m[2]) * wwww(m[3]); // {k,k,k,k}*{p,p,p,p} = {kp,kp,kp,kp}
    __m128 lo = wwww(m[2]) * zzzz(m[3]); // {l,l,l,l}*{o,o,o,o} = {lo,lo,lo,lo}
    __m128 ag = xxxx(m[0]) * zzzz(m[1]); // {a,a,a,a}*{g,g,g,g} = {ag,ag,ag,ag}
    __m128 ce = zzzz(m[0]) * xxxx(m[1]); // {c,c,c,c}*{e,e,e,e} = {ce,ce,ce,ce}
    __m128 jp = yyyy(m[2]) * wwww(m[3]); // {j,j,j,j}*{p,p,p,p} = {jp,jp,jp,jp}
    __m128 ln = wwww(m[2]) * yyyy(m[3]); // {l,l,l,l}*{n,n,n,n} = {ln,ln,ln,ln}
    __m128 ah = xxxx(m[0]) * wwww(m[1]); // {a,a,a,a}*{h,h,h,h} = {ah,ah,ah,ah}
    __m128 de = wwww(m[0]) * xxxx(m[1]); // {d,d,d,d}*{e,e,e,e} = {de,de,de,de}
    __m128 jo = yyyy(m[2]) * zzzz(m[3]); // {j,j,j,j}*{o,o,o,o} = {jo,jo,jo,jo}
    __m128 kn = zzzz(m[2]) * yyyy(m[3]); // {k,k,k,k}*{n,n,n,n} = {kn,kn,kn,kn}
    __m128 bg = yyyy(m[0]) * zzzz(m[1]); // {b,b,b,b}*{g,g,g,g} = {bg,bg,bg,bg}
    __m128 cf = zzzz(m[0]) * yyyy(m[1]); // {c,c,c,c}*{f,f,f,f} = {cf,cf,cf,cf}
    __m128 ip = xxxx(m[2]) * wwww(m[3]); // {i,i,i,i}*{p,p,p,p} = {ip,ip,ip,ip}
    __m128 lm = wwww(m[2]) * xxxx(m[3]); // {l,l,l,l}*{m,m,m,m} = {lm,lm,lm,lm}
    __m128 bh = yyyy(m[0]) * wwww(m[1]); // {b,b,b,b}*{h,h,h,h} = {bh,bh,bh,bh}
    __m128 df = wwww(m[0]) * yyyy(m[1]); // {d,d,d,d}*{f,f,f,f} = {df,df,df,df}
    __m128 io = xxxx(m[2]) * zzzz(m[3]); // {i,i,i,i}*{o,o,o,o} = {io,io,io,io}
    __m128 km = zzzz(m[2]) * xxxx(m[3]); // {k,k,k,k}*{m,m,m,m} = {km,km,km,km}
    __m128 ch = zzzz(m[0]) * wwww(m[1]); // {c,c,c,c}*{h,h,h,h} = {ch,ch,ch,ch}
    __m128 dg = wwww(m[0]) * zzzz(m[1]); // {d,d,d,d}*{g,g,g,g} = {dg,dg,dg,dg}
    __m128 in = xxxx(m[2]) * yyyy(m[3]); // {i,i,i,i}*{n,n,n,n} = {in,in,in,in}
    __m128 jm = yyyy(m[2]) * xxxx(m[3]); // {j,j,j,j}*{m,m,m,m} = {jm,jm,jm,jm}
    return _mm_cvtss_f32(
        (af-be)*(kp-lo) - (ag-ce)*(jp-ln) + (ah-de)*(jo-kn) +
        (bg-cf)*(ip-lm) - (bh-df)*(io-km) + (ch-dg)*(in-jm)
    );
    */
    // 48shuffles, 30muls, 17adds/subs
    // bei shuffles und adds/subs einfach und muls doppelt: 48+2*30+17 = 125 ops

    /* --- */
    // a{f,g,h}, b{e,g,h}, c{e,f,h}, d{e,f,g}, i{p,o,n}, j{p,o,m}, k{p,n,m}, l{o,n,m}
    mat4f_t::packed_t afgh = xxxx(m[0]) * m[1]; // {ae,af,ag,ah}
    mat4f_t::packed_t begh = yyyy(m[0]) * m[1]; // {be,bf,bg,bh}
    mat4f_t::packed_t cefh = zzzz(m[0]) * m[1]; // {ce,cf,cg,ch}
    mat4f_t::packed_t defg = wwww(m[0]) * m[1]; // {de,df,dg,dh}
    mat4f_t::packed_t ipon = xxxx(m[2]) * m[3]; // {im,in,io,ip}
    mat4f_t::packed_t jpom = yyyy(m[2]) * m[3]; // {jm,jn,jo,jp}
    mat4f_t::packed_t kpnm = zzzz(m[2]) * m[3]; // {km,kn,ko,kp}
    mat4f_t::packed_t lonm = wwww(m[2]) * m[3]; // {lm,ln,lo,lp}

    mat4f_t::packed_t result =
        ((yyyy(afgh) - xxxx(begh)) * (wwww(kpnm) - zzzz(lonm))) - // {(af-be)*(kp-lo),..}
        ((zzzz(afgh) - xxxx(cefh)) * (wwww(jpom) - yyyy(lonm))) + // {(ag-ce)*(jp-ln),..}
        ((wwww(afgh) - xxxx(defg)) * (zzzz(jpom) - yyyy(kpnm))) + // {(ah-de)*(jo-kn),..}
        ((zzzz(begh) - yyyy(cefh)) * (wwww(ipon) - xxxx(lonm))) - // {(bg-cf)*(ip-lm),..}
        ((wwww(begh) - yyyy(defg)) * (zzzz(ipon) - xxxx(kpnm))) + // {(bh-df)*(io-km),..}
        ((wwww(cefh) - zzzz(defg)) * (yyyy(ipon) - xxxx(jpom)))   // {(ch-dg)*(in-jm),..}
        ;
#if defined(_M_IX86) || defined(_M_X64)
    return _mm_cvtss_f32(result);
#elif defined(_M_ARM)
    return vget_lane_f32(vget_low_f32(result), 0);
#endif

    // 32shuffles, 14muls, 17adds/subs
    // bei shuffles und adds/subs einfach und muls doppelt: 32+14*2+17 = 77 ops
}
template<> inline mat4f_t mat4f_t::adjugate() const
{
    // [a b c d]
    // [e f g h]
    // [i j k l]
    // [m n o p]

    // adj(M): transpose of the cofactor matrix of M
    // adj(M) = transpose(C)
    // minor(M,i,j) = det3x3(M3x3(delete row i, delete column j))
    // C(i,j) = (-1)^(i+j)*minor(M(i,j))

    // minor(M,0,0) = det3x3({f,g,h},{j,k,l},{n,o,p})
    // minor(M,0,1) = det3x3({e,g,h},{i,k,l},{m,o,p})
    // minor(M,0,2) = det3x3({e,f,h},{i,j,l},{m,n,p})
    // minor(M,0,3) = det3x3({e,f,g},{i,j,k},{m,n,o})
    // minor(M,1,0) = det3x3({b,c,d},{j,k,l},{n,o,p})
    // minor(M,1,1) = det3x3({a,c,d},{i,k,l},{m,o,p})
    // minor(M,1,2) = det3x3({a,b,d},{i,j,l},{m,n,p})
    // minor(M,1,3) = det3x3({a,b,c},{i,j,k},{m,n,o})
    // minor(M,2,0) = det3x3({b,c,d},{f,g,h},{n,o,p})
    // minor(M,2,1) = det3x3({a,c,d},{e,g,h},{m,o,p})
    // minor(M,2,2) = det3x3({a,b,d},{e,f,h},{m,n,p})
    // minor(M,2,3) = det3x3({a,b,c},{e,f,g},{m,n,o})
    // minor(M,3,0) = det3x3({b,c,d},{f,g,h},{j,k,l})
    // minor(M,3,1) = det3x3({a,c,d},{e,g,h},{i,k,l})
    // minor(M,3,2) = det3x3({a,b,d},{e,f,h},{i,j,l})
    // minor(M,3,3) = det3x3({a,b,c},{e,f,g},{i,j,k})

    // C(0,0) =  det3x3({f,g,h},
    //                  {j,k,l},
    //                  {n,o,p}) = f(kp-lo) - g(jp-ln) + h(jo-kn
    // C(0,1) = -det3x3({e,g,h},
    //                  {i,k,l},
    //                  {m,o,p}) = e(kp-lo) - g(ip-lm) + h(io-km)
    // C(0,2) =  det3x3({e,f,h},
    //                  {i,j,l},
    //                  {m,n,p}) = e(jp-ln) - f(ip-lm) + h(in-jm)
    // C(0,3) = -det3x3({e,f,g},
    //                  {i,j,k},
    //                  {m,n,o}) = e(jo-kn) - f(io-km) + g(in-jm)
    // C(1,0) = -det3x3({b,c,d},
    //                  {j,k,l},
    //                  {n,o,p}) = b(kp-lo) - c(jp-ln) + d(jo-kn)
    // C(1,1) =  det3x3({a,c,d},
    //                  {i,k,l},
    //                  {m,o,p}) = a(kp-lo) - c(ip-lm) + d(io-km)
    // C(1,2) = -det3x3({a,b,d},
    //                  {i,j,l},
    //                  {m,n,p}) = a(jp-ln) - b(ip-lm) + d(in-jm)
    // C(1,3) =  det3x3({a,b,c},
    //                  {i,j,k},
    //                  {m,n,o}) = a(jo-kn) - b(io-km) + c(in-jm)
    // C(2,0) =  det3x3({b,c,d},
    //                  {f,g,h},
    //                  {n,o,p}) = b(gp-ho) - c(fp-hn) + d(fo-gn)
    // C(2,1) = -det3x3({a,c,d},
    //                  {e,g,h},
    //                  {m,o,p}) = a(gp-ho) - c(ep-hm) + d(eo-gm)
    // C(2,2) =  det3x3({a,b,d},
    //                  {e,f,h},
    //                  {m,n,p}) = a(fp-hn) - b(ep-hm) + d(en-fm)
    // C(2,3) = -det3x3({a,b,c},
    //                  {e,f,g},
    //                  {m,n,o}) = a(fo-gn) - b(eo-gm) + c(en-fm)
    // C(3,0) = -det3x3({b,c,d},
    //                  {f,g,h},
    //                  {j,k,l}) = b(gl-hk) - c(fl-hj) + d(fk-gj)
    // C(3,1) =  det3x3({a,c,d},
    //                  {e,g,h},
    //                  {i,k,l}) = a(gl-hk) - c(el-hi) + d(ek-gi)
    // C(3,2) = -det3x3({a,b,d},
    //                  {e,f,h},
    //                  {i,j,l}) = a(fl-hj) - b(el-hi) + d(ej-fi)
    // C(3,3) =  det3x3({a,b,c},
    //                  {e,f,g},
    //                  {i,j,k}) = a(fk-gj) - b(ek-gi) + c(ej-fi)

    // det3x3:
    // |[a b c]|    |e f|    |d f|    |d e|
    // |[d e f]| = a|h i| - b|g i| + c|g h| = a(ei-fh) - b(di-fg) + c(dh-eg)
    // |[g h i]|

    mat4f_t::packed_t eeee = xxxx(m[1]), ffff = yyyy(m[1]), gggg = zzzz(m[1]), hhhh = wwww(m[1]);
    mat4f_t::packed_t iiii = xxxx(m[2]), jjjj = yyyy(m[2]), kkkk = zzzz(m[2]), llll = wwww(m[2]);
    mat4f_t::packed_t mmmm = xxxx(m[3]), nnnn = yyyy(m[3]), oooo = zzzz(m[3]), pppp = wwww(m[3]);
    
    // [a b c d]   [ 1  2  3  4]
    // [e f g h]   [ 5  6  7  8]
    // [i j k l]   [ 9 10 11 12]
    // [m n o p] = [13 14 15 16]

    mat4f_t::packed_t baxx = yxxx(m[0]), caxx = zxxx(m[0]), daxx = wxxx(m[0]);
    mat4f_t::packed_t cbxx = zyxx(m[0]), dbxx = wyxx(m[0]), dcxx = wzxx(m[0]);

    mat4f_t::packed_t
        feba_kp_lo = yxyx(m[1], m[0]) * ((kkkk*pppp)-(llll*oooo)), geca_jp_ln = zxzx(m[1], m[0]) * ((jjjj*pppp)-(llll*nnnn)), heda_jo_kn = wxwx(m[1], m[0]) * ((jjjj*oooo)-(kkkk*nnnn)),
        gfcb_ip_lm = zyzy(m[1], m[0]) * ((iiii*pppp)-(llll*mmmm)), hfdb_io_km = wywy(m[1], m[0]) * ((iiii*oooo)-(kkkk*mmmm)), hgdc_in_jm = wzwz(m[1], m[0]) * ((iiii*nnnn)-(jjjj*mmmm)),
        baxx_gp_ho = baxx * ((gggg*pppp)-(hhhh*oooo)), caxx_fp_hn = caxx * ((ffff*pppp)-(hhhh*nnnn)), daxx_fo_gn = daxx * ((ffff*oooo)-(gggg*nnnn)),
        cbxx_ep_hm = cbxx * ((eeee*pppp)-(hhhh*mmmm)), dbxx_eo_gm = dbxx * ((eeee*oooo)-(gggg*mmmm)), dcxx_en_fm = dcxx * ((eeee*nnnn)-(ffff*mmmm)),
        baxx_gl_hk = baxx * ((gggg*llll)-(hhhh*kkkk)), caxx_fl_hj = caxx * ((ffff*llll)-(hhhh*jjjj)), daxx_fk_gj = daxx * ((ffff*kkkk)-(gggg*jjjj)),
        cbxx_el_hi = cbxx * ((eeee*llll)-(hhhh*iiii)), dbxx_ek_gi = dbxx * ((eeee*kkkk)-(gggg*iiii)), dcxx_ej_fi = dcxx * ((eeee*jjjj)-(ffff*iiii));

    mat4f_t::packed_t tmp = heda_jo_kn - hfdb_io_km + hgdc_in_jm ; // {-,e(jo-kn) - f(io-km) + g(in-jm),-,a(jo-kn) - b(io-km) + c(in-jm)}

    mat4f_t ret;
    ret.m[0] =
        xyxz(
            xzxx(feba_kp_lo-geca_jp_ln+heda_jo_kn),      // {f(kp-lo) - g(jp-ln) + h(jo-kn),b(kp-lo) - c(jp-ln) + d(jo-kn),-,-}
            xxxx(baxx_gp_ho-caxx_fp_hn+daxx_fo_gn, baxx_gl_hk-caxx_fl_hj+daxx_fk_gj) // {b(gp-ho) - c(fp-hn) + d(fo-gn),-,b(gl-hk) - c(fl-hj) + d(fk-gj),-}
        ); // {f(kp-lo) - g(jp-ln) + h(jo-kn),b(kp-lo) - c(jp-ln) + d(jo-kn),b(gp-ho) - c(fp-hn) + d(fo-gn),b(gl-hk) - c(fl-hj) + d(fk-gj)}
    ret.m[1] =
        xzxz(
            xxww(yyyy(feba_kp_lo)-gfcb_ip_lm+hfdb_io_km, feba_kp_lo-zzzz(gfcb_ip_lm)+zzzz(hfdb_io_km)),// {e(kp-lo) - g(ip-lm) + h(io-km),..,a(kp-lo) - c(ip-lm) + d(io-km),..}
            xxxx(yyyy(baxx_gp_ho)-cbxx_ep_hm+dbxx_eo_gm, yyyy(baxx_gl_hk)-cbxx_el_hi+dbxx_ek_gi) // {a(gp-ho) - c(ep-hm) + d(eo-gm),..,a(gl-hk) - c(el-hi) + d(ek-gi),..}
        ); // {e(kp-lo) - g(ip-lm) + h(io-km),a(kp-lo) - c(ip-lm) + d(io-km),a(gp-ho) - c(ep-hm) + d(eo-gm),a(gl-hk) - c(el-hi) + d(ek-gi)}
    ret.m[2] =
        xzxz(
            yyww(geca_jp_ln-gfcb_ip_lm+xxxx(hgdc_in_jm), geca_jp_ln-gfcb_ip_lm+zzzz(hgdc_in_jm)),// {e(jp-ln) - f(ip-lm) + h(in-jm),..,a(jp-ln) - b(ip-lm) + d(in-jm),..}
            yyyy(caxx_fp_hn-cbxx_ep_hm+xxxx(dcxx_en_fm), caxx_fl_hj-cbxx_el_hi+xxxx(dcxx_ej_fi)) // {a(fp-hn) - b(ep-hm) + d(en-fm),..,a(fl-hj) - b(el-hi) + d(ej-fi),..}
        ); // {e(jp-ln) - f(ip-lm) + h(in-jm),a(jp-ln) - b(ip-lm) + d(in-jm),a(fp-hn) - b(ep-hm) + d(en-fm),a(fl-hj) - b(el-hi) + d(ej-fi)}
    ret.m[3] =
        xzxz(
            yyww(tmp, tmp),// {e(jo-kn) - f(io-km) + g(in-jm),..,a(jo-kn) - b(io-km) + c(in-jm),..}
            yyyy(daxx_fo_gn-dbxx_eo_gm+dcxx_en_fm, daxx_fk_gj-dbxx_ek_gi+dcxx_ej_fi) // {a(fo-gn) - b(eo-gm) + c(en-fm),..,a(fk-gj) - b(ek-gi) + c(ej-fi),..}
        ); // {e(jo-kn) - f(io-km) + g(in-jm),a(jo-kn) - b(io-km) + c(in-jm),a(fo-gn) - b(eo-gm) + c(en-fm),a(fk-gj) - b(ek-gi) + c(ej-fi)}
    
    // [a b c d]
    // [e f g h]
    // [i j k l]
    // [m n o p]
    // adj(0,0) = f(kp-lo) - g(jp-ln) + h(jo-kn) x
    // adj(0,1) = b(kp-lo) - c(jp-ln) + d(jo-kn) x
    // adj(0,2) = b(gp-ho) - c(fp-hn) + d(fo-gn) x
    // adj(0,3) = b(gl-hk) - c(fl-hj) + d(fk-gj) x
    //   -> adj.m[0] = {f*kp_lo - g*jp_ln + h*jo_kn, b*kp_lo - c*jp_ln + d*jo_kn, b*gp_ho - c*fp_hn + d*fo_gn, b*gl_hk - c*fl_hj + d*fk_gj}
    //               = {xxxx(feba_kp_lo) - xxxx(geca_jp_ln) + xxxx(heda_jo_kn),
    //                  zzzz(feba_kp_lo) - zzzz(geca_jp_ln) + zzzz(heda_jo_kn),
    //                  xxxx(baxx_gp_ho) - xxxx(caxx_fp_hn) + xxxx(daxx_fo_gn),
    //                  xxxx(baxx_gl_hk) - xxxx(caxx_fl_hj) + xxxx(daxx_fk_gj) }
    // adj(1,0) = e(kp-lo) - g(ip-lm) + h(io-km) x
    // adj(1,1) = a(kp-lo) - c(ip-lm) + d(io-km) x
    // adj(1,2) = a(gp-ho) - c(ep-hm) + d(eo-gm) x
    // adj(1,3) = a(gl-hk) - c(el-hi) + d(ek-gi) x
    //   -> adj.m[1] = {e*kp_lo - g*ip_lm + h*io_km, a*kp_lo - c*ip_lm + d*io_km, a*gp_ho - c*ep_hm + d*eo_gm, a*gl_hk - c*el_hi + d*ek_gi}
    //               = {yyyy(feba_kp_lo) - xxxx(gfcb_ip_lm) + xxxx(hfdb_io_km),
    //                  wwww(feba_kp_lo) - zzzz(gfcb_ip_lm) + wwww(hfdb_io_km),
    //                  yyyy(baxx_gp_ho) - xxxx(cbxx_ep_hm) + xxxx(dbxx_eo_gm),
    //                  yyyy(baxx_gl_hk) - xxxx(cbxx_el_hi) + xxxx(dbxx_ek_gi) }
    // adj(2,0) = e(jp-ln) - f(ip-lm) + h(in-jm) x
    // adj(2,1) = a(jp-ln) - b(ip-lm) + d(in-jm) x
    // adj(2,2) = a(fp-hn) - b(ep-hm) + d(en-fm) x
    // adj(2,3) = a(fl-hj) - b(el-hi) + d(ej-fi) x
    //   -> adj.m[2] = {e*jp_ln - f*ip_lm + h*in_jm, a*jp_ln - b*ip_lm + d*in_jm, a*fp_hn - b*ep_hm + d*en_fm, a*fl_hj - b*el_hi + d*ej_fi}
    //               = {yyyy(geca_jp_ln) - yyyy(gfcb_ip_lm) + xxxx(hgdc_in_jm),
    //                  wwww(geca_jp_ln) - wwww(gfcb_ip_lm) + zzzz(hgdc_in_jm),
    //                  yyyy(caxx_fp_hn) - yyyy(cbxx_ep_hm) + xxxx(dcxx_en_fm),
    //                  yyyy(caxx_fl_hj) - yyyy(cbxx_el_hi) + xxxx(dcxx_ej_fi)}
    // adj(3,0) = e(jo-kn) - f(io-km) + g(in-jm) x
    // adj(3,1) = a(jo-kn) - b(io-km) + c(in-jm) x
    // adj(3,2) = a(fo-gn) - b(eo-gm) + c(en-fm) x
    // adj(3,3) = a(fk-gj) - b(ek-gi) + c(ej-fi) x
    //   -> adj.m[3] = {e*jo_kn - f*io_km + g*in_jm, a*jo_kn - b*io_km + c*in_jm, a*fo_gn - b*eo_gm + c*en_fm, a*fk_gj - b*ek_gi + c*ej_fi}
    //               = {yyyy(heda_jo_kn) - yyyy(hfdb_io_km) + yyyy(hgdc_in_jm),
    //                  wwww(heda_jo_kn) - wwww(hfdb_io_km) + wwww(hgdc_in_jm),
    //                  yyyy(daxx_fo_gn) - yyyy(dbxx_eo_gm) + yyyy(dcxx_en_fm),
    //                  yyyy(daxx_fk_gj) - yyyy(dbxx_ek_gi) + yyyy(dcxx_ej_fi)}
    
    // brauche:
    // (f,e,b,a)*(kp-lo,kp-lo,kp-lo,kp-lo) = feba * kp_lo;
    // {g,e,c,a}*{jp-ln,jp-ln,jp-ln,jp-ln} = geca * jp_ln;
    // {h,e,d,a}*{jo-kn,jo-kn,jo-kn,jo-kn} = heda * jo_kn;
    // {g,f,c,b}*{ip-lm,ip-lm,ip-lm,ip-lm} = gfcb * ip_lm;
    // {h,f,d,b}*{io-km,io-km,io-km,io-km} = hfdb * io_km;
    // {h,g,d,c}*{in-jm,in-jm,in-jm,in-jm} = hgdc * in_jm;
    // {b,a,?,?}*{gp-ho,gp-ho,gp-ho,gp-ho} = baxx * gp_ho;
    // {c,a,?,?}*{fp-hn,fp-hn,fp-hn,fp-hn} = caxx * fp_hn;
    // {d,a,?,?}*{fo-gn,fo-gn,fo-gn,fo-gn} = daxx * fo_gn;
    // {c,b,?,?}*{ep-hm,ep-hm,ep-hm,ep-hm} = cbxx * ep_hm;
    // {d,b,?,?}*{eo-gm,eo-gm,eo-gm,eo-gm} = dbxx * eo_gm;
    // {d,c,?,?}*{en-fm,en-fm,en-fm,en-fm} = dcxx * en_fm;
    // {b,a,?,?}*{gl-hk,gl-hk,gl-hk,gl-hk} = baxx * gl_hk;
    // {c,a,?,?}*{fl-hj,fl-hj,fl-hj,fl-hj} = caxx * fl_hj;
    // {d,a,?,?}*{fk-gj,fk-gj,fk-gj,fk-gj} = daxx * fk_gj;
    // {c,b,?,?}*{el-hi,el-hi,el-hi,el-hi} = cbxx * el_hi;
    // {d,b,?,?}*{ek-gi,ek-gi,ek-gi,ek-gi} = dbxx * ek_gi;
    // {d,c,?,?}*{ej-fi,ej-fi,ej-fi,ej-fi} = dcxx * ej_fi;

    // [ 1  2  3  4]            [0 0 0 0]
    // [ 5  6  7  8]            [0 0 0 0]
    // [ 9 10 11 12] -> adj() = [0 0 0 0]
    // [13 14 15 16]            [0 0 0 0]
}
template<> inline void mat4f_t::adjugate()
{
    mat4f_t::packed_t eeee = xxxx(m[1]), ffff = yyyy(m[1]), gggg = zzzz(m[1]), hhhh = wwww(m[1]);
    mat4f_t::packed_t iiii = xxxx(m[2]), jjjj = yyyy(m[2]), kkkk = zzzz(m[2]), llll = wwww(m[2]);
    mat4f_t::packed_t mmmm = xxxx(m[3]), nnnn = yyyy(m[3]), oooo = zzzz(m[3]), pppp = wwww(m[3]);

    mat4f_t::packed_t baxx = yxxx(m[0]), caxx = zxxx(m[0]), daxx = wxxx(m[0]);
    mat4f_t::packed_t cbxx = zyxx(m[0]), dbxx = wyxx(m[0]), dcxx = wzxx(m[0]);

    mat4f_t::packed_t
        feba_kp_lo = yxyx(m[1], m[0]) * ((kkkk*pppp)-(llll*oooo)),
        geca_jp_ln = zxzx(m[1], m[0]) * ((jjjj*pppp)-(llll*nnnn)),
        heda_jo_kn = wxwx(m[1], m[0]) * ((jjjj*oooo)-(kkkk*nnnn)),
        gfcb_ip_lm = zyzy(m[1], m[0]) * ((iiii*pppp)-(llll*mmmm)),
        hfdb_io_km = wywy(m[1], m[0]) * ((iiii*oooo)-(kkkk*mmmm)),
        hgdc_in_jm = wzwz(m[1], m[0]) * ((iiii*nnnn)-(jjjj*mmmm)),
        baxx_gp_ho = baxx * ((gggg*pppp)-(hhhh*oooo)),
        caxx_fp_hn = caxx * ((ffff*pppp)-(hhhh*nnnn)),
        daxx_fo_gn = daxx * ((ffff*oooo)-(gggg*nnnn)),
        cbxx_ep_hm = cbxx * ((eeee*pppp)-(hhhh*mmmm)),
        dbxx_eo_gm = dbxx * ((eeee*oooo)-(gggg*mmmm)),
        dcxx_en_fm = dcxx * ((eeee*nnnn)-(ffff*mmmm)),
        baxx_gl_hk = baxx * ((gggg*llll)-(hhhh*kkkk)),
        caxx_fl_hj = caxx * ((ffff*llll)-(hhhh*jjjj)),
        daxx_fk_gj = daxx * ((ffff*kkkk)-(gggg*jjjj)),
        cbxx_el_hi = cbxx * ((eeee*llll)-(hhhh*iiii)),
        dbxx_ek_gi = dbxx * ((eeee*kkkk)-(gggg*iiii)),
        dcxx_ej_fi = dcxx * ((eeee*jjjj)-(ffff*iiii));
    mat4f_t::packed_t tmp = heda_jo_kn - hfdb_io_km + hgdc_in_jm ; // {-,e(jo-kn) - f(io-km) + g(in-jm),-,a(jo-kn) - b(io-km) + c(in-jm)}
    m[0] =
        xyxz(
            xzxx(feba_kp_lo-geca_jp_ln+heda_jo_kn),      // {f(kp-lo) - g(jp-ln) + h(jo-kn),b(kp-lo) - c(jp-ln) + d(jo-kn),-,-}
            xxxx(baxx_gp_ho-caxx_fp_hn+daxx_fo_gn, baxx_gl_hk-caxx_fl_hj+daxx_fk_gj) // {b(gp-ho) - c(fp-hn) + d(fo-gn),-,b(gl-hk) - c(fl-hj) + d(fk-gj),-}
        ); // {f(kp-lo) - g(jp-ln) + h(jo-kn),b(kp-lo) - c(jp-ln) + d(jo-kn),b(gp-ho) - c(fp-hn) + d(fo-gn),b(gl-hk) - c(fl-hj) + d(fk-gj)}
    m[1] =
        xzxz(
            xxww(yyyy(feba_kp_lo)-gfcb_ip_lm+hfdb_io_km, feba_kp_lo-zzzz(gfcb_ip_lm)+zzzz(hfdb_io_km)),// {e(kp-lo) - g(ip-lm) + h(io-km),..,a(kp-lo) - c(ip-lm) + d(io-km),..}
            xxxx(yyyy(baxx_gp_ho)-cbxx_ep_hm+dbxx_eo_gm, yyyy(baxx_gl_hk)-cbxx_el_hi+dbxx_ek_gi) // {a(gp-ho) - c(ep-hm) + d(eo-gm),..,a(gl-hk) - c(el-hi) + d(ek-gi),..}
        ); // {e(kp-lo) - g(ip-lm) + h(io-km),a(kp-lo) - c(ip-lm) + d(io-km),a(gp-ho) - c(ep-hm) + d(eo-gm),a(gl-hk) - c(el-hi) + d(ek-gi)}
    m[2] =
        xzxz(
            yyww(geca_jp_ln-gfcb_ip_lm+xxxx(hgdc_in_jm), geca_jp_ln-gfcb_ip_lm+zzzz(hgdc_in_jm)),// {e(jp-ln) - f(ip-lm) + h(in-jm),..,a(jp-ln) - b(ip-lm) + d(in-jm),..}
            yyyy(caxx_fp_hn-cbxx_ep_hm+xxxx(dcxx_en_fm), caxx_fl_hj-cbxx_el_hi+xxxx(dcxx_ej_fi)) // {a(fp-hn) - b(ep-hm) + d(en-fm),..,a(fl-hj) - b(el-hi) + d(ej-fi),..}
        ); // {e(jp-ln) - f(ip-lm) + h(in-jm),a(jp-ln) - b(ip-lm) + d(in-jm),a(fp-hn) - b(ep-hm) + d(en-fm),a(fl-hj) - b(el-hi) + d(ej-fi)}
    m[3] =
        xzxz(
            yyww(tmp, tmp),// {e(jo-kn) - f(io-km) + g(in-jm),..,a(jo-kn) - b(io-km) + c(in-jm),..}
            yyyy(daxx_fo_gn-dbxx_eo_gm+dcxx_en_fm, daxx_fk_gj-dbxx_ek_gi+dcxx_ej_fi) // {a(fo-gn) - b(eo-gm) + c(en-fm),..,a(fk-gj) - b(ek-gi) + c(ej-fi),..}
        ); // {e(jo-kn) - f(io-km) + g(in-jm),a(jo-kn) - b(io-km) + c(in-jm),a(fo-gn) - b(eo-gm) + c(en-fm),a(fk-gj) - b(ek-gi) + c(ej-fi)}
}


inline mat4f_t operator*(float s, const mat4f_t & m)
{ return mat4f_t(m) *= s; }

template<> inline mat4f_t mat4f_t::inverse() const
{ float det_ = det();
  if(almost_equal(det_, 0.0f)) return mat4f_t(); // no inverse -> zero matrix
  return (1.0f / det_) * adjugate(); }
template<> inline void mat4f_t::inverse()
{ float det_ = det();
  if(almost_equal(det_, 0.0f)) m[0] = m[1] = m[2] = m[3] = math_t::zeroes();
  adjugate();
  float inv = 1.0f / det_;
  m[0] = inv * m[0]; m[1] = inv * m[1];
  m[2] = inv * m[2]; m[3] = inv * m[3]; }


template<> inline void mat4f_t::transform(vec4f_t & vec) const
{  vec.p =
       xzxz(
           xxxx(
               math_t::dot_packed(m[0], vec.p), // {dot0,..}
               math_t::dot_packed(m[1], vec.p)  // {dot1,..}
           ), // {dot0,dot0,dot1,dot1}
           xxxx(
               math_t::dot_packed(m[2], vec.p), // {dot2,..}
               math_t::dot_packed(m[3], vec.p)  // {dot3,..}
           )  // {dot2,dot2,dot3,dot3}
       ); // {dot0,dot1,dot2,dot3}
}
template<> inline vec4f_t mat4f_t::transform(const vec4f_t & vec) const
{ vec4f_t ret(vec); transform(ret); return ret; }


inline mat4f_t operator+(const mat4f_t & m1, const mat4f_t & m2)
{ return mat4f_t(m1) += m2; }
inline mat4f_t operator-(const mat4f_t & m1, const mat4f_t & m2)
{ return mat4f_t(m1) -= m2; }
inline mat4f_t operator*(const mat4f_t & m1, const mat4f_t & m2)
{ return mat4f_t(m1) *= m2; }
inline mat4f_t operator*(const mat4f_t & m, float s) { return s * m; }
inline vec4f_t operator*(const mat4f_t & m, const vec4f_t & vec) { return m.transform(vec); }

//#endif // ARM NEON shuffles for 2 operands missing


#if defined(PVECF_INTEL)

// packed double precision floating point calculations are supported starting
// with SSE2
#if defined(SSE2) || defined(AVX)
//
// mat2d_t implementation
//
template<> inline mat2d_t::mat_t()
{ m[0] = m[1] = math_t::zeroes(); }
template<> inline mat2d_t mat2d_t::identity()
{ static const __m128d rows[] = {{1.0, 0.0}, {0.0, 1.0}}; return mat2d_t(rows); }
template<> inline mat2d_t mat2d_t::zero() { return mat2d_t(); }
template<> inline void mat2d_t::set_identity()
{ static const __m128d rows[] = {{1.0, 0.0}, {0.0, 1.0}};
  m[0] = rows[0]; m[1] = rows[1]; }
template<> inline vec2d_t mat2d_t::row(size_t row) const
{ return vec2d_t(m[row]);  }
template<> inline vec2d_t mat2d_t::col(size_t col) const
{
    // [a b]
    // [c d]
    switch(col) {
    case 0:
        return vec2d_t(
            xx(
                xx(m[0], m[0]), // {a,a}
                xx(m[1], m[1])  // {c,c}
            ) // {a,c}
        );
    case 1:
        return vec2d_t(
            xx(
                yy(m[0], m[0]), // {b,b}
                yy(m[1], m[1])  // {d,d}
            ) // {b,d}
        );
    }
    return vec2d_t(); // zero vector
}
#define MAT2D_CMP_OP(op,mat,vec) \
template<> inline bool mat::operator op(const mat & mat_) const \
{ return (vec(m[0]) op mat_.m[0]) && (vec(m[1]) op mat_.m[1]); }
MAT2D_CMP_OP(==,mat2d_t,vec2d_t)
MAT2D_CMP_OP(!=,mat2d_t,vec2d_t)
MAT2D_CMP_OP(< ,mat2d_t,vec2d_t)
MAT2D_CMP_OP(<=,mat2d_t,vec2d_t)
MAT2D_CMP_OP(> ,mat2d_t,vec2d_t)
MAT2D_CMP_OP(>=,mat2d_t,vec2d_t)
#undef MAT2D_CMP_OP

#ifndef HAS_LOOP_UNROLLING
template<> inline mat2d_t & mat2d_t::operator+=(const mat2d_t & mat)
{ m[0] = m[0] + mat.m[0]; m[1] = m[1] + mat.m[1]; return *this; }
template<> inline mat2d_t & mat2d_t::operator-=(const mat2d_t & mat)
{ m[0] = m[0] - mat.m[0]; m[1] = m[1] - mat.m[1]; return *this; }
template<> inline mat2d_t & mat2d_t::operator*=(const mat2d_t & mat)
{ // [a b]   [e f]   [ae+bg af+bh]
  // [c d] * [g h] = [ce+dg cf+dh]
  mat2d_t tmp_mat;
  vec2d_t tmp_vec;
  tmp_vec.v[0] = mat.v[0]; tmp_vec.v[1] = mat.v[2]; // {e,g}
  tmp_mat.v[0] = vec2d_t(m[0]).dot(tmp_vec);        // ae+bg
  tmp_mat.v[2] = vec2d_t(m[1]).dot(tmp_vec);        // ce+dg
  tmp_vec.v[0] = mat.v[1]; tmp_vec.v[1] = mat.v[3]; // {f,h}
  tmp_mat.v[1] = vec2d_t(m[0]).dot(tmp_vec);        // af+bh
  tmp_mat.v[3] = vec2d_t(m[1]).dot(tmp_vec);        // cf+dh
  return *this; }
template<> inline mat2d_t & mat2d_t::operator*=(double s)
{ m[0] = m[0] * s; m[1] = m[1] * s; return *this; }
template<> inline mat2d_t & mat2d_t::operator/=(double s)
{ m[0] = m[0] / s; m[1] = m[1] / s; return *this; }
template<> inline mat2d_t mat2d_t::transpose() const
{ // [a b]    [a c]
  // [c d] -> [b d]
  mat2d_t tmp_mat;
  tmp_mat.m[0] = xx(m[0], m[1]); // {a,c}
  tmp_mat.m[1] = yy(m[0], m[1]); // {b,d}
  return tmp_mat;
}
template<> inline void mat2d_t::transpose()
{ // [a b]    [a c]
  // [c d] -> [b d]
  __m128d tmp0 = xx(m[0], m[1]);
  __m128d tmp1 = yy(m[0], m[1]);
  m[0] = tmp0; m[1] = tmp1;
}
#endif // !HAS_LOOP_UNROLLING

template<> inline double mat2d_t::det() const
{ return _mm_cvtsd_f64(xy(m[0], m[1])*yx(m[1], m[0])-xy(m[1], m[0])*yx(m[0], m[1])); } // 2muls, 1sub, 4shuffles
template<> inline mat2d_t mat2d_t::adjugate() const
{ //     [a b]    [ d -b]
  // M = [c d] -> [-c  a]
  mat2d_t ret;
  ret.m[0] = yy(m[1], math_t::zeroes() - m[0]); // { d,-b}
  ret.m[1] = xx(math_t::zeroes() - m[1], m[0]); // {-c, a}
  return ret;
  // adj(M): transpose of the cofactor matrix of M
  // adj(M) = transpose(C)
  // minor(M(i,j)) = det(M3x3(delete row i, delete column j))
  // C(i,j) = (-1)^(i+j)*minor(M(i,j))
}
template<> inline void mat2d_t::adjugate()
{ m[0] = yy(m[1], math_t::zeroes() - m[0]); // { d,-b}
  m[1] = xx(math_t::zeroes() - m[1], m[0]); // {-c, a}
}

inline mat2d_t operator*(double s, const mat2d_t & m) { return s * m; }

template<> inline mat2d_t mat2d_t::inverse() const
{ double det_ = det();
  if(almost_equal(det_, 0.0)) return mat2d_t(); // no inverse -> zero matrix
  return (1.0 / det_) * adjugate(); }
template<> inline void mat2d_t::inverse()
{ double det_ = det();
  if(almost_equal(det_, 0.0)) m[0] = m[1] = math_t::zeroes();
  adjugate();
  double inv = 1.0 / det_;
  m[0] = inv * m[0]; m[1] = inv * m[1]; }


template<> inline void mat2d_t::transform(vec2d_t & vec) const
{ vec.p =
      xx(
          math_t::dot_packed(m[0], vec.p), // {dot0,dot0}
          math_t::dot_packed(m[1], vec.p)  // {dot1,dot1}
      ); // {dot0,dot1}
}
template<> inline vec2d_t mat2d_t::transform(const vec2d_t & vec) const
{ vec2d_t ret(vec); transform(ret); return ret; }

inline mat2d_t operator+(const mat2d_t & m1, const mat2d_t & m2) { return m1 + m2; }
inline mat2d_t operator-(const mat2d_t & m1, const mat2d_t & m2) { return m1 - m2; }
inline mat2d_t operator*(const mat2d_t & m1, const mat2d_t & m2) { return m1 * m2; }
inline mat2d_t operator*(const mat2d_t & m, double s) { return m * s; }
inline vec2d_t operator*(const mat2d_t & m, const vec2d_t & vec) { return m.transform(vec); }
#endif // SSE2 || AVX

#endif // defined(PVECF_INTEL)

} // namespace math

#endif // !defined(PVECF_H)

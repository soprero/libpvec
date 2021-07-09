/*******************************************************************************
 * pveci.h                                                                     *
 *                                                                             *
 * Copyright (c) 2015-2018 Ronny Press                                         *
 *                                                                             *
 * rpress@soprero.de                                                           *
 *                                                                             *
 * All rights reserved.                                                        *
 *******************************************************************************/
#ifndef PVECI_H
#define PVECI_H

#include <pmathi.h>
#include <algorithm>


// this header file assumes at least SSE2 support (without checking the
// corresponding define) currently

// preprocessor shortcuts

#if (defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
#define PVECI_INTEL
#elif (defined(_MSC_VER) && defined(_M_ARM))
#define PVECI_ARM
#elif (defined(__GNUC__) && defined(__ARM_ARCH) && defined(__ARM_NEON))
#define PVECI_ARM
#define PVECI_ARM_GCC
#endif



// TODO:
// - free-standing operator{*,/} for all types (standard multi (no high/low, no widening))
// - shuffle operations for [u]int32x4_t and [u]int64x2_t (for [u]int8_t and [u]int16_t elements
//   the number of combos is too large)
// - shift operations
//   * element-wise
//   * full register shifts (identical impl for all types)
// - rotation operations
//   * element-wise
//   * full register rotation (identical impl for all types)

// - add C++03 compatibility (or even C++98 but low pri)


/*
unsigned comparisons
--------------------
for op={<,>}
  unsigned_compare<uint8_t>(op,a,b) = (a xor 0x80) op (b xor 0x80)
  unsigned_compare<uint16_t>(op,a,b) = (a xor 0x8000) op (b xor 0x8000)
  unsigned_compare<uint32_t>(op,a,b) = (a xor 0x80000000) op (b xor 0x80000000)
  unsigned_compare<uint64_t>(op,a,b) = (a xor 0x8000000000000000) op (b xor 0x8000000000000000)
    NOTE: there are no int64_t comparisons to be used as base in SSE2

memory accesses are to be avoided (don't want to thrash a client's cache line), so the
constants above have to be generated:
  uint32_t: wanted = {0x80000000}*4
            {X}*4=={X}*4 = {0xFFFFFFFF}*4 -> 31 element-wise left shifts = {0x80000000}*4
            (shift intrinsic to be used: _mm_slli_epi32())

  can use other sizes for the initial step (generating all bits set)
*/

// TODO:
// - get rid of the operator overloads for packed_t by providing a conversion
//   operator t_packed t_packed() const { return p; }
// - shift ops
// - rotate ops (XOP has instructions, AMD only)
// - ...

namespace math {

namespace ipriv {
    // helpers

    // ...
    template<bool cond, typename T = void> struct enable_if {};
    template<typename T> struct enable_if<true,T> { typedef T type; };
} // namespace ipriv


template<typename t_type, unsigned t_n, typename t_packed>
class veci_t
{
public:
    static const unsigned N = t_n;
    static_assert(
        sizeof(t_packed) == (sizeof(t_type) * N),
        "veci_t: wrong combo of t_type, N and t_packed template parameters used"
    );
    static_assert(
        std::is_integral<t_type>::value,
        "veci_t: supports integer types only"
    );
    static_assert(
        sizeof(t_type) == 1 || sizeof(t_type) == 2 ||
        sizeof(t_type) == 4 || sizeof(t_type) == 8,
        "veci_t: supports [u]int{8,16,32,64}_t only (sizeof() in {1,2,4,8})"
    );

    typedef t_type type_t;
    typedef t_packed packed_t;
    typedef typename math::imath_t<t_type,packed_t> math_t;

    inline veci_t();
    inline veci_t(t_type v0);
    inline veci_t(t_type v0, t_type v1);

    template<typename T = t_type, typename = typename ipriv::enable_if<(sizeof(T)<sizeof(uint64_t))>::type>
    inline veci_t(
        t_type v0, t_type v1, t_type v2
    );
    template<typename T = t_type, typename = typename ipriv::enable_if<(sizeof(T)<sizeof(uint64_t))>::type>
    inline veci_t(
        t_type v0, t_type v1, t_type v2, t_type v3
    );
    template<typename T = t_type, typename = typename ipriv::enable_if<(sizeof(T)<sizeof(uint32_t))>::type>
    inline veci_t(
        t_type v0, t_type v1, t_type v2, t_type v3, t_type v4
    );
    template<typename T = t_type, typename = typename ipriv::enable_if<(sizeof(T)<sizeof(uint32_t))>::type>
    inline veci_t(
        t_type v0, t_type v1, t_type v2, t_type v3, t_type v4, t_type v5
    );
    template<typename T = t_type, typename = typename ipriv::enable_if<(sizeof(T)<sizeof(uint32_t))>::type>
    inline veci_t(
        t_type v0, t_type v1, t_type v2, t_type v3, t_type v4, t_type v5, t_type v6
    );
    template<typename T = t_type, typename = typename ipriv::enable_if<(sizeof(T)<sizeof(uint32_t))>::type>
    inline veci_t(
        t_type v0, t_type v1, t_type v2, t_type v3, t_type v4, t_type v5, t_type v6, t_type v7
    );
    template<typename T = t_type, typename = typename ipriv::enable_if<(sizeof(T)<sizeof(uint16_t))>::type>
    inline veci_t(
        t_type v0, t_type v1, t_type v2, t_type v3, t_type v4, t_type v5, t_type v6, t_type v7,
        t_type v8
    );
    template<typename T = t_type, typename = typename ipriv::enable_if<(sizeof(T)<sizeof(uint16_t))>::type>
    inline veci_t(
        t_type v0, t_type v1, t_type v2, t_type v3, t_type v4, t_type v5, t_type v6, t_type v7,
        t_type v8, t_type v9
    );
    template<typename T = t_type, typename = typename ipriv::enable_if<(sizeof(T)<sizeof(uint16_t))>::type>
    inline veci_t(
        t_type v0, t_type v1, t_type v2, t_type v3, t_type v4, t_type v5, t_type v6, t_type v7,
        t_type v8, t_type v9, t_type va
    );
    template<typename T = t_type, typename = typename ipriv::enable_if<(sizeof(T)<sizeof(uint16_t))>::type>
    inline veci_t(
        t_type v0, t_type v1, t_type v2, t_type v3, t_type v4, t_type v5, t_type v6, t_type v7,
        t_type v8, t_type v9, t_type va, t_type vb
    );
    template<typename T = t_type, typename = typename ipriv::enable_if<(sizeof(T)<sizeof(uint16_t))>::type>
    inline veci_t(
        t_type v0, t_type v1, t_type v2, t_type v3, t_type v4, t_type v5, t_type v6, t_type v7,
        t_type v8, t_type v9, t_type va, t_type vb, t_type vc
    );
    template<typename T = t_type, typename = typename ipriv::enable_if<(sizeof(T)<sizeof(uint16_t))>::type>
    inline veci_t(
        t_type v0, t_type v1, t_type v2, t_type v3, t_type v4, t_type v5, t_type v6, t_type v7,
        t_type v8, t_type v9, t_type va, t_type vb, t_type vc, t_type vd
    );
    template<typename T = t_type, typename = typename ipriv::enable_if<(sizeof(T)<sizeof(uint16_t))>::type>
    inline veci_t(
        t_type v0, t_type v1, t_type v2, t_type v3, t_type v4, t_type v5, t_type v6, t_type v7,
        t_type v8, t_type v9, t_type va, t_type vb, t_type vc, t_type vd, t_type ve
    );
    template<typename T = t_type, typename = typename ipriv::enable_if<(sizeof(T)<sizeof(uint16_t))>::type>
    inline veci_t(
        t_type v0, t_type v1, t_type v2, t_type v3, t_type v4, t_type v5, t_type v6, t_type v7,
        t_type v8, t_type v9, t_type va, t_type vb, t_type vc, t_type vd, t_type ve, t_type vf
    );
    // remaining overloads for AVX/AVX2 (t_type==uint8_t, N==32) considered impractical
    // remaining overloads for AVX512 considered impractical

    inline veci_t(const t_type * p) { for(unsigned i = 0; i < N; ++i) v[i] = p[i]; }
    inline explicit veci_t(packed_t v) { p = v; }

    inline veci_t(std::initializer_list<t_type> l);

    inline veci_t & operator=(packed_t v) { p = v; return *this; }

    inline t_type & operator[](size_t idx) { return v[idx]; }
    inline t_type operator[](size_t idx) const { return v[idx]; }

    inline explicit operator const t_type *() const { return v; }
    inline explicit operator t_type *() const { return v; }

    // vector addition, subtraction (not saturated)
    // and logical operations
    inline veci_t & operator+=(const veci_t & v2);
    inline veci_t & operator-=(const veci_t & v2);
    inline veci_t & operator&=(const veci_t & v2);
    inline veci_t & operator|=(const veci_t & v2);
    inline veci_t & operator^=(const veci_t & v2);

    // mul and div postponed -> need to find out the possible intrinsics first
    //vec_t & operator*=(const vec_t & v2);
    //vec_t & operator/=(const vec_t & v2);

    inline veci_t & operator+=(packed_t v2);
    inline veci_t & operator-=(packed_t v2);
    inline veci_t & operator&=(packed_t v2);
    inline veci_t & operator|=(packed_t v2);
    inline veci_t & operator^=(packed_t v2);

    // vector comparisons
    inline bool operator==(const veci_t &) const;
    inline bool operator!=(const veci_t &) const;
    inline bool operator< (const veci_t &) const;
    inline bool operator<=(const veci_t &) const;
    inline bool operator> (const veci_t &) const;
    inline bool operator>=(const veci_t &) const;
    // returns true if at least one value pair is unequal
    inline bool neq_one(const veci_t &) const;

    // packed vector comparisons
    bool operator==(packed_t) const;
    bool operator!=(packed_t) const;
    bool operator< (packed_t) const;
    bool operator<=(packed_t) const;
    bool operator> (packed_t) const;
    bool operator>=(packed_t) const;
    // returns true if at least one value pair is unequal
    bool neq_one(packed_t) const;

    // SSE2 supports for int16_t*8 and uint8_t*16 only
    // AVX: ???
    // TODO: either disable these per enable_if<> for all other combos
    //       or provide a scalar implementation (emit a performance warning
    //       message, too)
    static inline veci_t min_(const veci_t &, const veci_t &);
    static inline veci_t min_(packed_t, packed_t);
    static inline veci_t max_(const veci_t &, const veci_t &);
    static inline veci_t max_(packed_t, packed_t);

    // saturated adds and subs
    // SSE2 supports for [u]int8_t*16, [u]int16_t*8 only
    // (probably rarely needed for [u]int{32,64}_t)
    template<typename T = t_type, typename = typename ipriv::enable_if<(sizeof(T)<=sizeof(uint16_t))>::type>
    inline veci_t & add_sat(const veci_t &);
    template<typename T = t_type, typename = typename ipriv::enable_if<(sizeof(T)<=sizeof(uint16_t))>::type>
    inline veci_t & add_sat(packed_t);
    template<typename T = t_type, typename = typename ipriv::enable_if<(sizeof(T)<=sizeof(uint16_t))>::type>
    inline veci_t & sub_sat(const veci_t &);
    template<typename T = t_type, typename = typename ipriv::enable_if<(sizeof(T)<=sizeof(uint16_t))>::type>
    inline veci_t & sub_sat(packed_t);

    template<typename T = t_type, typename = typename ipriv::enable_if<(std::is_signed<T>::value)>::type>
    inline veci_t abs_() const;
    template<typename T = t_type, typename = typename ipriv::enable_if<(std::is_signed<T>::value)>::type>
    inline void abs_();

#if defined(PVECI_INTEL)
    inline veci_t operator~() const
    {
        // TODO: this impl is SSE2, not AVX/AVX2 (128 bits wide vectors only)
        return veci_t(_mm_xor_si128(p, math_t::onebits()));
    }
#elif defined(PVECI_ARM)
    inline veci_t operator~() const;
#endif
    
    // load aligned
    inline void loada(const t_type * p);
    // load unaligned
    inline void loadu(const t_type * p);

    // store unaligned
    inline void storeu(t_type *p);
    
    union {
        t_type v[N];
        packed_t p;
    };

};


/*****************************************************************************
 *                                                                           *
 * veci_i8x16_t implementation                                               *
 *                                                                           *
 *****************************************************************************/
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
// x86/x64 with MSVC
#  define veci_i8x16_t veci_t<int8_t,16,__m128i>
#elif defined(_MSC_VER) && defined(_M_ARM)
// ARM NEON with MSVC
#  define veci_i8x16_t veci_t<int8_t,16,__n128>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
// x86/x64 with GCC / MinGW
#  define veci_i8x16_t veci_t<int8_t,16,__m128i>
#elif defined(__GNUC__) && defined(__ARM_ARCH) && defined(__ARM_NEON)
// ARM NEON with GCC
#  define veci_i8x16_t veci_t<int8_t,16,int8x16_t>
#endif


template<> inline veci_i8x16_t::veci_t()
{ p = math_t::zeroes(); }

template<> inline veci_i8x16_t::veci_t(int8_t v0)
{ p = math_t::zeroes();
  v[0] = v0; }

template<> inline veci_i8x16_t::veci_t(int8_t v0, int8_t v1)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; }

template<> template<> inline veci_i8x16_t::veci_t(int8_t v0, int8_t v1, int8_t v2)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; }

template<> template<> inline veci_i8x16_t::veci_t(int8_t v0, int8_t v1, int8_t v2, int8_t v3)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3; }

template<> template<> inline veci_i8x16_t::veci_t(
        int8_t v0, int8_t v1, int8_t v2, int8_t v3,
        int8_t v4
)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; }

template<> template<> inline veci_i8x16_t::veci_t(
        int8_t v0, int8_t v1, int8_t v2, int8_t v3,
        int8_t v4, int8_t v5
)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; v[5] = v5; }

template<> template<> inline veci_i8x16_t::veci_t(
        int8_t v0, int8_t v1, int8_t v2, int8_t v3,
        int8_t v4, int8_t v5, int8_t v6
)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; v[5] = v5; v[6] = v6; }

template<> template<> inline veci_i8x16_t::veci_t(
        int8_t v0, int8_t v1, int8_t v2, int8_t v3,
        int8_t v4, int8_t v5, int8_t v6, int8_t v7
)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; v[5] = v5; v[6] = v6; v[7] = v7; }

template<> template<> inline veci_i8x16_t::veci_t(
        int8_t v0, int8_t v1, int8_t v2, int8_t v3,
        int8_t v4, int8_t v5, int8_t v6, int8_t v7,
        int8_t v8
)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; v[5] = v5; v[6] = v6; v[7] = v7;
  v[8] = v8; }

template<> template<> inline veci_i8x16_t::veci_t(
        int8_t v0, int8_t v1, int8_t v2, int8_t v3,
        int8_t v4, int8_t v5, int8_t v6, int8_t v7,
        int8_t v8, int8_t v9
)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; v[5] = v5; v[6] = v6; v[7] = v7;
  v[8] = v8; v[9] = v9; }

template<> template<> inline veci_i8x16_t::veci_t(
        int8_t v0, int8_t v1, int8_t v2, int8_t v3,
        int8_t v4, int8_t v5, int8_t v6, int8_t v7,
        int8_t v8, int8_t v9, int8_t va
)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; v[5] = v5; v[6] = v6; v[7] = v7;
  v[8] = v8; v[9] = v9; v[10] = va; }

template<> template<> inline veci_i8x16_t::veci_t(
        int8_t v0, int8_t v1, int8_t v2, int8_t v3,
        int8_t v4, int8_t v5, int8_t v6, int8_t v7,
        int8_t v8, int8_t v9, int8_t va, int8_t vb
)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; v[5] = v5; v[6] = v6; v[7] = v7;
  v[8] = v8; v[9] = v9; v[10] = va; v[11] = vb; }

template<> template<> inline veci_i8x16_t::veci_t(
        int8_t v0, int8_t v1, int8_t v2, int8_t v3,
        int8_t v4, int8_t v5, int8_t v6, int8_t v7,
        int8_t v8, int8_t v9, int8_t va, int8_t vb,
        int8_t vc
)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; v[5] = v5; v[6] = v6; v[7] = v7;
  v[8] = v8; v[9] = v9; v[10] = va; v[11] = vb;
  v[12] = vc; }

template<> template<> inline veci_i8x16_t::veci_t(
        int8_t v0, int8_t v1, int8_t v2, int8_t v3,
        int8_t v4, int8_t v5, int8_t v6, int8_t v7,
        int8_t v8, int8_t v9, int8_t va, int8_t vb,
        int8_t vc, int8_t vd
)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; v[5] = v5; v[6] = v6; v[7] = v7;
  v[8] = v8; v[9] = v9; v[10] = va; v[11] = vb;
  v[12] = vc; v[13] = vd; }

template<> template<> inline veci_i8x16_t::veci_t(
        int8_t v0, int8_t v1, int8_t v2, int8_t v3,
        int8_t v4, int8_t v5, int8_t v6, int8_t v7,
        int8_t v8, int8_t v9, int8_t va, int8_t vb,
        int8_t vc, int8_t vd, int8_t ve
)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; v[5] = v5; v[6] = v6; v[7] = v7;
  v[8] = v8; v[9] = v9; v[10] = va; v[11] = vb;
  v[12] = vc; v[13] = vd; v[14] = ve; }

template<> template<> inline veci_i8x16_t::veci_t(
        int8_t v0, int8_t v1, int8_t v2, int8_t v3,
        int8_t v4, int8_t v5, int8_t v6, int8_t v7,
        int8_t v8, int8_t v9, int8_t va, int8_t vb,
        int8_t vc, int8_t vd, int8_t ve, int8_t vf
)
{ v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; v[5] = v5; v[6] = v6; v[7] = v7;
  v[8] = v8; v[9] = v9; v[10] = va; v[11] = vb;
  v[12] = vc; v[13] = vd; v[14] = ve; v[15] = vf; }

template<> inline veci_i8x16_t::veci_t(std::initializer_list<type_t> l)
{
    p = math_t::zeroes();
    int count = int(l.size() <= 16 ? l.size() : 16);
    int i = 0; for(auto it = l.begin(); i < count; ++i, ++it) v[i] = *it;
}

// vector addition, subtraction (not saturated)
// and logical operations
template<> inline veci_i8x16_t & veci_i8x16_t::operator+=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_add_epi8(p, v2);
#elif defined(PVECI_ARM)
    p = vaddq_s8(p, v2);
#endif
    return *this;
}

template<> inline veci_i8x16_t & veci_i8x16_t::operator-=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_sub_epi8(p, v2);
#elif defined(PVECI_ARM)
    p = vsubq_s8(p, v2);
#endif
    return *this;
}

template<> inline veci_i8x16_t & veci_i8x16_t::operator&=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_and_si128(p, v2);
#elif defined(PVECI_ARM)
    p = vandq_s8(p, v2);
#endif
    return *this;
}

template<> inline veci_i8x16_t & veci_i8x16_t::operator|=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_or_si128(p, v2);
#elif defined(PVECI_ARM)
    p = vorrq_s8(p, v2);
#endif
    return *this;
}

template<> inline veci_i8x16_t & veci_i8x16_t::operator^=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_xor_si128(p, v2);
#elif defined(PVECI_ARM)
    p = veorq_s8(p, v2);
#endif
    return *this;
}

template<> inline veci_i8x16_t & veci_i8x16_t::operator+=(const veci_i8x16_t & v2)
{ return operator+=(v2.p); }
template<> inline veci_i8x16_t & veci_i8x16_t::operator-=(const veci_i8x16_t & v2)
{ return operator-=(v2.p); }
template<> inline veci_i8x16_t & veci_i8x16_t::operator&=(const veci_i8x16_t & v2)
{ return operator&=(v2.p); }
template<> inline veci_i8x16_t & veci_i8x16_t::operator|=(const veci_i8x16_t & v2)
{ return operator|=(v2.p); }
template<> inline veci_i8x16_t & veci_i8x16_t::operator^=(const veci_i8x16_t & v2)
{ return operator^=(v2.p); }


// packed vector comparisons
// (impl postponed for ARM NEON)
#if defined(PVECI_INTEL)
template<> inline bool veci_i8x16_t::operator==(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmpeq_epi8(p, v2)) == 0xFFFF;
#elif defined(PVECI_ARM)
    XXX
    vceqq_s8(p, v2)
#endif
}

template<> inline bool veci_i8x16_t::operator!=(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmpeq_epi8(p, v2)) == 0x0000;
#elif defined(PVECI_ARM)
    XXX
#endif
}

template<> inline bool veci_i8x16_t::operator<(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmplt_epi8(p, v2)) == 0xFFFF;
#elif defined(PVECI_ARM)
    XXX
#endif
}

template<> inline bool veci_i8x16_t::operator<=(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmpgt_epi8(p, v2)) == 0x0000;
#elif defined(PVECI_ARM)
    XXX
#endif
}

template<> inline bool veci_i8x16_t::operator>(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmpgt_epi8(p, v2)) == 0xFFFF;
#elif defined(PVECI_ARM)
    XXX
#endif
}

template<> inline bool veci_i8x16_t::operator>=(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmplt_epi8(p, v2)) == 0x0000;
#elif defined(PVECI_ARM)
    XXX
#endif
}

// returns true if at least one value pair is unequal
template<> inline bool veci_i8x16_t::neq_one(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmpeq_epi8(p, v2)) != 0xFFFF;
#elif defined(PVECI_ARM)
    XXX
#endif
}


// vector comparisons
template<> inline bool veci_i8x16_t::operator==(const veci_i8x16_t & v2) const
{ return operator==(v2.p); }
template<> inline bool veci_i8x16_t::operator!=(const veci_i8x16_t & v2) const
{ return operator!=(v2.p); }
template<> inline bool veci_i8x16_t::operator<(const veci_i8x16_t & v2) const
{ return operator<(v2.p); }
template<> inline bool veci_i8x16_t::operator<=(const veci_i8x16_t & v2) const
{ return operator<=(v2.p); }
template<> inline bool veci_i8x16_t::operator>(const veci_i8x16_t & v2) const
{ return operator>(v2.p); }
template<> inline bool veci_i8x16_t::operator>=(const veci_i8x16_t & v2) const
{ return operator>=(v2.p); }
template<> inline bool veci_i8x16_t::neq_one(const veci_i8x16_t & v2) const
{ return neq_one(v2.p); }

// (impl postponed for ARM NEON)
#endif // defined(PVECI_INTEL)


// SSE2 intrinsic support for int16_t*8 and uint8_t*16 only
// AVX2: ???
// TODO: either disable these per enable_if<> for all other combos
//       or provide a scalar implementation (emit a performance warning
//       message, too)
template<> inline veci_i8x16_t veci_i8x16_t::min_(packed_t p1, packed_t p2)
{
#if defined(PVECI_INTEL)
#   if defined(SSE4)
    return veci_i8x16_t(_mm_min_epi8(p1, p2));
#   else
#   pragma message("performance warning: SSE2 does not provide min() for packed int8_t x 16")
    veci_i8x16_t ret, v1(p1), v2(p2);
    for(int i = 0; i < 16; ++i)
        ret[i] = (std::min)(v1[i], v2[i]);
    return ret;
#endif
#elif defined(PVECI_ARM)
    return veci_i8x16_t(vminq_s8(p1, p2));
#endif
}

template<> inline veci_i8x16_t veci_i8x16_t::min_(const veci_i8x16_t & v1, const veci_i8x16_t & v2)
{ return min_(v1.p, v2.p); }

template<> inline veci_i8x16_t veci_i8x16_t::max_(packed_t p1, packed_t p2)
{
#if defined(PVECI_INTEL)
#   if defined(SSE4)
    return veci_i8x16_t(_mm_max_epi8(p1, p2));
#   else
#   pragma message("performance warning: SSE2 does not provide min() for packed int8_t x 16")
    veci_i8x16_t ret, v1(p1), v2(p2);
    for(int i = 0; i < 16; ++i)
        ret[i] = (std::max)(v1[i], v2[i]);
    return ret;
#endif
#elif defined(PVECI_ARM)
    return veci_i8x16_t(vmaxq_s8(p1, p2));
#endif
}

template<> inline veci_i8x16_t veci_i8x16_t::max_(const veci_i8x16_t & v1, const veci_i8x16_t & v2)
{ return max_(v1.p, v2.p); }


// saturated adds and subs
// SSE2 supports for [u]int8_t*16, [u]int16_t*8 only 
template<> template<> inline veci_i8x16_t & veci_i8x16_t::add_sat(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_adds_epi8(p, v2);
#elif defined(PVECI_ARM)
    p = vqaddq_s8(p, v2);
#endif
    return *this;
}

template<> template<> inline veci_i8x16_t & veci_i8x16_t::sub_sat(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_subs_epi8(p, v2);
#elif defined(PVECI_ARM)
    p = vqsubq_s8(p, v2);
#endif
    return *this;
}

template<> template<> inline veci_i8x16_t & veci_i8x16_t::add_sat(const veci_i8x16_t & v2)
{ return add_sat(v2.p); }
template<> template<> inline veci_i8x16_t & veci_i8x16_t::sub_sat(const veci_i8x16_t & v2)
{ return sub_sat(v2.p); }


template<> template<> inline void veci_i8x16_t::abs_()
{
#if defined(PVECI_INTEL)
# if defined(SSSE3)
    p = _mm_abs_epi8(p);
# else
#   pragma message("performance warning: SSE2 does not provide abs() for packed int8_t x 16")
    __m128i negmask = _mm_cmplt_epi8(p, _mm_setzero_si128()); // v>=0 -> 0x00, v<0 -> 0xFF
    p = 
        _mm_sub_epi8(
            _mm_xor_si128(p, negmask), // negmask[i]==0x00 -> p[i], negmask[i]==0xFF -> ~p[i]
            negmask
        );                             // negmask[i]==0x00 -> p[i]-0 = p[i]
                                       // negmask[i]==0xFF -> ~p[i]-(-1) = ~p[i]+1 = -p[i]
# endif
#elif defined(PVECI_ARM)
    p = vabsq_s8(p);
#endif
}

template<> template<> inline veci_i8x16_t veci_i8x16_t::abs_() const
{ veci_i8x16_t ret(p); ret.abs_(); return ret; }


// unary minus
inline veci_i8x16_t operator-(const veci_i8x16_t & v)
{
#if defined(PVECI_INTEL)
    return veci_i8x16_t(math::imath_t<int8_t,__m128i>::zeroes()) -= v;
#elif defined(PVECI_ARM)
#ifndef PVECI_ARM_GCC
    return veci_i8x16_t(math::imath_t<int8_t,__n128>::zeroes()) -= v;
#else
    return veci_i8x16_t(math::imath_t<int8_t,int8x16_t>::zeroes()) -= v;
#endif
#endif
}


// load aligned
template<> inline void veci_i8x16_t::loada(const int8_t * ptr)
{
#if defined(PVECI_INTEL)
    p = _mm_load_si128(reinterpret_cast<const packed_t *>(ptr));
#elif defined(PVECI_ARM)
    p = *reinterpret_cast<const packed_t *>(ptr);
#endif
}

// load unaligned
template<> inline void veci_i8x16_t::loadu(const int8_t * ptr)
{
#if defined(PVECI_INTEL)
    p = _mm_loadu_si128(reinterpret_cast<const packed_t *>(ptr));
#elif defined(PVECI_ARM)
    // this may not even be supported (or maybe it is supported per VLDx)
    p = *reinterpret_cast<const packed_t *>(ptr);
#endif
}

// store unaligned
template<> inline void veci_i8x16_t::storeu(int8_t * ptr)
{
#if defined(PVECI_INTEL)
    _mm_storeu_si128(reinterpret_cast<packed_t *>(ptr), p);
#elif defined(PVECI_ARM)
    // this may not even be supported (or maybe it is supported per VLDx)
    p = *reinterpret_cast<const packed_t *>(ptr);
#endif
}


#ifdef PVECI_ARM
template<> inline veci_i8x16_t veci_i8x16_t::operator~() const
{ return veci_i8x16_t(veorq_s8(p, math_t::onebits())); }
#endif



// free-standing arithmetic operations (element-wise)
inline veci_i8x16_t operator+(const veci_i8x16_t & v1, const veci_i8x16_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_i8x16_t(_mm_add_epi8(v1.p, v2.p));
#elif defined(PVECI_ARM)
    return veci_i8x16_t(vaddq_s8(v1.p, v2.p));
#endif
}

inline veci_i8x16_t operator+(const veci_i8x16_t & v, int8_t s)
{
#if defined(PVECI_INTEL)
    return veci_i8x16_t(_mm_add_epi8(v.p, _mm_set1_epi8(s)));
#elif defined(PVECI_ARM)
    return veci_i8x16_t(vaddq_s8(v.p, vdupq_n_s8(s)));
#endif
}

inline veci_i8x16_t operator+(int8_t s, const veci_i8x16_t & v)
{
#if defined(PVECI_INTEL)
    return veci_i8x16_t(_mm_add_epi8(_mm_set1_epi8(s), v.p));
#elif defined(PVECI_ARM)
    return veci_i8x16_t(vaddq_s8(vdupq_n_s8(s), v.p));
#endif
}

inline veci_i8x16_t operator+(const veci_i8x16_t & v1, veci_i8x16_t::packed_t v2)
{
#if defined(PVECI_INTEL)
    return veci_i8x16_t(_mm_add_epi8(v1.p, v2));
#elif defined(PVECI_ARM)
    return veci_i8x16_t(vaddq_s8(v1.p, v2));
#endif
}

inline veci_i8x16_t operator+(veci_i8x16_t::packed_t v1, const veci_i8x16_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_i8x16_t(_mm_add_epi8(v1, v2.p));
#elif defined(PVECI_ARM)
    return veci_i8x16_t(vaddq_s8(v1, v2.p));
#endif
}

inline veci_i8x16_t operator-(const veci_i8x16_t & v1, const veci_i8x16_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_i8x16_t(_mm_sub_epi8(v1.p, v2.p));
#elif defined(PVECI_ARM)
    return veci_i8x16_t(vsubq_s8(v1.p, v2.p));
#endif
}

inline veci_i8x16_t operator-(const veci_i8x16_t & v, int8_t s)
{
#if defined(PVECI_INTEL)
    return veci_i8x16_t(_mm_sub_epi8(v.p, _mm_set1_epi8(s)));
#elif defined(PVECI_ARM)
    return veci_i8x16_t(vsubq_s8(v.p, vdupq_n_s8(s)));
#endif
}

inline veci_i8x16_t operator-(int8_t s, const veci_i8x16_t & v)
{
#if defined(PVECI_INTEL)
    return veci_i8x16_t(_mm_sub_epi8(_mm_set1_epi8(s), v.p));
#elif defined(PVECI_ARM)
    return veci_i8x16_t(vsubq_s8(vdupq_n_s8(s), v.p));
#endif
}

inline veci_i8x16_t operator-(const veci_i8x16_t & v1, veci_i8x16_t::packed_t v2)
{
#if defined(PVECI_INTEL)
    return veci_i8x16_t(_mm_sub_epi8(v1.p, v2));
#elif defined(PVECI_ARM)
    return veci_i8x16_t(vsubq_s8(v1.p, v2));
#endif
}

inline veci_i8x16_t operator-(veci_i8x16_t::packed_t v1, const veci_i8x16_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_i8x16_t(_mm_sub_epi8(v1, v2.p));
#elif defined(PVECI_ARM)
    return veci_i8x16_t(vsubq_s8(v1, v2.p));
#endif
}


// free-standing bit-wise logical operations
inline veci_i8x16_t operator&(veci_i8x16_t op1, veci_i8x16_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i8x16_t(_mm_and_si128(op1.p, op2.p));
#elif defined(PVECI_ARM)
    return veci_i8x16_t(vandq_s8(op1.p, op2.p));
#endif
}

inline veci_i8x16_t operator&(veci_i8x16_t::packed_t op1, veci_i8x16_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i8x16_t(_mm_and_si128(op1, op2.p));
#elif defined(PVECI_ARM)
    return veci_i8x16_t(vandq_s8(op1, op2.p));
#endif
}

inline veci_i8x16_t operator&(veci_i8x16_t op1, veci_i8x16_t::packed_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i8x16_t(_mm_and_si128(op1.p, op2));
#elif defined(PVECI_ARM)
    return veci_i8x16_t(vandq_s8(op1.p, op2));
#endif
}

inline veci_i8x16_t operator|(veci_i8x16_t op1, veci_i8x16_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i8x16_t(_mm_or_si128(op1.p, op2.p));
#elif defined(PVECI_ARM)
    return veci_i8x16_t(vorrq_s8(op1.p, op2.p));
#endif
}

inline veci_i8x16_t operator|(veci_i8x16_t::packed_t op1, veci_i8x16_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i8x16_t(_mm_or_si128(op1, op2.p));
#elif defined(PVECI_ARM)
    return veci_i8x16_t(vorrq_s8(op1, op2.p));
#endif
}

inline veci_i8x16_t operator|(veci_i8x16_t op1, veci_i8x16_t::packed_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i8x16_t(_mm_or_si128(op1.p, op2));
#elif defined(PVECI_ARM)
    return veci_i8x16_t(vorrq_s8(op1.p, op2));
#endif
}

inline veci_i8x16_t operator^(veci_i8x16_t op1, veci_i8x16_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i8x16_t(_mm_xor_si128(op1.p, op2.p));
#elif defined(PVECI_ARM)
    return veci_i8x16_t(veorq_s8(op1.p, op2.p));
#endif
}

inline veci_i8x16_t operator^(veci_i8x16_t::packed_t op1, veci_i8x16_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i8x16_t(_mm_xor_si128(op1, op2.p));
#elif defined(PVECI_ARM)
    return veci_i8x16_t(veorq_s8(op1, op2.p));
#endif
}

inline veci_i8x16_t operator^(veci_i8x16_t op1, veci_i8x16_t::packed_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i8x16_t(_mm_xor_si128(op1.p, op2));
#elif defined(PVECI_ARM)
    return veci_i8x16_t(veorq_s8(op1.p, op2));
#endif
}


#undef veci_i8x16_t


/*****************************************************************************
 *                                                                           *
 * veci_ui8x16_t implementation                                              *
 *                                                                           *
 *****************************************************************************/
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
// x86/x64 with MSVC
#  define veci_ui8x16_t veci_t<uint8_t,16,__m128i>
#elif defined(_MSC_VER) && defined(_M_ARM)
// ARM NEON with MSVC
#  define veci_ui8x16_t veci_t<uint8_t,16,__n128>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
// x86/x64 with GCC / MinGW
#  define veci_ui8x16_t veci_t<uint8_t,16,__m128i>
#elif defined(__GNUC__) && defined(__ARM_ARCH) && defined(__ARM_NEON)
// ARM NEON with GCC
#  define veci_ui8x16_t veci_t<uint8_t,16,uint8x16_t>
#endif


template<> inline veci_ui8x16_t::veci_t()
{ p = math_t::zeroes(); }
template<> inline veci_ui8x16_t::veci_t(uint8_t v0)
{ p = math_t::zeroes();
  v[0] = v0; }
template<> inline veci_ui8x16_t::veci_t(uint8_t v0, uint8_t v1)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; }
template<> template<> inline veci_ui8x16_t::veci_t(uint8_t v0, uint8_t v1, uint8_t v2)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; }
template<> template<> inline veci_ui8x16_t::veci_t(uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3; }
template<> template<> inline veci_ui8x16_t::veci_t(
        uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3,
        uint8_t v4
)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; }
template<> template<> inline veci_ui8x16_t::veci_t(
        uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3,
        uint8_t v4, uint8_t v5
)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; v[5] = v5; }
template<> template<> inline veci_ui8x16_t::veci_t(
        uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3,
        uint8_t v4, uint8_t v5, uint8_t v6
)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; v[5] = v5; v[6] = v6; }
template<> template<> inline veci_ui8x16_t::veci_t(
        uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3,
        uint8_t v4, uint8_t v5, uint8_t v6, uint8_t v7
)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; v[5] = v5; v[6] = v6; v[7] = v7; }
template<> template<> inline veci_ui8x16_t::veci_t(
        uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3,
        uint8_t v4, uint8_t v5, uint8_t v6, uint8_t v7,
        uint8_t v8
)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; v[5] = v5; v[6] = v6; v[7] = v7;
  v[8] = v8; }
template<> template<> inline veci_ui8x16_t::veci_t(
        uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3,
        uint8_t v4, uint8_t v5, uint8_t v6, uint8_t v7,
        uint8_t v8, uint8_t v9
)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; v[5] = v5; v[6] = v6; v[7] = v7;
  v[8] = v8; v[9] = v9; }
template<> template<> inline veci_ui8x16_t::veci_t(
        uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3,
        uint8_t v4, uint8_t v5, uint8_t v6, uint8_t v7,
        uint8_t v8, uint8_t v9, uint8_t va
)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; v[5] = v5; v[6] = v6; v[7] = v7;
  v[8] = v8; v[9] = v9; v[10] = va; }
template<> template<> inline veci_ui8x16_t::veci_t(
        uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3,
        uint8_t v4, uint8_t v5, uint8_t v6, uint8_t v7,
        uint8_t v8, uint8_t v9, uint8_t va, uint8_t vb
)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; v[5] = v5; v[6] = v6; v[7] = v7;
  v[8] = v8; v[9] = v9; v[10] = va; v[11] = vb; }
template<> template<> inline veci_ui8x16_t::veci_t(
        uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3,
        uint8_t v4, uint8_t v5, uint8_t v6, uint8_t v7,
        uint8_t v8, uint8_t v9, uint8_t va, uint8_t vb,
        uint8_t vc
)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; v[5] = v5; v[6] = v6; v[7] = v7;
  v[8] = v8; v[9] = v9; v[10] = va; v[11] = vb;
  v[12] = vc; }
template<> template<> inline veci_ui8x16_t::veci_t(
        uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3,
        uint8_t v4, uint8_t v5, uint8_t v6, uint8_t v7,
        uint8_t v8, uint8_t v9, uint8_t va, uint8_t vb,
        uint8_t vc, uint8_t vd
)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; v[5] = v5; v[6] = v6; v[7] = v7;
  v[8] = v8; v[9] = v9; v[10] = va; v[11] = vb;
  v[12] = vc; v[13] = vd; }
template<> template<> inline veci_ui8x16_t::veci_t(
        uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3,
        uint8_t v4, uint8_t v5, uint8_t v6, uint8_t v7,
        uint8_t v8, uint8_t v9, uint8_t va, uint8_t vb,
        uint8_t vc, uint8_t vd, uint8_t ve
)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; v[5] = v5; v[6] = v6; v[7] = v7;
  v[8] = v8; v[9] = v9; v[10] = va; v[11] = vb;
  v[12] = vc; v[13] = vd; v[14] = ve; }
template<> template<> inline veci_ui8x16_t::veci_t(
        uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3,
        uint8_t v4, uint8_t v5, uint8_t v6, uint8_t v7,
        uint8_t v8, uint8_t v9, uint8_t va, uint8_t vb,
        uint8_t vc, uint8_t vd, uint8_t ve, uint8_t vf
)
{ v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; v[5] = v5; v[6] = v6; v[7] = v7;
  v[8] = v8; v[9] = v9; v[10] = va; v[11] = vb;
  v[12] = vc; v[13] = vd; v[14] = ve; v[15] = vf; }

template<> inline veci_ui8x16_t::veci_t(std::initializer_list<type_t> l)
{
    p = math_t::zeroes();
    int count = int(l.size() <= 16 ? l.size() : 16);
    int i = 0; for(auto it = l.begin(); i < count; ++i, ++it) v[i] = *it;
}


// vector addition, subtraction (not saturated)
// and logical operations
template<> inline veci_ui8x16_t & veci_ui8x16_t::operator+=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_add_epi8(p, v2);
#elif defined(PVECI_ARM)
    p = vaddq_u8(p, v2);
#endif
    return *this;
}
template<> inline veci_ui8x16_t & veci_ui8x16_t::operator-=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_sub_epi8(p, v2);
#elif defined(PVECI_ARM)
    p = vsubq_u8(p, v2);
#endif
    return *this;
}
template<> inline veci_ui8x16_t & veci_ui8x16_t::operator&=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_and_si128(p, v2);
#elif defined(PVECI_ARM)
    p = vandq_u8(p, v2);
#endif
    return *this;
}
template<> inline veci_ui8x16_t & veci_ui8x16_t::operator|=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_or_si128(p, v2);
#elif defined(PVECI_ARM)
    p = vorrq_u8(p, v2);
#endif
    return *this;
}
template<> inline veci_ui8x16_t & veci_ui8x16_t::operator^=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_xor_si128(p, v2);
#elif defined(PVECI_ARM)
    p = veorq_u8(p, v2);
#endif
    return *this;
}

template<> inline veci_ui8x16_t & veci_ui8x16_t::operator+=(const veci_ui8x16_t & v2)
{ return operator+=(v2.p); }
template<> inline veci_ui8x16_t & veci_ui8x16_t::operator-=(const veci_ui8x16_t & v2)
{ return operator-=(v2.p); }
template<> inline veci_ui8x16_t & veci_ui8x16_t::operator&=(const veci_ui8x16_t & v2)
{ return operator&=(v2.p); }
template<> inline veci_ui8x16_t & veci_ui8x16_t::operator|=(const veci_ui8x16_t & v2)
{ return operator|=(v2.p); }
template<> inline veci_ui8x16_t & veci_ui8x16_t::operator^=(const veci_ui8x16_t & v2)
{ return operator^=(v2.p); }



// packed vector comparisons
// (impl postponed for ARM NEON)
#ifdef PVECI_INTEL
template<> inline bool veci_ui8x16_t::operator==(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmpeq_epi8(p, v2)) == 0xFFFF;
#elif defined(PVECI_ARM)
    XXX
#endif
}

template<> inline bool veci_ui8x16_t::operator!=(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmpeq_epi8(p, v2)) == 0x0000;
#elif defined(PVECI_ARM)
    XXX
#endif
}

template<> inline bool veci_ui8x16_t::operator<(packed_t v2) const
{
#if defined(PVECI_INTEL)
#pragma message("performance warning: SSE2 does not provide comparison operations for packed uint8_t x 16")
    return
        _mm_movemask_epi8(
            _mm_cmplt_epi8(
                _mm_xor_si128(p, math_t::sign_mask()),
                _mm_xor_si128(v2, math_t::sign_mask())
            )
        ) == 0xFFFF;
#elif defined(PVECI_ARM)
    XXX
#endif
}

template<> inline bool veci_ui8x16_t::operator<=(packed_t v2) const
{
#if defined(PVECI_INTEL)
#pragma message("performance warning: SSE2 does not provide comparison operations for packed uint8_t x 16")
    //static_assert(false, "not implemented yet");
    return
        _mm_movemask_epi8(
            _mm_cmpgt_epi8(
                _mm_xor_si128(p, math_t::sign_mask()),
                _mm_xor_si128(v2, math_t::sign_mask())
            )
        ) == 0x0000;
#elif defined(PVECI_ARM)
    XXX
#endif
}

template<> inline bool veci_ui8x16_t::operator>(packed_t v2) const
{
#if defined(PVECI_INTEL)
#pragma message("performance warning: SSE2 does not provide comparison operations for packed uint8_t x 16")
    return
        _mm_movemask_epi8(
            _mm_cmpgt_epi8(
                _mm_xor_si128(p, math_t::sign_mask()),
                _mm_xor_si128(v2, math_t::sign_mask())
            )
        ) == 0xFFFF;
#elif defined(PVECI_ARM)
    XXX
#endif
}

template<> inline bool veci_ui8x16_t::operator>=(packed_t v2) const
{
#if defined(PVECI_INTEL)
#pragma message("performance warning: SSE2 does not provide comparison operations for packed uint8_t x 16")
    return
        _mm_movemask_epi8(
            _mm_cmplt_epi8(
                _mm_xor_si128(p, math_t::sign_mask()),
                _mm_xor_si128(v2, math_t::sign_mask())
            )
        ) == 0x0000;
#elif defined(PVECI_ARM)
    XXX
#endif
}

// returns true if at least one value pair is unequal
template<> inline bool veci_ui8x16_t::neq_one(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmpeq_epi8(p, v2)) != 0xFFFF;
#elif defined(PVECI_ARM)
    XXX
#endif
}


// vector comparisons
template<> inline bool veci_ui8x16_t::operator==(const veci_ui8x16_t & v2) const
{ return operator==(v2.p); }
template<> inline bool veci_ui8x16_t::operator!=(const veci_ui8x16_t & v2) const
{ return operator!=(v2.p); }
template<> inline bool veci_ui8x16_t::operator<(const veci_ui8x16_t & v2) const
{ return operator<(v2.p); }
template<> inline bool veci_ui8x16_t::operator<=(const veci_ui8x16_t & v2) const
{ return operator<=(v2.p); }
template<> inline bool veci_ui8x16_t::operator>(const veci_ui8x16_t & v2) const
{ return operator>(v2.p); }
template<> inline bool veci_ui8x16_t::operator>=(const veci_ui8x16_t & v2) const
{ return operator>=(v2.p); }
// returns true if at least one value pair is unequal
template<> inline bool veci_ui8x16_t::neq_one(const veci_ui8x16_t & v2) const
{ return neq_one(v2.p); }

// (impl postponed for ARM NEON)
#endif // defined(PVECI_INTEL)


// SSE2 intrinsic support for int16_t*8 and uint8_t*16 only
// AVX2: ???
// TODO: either disable these per enable_if<> for all other combos
//       or provide a scalar implementation (emit a performance warning
//       message, too)
template<> inline veci_ui8x16_t veci_ui8x16_t::min_(packed_t p1, packed_t p2)
{
#if defined(PVECI_INTEL)
    return veci_ui8x16_t(_mm_min_epu8(p1, p2));
#elif defined(PVECI_ARM)
    return veci_ui8x16_t(vminq_u8(p1, p2));
#endif
}

template<> inline veci_ui8x16_t veci_ui8x16_t::min_(const veci_ui8x16_t & v1, const veci_ui8x16_t & v2)
{ return min_(v1.p, v2.p); }

template<> inline veci_ui8x16_t veci_ui8x16_t::max_(packed_t p1, packed_t p2)
{
#if defined(PVECI_INTEL)
    return veci_ui8x16_t(_mm_max_epu8(p1, p2));
#elif defined(PVECI_ARM)
    return veci_ui8x16_t(vmaxq_u8(p1, p2));
#endif
}

template<> inline veci_ui8x16_t veci_ui8x16_t::max_(const veci_ui8x16_t & v1, const veci_ui8x16_t & v2)
{ return max_(v1.p, v2.p); }


// saturated adds and subs
// SSE2 supports for [u]int8_t*16, [u]int16_t*8 only 
template<> template<> inline veci_ui8x16_t & veci_ui8x16_t::add_sat(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_adds_epu8(p, v2);
#elif defined(PVECI_ARM)
    p = vqaddq_u8(p, v2);
#endif
    return *this;
}

template<> template<> inline veci_ui8x16_t & veci_ui8x16_t::sub_sat(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_subs_epu8(p, v2);
#elif defined(PVECI_ARM)
    p = vqsubq_u8(p, v2);
#endif
    return *this;
}

template<> template<> inline veci_ui8x16_t & veci_ui8x16_t::add_sat(const veci_ui8x16_t & v2)
{ return add_sat(v2.p); }
template<> template<> inline veci_ui8x16_t & veci_ui8x16_t::sub_sat(const veci_ui8x16_t & v2)
{ return sub_sat(v2.p); }


// load aligned
template<> inline void veci_ui8x16_t::loada(const uint8_t * ptr)
{
#if defined(PVECI_INTEL)
    p = _mm_load_si128(reinterpret_cast<const packed_t *>(ptr));
#elif defined(PVECI_ARM)
    p = *reinterpret_cast<const packed_t *>(ptr);
#endif
}

// load unaligned
template<> inline void veci_ui8x16_t::loadu(const uint8_t * ptr)
{
#if defined(PVECI_INTEL)
    p = _mm_loadu_si128(reinterpret_cast<const packed_t *>(ptr));
#elif defined(PVECI_ARM)
    p = *reinterpret_cast<const packed_t *>(ptr);
#endif
}

// store unaligned
template<> inline void veci_ui8x16_t::storeu(uint8_t * ptr)
{
#if defined(PVECI_INTEL)
    _mm_storeu_si128(reinterpret_cast<packed_t *>(ptr), p);
#elif defined(PVECI_ARM)
    // this may not even be supported (or maybe it is supported per VLDx)
    p = *reinterpret_cast<const packed_t *>(ptr);
#endif
}


#ifdef PVECI_ARM
template<> inline veci_ui8x16_t veci_ui8x16_t::operator~() const
{ return veci_ui8x16_t(veorq_u8(p, math_t::onebits())); }
#endif



// free-standing arithmetic operations (element-wise)
inline veci_ui8x16_t operator+(const veci_ui8x16_t & v1, const veci_ui8x16_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_ui8x16_t(_mm_add_epi8(v1.p, v2.p));
#elif defined(PVECI_ARM)
    return veci_ui8x16_t(vaddq_u8(v1.p, v2.p));
#endif
}

inline veci_ui8x16_t operator+(const veci_ui8x16_t & v, uint8_t s)
{
#if defined(PVECI_INTEL)
    return veci_ui8x16_t(_mm_add_epi8(v.p, _mm_set1_epi8(s)));
#elif defined(PVECI_ARM)
    return veci_ui8x16_t(vaddq_u8(v.p, vdupq_n_u8(s)));
#endif
}

inline veci_ui8x16_t operator+(uint8_t s, const veci_ui8x16_t & v)
{
#if defined(PVECI_INTEL)
    return veci_ui8x16_t(_mm_add_epi8(_mm_set1_epi8(s), v.p));
#elif defined(PVECI_ARM)
    return veci_ui8x16_t(vaddq_u8(vdupq_n_u8(s), v.p));
#endif
}

inline veci_ui8x16_t operator+(const veci_ui8x16_t & v1, veci_ui8x16_t::packed_t v2)
{
#if defined(PVECI_INTEL)
    return veci_ui8x16_t(_mm_add_epi8(v1.p, v2));
#elif defined(PVECI_ARM)
    return veci_ui8x16_t(vaddq_u8(v1.p, v2));
#endif
}

inline veci_ui8x16_t operator+(veci_ui8x16_t::packed_t v1, const veci_ui8x16_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_ui8x16_t(_mm_add_epi8(v1, v2.p));
#elif defined(PVECI_ARM)
    return veci_ui8x16_t(vaddq_u8(v1, v2.p));
#endif
}

inline veci_ui8x16_t operator-(const veci_ui8x16_t & v1, const veci_ui8x16_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_ui8x16_t(_mm_sub_epi8(v1.p, v2.p));
#elif defined(PVECI_ARM)
    return veci_ui8x16_t(vsubq_u8(v1.p, v2.p));
#endif
}

inline veci_ui8x16_t operator-(const veci_ui8x16_t & v, uint8_t s)
{
#if defined(PVECI_INTEL)
    return veci_ui8x16_t(_mm_sub_epi8(v.p, _mm_set1_epi8(s)));
#elif defined(PVECI_ARM)
    return veci_ui8x16_t(vsubq_u8(v.p, vdupq_n_u8(s)));
#endif
}

inline veci_ui8x16_t operator-(uint8_t s, const veci_ui8x16_t & v)
{
#if defined(PVECI_INTEL)
    return veci_ui8x16_t(_mm_sub_epi8(_mm_set1_epi8(s), v.p));
#elif defined(PVECI_ARM)
    return veci_ui8x16_t(vsubq_u8(vdupq_n_u8(s), v.p));
#endif
}

inline veci_ui8x16_t operator-(const veci_ui8x16_t & v1, veci_ui8x16_t::packed_t v2)
{
#if defined(PVECI_INTEL)
    return veci_ui8x16_t(_mm_sub_epi8(v1.p, v2));
#elif defined(PVECI_ARM)
    return veci_ui8x16_t(vsubq_u8(v1.p, v2));
#endif
}

inline veci_ui8x16_t operator-(veci_ui8x16_t::packed_t v1, const veci_ui8x16_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_ui8x16_t(_mm_sub_epi8(v1, v2.p));
#elif defined(PVECI_ARM)
    return veci_ui8x16_t(vsubq_u8(v1, v2.p));
#endif
}

// free-standing bit-wise logical operations
inline veci_ui8x16_t operator&(veci_ui8x16_t op1, veci_ui8x16_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui8x16_t(_mm_and_si128(op1.p, op2.p));
#elif defined(PVECI_ARM)
    return veci_ui8x16_t(vandq_u8(op1.p, op2.p));
#endif
}

inline veci_ui8x16_t operator&(veci_ui8x16_t::packed_t op1, veci_ui8x16_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui8x16_t(_mm_and_si128(op1, op2.p));
#elif defined(PVECI_ARM)
    return veci_ui8x16_t(vandq_u8(op1, op2.p));
#endif
}

inline veci_ui8x16_t operator&(veci_ui8x16_t op1, veci_ui8x16_t::packed_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui8x16_t(_mm_and_si128(op1.p, op2));
#elif defined(PVECI_ARM)
    return veci_ui8x16_t(vandq_u8(op1.p, op2));
#endif
}

inline veci_ui8x16_t operator|(veci_ui8x16_t op1, veci_ui8x16_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui8x16_t(_mm_or_si128(op1.p, op2.p));
#elif defined(PVECI_ARM)
    return veci_ui8x16_t(vorrq_u8(op1.p, op2.p));
#endif
}

inline veci_ui8x16_t operator|(veci_ui8x16_t::packed_t op1, veci_ui8x16_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui8x16_t(_mm_or_si128(op1, op2.p));
#elif defined(PVECI_ARM)
    return veci_ui8x16_t(vorrq_u8(op1, op2.p));
#endif
}

inline veci_ui8x16_t operator|(veci_ui8x16_t op1, veci_ui8x16_t::packed_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui8x16_t(_mm_or_si128(op1.p, op2));
#elif defined(PVECI_ARM)
    return veci_ui8x16_t(vorrq_u8(op1.p, op2));
#endif
}

inline veci_ui8x16_t operator^(veci_ui8x16_t op1, veci_ui8x16_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui8x16_t(_mm_xor_si128(op1.p, op2.p));
#elif defined(PVECI_ARM)
    return veci_ui8x16_t(veorq_u8(op1.p, op2.p));
#endif
}

inline veci_ui8x16_t operator^(veci_ui8x16_t::packed_t op1, veci_ui8x16_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui8x16_t(_mm_xor_si128(op1, op2.p));
#elif defined(PVECI_ARM)
    return veci_ui8x16_t(veorq_u8(op1, op2.p));
#endif
}

inline veci_ui8x16_t operator^(veci_ui8x16_t op1, veci_ui8x16_t::packed_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui8x16_t(_mm_xor_si128(op1.p, op2));
#elif defined(PVECI_ARM)
    return veci_ui8x16_t(veorq_u8(op1.p, op2));
#endif
}


#undef veci_ui8x16_t


/*****************************************************************************
 *                                                                           *
 * veci_i16x8_t implementation                                               *
 *                                                                           *
 *****************************************************************************/
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
// x86/x64 with MSVC
#  define veci_i16x8_t veci_t<int16_t,8,__m128i>
#elif defined(_MSC_VER) && defined(_M_ARM)
// ARM NEON with MSVC
#  define veci_i16x8_t veci_t<int16_t,8,__n128>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
// x86/x64 with GCC / MinGW
#  define veci_i16x8_t veci_t<int16_t,8,__m128i>
#elif defined(__GNUC__) && defined(__ARM_ARCH) && defined(__ARM_NEON)
// ARM NEON with GCC
#  define veci_i16x8_t veci_t<int16_t,8,int16x8_t>
#endif

template<> inline veci_i16x8_t::veci_t()
{ p = math_t::zeroes(); }
template<> inline veci_i16x8_t::veci_t(int16_t v0)
{ p = math_t::zeroes();
  v[0] = v0; }
template<> inline veci_i16x8_t::veci_t(int16_t v0, int16_t v1)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; }
template<> template<> inline veci_i16x8_t::veci_t(int16_t v0, int16_t v1, int16_t v2)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; }
template<> template<> inline veci_i16x8_t::veci_t(int16_t v0, int16_t v1, int16_t v2, int16_t v3)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3; }
template<> template<> inline veci_i16x8_t::veci_t(
        int16_t v0, int16_t v1, int16_t v2, int16_t v3,
        int16_t v4
)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; }
template<> template<> inline veci_i16x8_t::veci_t(
        int16_t v0, int16_t v1, int16_t v2, int16_t v3,
        int16_t v4, int16_t v5
)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; v[5] = v5; }
template<> template<> inline veci_i16x8_t::veci_t(
        int16_t v0, int16_t v1, int16_t v2, int16_t v3,
        int16_t v4, int16_t v5, int16_t v6
)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; v[5] = v5; v[6] = v6; }
template<> template<> inline veci_i16x8_t::veci_t(
        int16_t v0, int16_t v1, int16_t v2, int16_t v3,
        int16_t v4, int16_t v5, int16_t v6, int16_t v7
)
{ v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; v[5] = v5; v[6] = v6; v[7] = v7; }
template<> inline veci_i16x8_t::veci_t(std::initializer_list<type_t> l)
{
    p = math_t::zeroes();
    int count = int(l.size() <= 8 ? l.size() : 8);
    int i = 0; for(auto it = l.begin(); i < count; ++i, ++it) v[i] = *it;
}


// vector addition, subtraction (not saturated)
// and logical operations
template<> inline veci_i16x8_t & veci_i16x8_t::operator+=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_add_epi16(p, v2);
#elif defined(PVECI_ARM)
    p = vaddq_s16(p, v2);
#endif
    return *this;
}
template<> inline veci_i16x8_t & veci_i16x8_t::operator-=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_sub_epi16(p, v2);
#elif defined(PVECI_ARM)
    p = vsubq_s16(p, v2);
#endif
    return *this;
}
template<> inline veci_i16x8_t & veci_i16x8_t::operator&=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_and_si128(p, v2);
#elif defined(PVECI_ARM)
    p = vandq_s16(p, v2);
#endif
    return *this;
}
template<> inline veci_i16x8_t & veci_i16x8_t::operator|=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_or_si128(p, v2);
#elif defined(PVECI_ARM)
    p = vorrq_s16(p, v2);
#endif
    return *this;
}
template<> inline veci_i16x8_t & veci_i16x8_t::operator^=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_xor_si128(p, v2);
#elif defined(PVECI_ARM)
    p = veorq_s16(p, v2);
#endif
    return *this;
}

template<> inline veci_i16x8_t & veci_i16x8_t::operator+=(const veci_i16x8_t & v2)
{ return operator+=(v2.p); }
template<> inline veci_i16x8_t & veci_i16x8_t::operator-=(const veci_i16x8_t & v2)
{ return operator-=(v2.p); }
template<> inline veci_i16x8_t & veci_i16x8_t::operator&=(const veci_i16x8_t & v2)
{ return operator&=(v2.p); }
template<> inline veci_i16x8_t & veci_i16x8_t::operator|=(const veci_i16x8_t & v2)
{ return operator|=(v2.p); }
template<> inline veci_i16x8_t & veci_i16x8_t::operator^=(const veci_i16x8_t & v2)
{ return operator^=(v2.p); }



// packed vector comparisons
// (impl postponed for ARM NEON)
#ifdef PVECI_INTEL
template<> inline bool veci_i16x8_t::operator==(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmpeq_epi16(p, v2)) == 0xFFFF;
#elif defined(PVECI_ARM)
    XXX
#endif
}
template<> inline bool veci_i16x8_t::operator!=(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmpeq_epi16(p, v2)) == 0x0000;
#elif defined(PVECI_ARM)
    XXX
#endif
}
template<> inline bool veci_i16x8_t::operator<(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmplt_epi16(p, v2)) == 0xFFFF;
#elif defined(PVECI_ARM)
    XXX
#endif
}
template<> inline bool veci_i16x8_t::operator<=(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmpgt_epi16(p, v2)) == 0x0000;
#elif defined(PVECI_ARM)
    XXX
#endif
}
template<> inline bool veci_i16x8_t::operator>(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmpgt_epi16(p, v2)) == 0xFFFF;
#elif defined(PVECI_ARM)
    XXX
#endif
}
template<> inline bool veci_i16x8_t::operator>=(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmplt_epi16(p, v2)) == 0x0000;
#elif defined(PVECI_ARM)
    XXX
#endif
}
// returns true if at least one value pair is unequal
template<> inline bool veci_i16x8_t::neq_one(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmpeq_epi16(p, v2)) != 0xFFFF;
#elif defined(PVECI_ARM)
    XXX
#endif
}


// vector comparisons
template<> inline bool veci_i16x8_t::operator==(const veci_i16x8_t & v2) const
{ return operator==(v2.p); }
template<> inline bool veci_i16x8_t::operator!=(const veci_i16x8_t & v2) const
{ return operator!=(v2.p); }
template<> inline bool veci_i16x8_t::operator<(const veci_i16x8_t & v2) const
{ return operator<(v2.p); }
template<> inline bool veci_i16x8_t::operator<=(const veci_i16x8_t & v2) const
{ return operator<=(v2.p); }
template<> inline bool veci_i16x8_t::operator>(const veci_i16x8_t & v2) const
{ return operator>(v2.p); }
template<> inline bool veci_i16x8_t::operator>=(const veci_i16x8_t & v2) const
{ return operator>=(v2.p); }
// returns true if at least one value pair is unequal
template<> inline bool veci_i16x8_t::neq_one(const veci_i16x8_t & v2) const
{ return neq_one(v2.p); }

// (impl postponed for ARM NEON)
#endif // defined(PVECI_INTEL)


// SSE2 intrinsic support for int16_t*8 and uint8_t*16 only
// AVX2: ???
// TODO: either disable these per enable_if<> for all other combos
//       or provide a scalar implementation (emit a performance warning
//       message, too)
template<> inline veci_i16x8_t veci_i16x8_t::min_(packed_t p1, packed_t p2)
{
#if defined(PVECI_INTEL)
    return veci_i16x8_t(_mm_min_epi16(p1, p2));
#elif defined(PVECI_ARM)
    return veci_i16x8_t(vminq_s16(p1, p2));
#endif
}
template<> inline veci_i16x8_t veci_i16x8_t::min_(const veci_i16x8_t & v1, const veci_i16x8_t & v2)
{ return min_(v1.p, v2.p); }

template<> inline veci_i16x8_t veci_i16x8_t::max_(packed_t p1, packed_t p2)
{
#if defined(PVECI_INTEL)
    return veci_i16x8_t(_mm_max_epi16(p1, p2));
#elif defined(PVECI_ARM)
    return veci_i16x8_t(vmaxq_s16(p1, p2));
#endif
}
template<> inline veci_i16x8_t veci_i16x8_t::max_(const veci_i16x8_t & v1, const veci_i16x8_t & v2)
{ return max_(v1.p, v2.p); }

// saturated adds and subs
// SSE2 supports for [u]int8_t*16, [u]int16_t*8 only 
template<> template<> inline veci_i16x8_t & veci_i16x8_t::add_sat(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_adds_epi16(p, v2);
#elif defined(PVECI_ARM)
    p = vqaddq_s16(p, v2);
#endif
    return *this;
}
template<> template<> inline veci_i16x8_t & veci_i16x8_t::sub_sat(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_subs_epi16(p, v2);
#elif defined(PVECI_ARM)
    p = vqsubq_s16(p, v2);
#endif
    return *this;
}
template<> template<> inline veci_i16x8_t & veci_i16x8_t::add_sat(const veci_i16x8_t & v2)
{ return add_sat(v2.p); }
template<> template<> inline veci_i16x8_t & veci_i16x8_t::sub_sat(const veci_i16x8_t & v2)
{ return sub_sat(v2.p); }


template<> template<> inline void veci_i16x8_t::abs_()
{
#if defined(PVECI_INTEL)
# if defined(SSSE3)
    p = _mm_abs_epi16(p);
# else
#   pragma message("performance warning: SSE2 does not provide abs() for packed int16_t x 8")
    __m128i mask = _mm_srai_epi16(p, 15); // (...SSSS,...) -> 0 for positive, -1 for negative
    // result = S==0 ? x-0 : ~x-(-1)
    p = 
        _mm_sub_epi16(
            _mm_xor_si128(p, mask), // (...PPPP for S==0, ~...PPPP for S==1)
            mask
        );
# endif
#elif defined(PVECI_ARM)
    p = vabsq_s16(p);
#endif
}
template<> template<> inline veci_i16x8_t veci_i16x8_t::abs_() const
{ veci_i16x8_t ret(p); ret.abs_(); return ret; }


// unary minus
inline veci_i16x8_t operator-(const veci_i16x8_t & v)
{
#if defined(PVECI_INTEL)
    return veci_i16x8_t(math::imath_t<int16_t,__m128i>::zeroes()) -= v;
#elif defined(PVECI_ARM)
#ifndef PVECI_ARM_GCC
    return veci_i16x8_t(math::imath_t<int16_t,__n128>::zeroes()) -= v;
#else
    return veci_i16x8_t(math::imath_t<int16_t,int16x8_t>::zeroes()) -= v;
#endif
#endif
}


// load aligned
template<> inline void veci_i16x8_t::loada(const int16_t * ptr)
{
#if defined(PVECI_INTEL)
    p = _mm_load_si128(reinterpret_cast<const packed_t *>(ptr));
#elif defined(PVECI_ARM)
    p = *reinterpret_cast<const packed_t *>(ptr);
#endif
}

// load unaligned
template<> inline void veci_i16x8_t::loadu(const int16_t * ptr)
{
#if defined(PVECI_INTEL)
    p = _mm_loadu_si128(reinterpret_cast<const packed_t *>(ptr));
#elif defined(PVECI_ARM)
    p = *reinterpret_cast<const packed_t *>(ptr);
#endif
}

// store unaligned
template<> inline void veci_i16x8_t::storeu(int16_t * ptr)
{
#if defined(PVECI_INTEL)
    _mm_storeu_si128(reinterpret_cast<packed_t *>(ptr), p);
#elif defined(PVECI_ARM)
    // this may not even be supported (or maybe it is supported per VLDx)
    p = *reinterpret_cast<const packed_t *>(ptr);
#endif
}


#ifdef PVECI_ARM
template<> inline veci_i16x8_t veci_i16x8_t::operator~() const
{ return veci_i16x8_t(veorq_s16(p, math_t::onebits())); }
#endif



// free-standing arithmetic operations (element-wise)
inline veci_i16x8_t operator+(const veci_i16x8_t & v1, const veci_i16x8_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_i16x8_t(_mm_add_epi16(v1.p, v2.p));
#elif defined(PVECI_ARM)
    return veci_i16x8_t(vaddq_s16(v1.p, v2.p));
#endif
}
inline veci_i16x8_t operator+(const veci_i16x8_t & v, int16_t s)
{
#if defined(PVECI_INTEL)
    return veci_i16x8_t(_mm_add_epi16(v.p, _mm_set1_epi16(s)));
#elif defined(PVECI_ARM)
    return veci_i16x8_t(vaddq_s16(v.p, vmovq_n_s16(s)));
#endif
}
inline veci_i16x8_t operator+(int16_t s, const veci_i16x8_t & v)
{
#if defined(PVECI_INTEL)
    return veci_i16x8_t(_mm_add_epi16(_mm_set1_epi16(s), v.p));
#elif defined(PVECI_ARM)
    return veci_i16x8_t(vaddq_s16(vmovq_n_s16(s), v.p));
#endif
}
inline veci_i16x8_t operator+(const veci_i16x8_t & v1, veci_i16x8_t::packed_t v2)
{
#if defined(PVECI_INTEL)
    return veci_i16x8_t(_mm_add_epi16(v1.p, v2));
#elif defined(PVECI_ARM)
    return veci_i16x8_t(vaddq_s16(v1.p, v2));
#endif
}
inline veci_i16x8_t operator+(veci_i16x8_t::packed_t v1, const veci_i16x8_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_i16x8_t(_mm_add_epi16(v1, v2.p));
#elif defined(PVECI_ARM)
    return veci_i16x8_t(vaddq_s16(v1, v2.p));
#endif
}

inline veci_i16x8_t operator-(const veci_i16x8_t & v1, const veci_i16x8_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_i16x8_t(_mm_sub_epi16(v1.p, v2.p));
#elif defined(PVECI_ARM)
    return veci_i16x8_t(vsubq_s16(v1.p, v2.p));
#endif
}
inline veci_i16x8_t operator-(const veci_i16x8_t & v, int16_t s)
{
#if defined(PVECI_INTEL)
    return veci_i16x8_t(_mm_sub_epi16(v.p, _mm_set1_epi16(s)));
#elif defined(PVECI_ARM)
    return veci_i16x8_t(vsubq_s16(v.p, vmovq_n_s16(s)));
#endif
}
inline veci_i16x8_t operator-(int16_t s, const veci_i16x8_t & v)
{
#if defined(PVECI_INTEL)
    return veci_i16x8_t(_mm_sub_epi16(_mm_set1_epi16(s), v.p));
#elif defined(PVECI_ARM)
    return veci_i16x8_t(vsubq_s16(vmovq_n_s16(s), v.p));
#endif
}
inline veci_i16x8_t operator-(const veci_i16x8_t & v1, veci_i16x8_t::packed_t v2)
{
#if defined(PVECI_INTEL)
    return veci_i16x8_t(_mm_sub_epi16(v1.p, v2));
#elif defined(PVECI_ARM)
    return veci_i16x8_t(vsubq_s16(v1.p, v2));
#endif
}
inline veci_i16x8_t operator-(veci_i16x8_t::packed_t v1, const veci_i16x8_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_i16x8_t(_mm_sub_epi16(v1, v2.p));
#elif defined(PVECI_ARM)
    return veci_i16x8_t(vsubq_s16(v1, v2.p));
#endif
}


// free-standing bit-wise logical operations
inline veci_i16x8_t operator&(veci_i16x8_t op1, veci_i16x8_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i16x8_t(_mm_and_si128(op1.p, op2.p));
#elif defined(PVECI_ARM)
    return veci_i16x8_t(vandq_s16(op1.p, op2.p));
#endif
}
inline veci_i16x8_t operator&(veci_i16x8_t::packed_t op1, veci_i16x8_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i16x8_t(_mm_and_si128(op1, op2.p));
#elif defined(PVECI_ARM)
    return veci_i16x8_t(vandq_s16(op1, op2.p));
#endif
}
inline veci_i16x8_t operator&(veci_i16x8_t op1, veci_i16x8_t::packed_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i16x8_t(_mm_and_si128(op1.p, op2));
#elif defined(PVECI_ARM)
    return veci_i16x8_t(vandq_s16(op1.p, op2));
#endif
}

inline veci_i16x8_t operator|(veci_i16x8_t op1, veci_i16x8_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i16x8_t(_mm_or_si128(op1.p, op2.p));
#elif defined(PVECI_ARM)
    return veci_i16x8_t(vorrq_s16(op1.p, op2.p));
#endif
}
inline veci_i16x8_t operator|(veci_i16x8_t::packed_t op1, veci_i16x8_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i16x8_t(_mm_or_si128(op1, op2.p));
#elif defined(PVECI_ARM)
    return veci_i16x8_t(vorrq_s16(op1, op2.p));
#endif
}
inline veci_i16x8_t operator|(veci_i16x8_t op1, veci_i16x8_t::packed_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i16x8_t(_mm_or_si128(op1.p, op2));
#elif defined(PVECI_ARM)
    return veci_i16x8_t(vorrq_s16(op1.p, op2));
#endif
}

inline veci_i16x8_t operator^(veci_i16x8_t op1, veci_i16x8_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i16x8_t(_mm_xor_si128(op1.p, op2.p));
#elif defined(PVECI_ARM)
    return veci_i16x8_t(veorq_s16(op1.p, op2.p));
#endif
}
inline veci_i16x8_t operator^(veci_i16x8_t::packed_t op1, veci_i16x8_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i16x8_t(_mm_xor_si128(op1, op2.p));
#elif defined(PVECI_ARM)
    return veci_i16x8_t(veorq_s16(op1, op2.p));
#endif
}
inline veci_i16x8_t operator^(veci_i16x8_t op1, veci_i16x8_t::packed_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i16x8_t(_mm_xor_si128(op1.p, op2));
#elif defined(PVECI_ARM)
    return veci_i16x8_t(veorq_s16(op1.p, op2));
#endif
}


#undef veci_i16x8_t


/*****************************************************************************
 *                                                                           *
 * veci_ui16x8_t implementation                                              *
 *                                                                           *
 *****************************************************************************/
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
// x86/x64 with MSVC
#  define veci_ui16x8_t veci_t<uint16_t,8,__m128i>
#elif defined(_MSC_VER) && defined(_M_ARM)
// ARM NEON with MSVC
#  define veci_ui16x8_t veci_t<uint16_t,8,__n128>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
// x86/x64 with GCC / MinGW
#  define veci_ui16x8_t veci_t<uint16_t,8,__m128i>
#elif defined(__GNUC__) && defined(__ARM_ARCH) && defined(__ARM_NEON)
// ARM NEON with GCC
#  define veci_ui16x8_t veci_t<uint16_t,8,uint16x8_t>
#endif

template<> inline veci_ui16x8_t::veci_t()
{ p = math_t::zeroes(); }
template<> inline veci_ui16x8_t::veci_t(uint16_t v0)
{ p = math_t::zeroes();
  v[0] = v0; }
template<> inline veci_ui16x8_t::veci_t(uint16_t v0, uint16_t v1)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; }
template<> template<> inline veci_ui16x8_t::veci_t(uint16_t v0, uint16_t v1, uint16_t v2)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; }
template<> template<> inline veci_ui16x8_t::veci_t(uint16_t v0, uint16_t v1, uint16_t v2, uint16_t v3)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3; }
template<> template<> inline veci_ui16x8_t::veci_t(
        uint16_t v0, uint16_t v1, uint16_t v2, uint16_t v3,
        uint16_t v4
)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; }
template<> template<> inline veci_ui16x8_t::veci_t(
        uint16_t v0, uint16_t v1, uint16_t v2, uint16_t v3,
        uint16_t v4, uint16_t v5
)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; v[5] = v5; }
template<> template<> inline veci_ui16x8_t::veci_t(
        uint16_t v0, uint16_t v1, uint16_t v2, uint16_t v3,
        uint16_t v4, uint16_t v5, uint16_t v6
)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; v[5] = v5; v[6] = v6; }
template<> template<> inline veci_ui16x8_t::veci_t(
        uint16_t v0, uint16_t v1, uint16_t v2, uint16_t v3,
        uint16_t v4, uint16_t v5, uint16_t v6, uint16_t v7
)
{ v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
  v[4] = v4; v[5] = v5; v[6] = v6; v[7] = v7; }
template<> inline veci_ui16x8_t::veci_t(std::initializer_list<type_t> l)
{
    p = math_t::zeroes();
    int count = int(l.size() <= 8 ? l.size() : 8);
    int i = 0; for(auto it = l.begin(); i < count; ++i, ++it) v[i] = *it;
}


// vector addition, subtraction (not saturated)
// and logical operations
template<> inline veci_ui16x8_t & veci_ui16x8_t::operator+=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_add_epi16(p, v2);
#elif defined(PVECI_ARM)
    p = vaddq_u16(p, v2);
#endif
    return *this;
}
template<> inline veci_ui16x8_t & veci_ui16x8_t::operator-=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_sub_epi16(p, v2);
#elif defined(PVECI_ARM)
    p = vsubq_u16(p, v2);
#endif
    return *this;
}
template<> inline veci_ui16x8_t & veci_ui16x8_t::operator&=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_and_si128(p, v2);
#elif defined(PVECI_ARM)
    p = vandq_u16(p, v2);
#endif
    return *this;
}
template<> inline veci_ui16x8_t & veci_ui16x8_t::operator|=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_or_si128(p, v2);
#elif defined(PVECI_ARM)
    p = vorrq_u16(p, v2);
#endif
    return *this;
}
template<> inline veci_ui16x8_t & veci_ui16x8_t::operator^=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_xor_si128(p, v2);
#elif defined(PVECI_ARM)
    p = veorq_u16(p, v2);
#endif
    return *this;
}

template<> inline veci_ui16x8_t & veci_ui16x8_t::operator+=(const veci_ui16x8_t & v2)
{ return operator+=(v2.p); }
template<> inline veci_ui16x8_t & veci_ui16x8_t::operator-=(const veci_ui16x8_t & v2)
{ return operator-=(v2.p); }
template<> inline veci_ui16x8_t & veci_ui16x8_t::operator&=(const veci_ui16x8_t & v2)
{ return operator&=(v2.p); }
template<> inline veci_ui16x8_t & veci_ui16x8_t::operator|=(const veci_ui16x8_t & v2)
{ return operator|=(v2.p); }
template<> inline veci_ui16x8_t & veci_ui16x8_t::operator^=(const veci_ui16x8_t & v2)
{ return operator^=(v2.p); }



// packed vector comparisons
// (impl postponed for ARM NEON)
#ifdef PVECI_INTEL
template<> inline bool veci_ui16x8_t::operator==(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmpeq_epi16(p, v2)) == 0xFFFF;
#elif defined(PVECI_ARM)
    XXX
#endif
}
template<> inline bool veci_ui16x8_t::operator!=(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmpeq_epi16(p, v2)) == 0x0000;
#elif defined(PVECI_ARM)
    XXX
#endif
}
template<> inline bool veci_ui16x8_t::operator<(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return
        _mm_movemask_epi8(
            _mm_cmplt_epi16(
                _mm_xor_si128(p, math_t::sign_mask()),
                _mm_xor_si128(v2, math_t::sign_mask())
            )
        ) == 0xFFFF;
#elif defined(PVECI_ARM)
    XXX
#endif
}
template<> inline bool veci_ui16x8_t::operator<=(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return
        _mm_movemask_epi8(
            _mm_cmpgt_epi16(
                _mm_xor_si128(p, math_t::sign_mask()),
                _mm_xor_si128(v2, math_t::sign_mask())
            )
        ) == 0x0000;
#elif defined(PVECI_ARM)
    XXX
#endif
}
template<> inline bool veci_ui16x8_t::operator>(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return
        _mm_movemask_epi8(
            _mm_cmpgt_epi16(
                _mm_xor_si128(p, math_t::sign_mask()),
                _mm_xor_si128(v2, math_t::sign_mask())
            )
        ) == 0xFFFF;
#elif defined(PVECI_ARM)
    XXX
#endif
}
template<> inline bool veci_ui16x8_t::operator>=(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return
        _mm_movemask_epi8(
            _mm_cmplt_epi16(
                _mm_xor_si128(p, math_t::sign_mask()),
                _mm_xor_si128(v2, math_t::sign_mask())
            )
        ) == 0x0000;
#elif defined(PVECI_ARM)
    XXX
#endif
}
// returns true if at least one value pair is unequal
template<> inline bool veci_ui16x8_t::neq_one(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmpeq_epi16(p, v2)) != 0xFFFF;
#elif defined(PVECI_ARM)
    XXX
#endif
}

// vector comparisons
template<> inline bool veci_ui16x8_t::operator==(const veci_ui16x8_t & v2) const
{ return operator==(v2.p); }
template<> inline bool veci_ui16x8_t::operator!=(const veci_ui16x8_t & v2) const
{ return operator!=(v2.p); }
template<> inline bool veci_ui16x8_t::operator<(const veci_ui16x8_t & v2) const
{ return operator<(v2.p); }
template<> inline bool veci_ui16x8_t::operator<=(const veci_ui16x8_t & v2) const
{ return operator<=(v2.p); }
template<> inline bool veci_ui16x8_t::operator>(const veci_ui16x8_t & v2) const
{ return operator>(v2.p); }
template<> inline bool veci_ui16x8_t::operator>=(const veci_ui16x8_t & v2) const
{ return operator>=(v2.p); }
// returns true if at least one value pair is unequal
template<> inline bool veci_ui16x8_t::neq_one(const veci_ui16x8_t & v2) const
{ return neq_one(v2.p); }

// (impl postponed for ARM NEON)
#endif // defined(PVECI_INTEL)


// SSE2 intrinsic support for int16_t*8 and uint8_t*16 only
// AVX2: ???
// TODO: either disable these per enable_if<> for all other combos
//       or provide a scalar implementation (emit a performance warning
//       message, too)
template<> inline veci_ui16x8_t veci_ui16x8_t::min_(packed_t p1, packed_t p2)
{
#if defined(PVECI_INTEL)
# if defined(SSE4)
    return veci_ui16x8_t(_mm_min_epu16(p1, p2));
# else
#   pragma message("performance warning: SSE2 does not provide min() for packed uint16_t x 8")
#   if 1
    veci_ui16x8_t ret, v1(p1), v2(p2);
    for(int i = 0; i < 8; ++i) ret[i] = (std::min)(v1[i], v2[i]);
    return ret;
#   else
    // TODO: this may be wrong because _mm_cmplt_epi16() does a signed comparison
    __m128i mask = _mm_cmplt_epi16(p1, p2); // p1[i]<p[2]->0xFFFF, p1[i]>=p2[i]->0x0000
    return
        uivec16x8_t(
            _mm_or_si128(
                _mm_and_si128(mask, p1),    // mask[i]==0x0000->0, mask[i]==0xFFFF->p1[i]
                _mm_andnot_si128(mask, p2)  // mask[i]==0x0000->p2[i], mask[i]==0xFFFF->0
            )
        );
#   endif
# endif
#elif defined(PVECI_ARM)
    return veci_ui16x8_t(vminq_u16(p1, p2));
#endif
}
template<> inline veci_ui16x8_t veci_ui16x8_t::min_(const veci_ui16x8_t & v1, const veci_ui16x8_t & v2)
{ return min_(v1.p, v2.p); }

template<> inline veci_ui16x8_t veci_ui16x8_t::max_(packed_t p1, packed_t p2)
{
#if defined(PVECI_INTEL)
# if defined(SSE4)
    return veci_ui16x8_t(_mm_max_epu16(p1, p2));
# else
#   pragma message("performance warning: SSE2 does not provide max() for packed uint16_t x 8")
#   if 1
    veci_ui16x8_t ret, v1(p1), v2(p2);
    for(int i = 0; i < 8; ++i) ret[i] = (std::max)(v1[i], v2[i]);
    return ret;
#   else
    // TODO: this may be wrong because _mm_cmpgt_epi16() does a signed comparison
    __m128i mask = _mm_cmpgt_epi16(p1, p2); // p1[i]>p[2]->0xFFFF, p1[i]<=p2[i]->0x0000
    return
        uivec16x8_t(
            _mm_or_si128(
                _mm_and_si128(mask, p1),    // mask[i]==0x0000->0, mask[i]==0xFFFF->p1[i]
                _mm_andnot_si128(mask, p2)  // mask[i]==0x0000->p2[i], mask[i]==0xFFFF->0
            )
        );
#   endif
# endif
#elif defined(PVECI_ARM)
    return veci_ui16x8_t(vmaxq_u16(p1, p2));
#endif
}
template<> inline veci_ui16x8_t veci_ui16x8_t::max_(const veci_ui16x8_t & v1, const veci_ui16x8_t & v2)
{ return max_(v1.p, v2.p); }

// saturated adds and subs
// SSE2 supports for [u]int8_t*16, [u]int16_t*8 only 
template<> template<> inline veci_ui16x8_t & veci_ui16x8_t::add_sat(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_adds_epu16(p, v2);
#elif defined(PVECI_ARM)
    p = vqaddq_u16(p, v2);
#endif
    return *this;
}
template<> template<> inline veci_ui16x8_t & veci_ui16x8_t::sub_sat(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_subs_epu16(p, v2);
#elif defined(PVECI_ARM)
    p = vqsubq_u16(p, v2);
#endif
    return *this;
}
template<> template<> inline veci_ui16x8_t & veci_ui16x8_t::add_sat(const veci_ui16x8_t & v2)
{ return add_sat(v2.p); }
template<> template<> inline veci_ui16x8_t & veci_ui16x8_t::sub_sat(const veci_ui16x8_t & v2)
{ return sub_sat(v2.p); }


// load aligned
template<> inline void veci_ui16x8_t::loada(const uint16_t * ptr)
{
#if defined(PVECI_INTEL)
    p = _mm_load_si128(reinterpret_cast<const packed_t *>(ptr));
#elif defined(PVECI_ARM)
    p = *reinterpret_cast<const packed_t *>(ptr);
#endif
}

// load unaligned
template<> inline void veci_ui16x8_t::loadu(const uint16_t * ptr)
{
#if defined(PVECI_INTEL)
    p = _mm_loadu_si128(reinterpret_cast<const packed_t *>(ptr));
#elif defined(PVECI_ARM)
    p = *reinterpret_cast<const packed_t *>(ptr);
#endif
}

// store unaligned
template<> inline void veci_ui16x8_t::storeu(uint16_t * ptr)
{
#if defined(PVECI_INTEL)
    _mm_storeu_si128(reinterpret_cast<packed_t *>(ptr), p);
#elif defined(PVECI_ARM)
    // this may not even be supported (or maybe it is supported per VLDx)
    p = *reinterpret_cast<const packed_t *>(ptr);
#endif
}


#ifdef PVECI_ARM
template<> inline veci_ui16x8_t veci_ui16x8_t::operator~() const
{ return veci_ui16x8_t(veorq_u16(p, math_t::onebits())); }
#endif


// free-standing arithmetic operations (element-wise)
inline veci_ui16x8_t operator+(const veci_ui16x8_t & v1, const veci_ui16x8_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_ui16x8_t(_mm_add_epi16(v1.p, v2.p));
#elif defined(PVECI_ARM)
    return veci_ui16x8_t(vaddq_u16(v1.p, v2.p));
#endif
}
inline veci_ui16x8_t operator+(const veci_ui16x8_t & v, uint16_t s)
{
#if defined(PVECI_INTEL)
    return veci_ui16x8_t(_mm_add_epi16(v.p, _mm_set1_epi16(s)));
#elif defined(PVECI_ARM)
    return veci_ui16x8_t(vaddq_u16(v.p, vmovq_n_u16(s)));
#endif
}
inline veci_ui16x8_t operator+(uint16_t s, const veci_ui16x8_t & v)
{
#if defined(PVECI_INTEL)
    return veci_ui16x8_t(_mm_add_epi16(_mm_set1_epi16(s), v.p));
#elif defined(PVECI_ARM)
    return veci_ui16x8_t(vaddq_u16(vmovq_n_u16(s), v.p));
#endif
}
inline veci_ui16x8_t operator+(const veci_ui16x8_t & v1, veci_ui16x8_t::packed_t v2)
{
#if defined(PVECI_INTEL)
    return veci_ui16x8_t(_mm_add_epi16(v1.p, v2));
#elif defined(PVECI_ARM)
    return veci_ui16x8_t(vaddq_u16(v1.p, v2));
#endif
}
inline veci_ui16x8_t operator+(veci_ui16x8_t::packed_t v1, const veci_ui16x8_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_ui16x8_t(_mm_add_epi16(v1, v2.p));
#elif defined(PVECI_ARM)
    return veci_ui16x8_t(vaddq_u16(v1, v2.p));
#endif
}

inline veci_ui16x8_t operator-(const veci_ui16x8_t & v1, const veci_ui16x8_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_ui16x8_t(_mm_sub_epi16(v1.p, v2.p));
#elif defined(PVECI_ARM)
    return veci_ui16x8_t(vsubq_u16(v1.p, v2.p));
#endif
}
inline veci_ui16x8_t operator-(const veci_ui16x8_t & v, uint16_t s)
{
#if defined(PVECI_INTEL)
    return veci_ui16x8_t(_mm_sub_epi16(v.p, _mm_set1_epi16(s)));
#elif defined(PVECI_ARM)
    return veci_ui16x8_t(vsubq_u16(v.p, vmovq_n_u16(s)));
#endif
}
inline veci_ui16x8_t operator-(uint16_t s, const veci_ui16x8_t & v)
{
#if defined(PVECI_INTEL)
    return veci_ui16x8_t(_mm_sub_epi16(_mm_set1_epi16(s), v.p));
#elif defined(PVECI_ARM)
    return veci_ui16x8_t(vsubq_u16(vmovq_n_u16(s), v.p));
#endif
}
inline veci_ui16x8_t operator-(const veci_ui16x8_t & v1, veci_ui16x8_t::packed_t v2)
{
#if defined(PVECI_INTEL)
    return veci_ui16x8_t(_mm_sub_epi16(v1.p, v2));
#elif defined(PVECI_ARM)
    return veci_ui16x8_t(vsubq_u16(v1.p, v2));
#endif
}
inline veci_ui16x8_t operator-(veci_ui16x8_t::packed_t v1, const veci_ui16x8_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_ui16x8_t(_mm_sub_epi16(v1, v2.p));
#elif defined(PVECI_ARM)
    return veci_ui16x8_t(vsubq_u16(v1, v2.p));
#endif
}


// free-standing bit-wise logical operations
inline veci_ui16x8_t operator&(veci_ui16x8_t op1, veci_ui16x8_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui16x8_t(_mm_and_si128(op1.p, op2.p));
#elif defined(PVECI_ARM)
    return veci_ui16x8_t(vandq_u16(op1.p, op2.p));
#endif
}
inline veci_ui16x8_t operator&(veci_ui16x8_t::packed_t op1, veci_ui16x8_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui16x8_t(_mm_and_si128(op1, op2.p));
#elif defined(PVECI_ARM)
    return veci_ui16x8_t(vandq_u16(op1, op2.p));
#endif
}
inline veci_ui16x8_t operator&(veci_ui16x8_t op1, veci_ui16x8_t::packed_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui16x8_t(_mm_and_si128(op1.p, op2));
#elif defined(PVECI_ARM)
    return veci_ui16x8_t(vandq_u16(op1.p, op2));
#endif
}

inline veci_ui16x8_t operator|(veci_ui16x8_t op1, veci_ui16x8_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui16x8_t(_mm_or_si128(op1.p, op2.p));
#elif defined(PVECI_ARM)
    return veci_ui16x8_t(vorrq_u16(op1.p, op2.p));
#endif
}
inline veci_ui16x8_t operator|(veci_ui16x8_t::packed_t op1, veci_ui16x8_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui16x8_t(_mm_or_si128(op1, op2.p));
#elif defined(PVECI_ARM)
    return veci_ui16x8_t(vorrq_u16(op1, op2.p));
#endif
}
inline veci_ui16x8_t operator|(veci_ui16x8_t op1, veci_ui16x8_t::packed_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui16x8_t(_mm_or_si128(op1.p, op2));
#elif defined(PVECI_ARM)
    return veci_ui16x8_t(vorrq_u16(op1.p, op2));
#endif
}

inline veci_ui16x8_t operator^(veci_ui16x8_t op1, veci_ui16x8_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui16x8_t(_mm_xor_si128(op1.p, op2.p));
#elif defined(PVECI_ARM)
    return veci_ui16x8_t(veorq_u16(op1.p, op2.p));
#endif
}
inline veci_ui16x8_t operator^(veci_ui16x8_t::packed_t op1, veci_ui16x8_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui16x8_t(_mm_xor_si128(op1, op2.p));
#elif defined(PVECI_ARM)
    return veci_ui16x8_t(veorq_u16(op1, op2.p));
#endif
}
inline veci_ui16x8_t operator^(veci_ui16x8_t op1, veci_ui16x8_t::packed_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui16x8_t(_mm_xor_si128(op1.p, op2));
#elif defined(PVECI_ARM)
    return veci_ui16x8_t(veorq_u16(op1.p, op2));
#endif
}


#undef veci_ui16x8_t


/*****************************************************************************
 *                                                                           *
 * veci_i32x4_t implementation                                               *
 *                                                                           *
 *****************************************************************************/
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
// x86/x64 with MSVC
#  define veci_i32x4_t veci_t<int32_t,4,__m128i>
#elif defined(_MSC_VER) && defined(_M_ARM)
// ARM NEON with MSVC
#  define veci_i32x4_t veci_t<int32_t,4,__n128>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
// x86/x64 with GCC / MinGW
#  define veci_i32x4_t veci_t<int32_t,4,__m128i>
#elif defined(__GNUC__) && defined(__ARM_ARCH) && defined(__ARM_NEON)
// ARM NEON with GCC
#  define veci_i32x4_t veci_t<int32_t,4,int32x4_t>
#endif

template<> inline veci_i32x4_t::veci_t()
{ p = math_t::zeroes(); }
template<> inline veci_i32x4_t::veci_t(int32_t v0)
{ p = math_t::zeroes();
  v[0] = v0; }
template<> inline veci_i32x4_t::veci_t(int32_t v0, int32_t v1)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; }
template<> template<> inline veci_i32x4_t::veci_t(int32_t v0, int32_t v1, int32_t v2)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; }
template<> template<> inline veci_i32x4_t::veci_t(int32_t v0, int32_t v1, int32_t v2, int32_t v3)
{ v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3; }
template<> inline veci_i32x4_t::veci_t(std::initializer_list<type_t> l)
{
    p = math_t::zeroes();
    int count = int(l.size() <= 4 ? l.size() : 4);
    int i = 0; for(auto it = l.begin(); i < count; ++i, ++it) v[i] = *it;
}


// vector addition, subtraction (not saturated)
// and logical operations
template<> inline veci_i32x4_t & veci_i32x4_t::operator+=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_add_epi32(p, v2);
#elif defined(PVECI_ARM)
    p = vaddq_s32(p, v2);
#endif
    return *this;
}
template<> inline veci_i32x4_t & veci_i32x4_t::operator-=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_sub_epi32(p, v2);
#elif defined(PVECI_ARM)
    p = vsubq_s32(p, v2);
#endif
    return *this;
}
template<> inline veci_i32x4_t & veci_i32x4_t::operator&=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_and_si128(p, v2);
#elif defined(PVECI_ARM)
    p = vandq_s32(p, v2);
#endif
    return *this;
}
template<> inline veci_i32x4_t & veci_i32x4_t::operator|=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_or_si128(p, v2);
#elif defined(PVECI_ARM)
    p = vorrq_s32(p, v2);
#endif
    return *this;
}
template<> inline veci_i32x4_t & veci_i32x4_t::operator^=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_xor_si128(p, v2);
#elif defined(PVECI_ARM)
    p = veorq_s32(p, v2);
#endif
    return *this;
}

template<> inline veci_i32x4_t & veci_i32x4_t::operator+=(const veci_i32x4_t & v2)
{ return operator+=(v2.p); }
template<> inline veci_i32x4_t & veci_i32x4_t::operator-=(const veci_i32x4_t & v2)
{ return operator-=(v2.p); }
template<> inline veci_i32x4_t & veci_i32x4_t::operator&=(const veci_i32x4_t & v2)
{ return operator&=(v2.p); }
template<> inline veci_i32x4_t & veci_i32x4_t::operator|=(const veci_i32x4_t & v2)
{ return operator|=(v2.p); }
template<> inline veci_i32x4_t & veci_i32x4_t::operator^=(const veci_i32x4_t & v2)
{ return operator^=(v2.p); }


// packed vector comparisons
// (impl postponed for ARM NEON)
#ifdef PVECI_INTEL
template<> inline bool veci_i32x4_t::operator==(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmpeq_epi32(p, v2)) == 0xFFFF;
#elif defined(PVECI_ARM)
    XXX
#endif
}
template<> inline bool veci_i32x4_t::operator!=(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmpeq_epi32(p, v2)) == 0x0000;
#elif defined(PVECI_ARM)
    XXX
#endif
}
template<> inline bool veci_i32x4_t::operator<(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmplt_epi32(p, v2)) == 0xFFFF;
#elif defined(PVECI_ARM)
    XXX
#endif
}
template<> inline bool veci_i32x4_t::operator<=(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmpgt_epi32(p, v2)) == 0x0000;
#elif defined(PVECI_ARM)
    XXX
#endif
}
template<> inline bool veci_i32x4_t::operator>(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmpgt_epi32(p, v2)) == 0xFFFF;
#elif defined(PVECI_ARM)
    XXX
#endif
}
template<> inline bool veci_i32x4_t::operator>=(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmplt_epi32(p, v2)) == 0x0000;
#elif defined(PVECI_ARM)
    XXX
#endif
}
// returns true if at least one value pair is unequal
template<> inline bool veci_i32x4_t::neq_one(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmpeq_epi8(p, v2)) != 0xFFFF;
#elif defined(PVECI_ARM)
    XXX
#endif
}

// vector comparisons
template<> inline bool veci_i32x4_t::operator==(const veci_i32x4_t & v2) const
{ return operator==(v2.p); }
template<> inline bool veci_i32x4_t::operator!=(const veci_i32x4_t & v2) const
{ return operator!=(v2.p); }
template<> inline bool veci_i32x4_t::operator<(const veci_i32x4_t & v2) const
{ return operator<(v2.p); }
template<> inline bool veci_i32x4_t::operator<=(const veci_i32x4_t & v2) const
{ return operator<=(v2.p); }
template<> inline bool veci_i32x4_t::operator>(const veci_i32x4_t & v2) const
{ return operator>(v2.p); }
template<> inline bool veci_i32x4_t::operator>=(const veci_i32x4_t & v2) const
{ return operator>=(v2.p); }
// returns true if at least one value pair is unequal
template<> inline bool veci_i32x4_t::neq_one(const veci_i32x4_t & v2) const
{ return neq_one(v2.p); }

// (impl postponed for ARM NEON)
#endif // defined(PVECI_INTEL)


// SSE2 intrinsic support for int16_t*8 and uint8_t*16 only
// AVX2: ???
// TODO: either disable these per enable_if<> for all other combos
//       or provide a scalar implementation (emit a performance warning
//       message, too)
template<> inline veci_i32x4_t veci_i32x4_t::min_(packed_t p1, packed_t p2)
{
#if defined(PVECI_INTEL)
# if defined(SSE4)
    return veci_i32x4_t(_mm_min_epi32(p1, p2));
# else
#   pragma message("performance warning: SSE2 does not provide min() for packed int32_t x 4")
    __m128i mask = _mm_cmplt_epi32(p1, p2); // p1[i]>=p2[i]->mask[i]=0x00, p1[i]<p2[i]->mask[i]=0xFF..
    return
        veci_i32x4_t(
            _mm_or_si128(
                _mm_and_si128(mask, p1),    // mask[i]==0->0, mask[i]==0xFF..->p1[i]
                _mm_andnot_si128(mask, p2)  // mask[i]==0->p2[i], mask[i]==0xFF..->0
            )
        );
# endif
#elif defined(PVECI_ARM)
    return veci_i32x4_t(vminq_s32(p1, p2));
#endif
}
template<> inline veci_i32x4_t veci_i32x4_t::min_(const veci_i32x4_t & v1, const veci_i32x4_t & v2)
{ return min_(v1.p, v2.p); }

template<> inline veci_i32x4_t veci_i32x4_t::max_(packed_t p1, packed_t p2)
{
#if defined(PVECI_INTEL)
# if defined(SSE4)
    return veci_i32x4_t(_mm_max_epi32(p1, p2));
# else
#   pragma message("performance warning: SSE2 does not provide max() for packed int32_t x 4")
    __m128i mask = _mm_cmpgt_epi32(p1, p2); // p1[i]>p2[i]->mask[i]=0xFF.., p1[i]<=p2[i]->mask[i]=0x00
    return
        veci_i32x4_t(
            _mm_or_si128(
                _mm_and_si128(mask, p1),    // mask[i]==0->0, mask[i]==0xFF..->p1[i]
                _mm_andnot_si128(mask, p2)  // mask[i]==0->p2[i], mask[i]==0xFF..->0
            )
        );
# endif
#elif defined(PVECI_ARM)
    return veci_i32x4_t(vmaxq_s32(p1, p2));
#endif
}
template<> inline veci_i32x4_t veci_i32x4_t::max_(const veci_i32x4_t & v1, const veci_i32x4_t & v2)
{ return max_(v1.p, v2.p); }



template<> template<> inline void veci_i32x4_t::abs_()
{
#if defined(PVECI_INTEL)
# if defined(SSSE3)
    p = _mm_abs_epi32(p);
# else
#   pragma message("performance warning: SSE2 does not provide abs() for packed int32_t x 4")
    __m128i mask = _mm_srai_epi32(p, 31); // extract sign bit into all elem bits
    p = _mm_sub_epi32(_mm_xor_si128(p, mask), mask); // subtract 0 if positive, invert an subtract -1 if negative
# endif
#elif defined(PVECI_ARM)
    // cause of some problems but not sure
    // (there's a comment in the mozilla webrtc sources about vabsq_s16() not changing -32768, so
    //  these instructions might have expected corner cases?)
    p = vabsq_s32(p);

#endif
}
template<> template<> inline veci_i32x4_t veci_i32x4_t::abs_() const
{ veci_i32x4_t ret(p); ret.abs_(); return ret; }


// unary minus
inline veci_i32x4_t operator-(const veci_i32x4_t & v)
{
    
#if defined(PVECI_INTEL)
    return veci_i32x4_t(math::imath_t<int32_t,__m128i>::zeroes()) -= v.p;
#elif defined(PVECI_ARM)
#ifndef PVECI_ARM_GCC
    return veci_i32x4_t(math::imath_t<int32_t,__n128>::zeroes()) -= v.p;
#else
    return veci_i32x4_t(math::imath_t<int32_t,int32x4_t>::zeroes()) -= v.p;
#endif
#endif
}


// load aligned
template<> inline void veci_i32x4_t::loada(const int32_t * ptr)
{
#if defined(PVECI_INTEL)
    p = _mm_load_si128(reinterpret_cast<const packed_t *>(ptr));
#elif defined(PVECI_ARM)
    p = *reinterpret_cast<const packed_t *>(ptr);
#endif
}

// load unaligned
template<> inline void veci_i32x4_t::loadu(const int32_t * ptr)
{
#if defined(PVECI_INTEL)
    p = _mm_loadu_si128(reinterpret_cast<const packed_t *>(ptr));
#elif defined(PVECI_ARM)
    p = *reinterpret_cast<const packed_t *>(ptr);
#endif
}

// store unaligned
template<> inline void veci_i32x4_t::storeu(int32_t * ptr)
{
#if defined(PVECI_INTEL)
    _mm_storeu_si128(reinterpret_cast<packed_t *>(ptr), p);
#elif defined(PVECI_ARM)
    // this may not even be supported (or maybe it is supported per VLDx)
    p = *reinterpret_cast<const packed_t *>(ptr);
#endif
}


#ifdef PVECI_ARM
template<> inline veci_i32x4_t veci_i32x4_t::operator~() const
{ return veci_i32x4_t(veorq_s32(p, math_t::onebits())); }
#endif


// free-standing arithmetic operations (element-wise)
inline veci_i32x4_t operator+(const veci_i32x4_t & v1, const veci_i32x4_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_i32x4_t(_mm_add_epi32(v1.p, v2.p));
#elif defined(PVECI_ARM)
    return veci_i32x4_t(vaddq_s32(v1.p, v2.p));
#endif
}
inline veci_i32x4_t operator+(const veci_i32x4_t & v, int32_t s)
{
#if defined(PVECI_INTEL)
    return veci_i32x4_t(_mm_add_epi32(v.p, _mm_set1_epi32(s)));
#elif defined(PVECI_ARM)
    return veci_i32x4_t(vaddq_s32(v.p, vmovq_n_s32(s)));
#endif
}
inline veci_i32x4_t operator+(int32_t s, const veci_i32x4_t & v)
{
#if defined(PVECI_INTEL)
    return veci_i32x4_t(_mm_add_epi32(_mm_set1_epi32(s), v.p));
#elif defined(PVECI_ARM)
    return veci_i32x4_t(vaddq_s32(vmovq_n_s32(s), v.p));
#endif
}
inline veci_i32x4_t operator+(const veci_i32x4_t & v1, veci_i32x4_t::packed_t v2)
{
#if defined(PVECI_INTEL)
    return veci_i32x4_t(_mm_add_epi32(v1.p, v2));
#elif defined(PVECI_ARM)
    return veci_i32x4_t(vaddq_s32(v1.p, v2));
#endif
}
inline veci_i32x4_t operator+(veci_i32x4_t::packed_t v1, const veci_i32x4_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_i32x4_t(_mm_add_epi32(v1, v2.p));
#elif defined(PVECI_ARM)
    return veci_i32x4_t(vaddq_s32(v1, v2.p));
#endif
}

inline veci_i32x4_t operator-(const veci_i32x4_t & v1, const veci_i32x4_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_i32x4_t(_mm_sub_epi32(v1.p, v2.p));
#elif defined(PVECI_ARM)
    return veci_i32x4_t(vsubq_s32(v1.p, v2.p));
#endif
}
inline veci_i32x4_t operator-(const veci_i32x4_t & v, int32_t s)
{
#if defined(PVECI_INTEL)
    return veci_i32x4_t(_mm_sub_epi32(v.p, _mm_set1_epi32(s)));
#elif defined(PVECI_ARM)
    return veci_i32x4_t(vsubq_s32(v.p, vmovq_n_s32(s)));
#endif
}
inline veci_i32x4_t operator-(int32_t s, const veci_i32x4_t & v)
{
#if defined(PVECI_INTEL)
    return veci_i32x4_t(_mm_sub_epi32(_mm_set1_epi32(s), v.p));
#elif defined(PVECI_ARM)
    return veci_i32x4_t(vsubq_s32(vmovq_n_s32(s), v.p));
#endif
}
inline veci_i32x4_t operator-(const veci_i32x4_t & v1, veci_i32x4_t::packed_t v2)
{
#if defined(PVECI_INTEL)
    return veci_i32x4_t(_mm_sub_epi32(v1.p, v2));
#elif defined(PVECI_ARM)
    return veci_i32x4_t(vsubq_s32(v1.p, v2));
#endif
}
inline veci_i32x4_t operator-(veci_i32x4_t::packed_t v1, const veci_i32x4_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_i32x4_t(_mm_sub_epi32(v1, v2.p));
#elif defined(PVECI_ARM)
    return veci_i32x4_t(vsubq_s32(v1, v2.p));
#endif
}


// free-standing bit-wise logical operations
inline veci_i32x4_t operator&(veci_i32x4_t op1, veci_i32x4_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i32x4_t(_mm_and_si128(op1.p, op2.p));
#elif defined(PVECI_ARM)
    return veci_i32x4_t(vandq_s32(op1.p, op2.p));
#endif
}
inline veci_i32x4_t operator&(veci_i32x4_t::packed_t op1, veci_i32x4_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i32x4_t(_mm_and_si128(op1, op2.p));
#elif defined(PVECI_ARM)
    return veci_i32x4_t(vandq_s32(op1, op2.p));
#endif
}
inline veci_i32x4_t operator&(veci_i32x4_t op1, veci_i32x4_t::packed_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i32x4_t(_mm_and_si128(op1.p, op2));
#elif defined(PVECI_ARM)
    return veci_i32x4_t(vandq_s32(op1.p, op2));
#endif
}

inline veci_i32x4_t operator|(veci_i32x4_t op1, veci_i32x4_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i32x4_t(_mm_or_si128(op1.p, op2.p));
#elif defined(PVECI_ARM)
    return veci_i32x4_t(vorrq_s32(op1.p, op2.p));
#endif
}
inline veci_i32x4_t operator|(veci_i32x4_t::packed_t op1, veci_i32x4_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i32x4_t(_mm_or_si128(op1, op2.p));
#elif defined(PVECI_ARM)
    return veci_i32x4_t(vorrq_s32(op1, op2.p));
#endif
}
inline veci_i32x4_t operator|(veci_i32x4_t op1, veci_i32x4_t::packed_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i32x4_t(_mm_or_si128(op1.p, op2));
#elif defined(PVECI_ARM)
    return veci_i32x4_t(vorrq_s32(op1.p, op2));
#endif
}

inline veci_i32x4_t operator^(veci_i32x4_t op1, veci_i32x4_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i32x4_t(_mm_xor_si128(op1.p, op2.p));
#elif defined(PVECI_ARM)
    return veci_i32x4_t(veorq_s32(op1.p, op2.p));
#endif
}
inline veci_i32x4_t operator^(veci_i32x4_t::packed_t op1, veci_i32x4_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i32x4_t(_mm_xor_si128(op1, op2.p));
#elif defined(PVECI_ARM)
    return veci_i32x4_t(veorq_s32(op1, op2.p));
#endif
}
inline veci_i32x4_t operator^(veci_i32x4_t op1, veci_i32x4_t::packed_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i32x4_t(_mm_xor_si128(op1.p, op2));
#elif defined(PVECI_ARM)
    return veci_i32x4_t(veorq_s32(op1.p, op2));
#endif
}

//
// swizzle ops for s32x4
//

// TODO: the two params version is still wrong (see paper)
// veci_ui32x4_t version wrong, too
// Q: for intel too, or for NEON only?
#if defined(PVECI_INTEL)
#define SWIZZLE_S32x4____(a,b,c,d,a_,b_,c_,d_)                                   \
inline veci_i32x4_t a##b##c##d(const veci_i32x4_t & v)                           \
{ return veci_i32x4_t(_mm_shuffle_epi32(v.p, _MM_SHUFFLE(d_,c_,b_,a_))); }       \
inline veci_i32x4_t a##b##c##d(const veci_i32x4_t & v1, const veci_i32x4_t & v2) \
{ return veci_i32x4_t(_mm_unpackhi_epi32(_mm_shuffle_epi32(v1.p, _MM_SHUFFLE(d_,c_,0,0)), _mm_shuffle_epi32(v2.p, _MM_SHUFFLE(a_,b_,0,0)))); }


#elif defined(PVECI_ARM)

#if 1
#define SWIZZLE_S32x4____(a,b,c,d,a_,b_,c_,d_) \
inline veci_i32x4_t a##b##c##d(const veci_i32x4_t & v) { return veci_i32x4_t(a##b##c##d(v.p)); } \
inline veci_i32x4_t a##b##c##d(const veci_i32x4_t & v1, const veci_i32x4_t & v2) { return veci_i32x4_t(a##b##c##d(v1.p, v2.p)); }

#else
// attempt to specialize some combos with less instructions

// identity
inline veci_i32x4_t xyzw(const veci_i32x4_t & v) { return v; }

// reverse
inline veci_i32x4_t wzyx(const veci_i32x4_t & v) { return veci_i32x4_t(vrev64q_s32(v.p)); }

inline veci_i32x4_t yxzw(const veci_i32x4_t & v) {
    return veci_i32x4_t(vcombine_s32(vrev64_s32(vget_high_s32(v.p)), vget_low_s32(v.p)));
}

inline veci_i32x4_t xywz(const veci_i32x4_t & v) {
    return veci_i32x4_t(vcombine_s32(vget_high_s32(v.p), vrev64_s32(vget_low_s32(v.p))));
}

inline veci_i32x4_t yxwz(const veci_i32x4_t & v) {
    return veci_i32x4_t(vcombine_s32(vrev64_s32(vget_high_s32(v.p)), vrev64_s32(vget_low_s32(v.p))));
}

inline veci_i32x4_t zwxy(const veci_i32x4_t & v) {
    return veci_i32x4_t(vcombine_s32(vget_low_s32(v.p), vget_high_s32(v.p)));
}

inline veci_i32x4_t wzxy(const veci_i32x4_t & v) {
    return veci_i32x4_t(vcombine_s32(vrev64_s32(vget_low_s32(v.p)), vget_high_s32(v.p)));
}

inline veci_i32x4_t zwyx(const veci_i32x4_t & v) {
    return veci_i32x4_t(vcombine_s32(vget_low_s32(v.p), vrev64_s32(vget_high_s32(v.p))));
}


// base structure for the "no element duplication" operations (pure shuffles) with Dreg splits:
// temp = vzipq_f32(vcombine_f32(vrev64_f32(vget_high_f32(x))), vcombine_f32(vrev64_f32(vget_low_f32(x))))
// vcombine_f32(vrev64_f32(vget_high_f32(temp)), vrev64_f32(vget_low_f32(temp)))

#define NOP(x) (x)
#define REV(x) vrev64_s32(x)
#define SHUFFLE(x,rev1,rev2,rev3,rev4) {       \
    int32x2_t temp1 = rev1(vget_high_s32(x));  \
    int32x2_t temp2 = rev2(vget_low_s32(x));   \
    int32x2x2_t temp = vzip_s32(temp1, temp2); \
    return veci_i32x4_t(vcombine_s32(rev3(temp.val[1]), rev4(temp.val[0]))); }
// xzyw()
// (1) == nop, (2) == nop, (3) == nop, (4) == nop (0000)
inline veci_i32x4_t xzyw(const veci_i32x4_t & v) SHUFFLE(v.p,NOP,NOP,NOP,NOP)
// xzwy()
// (1) == nop, (2) == nop, (3) == nop             (0001)
inline veci_i32x4_t xzwy(const veci_i32x4_t & v) SHUFFLE(v.p,NOP,NOP,NOP,REV)
// xwyz()
// (1) == nop, (3) == nop, (4) == nop             (0100)
inline veci_i32x4_t xwyz(const veci_i32x4_t & v) SHUFFLE(v.p,NOP,REV,NOP,NOP)
// xwzy()
// (1) == nop, (3) == nop                         (0101)
inline veci_i32x4_t xwzy(const veci_i32x4_t & v) SHUFFLE(v.p,NOP,REV,NOP,REV)
//
// yzxw()
// (2) == nop, (3) == nop, (4) == nop             (1000)
inline veci_i32x4_t yzxw(const veci_i32x4_t & v) SHUFFLE(v.p,REV,NOP,NOP,NOP)
// yzwx()
// (2) == nop, (3) == nop                         (1001)
inline veci_i32x4_t yzwx(const veci_i32x4_t & v) SHUFFLE(v.p,REV,NOP,NOP,REV)
// ywxz()
// (3) == nop, (4) == nop                         (1100)
inline veci_i32x4_t ywxz(const veci_i32x4_t & v) SHUFFLE(v.p,REV,REV,NOP,NOP)
// ywzx()
// (3) == nop                                     (1101)
inline veci_i32x4_t ywzx(const veci_i32x4_t & v) SHUFFLE(v.p,REV,REV,NOP,REV)
//
// zxyw()
// (1) == nop, (2) == nop, (4) == nop             (0010)
inline veci_i32x4_t zxyw(const veci_i32x4_t & v) SHUFFLE(v.p,NOP,NOP,REV,NOP)
// zxwy()
// (1) == nop, (2) == nop                         (0011)
inline veci_i32x4_t zxwy(const veci_i32x4_t & v) SHUFFLE(v.p,NOP,NOP,REV,REV)
// zyxw()
// (2) == nop, (4) == nop                         (1010)
inline veci_i32x4_t zyxw(const veci_i32x4_t & v) SHUFFLE(v.p,REV,NOP,REV,NOP)
// zywx()
// (2) == nop                                     (1011)
inline veci_i32x4_t zywx(const veci_i32x4_t & v) SHUFFLE(v.p,REV,NOP,REV,REV)
//
// wxyz()
// (1) == nop, (4) == nop                         (0110)
inline veci_i32x4_t wxyz(const veci_i32x4_t & v) SHUFFLE(v.p,NOP,REV,REV,NOP)
// wxzy()
// (1) == nop                                     (0111)
inline veci_i32x4_t wxzy(const veci_i32x4_t & v) SHUFFLE(v.p,NOP,REV,REV,REV)
// wyxz()
// (4) == nop                                     (1110)
inline veci_i32x4_t wyxz(const veci_i32x4_t & v) SHUFFLE(v.p,REV,REV,REV,NOP)
// wyzx()
//                                                (1111)
inline veci_i32x4_t wyzx(const veci_i32x4_t & v) SHUFFLE(v.p,REV,REV,REV,REV)

#undef SHUFFLE

// count: 24

inline veci_i32x4_t xzxz(const veci_i32x4_t & v) {
    int32x2_t temp1 = vdup_lane_s32(vget_high_s32(v.p), 1); // (x,x)
    int32x2_t temp2 = vdup_lane_s32(vget_low_s32(v.p), 1);  // (z,z)
    int32x2x2_t temp = vzip_s32(temp1, temp2); // (x,z),(x,z)
    return veci_i32x4_t(vcombine_s32(temp.val[1], temp.val[0])); }

inline veci_i32x4_t ywyw(const veci_i32x4_t & v) {
    int32x2_t temp1 = vdup_lane_s32(vget_high_s32(v.p), 0); // (y,y)
    int32x2_t temp2 = vdup_lane_s32(vget_low_s32(v.p), 0);  // (w,w)
    int32x2x2_t temp = vzip_s32(temp1, temp2); // (y,w),(y,w)
    return veci_i32x4_t(vcombine_s32(temp.val[1], temp.val[0])); }

inline veci_i32x4_t xyxy(const veci_i32x4_t & v) {
    int32x2_t temp = vget_high_s32(v.p); // (x,y)
    return veci_i32x4_t(vcombine_s32(temp, temp)); }

inline veci_i32x4_t zwzw(const veci_i32x4_t & v) {
    int32x2_t temp = vget_low_s32(v.p); // (z,w)
    return veci_i32x4_t(vcombine_s32(temp, temp)); }

// count: 28

// xxx{y,z,w}()
inline veci_i32x4_t xxxy(const veci_i32x4_t & v) {
    int32x2_t temp1 = vdup_lane_s32(vget_high_s32(v.p), 1); // (x,x)
    int32x2_t temp2 = vget_high_s32(v.p); // (x,y)
    return veci_i32x4_t(vcombine_s32(temp1, temp2)); }

inline veci_i32x4_t xxxz(const veci_i32x4_t & v) {
    int32x2_t temp1 = vdup_lane_s32(vget_high_s32(v.p), 1); // (x,x)
    int32x2_t temp2 = vdup_lane_s32(vget_low_s32(v.p), 1); // (z,z)
    int32x2x2_t temp = vzip_s32(temp1, temp2); // (x,z),(x,z)
    return veci_i32x4_t(vcombine_s32(vdup_lane_s32(temp.val[1], 1), temp.val[0])); }

inline veci_i32x4_t xxxw(const veci_i32x4_t & v) {
    int32x2_t temp1 = vdup_lane_s32(vget_high_s32(v.p), 1); // (x,x)
    int32x2_t temp2 = vdup_lane_s32(vget_low_s32(v.p), 0); // (w,w)
    int32x2x2_t temp = vzip_s32(temp1, temp2); // (x,w),(x,w)
    return veci_i32x4_t(vcombine_s32(vdup_lane_s32(temp.val[1], 1), temp.val[0])); }

// count: 31

// xx{y,z,w}x()
inline veci_i32x4_t xxyx(const veci_i32x4_t & v) {
    int32x2_t temp1 = vdup_lane_s32(vget_high_s32(v.p), 1); // (x,x)
    int32x2_t temp2 = vget_high_s32(v.p); // (x,y)
    return veci_i32x4_t(vcombine_s32(temp1, vrev64_s32(temp2))); }

inline veci_i32x4_t xxzx(const veci_i32x4_t & v) {
    int32x2_t temp1 = vdup_lane_s32(vget_high_s32(v.p), 1); // (x,x)
    int32x2_t temp2 = vdup_lane_s32(vget_low_s32(v.p), 1); // (z,z)
    int32x2x2_t temp = vzip_s32(temp1, temp2); // (x,z),(x,z)
    return veci_i32x4_t(vcombine_s32(vdup_lane_s32(temp.val[1], 1), vrev64_s32(temp.val[0]))); }

inline veci_i32x4_t xxwx(const veci_i32x4_t & v) {
    int32x2_t temp1 = vdup_lane_s32(vget_high_s32(v.p), 1); // (x,x)
    int32x2_t temp2 = vdup_lane_s32(vget_low_s32(v.p), 0); // (w,w)
    int32x2x2_t temp = vzip_s32(temp1, temp2); // (x,w),(x,w)
    return veci_i32x4_t(vcombine_s32(vdup_lane_s32(temp.val[1], 1), vrev64_s32(temp.val[0]))); }

// count: 34

// xx??() -> vdup_lane_f32(vget_high_f32(v), 1) = new_high
// yy??() -> vdup_lane_f32(vget_high_f32(v), 0) = new_high
// zz??() -> vdup_lane_f32(vget_low_f32 (v), 1) = new_high
// ww??() -> vdup_lane_f32(vget_low_f32 (v), 0) = new_high

// ??xx() -> as xx??() = new_low
// ??yy() -> as yy??() = new_low
// ??zz() -> as zz??() = new_low
// ??ww() -> as ww??() = new_low

#define SWIZZLE_22(a,b,hi,lo,combhi,comblo)              \
inline veci_i32x4_t a##a##b##b(const veci_i32x4_t & v) { \
    int32x2_t temp1 = vget_high_s32(v.p);                \
    int32x2_t temp2 = vget_low_s32(v.p);                 \
    return veci_i32x4_t(vcombine_s32(                    \
        vdup_lane_s32(temp##combhi, hi),                 \
        vdup_lane_s32(temp##comblo, lo)                  \
    ));                                                  \
}

SWIZZLE_22(x,x,1,1,1,1)
SWIZZLE_22(x,y,1,0,1,1)
SWIZZLE_22(x,z,1,1,1,2)
SWIZZLE_22(x,w,1,0,1,2)

SWIZZLE_22(y,x,0,1,1,1)
SWIZZLE_22(y,y,0,0,1,1)
SWIZZLE_22(y,z,0,1,1,2)
SWIZZLE_22(y,w,0,0,1,2)

SWIZZLE_22(z,x,1,1,2,1)
SWIZZLE_22(z,y,1,0,2,1)
SWIZZLE_22(z,z,1,1,2,2)
SWIZZLE_22(z,w,1,0,2,2)

SWIZZLE_22(w,x,0,1,2,1)
SWIZZLE_22(w,y,0,0,2,1)
SWIZZLE_22(w,z,0,1,2,2)
SWIZZLE_22(w,w,0,0,2,2)

#undef SWIZZLE_22

// count: 50

// {xx,yy}zw(): vcombine_f32(vdup_lane_f32(vget_high_f32(v), 1/0), vget_low_f32(v));
// {xx,yy}wz(): vcombine_f32(vdup_lane_f32(vget_high_f32(v), 1/0), vrev64_f32(vget_low_f32(v)));
#define SWIZZLE_23(a,b,c,hi,lorev)                      \
inline veci_i32x4_t a##b##c(const veci_i32x4_t & v) {   \
    int32x2_t temp1 = vget_high_s32(v.p);               \
    int32x2_t temp2 = vget_low_s32(v.p);                \
    return veci_i32x4_t(vcombine_s32(vdup_lane_s32(temp1, hi), lorev(temp2))); }

SWIZZLE_23(xx,z,w,1,NOP)
SWIZZLE_23(xx,w,z,1,REV)
SWIZZLE_23(yy,z,w,0,NOP)
SWIZZLE_23(yy,w,z,0,REV)

#undef SWIZZLE_23

// count: 54

// zw{xx,yy}(): vcombine_f32(vget_low_f32(v), vdup_lane_f32(vget_high_f32(v), 1/0));
// wz{xx,yy}(): vcombine_f32(vrev64_f32(vget_low_f32(v)), vdup_lane_f32(vget_high_f32(v), 1/0));
#define SWIZZLE_23(a,b,c,lo,hirev)                               \
inline veci_i32x4_t a##b##c(const veci_i32x4_t & v) {            \
    int32x2_t temp1 = vget_low_s32(v.p);                         \
    int32x2_t temp2 = vget_high_s32(v.p);                        \
    return veci_i32x4_t(vcombine_s32(hirev(temp1), vdup_lane_s32(temp1, lo))); }

SWIZZLE_23(z,w,xx,1,NOP)
SWIZZLE_23(z,w,yy,0,NOP)
SWIZZLE_23(w,z,xx,1,REV)
SWIZZLE_23(w,z,yy,0,REV)

#undef SWIZZLE_23

// count: 58

#define SWIZZLE_23(x,y,z,b_,a_,a,b,c)                       \
inline veci_i32x4_t x##y##z(const veci_i32x4_t & v) {       \
    return                                                  \
        veci_i32x4_t(                                       \
            vcombine_s32(                                   \
                vdup_lane_s32(vget_##a_##_s32(v.p), a),     \
                vzip_s32(                                   \
                    vdup_lane_s32(vget_##a_##_s32(v.p), b), \
                    vdup_lane_s32(vget_##b_##_s32(v.p), c)  \
                ).val[1]                                    \
            )                                               \
        );                                                  \
}


SWIZZLE_23(xx,y,w,low ,high,1,0,0)
SWIZZLE_23(xx,y,z,low ,high,1,0,1)

SWIZZLE_23(yy,y,w,low ,high,0,0,0)
SWIZZLE_23(yy,y,z,low ,high,0,0,1)
SWIZZLE_23(yy,x,w,low ,high,0,1,0)
SWIZZLE_23(yy,x,z,low ,high,0,1,1)

SWIZZLE_23(zz,y,w,low ,low ,1,0,0)
SWIZZLE_23(zz,y,z,low ,low ,1,0,1)
SWIZZLE_23(zz,x,w,low ,low ,1,1,0)
SWIZZLE_23(zz,x,z,low ,low ,1,1,1)

SWIZZLE_23(ww,y,w,low ,low ,0,0,0)
SWIZZLE_23(ww,y,z,low ,low ,0,0,1)
SWIZZLE_23(ww,x,w,low ,low ,0,1,0)
SWIZZLE_23(ww,x,z,low ,low ,0,1,1)

SWIZZLE_23(yy,y,x,high,high,0,0,1)
SWIZZLE_23(yy,x,y,high,high,0,1,0)

SWIZZLE_23(zz,w,y,high,low ,1,0,0)
SWIZZLE_23(zz,w,x,high,low ,1,0,1)
SWIZZLE_23(zz,z,y,high,low ,1,1,0)
SWIZZLE_23(zz,z,x,high,low ,1,1,1)

SWIZZLE_23(ww,w,y,high,low ,0,0,0)
SWIZZLE_23(ww,w,x,high,low ,0,0,1)
SWIZZLE_23(ww,z,y,high,low ,0,1,0)
SWIZZLE_23(ww,z,x,high,low ,0,1,1)


#undef SWIZZLE_23

// count: 82


#define SWIZZLE_23(x,y,z,b_,a_,a,b,c)                       \
inline veci_i32x4_t x##y##z(const veci_i32x4_t & v) {       \
    return                                                  \
        veci_i32x4_t(                                       \
            vcombine_s32(                                   \
                vdup_lane_s32(vget_##b_##_s32(v.p), a),     \
                vzip_s32(                                   \
                    vdup_lane_s32(vget_##b_##_s32(v.p), b), \
                    vdup_lane_s32(vget_##a_##_s32(v.p), c)  \
                ).val[1]                                    \
            )                                               \
        );                                                  \
}

SWIZZLE_23(ww,w,z,low ,low ,0,0,1)
SWIZZLE_23(ww,z,w,low ,low ,0,1,0)

SWIZZLE_23(zz,w,z,low ,low ,1,0,1)
SWIZZLE_23(zz,z,w,low ,low ,1,1,0)

#undef SWIZZLE_23

// count: 86


#define SWIZZLE_23(x,y,z,b_,a_,a,b,c)                       \
inline veci_i32x4_t x##y##z(const veci_i32x4_t & v) {       \
    return                                                  \
        veci_i32x4_t(                                       \
            vcombine_s32(                                   \
                vdup_lane_s32(vget_##a_##_s32(v.p), a),     \
                vzip_s32(                                   \
                    vdup_lane_s32(vget_##b_##_s32(v.p), b), \
                    vdup_lane_s32(vget_##a_##_s32(v.p), c)  \
                ).val[1]                                    \
            )                                               \
        );                                                  \
}

SWIZZLE_23(yy,w,y,low ,high,0,0,0)
SWIZZLE_23(yy,w,x,low ,high,0,0,1)
SWIZZLE_23(yy,z,y,low ,high,0,1,0)
SWIZZLE_23(yy,z,x,low ,high,0,1,1)
SWIZZLE_23(xx,w,y,low ,high,1,0,0)
SWIZZLE_23(xx,z,y,low ,high,1,1,0)

#undef SWIZZLE_23

// count: 92


#define SWIZZLE_32XX(z,a)                            \
inline veci_i32x4_t xy##z(const veci_i32x4_t & v) {  \
    return                                           \
        veci_i32x4_t(                                \
            vcombine_s32(                            \
                vget_high_s32(v.p),                  \
                vdup_lane_s32(vget_high_s32(v.p), a) \
            )                                        \
        );                                           \
}

SWIZZLE_32XX(xx,1)
SWIZZLE_32XX(yy,0)

#undef SWIZZLE_32XX

// count: 94


#define SWIZZLE_323X(z,hi_,hi,lo_,lo)                         \
inline veci_i32x4_t xy##z(const veci_i32x4_t & v) {           \
    return                                                    \
        veci_i32x4_t(                                         \
            vcombine_s32(                                     \
                vget_high_s32(v.p),                           \
                vzip_s32(                                     \
                    vdup_lane_s32(vget_##hi_##_s32(v.p), hi), \
                    vdup_lane_s32(vget_##lo_##_s32(v.p), lo)  \
                ).val[1]                                      \
            )                                                 \
        );                                                    \
}

SWIZZLE_323X(yw,high,0,low ,0)
SWIZZLE_323X(yz,high,0,low ,1)
SWIZZLE_323X(xw,high,1,low ,0)
SWIZZLE_323X(xz,high,1,low ,1)

SWIZZLE_323X(wy,low ,0,high,0)
SWIZZLE_323X(wx,low ,0,high,1)
SWIZZLE_323X(zy,low ,1,high,0)
SWIZZLE_323X(zx,low ,1,high,1)

SWIZZLE_323X(ww,low ,0,low ,0)
SWIZZLE_323X(zz,low ,1,low ,1)

#undef SWIZZLE_323X

// count: 104

#define SWIZZLE_323X(z,hi_,hi,lo_,lo)                         \
inline veci_i32x4_t z##xy(const veci_i32x4_t & v) {           \
    return                                                    \
        veci_i32x4_t(                                         \
            vcombine_s32(                                     \
                vzip_s32(                                     \
                    vdup_lane_s32(vget_##hi_##_s32(v.p), hi), \
                    vdup_lane_s32(vget_##lo_##_s32(v.p), lo)  \
                ).val[1],                                     \
                vget_high_s32(v.p)                            \
            )                                                 \
        );                                                    \
}

SWIZZLE_323X(yw,high,0,low ,0)
SWIZZLE_323X(yz,high,0,low ,1)
SWIZZLE_323X(xw,high,1,low ,0)
SWIZZLE_323X(xz,high,1,low ,1)

SWIZZLE_323X(wy,low ,0,high,0)
SWIZZLE_323X(wx,low ,0,high,1)
SWIZZLE_323X(zy,low ,1,high,0)
SWIZZLE_323X(zx,low ,1,high,1)

SWIZZLE_323X(ww,low ,0,low ,0)
SWIZZLE_323X(zz,low ,1,low ,1)


#undef SWIZZLE_323X

// count: 114

#define SWIZZLE_323X(z,hi_,hi,lo_,lo)                         \
inline veci_i32x4_t z##xx(const veci_i32x4_t & v) {           \
    return                                                    \
        veci_i32x4_t(                                         \
            vcombine_s32(                                     \
                vzip_s32(                                     \
                    vdup_lane_s32(vget_##hi_##_s32(v.p), hi), \
                    vdup_lane_s32(vget_##lo_##_s32(v.p), lo)  \
                ).val[1],                                     \
                vdup_lane_s32(vget_high_s32(v.p), 1)          \
            )                                                 \
        );                                                    \
}

SWIZZLE_323X(yx,high,0,high,1)

SWIZZLE_323X(wy,low ,0,high,0)
SWIZZLE_323X(wx,low ,0,high,1)
SWIZZLE_323X(zy,low ,1,high,0)
SWIZZLE_323X(zx,low ,1,high,1)

SWIZZLE_323X(yw,high,0,low ,0)
SWIZZLE_323X(yz,high,0,low ,1)
SWIZZLE_323X(xw,high,1,low ,0)
SWIZZLE_323X(xz,high,1,low ,1)

#undef SWIZZLE_323X

// count: 123


#define SWIZZLE_REV(x,y,z,w,a_,b_,hiop,loop)             \
inline veci_i32x4_t x##y##z##w(const veci_i32x4_t & v) { \
    return                                               \
        veci_i32x4_t(                                    \
            vcombine_s32(                                \
                hiop(vget_##a_##_s32(v.p)),              \
                loop(vget_##b_##_s32(v.p))               \
            )                                            \
        );                                               \
}

SWIZZLE_REV(x,y,y,x,high,high,NOP,REV)
SWIZZLE_REV(z,w,w,z,low ,low ,NOP,REV)

SWIZZLE_REV(y,x,x,y,high,high,REV,NOP)
SWIZZLE_REV(w,z,z,w,low ,low ,REV,NOP)

SWIZZLE_REV(y,x,y,x,high,high,REV,REV)
SWIZZLE_REV(w,z,w,z,low ,low ,REV,REV)


#undef SWIZZLE_REV

// count: 129


#undef NOP
#undef REV



#define SWIZZLE1_GEN(n,hihi_,hihi,hilo_,hilo,lohi_,lohi,lolo_,lolo) \
inline veci_i32x4_t n(const veci_i32x4_t & v) {                   \
    return                                                        \
        veci_i32x4_t(                                             \
            vcombine_s32(                                         \
                vzip_s32(                                         \
                    vdup_lane_s32(vget_##hihi_##_s32(v.p), hihi), \
                    vdup_lane_s32(vget_##hilo_##_s32(v.p), hilo)  \
                ).val[1],                                         \
                vzip_s32(                                         \
                    vdup_lane_s32(vget_##lohi_##_s32(v.p), lohi), \
                    vdup_lane_s32(vget_##lolo_##_s32(v.p), lolo)  \
                ).val[1]                                          \
            )                                                     \
        );                                                        \
}

SWIZZLE1_GEN(yxyy,high,0,high,1,high,0,high,0)

SWIZZLE1_GEN(yxyw,high,0,high,1,high,0,low ,0)
SWIZZLE1_GEN(yxyz,high,0,high,1,high,0,low ,1)
SWIZZLE1_GEN(yxxw,high,0,high,1,high,1,low ,0)
SWIZZLE1_GEN(yxxz,high,0,high,1,high,1,low ,1)

SWIZZLE1_GEN(yxwy,high,0,high,1,low ,0,high,0)
SWIZZLE1_GEN(yxwx,high,0,high,1,low ,0,high,1)
SWIZZLE1_GEN(yxzy,high,0,high,1,low ,1,high,0)
SWIZZLE1_GEN(yxzx,high,0,high,1,low ,1,high,1)

SWIZZLE1_GEN(yxww,high,0,high,1,low ,0,low ,0)
SWIZZLE1_GEN(yxzz,high,0,high,1,low ,1,low ,1)

SWIZZLE1_GEN(ywyy,high,0,low ,0,high,0,high,0)
SWIZZLE1_GEN(ywyx,high,0,low ,0,high,0,high,1)

SWIZZLE1_GEN(ywyz,high,0,low ,0,high,0,low ,1)
SWIZZLE1_GEN(ywxw,high,0,low ,0,high,1,low ,0)

SWIZZLE1_GEN(ywwy,high,0,low ,0,low ,0,high,0)
SWIZZLE1_GEN(ywwx,high,0,low ,0,low ,0,high,1)
SWIZZLE1_GEN(ywzy,high,0,low ,0,low ,1,high,0)

SWIZZLE1_GEN(ywww,high,0,low ,0,low ,0,low ,0)
SWIZZLE1_GEN(ywwz,high,0,low ,0,low ,0,low ,1)
SWIZZLE1_GEN(ywzw,high,0,low ,0,low ,1,low ,0)
SWIZZLE1_GEN(ywzz,high,0,low ,0,low ,1,low ,1)


SWIZZLE1_GEN(yzyy,high,0,low ,1,high,0,high,0)
SWIZZLE1_GEN(yzyx,high,0,low ,1,high,0,high,1)

SWIZZLE1_GEN(yzyw,high,0,low ,1,high,0,low ,0)
SWIZZLE1_GEN(yzyz,high,0,low ,1,high,0,low ,1)
SWIZZLE1_GEN(yzxz,high,0,low ,1,high,1,low ,1)

SWIZZLE1_GEN(yzwy,high,0,low ,1,low ,0,high,0)
SWIZZLE1_GEN(yzzy,high,0,low ,1,low ,1,high,0)
SWIZZLE1_GEN(yzzx,high,0,low ,1,low ,1,high,1)

SWIZZLE1_GEN(yzww,high,0,low ,1,low ,0,low ,0)
SWIZZLE1_GEN(yzwz,high,0,low ,1,low ,0,low ,1)
SWIZZLE1_GEN(yzzw,high,0,low ,1,low ,1,low ,0)
SWIZZLE1_GEN(yzzz,high,0,low ,1,low ,1,low ,1)


SWIZZLE1_GEN(xwyy,high,1,low ,0,high,0,high,0)
SWIZZLE1_GEN(xwyx,high,1,low ,0,high,0,high,1)

SWIZZLE1_GEN(xwyw,high,1,low ,0,high,0,low ,0)
SWIZZLE1_GEN(xwxw,high,1,low ,0,high,1,low ,0)
SWIZZLE1_GEN(xwxz,high,1,low ,0,high,1,low ,1)

SWIZZLE1_GEN(xwwy,high,1,low ,0,low ,0,high,0)
SWIZZLE1_GEN(xwwx,high,1,low ,0,low ,0,high,1)
SWIZZLE1_GEN(xwzx,high,1,low ,0,low ,1,high,1)

SWIZZLE1_GEN(xwww,high,1,low ,0,low ,0,low ,0)
SWIZZLE1_GEN(xwwz,high,1,low ,0,low ,0,low ,1)
SWIZZLE1_GEN(xwzw,high,1,low ,0,low ,1,low ,0)
SWIZZLE1_GEN(xwzz,high,1,low ,0,low ,1,low ,1)


SWIZZLE1_GEN(xzyy,high,1,low ,1,high,0,high,0)
SWIZZLE1_GEN(xzyx,high,1,low ,1,high,0,high,1)

SWIZZLE1_GEN(xzyz,high,1,low ,1,high,0,low ,1)
SWIZZLE1_GEN(xzxw,high,1,low ,1,high,1,low ,0)

SWIZZLE1_GEN(xzwx,high,1,low ,1,low ,0,high,1)
SWIZZLE1_GEN(xzzy,high,1,low ,1,low ,1,high,0)
SWIZZLE1_GEN(xzzx,high,1,low ,1,low ,1,high,1)

SWIZZLE1_GEN(xzww,high,1,low ,1,low ,0,low ,0)
SWIZZLE1_GEN(xzwz,high,1,low ,1,low ,0,low ,1)
SWIZZLE1_GEN(xzzw,high,1,low ,1,low ,1,low ,0)
SWIZZLE1_GEN(xzzz,high,1,low ,1,low ,1,low ,1)


SWIZZLE1_GEN(wyyy,low ,0,high,0,high,0,high,0)
SWIZZLE1_GEN(wyyx,low ,0,high,0,high,0,high,1)

SWIZZLE1_GEN(wyyw,low ,0,high,0,high,0,low ,0)
SWIZZLE1_GEN(wyyz,low ,0,high,0,high,0,low ,1)
SWIZZLE1_GEN(wyxw,low ,0,high,0,high,1,low ,0)

SWIZZLE1_GEN(wywy,low ,0,high,0,low ,0,high,0)
SWIZZLE1_GEN(wywx,low ,0,high,0,low ,0,high,1)
SWIZZLE1_GEN(wyzy,low ,0,high,0,low ,1,high,0)

SWIZZLE1_GEN(wyww,low ,0,high,0,low ,0,low ,0)
SWIZZLE1_GEN(wywz,low ,0,high,0,low ,0,low ,1)
SWIZZLE1_GEN(wyzw,low ,0,high,0,low ,1,low ,0)
SWIZZLE1_GEN(wyzz,low ,0,high,0,low ,1,low ,1)


SWIZZLE1_GEN(wxyy,low ,0,high,1,high,0,high,0)
SWIZZLE1_GEN(wxyx,low ,0,high,1,high,0,high,1)

SWIZZLE1_GEN(wxyw,low ,0,high,1,high,0,low ,0)
SWIZZLE1_GEN(wxxw,low ,0,high,1,high,1,low ,0)
SWIZZLE1_GEN(wxxz,low ,0,high,1,high,1,low ,1)

SWIZZLE1_GEN(wxwy,low ,0,high,1,low ,0,high,0)
SWIZZLE1_GEN(wxwx,low ,0,high,1,low ,0,high,1)
SWIZZLE1_GEN(wxzx,low ,0,high,1,low ,1,high,1)

SWIZZLE1_GEN(wxww,low ,0,high,1,low ,0,low ,0)
SWIZZLE1_GEN(wxwz,low ,0,high,1,low ,0,low ,1)
SWIZZLE1_GEN(wxzw,low ,0,high,1,low ,1,low ,0)
SWIZZLE1_GEN(wxzz,low ,0,high,1,low ,1,low ,1)


SWIZZLE1_GEN(zyyy,low ,1,high,0,high,0,high,0)
SWIZZLE1_GEN(zyyx,low ,1,high,0,high,0,high,1)

SWIZZLE1_GEN(zyyw,low ,1,high,0,high,0,low ,0)
SWIZZLE1_GEN(zyyz,low ,1,high,0,high,0,low ,1)
SWIZZLE1_GEN(zyxz,low ,1,high,0,high,1,low ,1)

SWIZZLE1_GEN(zywy,low ,1,high,0,low ,0,high,0)
SWIZZLE1_GEN(zyzy,low ,1,high,0,low ,1,high,0)
SWIZZLE1_GEN(zyzx,low ,1,high,0,low ,1,high,1)

SWIZZLE1_GEN(zyww,low ,1,high,0,low ,0,low ,0)
SWIZZLE1_GEN(zywz,low ,1,high,0,low ,0,low ,1)
SWIZZLE1_GEN(zyzw,low ,1,high,0,low ,1,low ,0)
SWIZZLE1_GEN(zyzz,low ,1,high,0,low ,1,low ,1)


SWIZZLE1_GEN(zxyy,low ,1,high,1,high,0,high,0)
SWIZZLE1_GEN(zxyx,low ,1,high,1,high,0,high,1)

SWIZZLE1_GEN(zxyz,low ,1,high,1,high,0,low ,1)
SWIZZLE1_GEN(zxxw,low ,1,high,1,high,1,low ,0)
SWIZZLE1_GEN(zxxz,low ,1,high,1,high,1,low ,1)

SWIZZLE1_GEN(zxwx,low ,1,high,1,low ,0,high,1)
SWIZZLE1_GEN(zxzy,low ,1,high,1,low ,1,high,0)
SWIZZLE1_GEN(zxzx,low ,1,high,1,low ,1,high,1)

SWIZZLE1_GEN(zxww,low ,1,high,1,low ,0,low ,0)
SWIZZLE1_GEN(zxwz,low ,1,high,1,low ,0,low ,1)
SWIZZLE1_GEN(zxzw,low ,1,high,1,low ,1,low ,0)
SWIZZLE1_GEN(zxzz,low ,1,high,1,low ,1,low ,1)

SWIZZLE1_GEN(wwyx,low ,0,low ,0,high,0,high,1)

SWIZZLE1_GEN(wzyw,low ,0,low ,1,high,0,low ,0)
SWIZZLE1_GEN(wzyz,low ,0,low ,1,high,0,low ,1)
SWIZZLE1_GEN(wzxw,low ,0,low ,1,high,1,low ,0)
SWIZZLE1_GEN(wzxz,low ,0,low ,1,high,1,low ,1)

SWIZZLE1_GEN(wzwy,low ,0,low ,1,low ,0,high,0)
SWIZZLE1_GEN(wzwx,low ,0,low ,1,low ,0,high,1)
SWIZZLE1_GEN(wzzy,low ,0,low ,1,low ,1,high,0)
SWIZZLE1_GEN(wzzx,low ,0,low ,1,low ,1,high,1)

SWIZZLE1_GEN(wzww,low ,0,low ,1,low ,0,low ,0)
SWIZZLE1_GEN(wzzz,low ,0,low ,1,low ,1,low ,1)

SWIZZLE1_GEN(zwyw,low ,1,low ,0,high,0,low ,0)
SWIZZLE1_GEN(zwyz,low ,1,low ,0,high,0,low ,1)
SWIZZLE1_GEN(zwxw,low ,1,low ,0,high,1,low ,0)
SWIZZLE1_GEN(zwxz,low ,1,low ,0,high,1,low ,1)

SWIZZLE1_GEN(zwwy,low ,1,low ,0,low ,0,high,0)
SWIZZLE1_GEN(zwwx,low ,1,low ,0,low ,0,high,1)
SWIZZLE1_GEN(zwzy,low ,1,low ,0,low ,1,high,0)
SWIZZLE1_GEN(zwzx,low ,1,low ,0,low ,1,high,1)

SWIZZLE1_GEN(zwww,low ,1,low ,0,low ,0,low ,0)
SWIZZLE1_GEN(zwzz,low ,1,low ,0,low ,1,low ,1)

SWIZZLE1_GEN(zzyx,low ,1,low ,1,high,0,high,1)

#undef SWIZZLE1_GEN

// count: 123+??


#define SWIZZLE2_GEN(a,b,c,d,hihi_,hihi,hilo_,hilo,lohi_,lohi,lolo_,lolo)          \
inline veci_i32x4_t a##b##c##d(const veci_i32x4_t & v1, const veci_i32x4_t & v2) { \
    return                                                                         \
        veci_i32x4_t(                                                              \
            vcombine_s32(                                                          \
                vzip_s32(                                                          \
                    vdup_lane_s32(vget_##hihi_##_s32(v1.p), hihi),                 \
                    vdup_lane_s32(vget_##hilo_##_s32(v1.p), hilo)                  \
                ).val[1],                                                          \
                vzip_s32(                                                          \
                    vdup_lane_s32(vget_##lohi_##_s32(v2.p), lohi),                 \
                    vdup_lane_s32(vget_##lolo_##_s32(v2.p), lolo)                  \
                ).val[1]                                                           \
            )                                                                      \
        );                                                                         \
}

#define SWIZZLE_FLOAT_4___(a,b,c,a1_,a2_,b1_,b2_,c1_,c2_) \
    SWIZZLE2_GEN(a,b,c,x,a1_,a2_,b1_,b2_,c1_,c2_,high,1)  \
    SWIZZLE2_GEN(a,b,c,y,a1_,a2_,b1_,b2_,c1_,c2_,high,0)  \
    SWIZZLE2_GEN(a,b,c,z,a1_,a2_,b1_,b2_,c1_,c2_,low,1)   \
    SWIZZLE2_GEN(a,b,c,w,a1_,a2_,b1_,b2_,c1_,c2_,low,0)
#define SWIZZLE_FLOAT_4__(a,b,a1_,a2_,b1_,b2_)       \
    SWIZZLE_FLOAT_4___(a,b,x,a1_,a2_,b1_,b2_,high,1) \
    SWIZZLE_FLOAT_4___(a,b,y,a1_,a2_,b1_,b2_,high,0) \
    SWIZZLE_FLOAT_4___(a,b,z,a1_,a2_,b1_,b2_,low,1)  \
    SWIZZLE_FLOAT_4___(a,b,w,a1_,a2_,b1_,b2_,low,0)
#define SWIZZLE_FLOAT_4_(a,a1_,a2_)       \
    SWIZZLE_FLOAT_4__(a,x,a1_,a2_,high,1) \
    SWIZZLE_FLOAT_4__(a,y,a1_,a2_,high,0) \
    SWIZZLE_FLOAT_4__(a,z,a1_,a2_,low,1)  \
    SWIZZLE_FLOAT_4__(a,w,a1_,a2_,low,0)
#define SWIZZLE_FLOAT_4        \
    SWIZZLE_FLOAT_4_(x,high,1) \
    SWIZZLE_FLOAT_4_(y,high,0) \
    SWIZZLE_FLOAT_4_(z,low,1)  \
    SWIZZLE_FLOAT_4_(w,low,0)

SWIZZLE_FLOAT_4

#undef SWIZZLE_FLOAT_4
#undef SWIZZLE_FLOAT_4_
#undef SWIZZLE_FLOAT_4__
#undef SWIZZLE_FLOAT_4___

#undef SWIZZLE2_GEN

#endif // specialized ...

#endif // defined(PVECI_INTEL)

#define SWIZZLE_S32x4___(a,b,c,a_,b_,c_)  \
    SWIZZLE_S32x4____(a,b,c,x,a_,b_,c_,0) \
    SWIZZLE_S32x4____(a,b,c,y,a_,b_,c_,1) \
    SWIZZLE_S32x4____(a,b,c,z,a_,b_,c_,2) \
    SWIZZLE_S32x4____(a,b,c,w,a_,b_,c_,3)
#define SWIZZLE_S32x4__(a,b,a_,b_)  \
    SWIZZLE_S32x4___(a,b,x,a_,b_,0) \
    SWIZZLE_S32x4___(a,b,y,a_,b_,1) \
    SWIZZLE_S32x4___(a,b,z,a_,b_,2) \
    SWIZZLE_S32x4___(a,b,w,a_,b_,3)
#define SWIZZLE_S32x4_(a,a_)  \
    SWIZZLE_S32x4__(a,x,a_,0) \
    SWIZZLE_S32x4__(a,y,a_,1) \
    SWIZZLE_S32x4__(a,z,a_,2) \
    SWIZZLE_S32x4__(a,w,a_,3)
#define SWIZZLE_S32x4   \
    SWIZZLE_S32x4_(x,0) \
    SWIZZLE_S32x4_(y,1) \
    SWIZZLE_S32x4_(z,2) \
    SWIZZLE_S32x4_(w,3)

SWIZZLE_S32x4

#undef SWIZZLE_S32x4____
#undef SWIZZLE_S32x4___
#undef SWIZZLE_S32x4__
#undef SWIZZLE_S32x4_
#undef SWIZZLE_S32x4

#undef veci_i32x4_t


/*****************************************************************************
 *                                                                           *
 * veci_ui32x4_t implementation                                              *
 *                                                                           *
 *****************************************************************************/
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
// x86/x64 with MSVC
#  define veci_ui32x4_t veci_t<uint32_t,4,__m128i>
#elif defined(_MSC_VER) && defined(_M_ARM)
// ARM NEON with MSVC
#  define veci_ui32x4_t veci_t<uint32_t,4,__n128>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
// x86/x64 with GCC / MinGW
#  define veci_ui32x4_t veci_t<uint32_t,4,__m128i>
#elif defined(__GNUC__) && defined(__ARM_ARCH) && defined(__ARM_NEON)
// ARM NEON with GCC
#  define veci_ui32x4_t veci_t<uint32_t,4,uint32x4_t>
#endif

template<> inline veci_ui32x4_t::veci_t()
{ p = math_t::zeroes(); }
template<> inline veci_ui32x4_t::veci_t(uint32_t v0)
{ p = math_t::zeroes();
  v[0] = v0; }
template<> inline veci_ui32x4_t::veci_t(uint32_t v0, uint32_t v1)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; }
template<> template<> inline veci_ui32x4_t::veci_t(uint32_t v0, uint32_t v1, uint32_t v2)
{ p = math_t::zeroes();
  v[0] = v0; v[1] = v1; v[2] = v2; }
template<> template<> inline veci_ui32x4_t::veci_t(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3)
{ v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3; }
template<> inline veci_ui32x4_t::veci_t(std::initializer_list<type_t> l)
{
    p = math_t::zeroes();
    int count = int(l.size() <= 4 ? l.size() : 4);
    int i = 0; for(auto it = l.begin(); i < count; ++i, ++it) v[i] = *it;
}


// vector addition, subtraction (not saturated)
// and logical operations
template<> inline veci_ui32x4_t & veci_ui32x4_t::operator+=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_add_epi32(p, v2);
#elif defined(PVECI_ARM)
    p = vaddq_u32(p, v2);
#endif
    return *this;
}
template<> inline veci_ui32x4_t & veci_ui32x4_t::operator-=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_sub_epi32(p, v2);
#elif defined(PVECI_ARM)
    p = vsubq_u32(p, v2);
#endif
    return *this;
}
template<> inline veci_ui32x4_t & veci_ui32x4_t::operator&=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_and_si128(p, v2);
#elif defined(PVECI_ARM)
    p = vandq_u32(p, v2);
#endif
    return *this;
}
template<> inline veci_ui32x4_t & veci_ui32x4_t::operator|=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_or_si128(p, v2);
#elif defined(PVECI_ARM)
    p = vorrq_u32(p, v2);
#endif
    return *this;
}
template<> inline veci_ui32x4_t & veci_ui32x4_t::operator^=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_xor_si128(p, v2);
#elif defined(PVECI_ARM)
    p = veorq_u32(p, v2);
#endif
    return *this;
}

template<> inline veci_ui32x4_t & veci_ui32x4_t::operator+=(const veci_ui32x4_t & v2)
{ return operator+=(v2.p); }
template<> inline veci_ui32x4_t & veci_ui32x4_t::operator-=(const veci_ui32x4_t & v2)
{ return operator-=(v2.p); }
template<> inline veci_ui32x4_t & veci_ui32x4_t::operator&=(const veci_ui32x4_t & v2)
{ return operator&=(v2.p); }
template<> inline veci_ui32x4_t & veci_ui32x4_t::operator|=(const veci_ui32x4_t & v2)
{ return operator|=(v2.p); }
template<> inline veci_ui32x4_t & veci_ui32x4_t::operator^=(const veci_ui32x4_t & v2)
{ return operator^=(v2.p); }


// packed vector comparisons
// (impl postponed for ARM NEON)
#ifdef PVECI_INTEL
template<> inline bool veci_ui32x4_t::operator==(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmpeq_epi32(p, v2)) == 0xFFFF;
#elif defined(PVECI_ARM)
    XXX
#endif
}
template<> inline bool veci_ui32x4_t::operator!=(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmpeq_epi32(p, v2)) == 0x0000;
#elif defined(PVECI_ARM)
    XXX
#endif
}
template<> inline bool veci_ui32x4_t::operator<(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return
        _mm_movemask_epi8(
            _mm_cmplt_epi32(
                _mm_xor_si128(p, math_t::sign_mask()),
                _mm_xor_si128(v2, math_t::sign_mask())
            )
        ) == 0xFFFF;
#elif defined(PVECI_ARM)
    XXX
#endif
}
template<> inline bool veci_ui32x4_t::operator<=(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return
        _mm_movemask_epi8(
            _mm_cmpgt_epi32(
                _mm_xor_si128(p, math_t::sign_mask()),
                _mm_xor_si128(v2, math_t::sign_mask())
            )
        ) == 0x0000;
#elif defined(PVECI_ARM)
    XXX
#endif
}
template<> inline bool veci_ui32x4_t::operator>(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return
        _mm_movemask_epi8(
            _mm_cmpgt_epi32(
                _mm_xor_si128(p, math_t::sign_mask()),
                _mm_xor_si128(v2, math_t::sign_mask())
            )
        ) == 0xFFFF;
#elif defined(PVECI_ARM)
    XXX
#endif
}
template<> inline bool veci_ui32x4_t::operator>=(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return
        _mm_movemask_epi8(
            _mm_cmplt_epi32(
                _mm_xor_si128(p, math_t::sign_mask()),
                _mm_xor_si128(v2, math_t::sign_mask())
            )
        ) == 0x0000;
#elif defined(PVECI_ARM)
    XXX
#endif
}
// returns true if at least one value pair is unequal
template<> inline bool veci_ui32x4_t::neq_one(packed_t v2) const
{
#if defined(PVECI_INTEL)
    return _mm_movemask_epi8(_mm_cmpeq_epi32(p, v2)) != 0xFFFF;
#elif defined(PVECI_ARM)
    XXX
#endif
}


// vector comparisons
template<> inline bool veci_ui32x4_t::operator==(const veci_ui32x4_t & v2) const
{ return operator==(v2.p); }
template<> inline bool veci_ui32x4_t::operator!=(const veci_ui32x4_t & v2) const
{ return operator!=(v2.p); }
template<> inline bool veci_ui32x4_t::operator<(const veci_ui32x4_t & v2) const
{ return operator<(v2.p); }
template<> inline bool veci_ui32x4_t::operator<=(const veci_ui32x4_t & v2) const
{ return operator<=(v2.p); }
template<> inline bool veci_ui32x4_t::operator>(const veci_ui32x4_t & v2) const
{ return operator>(v2.p); }
template<> inline bool veci_ui32x4_t::operator>=(const veci_ui32x4_t & v2) const
{ return operator>=(v2.p); }
// returns true if at least one value pair is unequal
template<> inline bool veci_ui32x4_t::neq_one(const veci_ui32x4_t & v2) const
{ return neq_one(v2.p); }

// (impl postponed for ARM NEON)
#endif // defined(PVECI_INTEL)


// SSE2 intrinsic support for int16_t*8 and uint8_t*16 only
// AVX2: ???
// TODO: either disable these per enable_if<> for all other combos
//       or provide a scalar implementation (emit a performance warning
//       message, too)
template<> inline veci_ui32x4_t veci_ui32x4_t::min_(packed_t p1, packed_t p2)
{
#if defined(PVECI_INTEL)
# if defined(SSE4)
    return veci_ui32x4_t(_mm_min_epu32(p1, p2));
# else
#   pragma message("performance warning: SSE2 does not provide min() for packed uint32_t x 4")
    veci_ui32x4_t ret, v1(p1), v2(p2);
    for(int i = 0; i < 4; ++i)
        ret[i] = (std::min)(v1[i], v2[i]);
    return ret;
# endif
#elif defined(PVECI_ARM)
    return veci_ui32x4_t(vminq_u32(p1, p2));
#endif
}
template<> inline veci_ui32x4_t veci_ui32x4_t::min_(const veci_ui32x4_t & v1, const veci_ui32x4_t & v2)
{ return min_(v1.p, v2.p); }

template<> inline veci_ui32x4_t veci_ui32x4_t::max_(packed_t p1, packed_t p2)
{
#if defined(PVECI_INTEL)
# if defined(SSE4)
    return veci_ui32x4_t(_mm_max_epu32(p1, p2));
# else
#   pragma message("performance warning: SSE2 does not provide max() for packed uint32_t x 4")
    veci_ui32x4_t ret, v1(p1), v2(p2);
    for(int i = 0; i < 4; ++i)
        ret[i] = (std::max)(v1[i], v2[i]);
    return ret;
# endif
#elif defined(PVECI_ARM)
    return veci_ui32x4_t(vmaxq_u32(p1, p2));
#endif
}
template<> inline veci_ui32x4_t veci_ui32x4_t::max_(const veci_ui32x4_t & v1, const veci_ui32x4_t & v2)
{ return max_(v1.p, v2.p); }



// load aligned
template<> inline void veci_ui32x4_t::loada(const uint32_t * ptr)
{
#if defined(PVECI_INTEL)
    p = _mm_load_si128(reinterpret_cast<const packed_t *>(ptr));
#elif defined(PVECI_ARM)
    p = *reinterpret_cast<const packed_t *>(ptr);
#endif
}

// load unaligned
template<> inline void veci_ui32x4_t::loadu(const uint32_t * ptr)
{
#if defined(PVECI_INTEL)
    p = _mm_loadu_si128(reinterpret_cast<const packed_t *>(ptr));
#elif defined(PVECI_ARM)
    p = *reinterpret_cast<const packed_t *>(ptr);
#endif
}

// store unaligned
template<> inline void veci_ui32x4_t::storeu(uint32_t * ptr)
{
#if defined(PVECI_INTEL)
    _mm_storeu_si128(reinterpret_cast<packed_t *>(ptr), p);
#elif defined(PVECI_ARM)
    // this may not even be supported (or maybe it is supported per VLDx)
    p = *reinterpret_cast<const packed_t *>(ptr);
#endif
}


#ifdef PVECI_ARM
template<> inline veci_ui32x4_t veci_ui32x4_t::operator~() const
{ return veci_ui32x4_t(veorq_u32(p, math_t::onebits())); }
#endif


// free-standing arithmetic operations (element-wise)
inline veci_ui32x4_t operator+(const veci_ui32x4_t & v1, const veci_ui32x4_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_ui32x4_t(_mm_add_epi32(v1.p, v2.p));
#elif defined(PVECI_ARM)
    return veci_ui32x4_t(vaddq_u32(v1.p, v2.p));
#endif
}
inline veci_ui32x4_t operator+(const veci_ui32x4_t & v, uint32_t s)
{
#if defined(PVECI_INTEL)
    return veci_ui32x4_t(_mm_add_epi32(v.p, _mm_set1_epi32(s)));
#elif defined(PVECI_ARM)
    return veci_ui32x4_t(vaddq_u32(v.p, vmovq_n_u32(s)));
#endif
}
inline veci_ui32x4_t operator+(uint32_t s, const veci_ui32x4_t & v)
{
#if defined(PVECI_INTEL)
    return veci_ui32x4_t(_mm_add_epi32(_mm_set1_epi32(s), v.p));
#elif defined(PVECI_ARM)
    return veci_ui32x4_t(vaddq_u32(vmovq_n_u32(s), v.p));
#endif
}
inline veci_ui32x4_t operator+(const veci_ui32x4_t & v1, veci_ui32x4_t::packed_t v2)
{
#if defined(PVECI_INTEL)
    return veci_ui32x4_t(_mm_add_epi32(v1.p, v2));
#elif defined(PVECI_ARM)
    return veci_ui32x4_t(vaddq_u32(v1.p, v2));
#endif
}
inline veci_ui32x4_t operator+(veci_ui32x4_t::packed_t v1, const veci_ui32x4_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_ui32x4_t(_mm_add_epi32(v1, v2.p));
#elif defined(PVECI_ARM)
    return veci_ui32x4_t(vaddq_u32(v1, v2.p));
#endif
}

inline veci_ui32x4_t operator-(const veci_ui32x4_t & v1, const veci_ui32x4_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_ui32x4_t(_mm_sub_epi32(v1.p, v2.p));
#elif defined(PVECI_ARM)
    return veci_ui32x4_t(vsubq_u32(v1.p, v2.p));
#endif
}
inline veci_ui32x4_t operator-(const veci_ui32x4_t & v, uint32_t s)
{
#if defined(PVECI_INTEL)
    return veci_ui32x4_t(_mm_sub_epi32(v.p, _mm_set1_epi32(s)));
#elif defined(PVECI_ARM)
    return veci_ui32x4_t(vsubq_u32(v.p, vmovq_n_u32(s)));
#endif
}
inline veci_ui32x4_t operator-(uint32_t s, const veci_ui32x4_t & v)
{
#if defined(PVECI_INTEL)
    return veci_ui32x4_t(_mm_sub_epi32(_mm_set1_epi32(s), v.p));
#elif defined(PVECI_ARM)
    return veci_ui32x4_t(vsubq_u32(vmovq_n_u32(s), v.p));
#endif
}
inline veci_ui32x4_t operator-(const veci_ui32x4_t & v1, veci_ui32x4_t::packed_t v2)
{
#if defined(PVECI_INTEL)
    return veci_ui32x4_t(_mm_sub_epi32(v1.p, v2));
#elif defined(PVECI_ARM)
    return veci_ui32x4_t(vsubq_u32(v1.p, v2));
#endif
}
inline veci_ui32x4_t operator-(veci_ui32x4_t::packed_t v1, const veci_ui32x4_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_ui32x4_t(_mm_sub_epi32(v1, v2.p));
#elif defined(PVECI_ARM)
    return veci_ui32x4_t(vsubq_u32(v1, v2.p));
#endif
}


// free-standing bit-wise logical operations
inline veci_ui32x4_t operator&(veci_ui32x4_t op1, veci_ui32x4_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui32x4_t(_mm_and_si128(op1.p, op2.p));
#elif defined(PVECI_ARM)
    return veci_ui32x4_t(vandq_u32(op1.p, op2.p));
#endif
}
inline veci_ui32x4_t operator&(veci_ui32x4_t::packed_t op1, veci_ui32x4_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui32x4_t(_mm_and_si128(op1, op2.p));
#elif defined(PVECI_ARM)
    return veci_ui32x4_t(vandq_u32(op1, op2.p));
#endif
}
inline veci_ui32x4_t operator&(veci_ui32x4_t op1, veci_ui32x4_t::packed_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui32x4_t(_mm_and_si128(op1.p, op2));
#elif defined(PVECI_ARM)
    return veci_ui32x4_t(vandq_u32(op1.p, op2));
#endif
}

inline veci_ui32x4_t operator|(veci_ui32x4_t op1, veci_ui32x4_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui32x4_t(_mm_or_si128(op1.p, op2.p));
#elif defined(PVECI_ARM)
    return veci_ui32x4_t(vorrq_u32(op1.p, op2.p));
#endif
}
inline veci_ui32x4_t operator|(veci_ui32x4_t::packed_t op1, veci_ui32x4_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui32x4_t(_mm_or_si128(op1, op2.p));
#elif defined(PVECI_ARM)
    return veci_ui32x4_t(vorrq_u32(op1, op2.p));
#endif
}
inline veci_ui32x4_t operator|(veci_ui32x4_t op1, veci_ui32x4_t::packed_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui32x4_t(_mm_or_si128(op1.p, op2));
#elif defined(PVECI_ARM)
    return veci_ui32x4_t(vorrq_u32(op1.p, op2));
#endif
}

inline veci_ui32x4_t operator^(veci_ui32x4_t op1, veci_ui32x4_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui32x4_t(_mm_xor_si128(op1.p, op2.p));
#elif defined(PVECI_ARM)
    return veci_ui32x4_t(veorq_u32(op1.p, op2.p));
#endif
}
inline veci_ui32x4_t operator^(veci_ui32x4_t::packed_t op1, veci_ui32x4_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui32x4_t(_mm_xor_si128(op1, op2.p));
#elif defined(PVECI_ARM)
    return veci_ui32x4_t(veorq_u32(op1, op2.p));
#endif
}
inline veci_ui32x4_t operator^(veci_ui32x4_t op1, veci_ui32x4_t::packed_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui32x4_t(_mm_xor_si128(op1.p, op2));
#elif defined(PVECI_ARM)
    return veci_ui32x4_t(veorq_u32(op1.p, op2));
#endif
}


#if defined(PVECI_INTEL)


#define SWIZZLE_U32x4____(a,b,c,d,a_,b_,c_,d_)                                      \
inline veci_ui32x4_t a##b##c##d(const veci_ui32x4_t & v)                            \
{ return veci_ui32x4_t(_mm_shuffle_epi32(v.p, _MM_SHUFFLE(d_,c_,b_,a_))); }         \
inline veci_ui32x4_t a##b##c##d(const veci_ui32x4_t & v1, const veci_ui32x4_t & v2) \
{ return veci_ui32x4_t(_mm_unpackhi_epi32(_mm_shuffle_epi32(v1.p, _MM_SHUFFLE(d_,c_,0,0)), _mm_shuffle_epi32(v2.p, _MM_SHUFFLE(a_,b_,0,0)))); }
#define SWIZZLE_U32x4___(a,b,c,a_,b_,c_)  \
    SWIZZLE_U32x4____(a,b,c,x,a_,b_,c_,0) \
    SWIZZLE_U32x4____(a,b,c,y,a_,b_,c_,1) \
    SWIZZLE_U32x4____(a,b,c,z,a_,b_,c_,2) \
    SWIZZLE_U32x4____(a,b,c,w,a_,b_,c_,3)
#define SWIZZLE_U32x4__(a,b,a_,b_)  \
    SWIZZLE_U32x4___(a,b,x,a_,b_,0) \
    SWIZZLE_U32x4___(a,b,y,a_,b_,1) \
    SWIZZLE_U32x4___(a,b,z,a_,b_,2) \
    SWIZZLE_U32x4___(a,b,w,a_,b_,3)
#define SWIZZLE_U32x4_(a,a_)  \
    SWIZZLE_U32x4__(a,x,a_,0) \
    SWIZZLE_U32x4__(a,y,a_,1) \
    SWIZZLE_U32x4__(a,z,a_,2) \
    SWIZZLE_U32x4__(a,w,a_,3)
#define SWIZZLE_U32x4   \
    SWIZZLE_U32x4_(x,0) \
    SWIZZLE_U32x4_(y,1) \
    SWIZZLE_U32x4_(z,2) \
    SWIZZLE_U32x4_(w,3)

SWIZZLE_U32x4

#undef SWIZZLE_U32x4____
#undef SWIZZLE_U32x4___
#undef SWIZZLE_U32x4__
#undef SWIZZLE_U32x4_
#undef SWIZZLE_U32x4


#elif defined(PVECI_ARM)
#if 1

//
// swizzle ops for u32x4
//
#define SWIZZLE_U32_4____(a,b,c,d,a_,b_,c_,d_) \
inline veci_ui32x4_t a##b##c##d(const veci_ui32x4_t & v) { return veci_ui32x4_t(a##b##c##d(v.p)); } \
inline veci_ui32x4_t a##b##c##d(const veci_ui32x4_t & v1, const veci_ui32x4_t & v2) { return veci_ui32x4_t(a##b##c##d(v1.p, v2.p)); }
#define SWIZZLE_U32_4___(a,b,c,a_,b_,c_) \
    SWIZZLE_U32_4____(a,b,c,x,a_,b_,c_,0) \
    SWIZZLE_U32_4____(a,b,c,y,a_,b_,c_,1) \
    SWIZZLE_U32_4____(a,b,c,z,a_,b_,c_,2) \
    SWIZZLE_U32_4____(a,b,c,w,a_,b_,c_,3)
#define SWIZZLE_U32_4__(a,b,a_,b_) \
    SWIZZLE_U32_4___(a,b,x,a_,b_,0) \
    SWIZZLE_U32_4___(a,b,y,a_,b_,1) \
    SWIZZLE_U32_4___(a,b,z,a_,b_,2) \
    SWIZZLE_U32_4___(a,b,w,a_,b_,3)
#define SWIZZLE_U32_4_(a,a_)  \
    SWIZZLE_U32_4__(a,x,a_,0) \
    SWIZZLE_U32_4__(a,y,a_,1) \
    SWIZZLE_U32_4__(a,z,a_,2) \
    SWIZZLE_U32_4__(a,w,a_,3)
#define SWIZZLE_U32_4   \
    SWIZZLE_U32_4_(x,0) \
    SWIZZLE_U32_4_(y,1) \
    SWIZZLE_U32_4_(z,2) \
    SWIZZLE_U32_4_(w,3)

SWIZZLE_U32_4

#undef SWIZZLE_U32_4____
#undef SWIZZLE_U32_4___
#undef SWIZZLE_U32_4__
#undef SWIZZLE_U32_4_
#undef SWIZZLE_U32_4

#else
// attempt to specialize some combos with less instructions

// identity
inline veci_ui32x4_t xyzw(const veci_ui32x4_t & v) { return v; }

// reverse
inline veci_ui32x4_t wzyx(const veci_ui32x4_t & v) { return veci_ui32x4_t(vrev64q_u32(v.p)); }

inline veci_ui32x4_t yxzw(const veci_ui32x4_t & v) {
    return veci_ui32x4_t(vcombine_u32(vrev64_u32(vget_high_u32(v.p)), vget_low_u32(v.p)));
}

inline veci_ui32x4_t xywz(const veci_ui32x4_t & v) {
    return veci_ui32x4_t(vcombine_u32(vget_high_u32(v.p), vrev64_u32(vget_low_u32(v.p))));
}

inline veci_ui32x4_t yxwz(const veci_ui32x4_t & v) {
    return veci_ui32x4_t(vcombine_u32(vrev64_u32(vget_high_u32(v.p)), vrev64_u32(vget_low_u32(v.p))));
}

inline veci_ui32x4_t zwxy(const veci_ui32x4_t & v) {
    return veci_ui32x4_t(vcombine_u32(vget_low_u32(v.p), vget_high_u32(v.p)));
}

inline veci_ui32x4_t wzxy(const veci_ui32x4_t & v) {
    return veci_ui32x4_t(vcombine_u32(vrev64_u32(vget_low_u32(v.p)), vget_high_u32(v.p)));
}

inline veci_ui32x4_t zwyx(const veci_ui32x4_t & v) {
    return veci_ui32x4_t(vcombine_u32(vget_low_u32(v.p), vrev64_u32(vget_high_u32(v.p))));
}


// base structure for the "no element duplication" operations (pure shuffles) with Dreg splits:
// temp = vzipq_f32(vcombine_f32(vrev64_f32(vget_high_f32(x))), vcombine_f32(vrev64_f32(vget_low_f32(x))))
// vcombine_f32(vrev64_f32(vget_high_f32(temp)), vrev64_f32(vget_low_f32(temp)))

#define NOP(x) (x)
#define REV(x) vrev64_u32(x)
#define SHUFFLE(x,rev1,rev2,rev3,rev4) {        \
    uint32x2_t temp1 = rev1(vget_high_u32(x));  \
    uint32x2_t temp2 = rev2(vget_low_u32(x));   \
    uint32x2x2_t temp = vzip_u32(temp1, temp2); \
    return veci_ui32x4_t(vcombine_u32(rev3(temp.val[1]), rev4(temp.val[0]))); }
// xzyw()
// (1) == nop, (2) == nop, (3) == nop, (4) == nop (0000)
inline veci_ui32x4_t xzyw(const veci_ui32x4_t & v) SHUFFLE(v.p,NOP,NOP,NOP,NOP)
// xzwy()
// (1) == nop, (2) == nop, (3) == nop             (0001)
inline veci_ui32x4_t xzwy(const veci_ui32x4_t & v) SHUFFLE(v.p,NOP,NOP,NOP,REV)
// xwyz()
// (1) == nop, (3) == nop, (4) == nop             (0100)
inline veci_ui32x4_t xwyz(const veci_ui32x4_t & v) SHUFFLE(v.p,NOP,REV,NOP,NOP)
// xwzy()
// (1) == nop, (3) == nop                         (0101)
inline veci_ui32x4_t xwzy(const veci_ui32x4_t & v) SHUFFLE(v.p,NOP,REV,NOP,REV)
//
// yzxw()
// (2) == nop, (3) == nop, (4) == nop             (1000)
inline veci_ui32x4_t yzxw(const veci_ui32x4_t & v) SHUFFLE(v.p,REV,NOP,NOP,NOP)
// yzwx()
// (2) == nop, (3) == nop                         (1001)
inline veci_ui32x4_t yzwx(const veci_ui32x4_t & v) SHUFFLE(v.p,REV,NOP,NOP,REV)
// ywxz()
// (3) == nop, (4) == nop                         (1100)
inline veci_ui32x4_t ywxz(const veci_ui32x4_t & v) SHUFFLE(v.p,REV,REV,NOP,NOP)
// ywzx()
// (3) == nop                                     (1101)
inline veci_ui32x4_t ywzx(const veci_ui32x4_t & v) SHUFFLE(v.p,REV,REV,NOP,REV)
//
// zxyw()
// (1) == nop, (2) == nop, (4) == nop             (0010)
inline veci_ui32x4_t zxyw(const veci_ui32x4_t & v) SHUFFLE(v.p,NOP,NOP,REV,NOP)
// zxwy()
// (1) == nop, (2) == nop                         (0011)
inline veci_ui32x4_t zxwy(const veci_ui32x4_t & v) SHUFFLE(v.p,NOP,NOP,REV,REV)
// zyxw()
// (2) == nop, (4) == nop                         (1010)
inline veci_ui32x4_t zyxw(const veci_ui32x4_t & v) SHUFFLE(v.p,REV,NOP,REV,NOP)
// zywx()
// (2) == nop                                     (1011)
inline veci_ui32x4_t zywx(const veci_ui32x4_t & v) SHUFFLE(v.p,REV,NOP,REV,REV)
//
// wxyz()
// (1) == nop, (4) == nop                         (0110)
inline veci_ui32x4_t wxyz(const veci_ui32x4_t & v) SHUFFLE(v.p,NOP,REV,REV,NOP)
// wxzy()
// (1) == nop                                     (0111)
inline veci_ui32x4_t wxzy(const veci_ui32x4_t & v) SHUFFLE(v.p,NOP,REV,REV,REV)
// wyxz()
// (4) == nop                                     (1110)
inline veci_ui32x4_t wyxz(const veci_ui32x4_t & v) SHUFFLE(v.p,REV,REV,REV,NOP)
// wyzx()
//                                                (1111)
inline veci_ui32x4_t wyzx(const veci_ui32x4_t & v) SHUFFLE(v.p,REV,REV,REV,REV)

#undef SHUFFLE

// count: 24

inline veci_ui32x4_t xzxz(const veci_ui32x4_t & v) {
    uint32x2_t temp1 = vdup_lane_u32(vget_high_u32(v.p), 1); // (x,x)
    uint32x2_t temp2 = vdup_lane_u32(vget_low_u32(v.p), 1);  // (z,z)
    uint32x2x2_t temp = vzip_u32(temp1, temp2); // (x,z),(x,z)
    return veci_ui32x4_t(vcombine_u32(temp.val[1], temp.val[0])); }

inline veci_ui32x4_t ywyw(const veci_ui32x4_t & v) {
    uint32x2_t temp1 = vdup_lane_u32(vget_high_u32(v.p), 0); // (y,y)
    uint32x2_t temp2 = vdup_lane_u32(vget_low_u32(v.p), 0);  // (w,w)
    uint32x2x2_t temp = vzip_u32(temp1, temp2); // (y,w),(y,w)
    return veci_ui32x4_t(vcombine_u32(temp.val[1], temp.val[0])); }

inline veci_ui32x4_t xyxy(const veci_ui32x4_t & v) {
    uint32x2_t temp = vget_high_u32(v.p); // (x,y)
    return veci_ui32x4_t(vcombine_u32(temp, temp)); }

inline veci_ui32x4_t zwzw(const veci_ui32x4_t & v) {
    uint32x2_t temp = vget_low_u32(v.p); // (z,w)
    return veci_ui32x4_t(vcombine_u32(temp, temp)); }

// count: 28

// xxx{y,z,w}()
inline veci_ui32x4_t xxxy(const veci_ui32x4_t & v) {
    uint32x2_t temp1 = vdup_lane_u32(vget_high_u32(v.p), 1); // (x,x)
    uint32x2_t temp2 = vget_high_u32(v.p); // (x,y)
    return veci_ui32x4_t(vcombine_u32(temp1, temp2)); }

inline veci_ui32x4_t xxxz(const veci_ui32x4_t & v) {
    uint32x2_t temp1 = vdup_lane_u32(vget_high_u32(v.p), 1); // (x,x)
    uint32x2_t temp2 = vdup_lane_u32(vget_low_u32(v.p), 1); // (z,z)
    uint32x2x2_t temp = vzip_u32(temp1, temp2); // (x,z),(x,z)
    return veci_ui32x4_t(vcombine_u32(vdup_lane_u32(temp.val[1], 1), temp.val[0])); }

inline veci_ui32x4_t xxxw(const veci_ui32x4_t & v) {
    uint32x2_t temp1 = vdup_lane_u32(vget_high_u32(v.p), 1); // (x,x)
    uint32x2_t temp2 = vdup_lane_u32(vget_low_u32(v.p), 0); // (w,w)
    uint32x2x2_t temp = vzip_u32(temp1, temp2); // (x,w),(x,w)
    return veci_ui32x4_t(vcombine_u32(vdup_lane_u32(temp.val[1], 1), temp.val[0])); }

// count: 31

// xx{y,z,w}x()
inline veci_ui32x4_t xxyx(const veci_ui32x4_t & v) {
    uint32x2_t temp1 = vdup_lane_u32(vget_high_u32(v.p), 1); // (x,x)
    uint32x2_t temp2 = vget_high_u32(v.p); // (x,y)
    return veci_ui32x4_t(vcombine_u32(temp1, vrev64_u32(temp2))); }

inline veci_ui32x4_t xxzx(const veci_ui32x4_t & v) {
    uint32x2_t temp1 = vdup_lane_u32(vget_high_u32(v.p), 1); // (x,x)
    uint32x2_t temp2 = vdup_lane_u32(vget_low_u32(v.p), 1); // (z,z)
    uint32x2x2_t temp = vzip_u32(temp1, temp2); // (x,z),(x,z)
    return veci_ui32x4_t(vcombine_u32(vdup_lane_u32(temp.val[1], 1), vrev64_u32(temp.val[0]))); }

inline veci_ui32x4_t xxwx(const veci_ui32x4_t & v) {
    uint32x2_t temp1 = vdup_lane_u32(vget_high_u32(v.p), 1); // (x,x)
    uint32x2_t temp2 = vdup_lane_u32(vget_low_u32(v.p), 0); // (w,w)
    uint32x2x2_t temp = vzip_u32(temp1, temp2); // (x,w),(x,w)
    return veci_ui32x4_t(vcombine_u32(vdup_lane_u32(temp.val[1], 1), vrev64_u32(temp.val[0]))); }

// count: 34

// xx??() -> vdup_lane_f32(vget_high_f32(v), 1) = new_high
// yy??() -> vdup_lane_f32(vget_high_f32(v), 0) = new_high
// zz??() -> vdup_lane_f32(vget_low_f32 (v), 1) = new_high
// ww??() -> vdup_lane_f32(vget_low_f32 (v), 0) = new_high

// ??xx() -> as xx??() = new_low
// ??yy() -> as yy??() = new_low
// ??zz() -> as zz??() = new_low
// ??ww() -> as ww??() = new_low

#define SWIZZLE_22(a,b,hi,lo,combhi,comblo)              \
inline veci_ui32x4_t a##a##b##b(const veci_ui32x4_t & v) { \
    uint32x2_t temp1 = vget_high_u32(v.p);                \
    uint32x2_t temp2 = vget_low_u32(v.p);                 \
    return veci_ui32x4_t(vcombine_u32(                    \
        vdup_lane_u32(temp##combhi, hi),                 \
        vdup_lane_u32(temp##comblo, lo)                  \
    ));                                                  \
}

SWIZZLE_22(x,x,1,1,1,1)
SWIZZLE_22(x,y,1,0,1,1)
SWIZZLE_22(x,z,1,1,1,2)
SWIZZLE_22(x,w,1,0,1,2)

SWIZZLE_22(y,x,0,1,1,1)
SWIZZLE_22(y,y,0,0,1,1)
SWIZZLE_22(y,z,0,1,1,2)
SWIZZLE_22(y,w,0,0,1,2)

SWIZZLE_22(z,x,1,1,2,1)
SWIZZLE_22(z,y,1,0,2,1)
SWIZZLE_22(z,z,1,1,2,2)
SWIZZLE_22(z,w,1,0,2,2)

SWIZZLE_22(w,x,0,1,2,1)
SWIZZLE_22(w,y,0,0,2,1)
SWIZZLE_22(w,z,0,1,2,2)
SWIZZLE_22(w,w,0,0,2,2)

#undef SWIZZLE_22

// count: 50

// {xx,yy}zw(): vcombine_f32(vdup_lane_f32(vget_high_f32(v), 1/0), vget_low_f32(v));
// {xx,yy}wz(): vcombine_f32(vdup_lane_f32(vget_high_f32(v), 1/0), vrev64_f32(vget_low_f32(v)));
#define SWIZZLE_23(a,b,c,hi,lorev)                      \
inline veci_ui32x4_t a##b##c(const veci_ui32x4_t & v) {   \
    uint32x2_t temp1 = vget_high_u32(v.p);               \
    uint32x2_t temp2 = vget_low_u32(v.p);                \
    return veci_ui32x4_t(vcombine_u32(vdup_lane_u32(temp1, hi), lorev(temp2))); }

SWIZZLE_23(xx,z,w,1,NOP)
SWIZZLE_23(xx,w,z,1,REV)
SWIZZLE_23(yy,z,w,0,NOP)
SWIZZLE_23(yy,w,z,0,REV)

#undef SWIZZLE_23

// count: 54

// zw{xx,yy}(): vcombine_f32(vget_low_f32(v), vdup_lane_f32(vget_high_f32(v), 1/0));
// wz{xx,yy}(): vcombine_f32(vrev64_f32(vget_low_f32(v)), vdup_lane_f32(vget_high_f32(v), 1/0));
#define SWIZZLE_23(a,b,c,lo,hirev)                               \
inline veci_ui32x4_t a##b##c(const veci_ui32x4_t & v) {            \
    uint32x2_t temp1 = vget_low_u32(v.p);                         \
    uint32x2_t temp2 = vget_high_u32(v.p);                        \
    return veci_ui32x4_t(vcombine_u32(hirev(temp1), vdup_lane_u32(temp1, lo))); }

SWIZZLE_23(z,w,xx,1,NOP)
SWIZZLE_23(z,w,yy,0,NOP)
SWIZZLE_23(w,z,xx,1,REV)
SWIZZLE_23(w,z,yy,0,REV)

#undef SWIZZLE_23

// count: 58

#define SWIZZLE_23(x,y,z,b_,a_,a,b,c)                       \
inline veci_ui32x4_t x##y##z(const veci_ui32x4_t & v) {       \
    return                                                  \
        veci_ui32x4_t(                                       \
            vcombine_u32(                                   \
                vdup_lane_u32(vget_##a_##_u32(v.p), a),     \
                vzip_u32(                                   \
                    vdup_lane_u32(vget_##a_##_u32(v.p), b), \
                    vdup_lane_u32(vget_##b_##_u32(v.p), c)  \
                ).val[1]                                    \
            )                                               \
        );                                                  \
}


SWIZZLE_23(xx,y,w,low ,high,1,0,0)
SWIZZLE_23(xx,y,z,low ,high,1,0,1)

SWIZZLE_23(yy,y,w,low ,high,0,0,0)
SWIZZLE_23(yy,y,z,low ,high,0,0,1)
SWIZZLE_23(yy,x,w,low ,high,0,1,0)
SWIZZLE_23(yy,x,z,low ,high,0,1,1)

SWIZZLE_23(zz,y,w,low ,low ,1,0,0)
SWIZZLE_23(zz,y,z,low ,low ,1,0,1)
SWIZZLE_23(zz,x,w,low ,low ,1,1,0)
SWIZZLE_23(zz,x,z,low ,low ,1,1,1)

SWIZZLE_23(ww,y,w,low ,low ,0,0,0)
SWIZZLE_23(ww,y,z,low ,low ,0,0,1)
SWIZZLE_23(ww,x,w,low ,low ,0,1,0)
SWIZZLE_23(ww,x,z,low ,low ,0,1,1)

SWIZZLE_23(yy,y,x,high,high,0,0,1)
SWIZZLE_23(yy,x,y,high,high,0,1,0)

SWIZZLE_23(zz,w,y,high,low ,1,0,0)
SWIZZLE_23(zz,w,x,high,low ,1,0,1)
SWIZZLE_23(zz,z,y,high,low ,1,1,0)
SWIZZLE_23(zz,z,x,high,low ,1,1,1)

SWIZZLE_23(ww,w,y,high,low ,0,0,0)
SWIZZLE_23(ww,w,x,high,low ,0,0,1)
SWIZZLE_23(ww,z,y,high,low ,0,1,0)
SWIZZLE_23(ww,z,x,high,low ,0,1,1)


#undef SWIZZLE_23

// count: 82


#define SWIZZLE_23(x,y,z,b_,a_,a,b,c)                       \
inline veci_ui32x4_t x##y##z(const veci_ui32x4_t & v) {       \
    return                                                  \
        veci_ui32x4_t(                                       \
            vcombine_u32(                                   \
                vdup_lane_u32(vget_##b_##_u32(v.p), a),     \
                vzip_u32(                                   \
                    vdup_lane_u32(vget_##b_##_u32(v.p), b), \
                    vdup_lane_u32(vget_##a_##_u32(v.p), c)  \
                ).val[1]                                    \
            )                                               \
        );                                                  \
}

SWIZZLE_23(ww,w,z,low ,low ,0,0,1)
SWIZZLE_23(ww,z,w,low ,low ,0,1,0)

SWIZZLE_23(zz,w,z,low ,low ,1,0,1)
SWIZZLE_23(zz,z,w,low ,low ,1,1,0)

#undef SWIZZLE_23

// count: 86


#define SWIZZLE_23(x,y,z,b_,a_,a,b,c)                       \
inline veci_ui32x4_t x##y##z(const veci_ui32x4_t & v) {       \
    return                                                  \
        veci_ui32x4_t(                                       \
            vcombine_u32(                                   \
                vdup_lane_u32(vget_##a_##_u32(v.p), a),     \
                vzip_u32(                                   \
                    vdup_lane_u32(vget_##b_##_u32(v.p), b), \
                    vdup_lane_u32(vget_##a_##_u32(v.p), c)  \
                ).val[1]                                    \
            )                                               \
        );                                                  \
}

SWIZZLE_23(yy,w,y,low ,high,0,0,0)
SWIZZLE_23(yy,w,x,low ,high,0,0,1)
SWIZZLE_23(yy,z,y,low ,high,0,1,0)
SWIZZLE_23(yy,z,x,low ,high,0,1,1)
SWIZZLE_23(xx,w,y,low ,high,1,0,0)
SWIZZLE_23(xx,z,y,low ,high,1,1,0)

#undef SWIZZLE_23

// count: 92


#define SWIZZLE_32XX(z,a)                            \
inline veci_ui32x4_t xy##z(const veci_ui32x4_t & v) {  \
    return                                           \
        veci_ui32x4_t(                                \
            vcombine_u32(                            \
                vget_high_u32(v.p),                  \
                vdup_lane_u32(vget_high_u32(v.p), a) \
            )                                        \
        );                                           \
}

SWIZZLE_32XX(xx,1)
SWIZZLE_32XX(yy,0)

#undef SWIZZLE_32XX

// count: 94


#define SWIZZLE_323X(z,hi_,hi,lo_,lo)                         \
inline veci_ui32x4_t xy##z(const veci_ui32x4_t & v) {           \
    return                                                    \
        veci_ui32x4_t(                                         \
            vcombine_u32(                                     \
                vget_high_u32(v.p),                           \
                vzip_u32(                                     \
                    vdup_lane_u32(vget_##hi_##_u32(v.p), hi), \
                    vdup_lane_u32(vget_##lo_##_u32(v.p), lo)  \
                ).val[1]                                      \
            )                                                 \
        );                                                    \
}

SWIZZLE_323X(yw,high,0,low ,0)
SWIZZLE_323X(yz,high,0,low ,1)
SWIZZLE_323X(xw,high,1,low ,0)
SWIZZLE_323X(xz,high,1,low ,1)

SWIZZLE_323X(wy,low ,0,high,0)
SWIZZLE_323X(wx,low ,0,high,1)
SWIZZLE_323X(zy,low ,1,high,0)
SWIZZLE_323X(zx,low ,1,high,1)

SWIZZLE_323X(ww,low ,0,low ,0)
SWIZZLE_323X(zz,low ,1,low ,1)

#undef SWIZZLE_323X

// count: 104

#define SWIZZLE_323X(z,hi_,hi,lo_,lo)                         \
inline veci_ui32x4_t z##xy(const veci_ui32x4_t & v) {           \
    return                                                    \
        veci_ui32x4_t(                                         \
            vcombine_u32(                                     \
                vzip_u32(                                     \
                    vdup_lane_u32(vget_##hi_##_u32(v.p), hi), \
                    vdup_lane_u32(vget_##lo_##_u32(v.p), lo)  \
                ).val[1],                                     \
                vget_high_u32(v.p)                            \
            )                                                 \
        );                                                    \
}

SWIZZLE_323X(yw,high,0,low ,0)
SWIZZLE_323X(yz,high,0,low ,1)
SWIZZLE_323X(xw,high,1,low ,0)
SWIZZLE_323X(xz,high,1,low ,1)

SWIZZLE_323X(wy,low ,0,high,0)
SWIZZLE_323X(wx,low ,0,high,1)
SWIZZLE_323X(zy,low ,1,high,0)
SWIZZLE_323X(zx,low ,1,high,1)

SWIZZLE_323X(ww,low ,0,low ,0)
SWIZZLE_323X(zz,low ,1,low ,1)


#undef SWIZZLE_323X

// count: 114

#define SWIZZLE_323X(z,hi_,hi,lo_,lo)                         \
inline veci_ui32x4_t z##xx(const veci_ui32x4_t & v) {           \
    return                                                    \
        veci_ui32x4_t(                                         \
            vcombine_u32(                                     \
                vzip_u32(                                     \
                    vdup_lane_u32(vget_##hi_##_u32(v.p), hi), \
                    vdup_lane_u32(vget_##lo_##_u32(v.p), lo)  \
                ).val[1],                                     \
                vdup_lane_u32(vget_high_u32(v.p), 1)          \
            )                                                 \
        );                                                    \
}

SWIZZLE_323X(yx,high,0,high,1)

SWIZZLE_323X(wy,low ,0,high,0)
SWIZZLE_323X(wx,low ,0,high,1)
SWIZZLE_323X(zy,low ,1,high,0)
SWIZZLE_323X(zx,low ,1,high,1)

SWIZZLE_323X(yw,high,0,low ,0)
SWIZZLE_323X(yz,high,0,low ,1)
SWIZZLE_323X(xw,high,1,low ,0)
SWIZZLE_323X(xz,high,1,low ,1)

#undef SWIZZLE_323X

// count: 123


#define SWIZZLE_REV(x,y,z,w,a_,b_,hiop,loop)             \
inline veci_ui32x4_t x##y##z##w(const veci_ui32x4_t & v) { \
    return                                               \
        veci_ui32x4_t(                                    \
            vcombine_u32(                                \
                hiop(vget_##a_##_u32(v.p)),              \
                loop(vget_##b_##_u32(v.p))               \
            )                                            \
        );                                               \
}

SWIZZLE_REV(x,y,y,x,high,high,NOP,REV)
SWIZZLE_REV(z,w,w,z,low ,low ,NOP,REV)

SWIZZLE_REV(y,x,x,y,high,high,REV,NOP)
SWIZZLE_REV(w,z,z,w,low ,low ,REV,NOP)

SWIZZLE_REV(y,x,y,x,high,high,REV,REV)
SWIZZLE_REV(w,z,w,z,low ,low ,REV,REV)


#undef SWIZZLE_REV

// count: 129


#undef NOP
#undef REV



#define SWIZZLE1_GEN(n,hihi_,hihi,hilo_,hilo,lohi_,lohi,lolo_,lolo) \
inline veci_ui32x4_t n(const veci_ui32x4_t & v) {                   \
    return                                                        \
        veci_ui32x4_t(                                             \
            vcombine_u32(                                         \
                vzip_u32(                                         \
                    vdup_lane_u32(vget_##hihi_##_u32(v.p), hihi), \
                    vdup_lane_u32(vget_##hilo_##_u32(v.p), hilo)  \
                ).val[1],                                         \
                vzip_u32(                                         \
                    vdup_lane_u32(vget_##lohi_##_u32(v.p), lohi), \
                    vdup_lane_u32(vget_##lolo_##_u32(v.p), lolo)  \
                ).val[1]                                          \
            )                                                     \
        );                                                        \
}

SWIZZLE1_GEN(yxyy,high,0,high,1,high,0,high,0)

SWIZZLE1_GEN(yxyw,high,0,high,1,high,0,low ,0)
SWIZZLE1_GEN(yxyz,high,0,high,1,high,0,low ,1)
SWIZZLE1_GEN(yxxw,high,0,high,1,high,1,low ,0)
SWIZZLE1_GEN(yxxz,high,0,high,1,high,1,low ,1)

SWIZZLE1_GEN(yxwy,high,0,high,1,low ,0,high,0)
SWIZZLE1_GEN(yxwx,high,0,high,1,low ,0,high,1)
SWIZZLE1_GEN(yxzy,high,0,high,1,low ,1,high,0)
SWIZZLE1_GEN(yxzx,high,0,high,1,low ,1,high,1)

SWIZZLE1_GEN(yxww,high,0,high,1,low ,0,low ,0)
SWIZZLE1_GEN(yxzz,high,0,high,1,low ,1,low ,1)

SWIZZLE1_GEN(ywyy,high,0,low ,0,high,0,high,0)
SWIZZLE1_GEN(ywyx,high,0,low ,0,high,0,high,1)

SWIZZLE1_GEN(ywyz,high,0,low ,0,high,0,low ,1)
SWIZZLE1_GEN(ywxw,high,0,low ,0,high,1,low ,0)

SWIZZLE1_GEN(ywwy,high,0,low ,0,low ,0,high,0)
SWIZZLE1_GEN(ywwx,high,0,low ,0,low ,0,high,1)
SWIZZLE1_GEN(ywzy,high,0,low ,0,low ,1,high,0)

SWIZZLE1_GEN(ywww,high,0,low ,0,low ,0,low ,0)
SWIZZLE1_GEN(ywwz,high,0,low ,0,low ,0,low ,1)
SWIZZLE1_GEN(ywzw,high,0,low ,0,low ,1,low ,0)
SWIZZLE1_GEN(ywzz,high,0,low ,0,low ,1,low ,1)


SWIZZLE1_GEN(yzyy,high,0,low ,1,high,0,high,0)
SWIZZLE1_GEN(yzyx,high,0,low ,1,high,0,high,1)

SWIZZLE1_GEN(yzyw,high,0,low ,1,high,0,low ,0)
SWIZZLE1_GEN(yzyz,high,0,low ,1,high,0,low ,1)
SWIZZLE1_GEN(yzxz,high,0,low ,1,high,1,low ,1)

SWIZZLE1_GEN(yzwy,high,0,low ,1,low ,0,high,0)
SWIZZLE1_GEN(yzzy,high,0,low ,1,low ,1,high,0)
SWIZZLE1_GEN(yzzx,high,0,low ,1,low ,1,high,1)

SWIZZLE1_GEN(yzww,high,0,low ,1,low ,0,low ,0)
SWIZZLE1_GEN(yzwz,high,0,low ,1,low ,0,low ,1)
SWIZZLE1_GEN(yzzw,high,0,low ,1,low ,1,low ,0)
SWIZZLE1_GEN(yzzz,high,0,low ,1,low ,1,low ,1)


SWIZZLE1_GEN(xwyy,high,1,low ,0,high,0,high,0)
SWIZZLE1_GEN(xwyx,high,1,low ,0,high,0,high,1)

SWIZZLE1_GEN(xwyw,high,1,low ,0,high,0,low ,0)
SWIZZLE1_GEN(xwxw,high,1,low ,0,high,1,low ,0)
SWIZZLE1_GEN(xwxz,high,1,low ,0,high,1,low ,1)

SWIZZLE1_GEN(xwwy,high,1,low ,0,low ,0,high,0)
SWIZZLE1_GEN(xwwx,high,1,low ,0,low ,0,high,1)
SWIZZLE1_GEN(xwzx,high,1,low ,0,low ,1,high,1)

SWIZZLE1_GEN(xwww,high,1,low ,0,low ,0,low ,0)
SWIZZLE1_GEN(xwwz,high,1,low ,0,low ,0,low ,1)
SWIZZLE1_GEN(xwzw,high,1,low ,0,low ,1,low ,0)
SWIZZLE1_GEN(xwzz,high,1,low ,0,low ,1,low ,1)


SWIZZLE1_GEN(xzyy,high,1,low ,1,high,0,high,0)
SWIZZLE1_GEN(xzyx,high,1,low ,1,high,0,high,1)

SWIZZLE1_GEN(xzyz,high,1,low ,1,high,0,low ,1)
SWIZZLE1_GEN(xzxw,high,1,low ,1,high,1,low ,0)

SWIZZLE1_GEN(xzwx,high,1,low ,1,low ,0,high,1)
SWIZZLE1_GEN(xzzy,high,1,low ,1,low ,1,high,0)
SWIZZLE1_GEN(xzzx,high,1,low ,1,low ,1,high,1)

SWIZZLE1_GEN(xzww,high,1,low ,1,low ,0,low ,0)
SWIZZLE1_GEN(xzwz,high,1,low ,1,low ,0,low ,1)
SWIZZLE1_GEN(xzzw,high,1,low ,1,low ,1,low ,0)
SWIZZLE1_GEN(xzzz,high,1,low ,1,low ,1,low ,1)


SWIZZLE1_GEN(wyyy,low ,0,high,0,high,0,high,0)
SWIZZLE1_GEN(wyyx,low ,0,high,0,high,0,high,1)

SWIZZLE1_GEN(wyyw,low ,0,high,0,high,0,low ,0)
SWIZZLE1_GEN(wyyz,low ,0,high,0,high,0,low ,1)
SWIZZLE1_GEN(wyxw,low ,0,high,0,high,1,low ,0)

SWIZZLE1_GEN(wywy,low ,0,high,0,low ,0,high,0)
SWIZZLE1_GEN(wywx,low ,0,high,0,low ,0,high,1)
SWIZZLE1_GEN(wyzy,low ,0,high,0,low ,1,high,0)

SWIZZLE1_GEN(wyww,low ,0,high,0,low ,0,low ,0)
SWIZZLE1_GEN(wywz,low ,0,high,0,low ,0,low ,1)
SWIZZLE1_GEN(wyzw,low ,0,high,0,low ,1,low ,0)
SWIZZLE1_GEN(wyzz,low ,0,high,0,low ,1,low ,1)


SWIZZLE1_GEN(wxyy,low ,0,high,1,high,0,high,0)
SWIZZLE1_GEN(wxyx,low ,0,high,1,high,0,high,1)

SWIZZLE1_GEN(wxyw,low ,0,high,1,high,0,low ,0)
SWIZZLE1_GEN(wxxw,low ,0,high,1,high,1,low ,0)
SWIZZLE1_GEN(wxxz,low ,0,high,1,high,1,low ,1)

SWIZZLE1_GEN(wxwy,low ,0,high,1,low ,0,high,0)
SWIZZLE1_GEN(wxwx,low ,0,high,1,low ,0,high,1)
SWIZZLE1_GEN(wxzx,low ,0,high,1,low ,1,high,1)

SWIZZLE1_GEN(wxww,low ,0,high,1,low ,0,low ,0)
SWIZZLE1_GEN(wxwz,low ,0,high,1,low ,0,low ,1)
SWIZZLE1_GEN(wxzw,low ,0,high,1,low ,1,low ,0)
SWIZZLE1_GEN(wxzz,low ,0,high,1,low ,1,low ,1)


SWIZZLE1_GEN(zyyy,low ,1,high,0,high,0,high,0)
SWIZZLE1_GEN(zyyx,low ,1,high,0,high,0,high,1)

SWIZZLE1_GEN(zyyw,low ,1,high,0,high,0,low ,0)
SWIZZLE1_GEN(zyyz,low ,1,high,0,high,0,low ,1)
SWIZZLE1_GEN(zyxz,low ,1,high,0,high,1,low ,1)

SWIZZLE1_GEN(zywy,low ,1,high,0,low ,0,high,0)
SWIZZLE1_GEN(zyzy,low ,1,high,0,low ,1,high,0)
SWIZZLE1_GEN(zyzx,low ,1,high,0,low ,1,high,1)

SWIZZLE1_GEN(zyww,low ,1,high,0,low ,0,low ,0)
SWIZZLE1_GEN(zywz,low ,1,high,0,low ,0,low ,1)
SWIZZLE1_GEN(zyzw,low ,1,high,0,low ,1,low ,0)
SWIZZLE1_GEN(zyzz,low ,1,high,0,low ,1,low ,1)


SWIZZLE1_GEN(zxyy,low ,1,high,1,high,0,high,0)
SWIZZLE1_GEN(zxyx,low ,1,high,1,high,0,high,1)

SWIZZLE1_GEN(zxyz,low ,1,high,1,high,0,low ,1)
SWIZZLE1_GEN(zxxw,low ,1,high,1,high,1,low ,0)
SWIZZLE1_GEN(zxxz,low ,1,high,1,high,1,low ,1)

SWIZZLE1_GEN(zxwx,low ,1,high,1,low ,0,high,1)
SWIZZLE1_GEN(zxzy,low ,1,high,1,low ,1,high,0)
SWIZZLE1_GEN(zxzx,low ,1,high,1,low ,1,high,1)

SWIZZLE1_GEN(zxww,low ,1,high,1,low ,0,low ,0)
SWIZZLE1_GEN(zxwz,low ,1,high,1,low ,0,low ,1)
SWIZZLE1_GEN(zxzw,low ,1,high,1,low ,1,low ,0)
SWIZZLE1_GEN(zxzz,low ,1,high,1,low ,1,low ,1)

SWIZZLE1_GEN(wwyx,low ,0,low ,0,high,0,high,1)

SWIZZLE1_GEN(wzyw,low ,0,low ,1,high,0,low ,0)
SWIZZLE1_GEN(wzyz,low ,0,low ,1,high,0,low ,1)
SWIZZLE1_GEN(wzxw,low ,0,low ,1,high,1,low ,0)
SWIZZLE1_GEN(wzxz,low ,0,low ,1,high,1,low ,1)

SWIZZLE1_GEN(wzwy,low ,0,low ,1,low ,0,high,0)
SWIZZLE1_GEN(wzwx,low ,0,low ,1,low ,0,high,1)
SWIZZLE1_GEN(wzzy,low ,0,low ,1,low ,1,high,0)
SWIZZLE1_GEN(wzzx,low ,0,low ,1,low ,1,high,1)

SWIZZLE1_GEN(wzww,low ,0,low ,1,low ,0,low ,0)
SWIZZLE1_GEN(wzzz,low ,0,low ,1,low ,1,low ,1)

SWIZZLE1_GEN(zwyw,low ,1,low ,0,high,0,low ,0)
SWIZZLE1_GEN(zwyz,low ,1,low ,0,high,0,low ,1)
SWIZZLE1_GEN(zwxw,low ,1,low ,0,high,1,low ,0)
SWIZZLE1_GEN(zwxz,low ,1,low ,0,high,1,low ,1)

SWIZZLE1_GEN(zwwy,low ,1,low ,0,low ,0,high,0)
SWIZZLE1_GEN(zwwx,low ,1,low ,0,low ,0,high,1)
SWIZZLE1_GEN(zwzy,low ,1,low ,0,low ,1,high,0)
SWIZZLE1_GEN(zwzx,low ,1,low ,0,low ,1,high,1)

SWIZZLE1_GEN(zwww,low ,1,low ,0,low ,0,low ,0)
SWIZZLE1_GEN(zwzz,low ,1,low ,0,low ,1,low ,1)

SWIZZLE1_GEN(zzyx,low ,1,low ,1,high,0,high,1)

#undef SWIZZLE1_GEN

// count: 123+??


#define SWIZZLE2_GEN(a,b,c,d,hihi_,hihi,hilo_,hilo,lohi_,lohi,lolo_,lolo)          \
inline veci_ui32x4_t a##b##c##d(const veci_ui32x4_t & v1, const veci_ui32x4_t & v2) { \
    return                                                                         \
        veci_ui32x4_t(                                                              \
            vcombine_u32(                                                          \
                vzip_u32(                                                          \
                    vdup_lane_u32(vget_##hihi_##_u32(v1.p), hihi),                 \
                    vdup_lane_u32(vget_##hilo_##_u32(v1.p), hilo)                  \
                ).val[1],                                                          \
                vzip_u32(                                                          \
                    vdup_lane_u32(vget_##lohi_##_u32(v2.p), lohi),                 \
                    vdup_lane_u32(vget_##lolo_##_u32(v2.p), lolo)                  \
                ).val[1]                                                           \
            )                                                                      \
        );                                                                         \
}

#define SWIZZLE_FLOAT_4___(a,b,c,a1_,a2_,b1_,b2_,c1_,c2_) \
    SWIZZLE2_GEN(a,b,c,x,a1_,a2_,b1_,b2_,c1_,c2_,high,1)  \
    SWIZZLE2_GEN(a,b,c,y,a1_,a2_,b1_,b2_,c1_,c2_,high,0)  \
    SWIZZLE2_GEN(a,b,c,z,a1_,a2_,b1_,b2_,c1_,c2_,low,1)   \
    SWIZZLE2_GEN(a,b,c,w,a1_,a2_,b1_,b2_,c1_,c2_,low,0)
#define SWIZZLE_FLOAT_4__(a,b,a1_,a2_,b1_,b2_)       \
    SWIZZLE_FLOAT_4___(a,b,x,a1_,a2_,b1_,b2_,high,1) \
    SWIZZLE_FLOAT_4___(a,b,y,a1_,a2_,b1_,b2_,high,0) \
    SWIZZLE_FLOAT_4___(a,b,z,a1_,a2_,b1_,b2_,low,1)  \
    SWIZZLE_FLOAT_4___(a,b,w,a1_,a2_,b1_,b2_,low,0)
#define SWIZZLE_FLOAT_4_(a,a1_,a2_)       \
    SWIZZLE_FLOAT_4__(a,x,a1_,a2_,high,1) \
    SWIZZLE_FLOAT_4__(a,y,a1_,a2_,high,0) \
    SWIZZLE_FLOAT_4__(a,z,a1_,a2_,low,1)  \
    SWIZZLE_FLOAT_4__(a,w,a1_,a2_,low,0)
#define SWIZZLE_FLOAT_4        \
    SWIZZLE_FLOAT_4_(x,high,1) \
    SWIZZLE_FLOAT_4_(y,high,0) \
    SWIZZLE_FLOAT_4_(z,low,1)  \
    SWIZZLE_FLOAT_4_(w,low,0)

SWIZZLE_FLOAT_4

#undef SWIZZLE_FLOAT_4
#undef SWIZZLE_FLOAT_4_
#undef SWIZZLE_FLOAT_4__
#undef SWIZZLE_FLOAT_4___

#undef SWIZZLE2_GEN

#endif // attempt to specialize some combos with less instructions

#endif



#undef veci_ui32x4_t

/*****************************************************************************
 *                                                                           *
 * veci_i64x2_t implementation                                               *
 *                                                                           *
 *****************************************************************************/
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
// x86/x64 with MSVC
#  define veci_i64x2_t veci_t<int64_t,2,__m128i>
#elif defined(_MSC_VER) && defined(_M_ARM)
// ARM NEON with MSVC
#  define veci_i64x2_t veci_t<int64_t,2,__n128>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
// x86/x64 with GCC / MinGW
#  define veci_i64x2_t veci_t<int64_t,2,__m128i>
#elif defined(__GNUC__) && defined(__ARM_ARCH) && defined(__ARM_NEON)
// ARM NEON with GCC
#  define veci_i64x2_t veci_t<int64_t,2,int64x2_t>
#endif

template<> inline veci_i64x2_t::veci_t()
{ p = math_t::zeroes(); }
template<> inline veci_i64x2_t::veci_t(int64_t v0)
{ p = math_t::zeroes();
  v[0] = v0; }
template<> inline veci_i64x2_t::veci_t(int64_t v0, int64_t v1)
{ v[0] = v0; v[1] = v1; }
template<> inline veci_i64x2_t::veci_t(std::initializer_list<type_t> l)
{
    p = math_t::zeroes();
    int count = int(l.size() <= 2 ? l.size() : 2);
    int i = 0; for(auto it = l.begin(); i < count; ++i, ++it) v[i] = *it;
}


// vector addition, subtraction (not saturated)
// and logical operations
template<> inline veci_i64x2_t & veci_i64x2_t::operator+=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_add_epi64(p, v2);
#elif defined(PVECI_ARM)
    p = vaddq_s64(p, v2);
#endif
    return *this;
}
template<> inline veci_i64x2_t & veci_i64x2_t::operator-=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_sub_epi64(p, v2);
#elif defined(PVECI_ARM)
    p = vsubq_s64(p, v2);
#endif
    return *this;
}
template<> inline veci_i64x2_t & veci_i64x2_t::operator&=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_and_si128(p, v2);
#elif defined(PVECI_ARM)
    p = vandq_s64(p, v2);
#endif
    return *this;
}
template<> inline veci_i64x2_t & veci_i64x2_t::operator|=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_or_si128(p, v2);
#elif defined(PVECI_ARM)
    p = vorrq_s64(p, v2);
#endif
    return *this;
}
template<> inline veci_i64x2_t & veci_i64x2_t::operator^=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_xor_si128(p, v2);
#elif defined(PVECI_ARM)
    p = veorq_s64(p, v2);
#endif
    return *this;
}

template<> inline veci_i64x2_t & veci_i64x2_t::operator+=(const veci_i64x2_t & v2)
{ return operator+=(v2.p); }
template<> inline veci_i64x2_t & veci_i64x2_t::operator-=(const veci_i64x2_t & v2)
{ return operator-=(v2.p); }
template<> inline veci_i64x2_t & veci_i64x2_t::operator&=(const veci_i64x2_t & v2)
{ return operator&=(v2.p); }
template<> inline veci_i64x2_t & veci_i64x2_t::operator|=(const veci_i64x2_t & v2)
{ return operator|=(v2.p); }
template<> inline veci_i64x2_t & veci_i64x2_t::operator^=(const veci_i64x2_t & v2)
{ return operator^=(v2.p); }


// packed vector comparisons
// SSE2 doesn't provide 64bit comparisons; operators ==,!=,neq_one() are
// implemented per other sized comparisons as only (un-)identical bit patterns
// are relevant

// (impl postponed for ARM NEON)
#ifdef PVECI_INTEL
template<> inline bool veci_i64x2_t::operator==(packed_t v2) const
{
#if defined(PVECI_INTEL)
# if defined(SSE4)
    return _mm_movemask_epi8(_mm_cmpeq_epi64(p, v2)) == 0xFFFF;
# else
    return _mm_movemask_epi8(_mm_cmpeq_epi32(p, v2)) == 0xFFFF;
# endif
#elif defined(PVECI_ARM)

#endif
}
template<> inline bool veci_i64x2_t::operator!=(packed_t v2) const
{
#if defined(PVECI_INTEL)
# if defined(SSE4)
    return _mm_movemask_epi8(_mm_cmpeq_epi64(p, v2)) == 0x0000;
# else
    return _mm_movemask_epi8(_mm_cmpeq_epi32(p, v2)) == 0x0000;
# endif
#elif defined(PVECI_ARM)

#endif
}
template<> inline bool veci_i64x2_t::operator<(packed_t v2) const
{
#if defined(PVECI_INTEL)
# if defined(SSE4_2)
#   pragma message("performance warning: SSE2/SSE3/SSSE3/SSE4.1/SSE4.2 do not provide this comparison operation for packed int64_t x 2")
    return _mm_movemask_epi8(_mm_or_si128(_mm_cmpeq_epi64(p, v2), _mm_cmpgt_epi64(p, v2))) == 0x0000;
# else
#   pragma message("performance warning: SSE2 does not provide comparison operations for packed int64_t x 2")
    int lt = _mm_movemask_epi8(_mm_cmplt_epi32(p, v2));
    int eq = _mm_movemask_epi8(_mm_cmpeq_epi32(p, v2));
    return (lt & 0xF0F0) || ((eq & 0xF0F0) && (lt & 0x0F0F));
# endif
#elif defined(PVECI_ARM)
    XXX
#endif
}
template<> inline bool veci_i64x2_t::operator<=(packed_t v2) const
{
#if defined(PVECI_INTEL)
# if defined(SSE4_2)
    return _mm_movemask_epi8(_mm_cmpgt_epi64(p, v2)) == 0x0000;
# else
#   pragma message("performance warning: SSE2 does not provide comparison operations for packed int64_t x 2")
    int lt = _mm_movemask_epi8(_mm_cmplt_epi32(p, v2));
    int eq = _mm_movemask_epi8(_mm_cmpeq_epi32(p, v2));
    return (lt & 0xF0F0) || ((eq & 0xF0F0) && ((lt & 0x0F0F) || (eq & 0x0F0F)));
# endif
#elif defined(PVECI_ARM)
    XXX
#endif
}
template<> inline bool veci_i64x2_t::operator>(packed_t v2) const
{
#if defined(PVECI_INTEL)
# if defined(SSE4_2)
    return _mm_movemask_epi8(_mm_cmpgt_epi64(p, v2)) == 0xFFFF;
# else
#   pragma message("performance warning: SSE2 does not provide comparison operations for packed int64_t x 2")
    int gt = _mm_movemask_epi8(_mm_cmpgt_epi32(p, v2));
    int eq = _mm_movemask_epi8(_mm_cmpeq_epi32(p, v2));
    return (gt & 0xF0F0) || ((eq & 0xF0F0) && (gt & 0x0F0F));
# endif
#elif defined(PVECI_ARM)
    XXX
#endif
}
template<> inline bool veci_i64x2_t::operator>=(packed_t v2) const
{
#if defined(PVECI_INTEL)
# if defined(SSE4_2)
    return _mm_movemask_epi8(_mm_or_si128(_mm_cmpeq_epi64(p, v2), _mm_cmpgt_epi64(p, v2)) == 0xFFFF;
# else
#   pragma message("performance warning: SSE2 does not provide comparison operations for packed int64_t x 2")
    int gt = _mm_movemask_epi8(_mm_cmpgt_epi32(p, v2));
    int eq = _mm_movemask_epi8(_mm_cmpeq_epi32(p, v2));
    return (gt & 0xF0F0) || ((eq & 0xF0F0) && ((gt & 0x0F0F) || (eq & 0x0F0F)));
# endif
#elif defined(PVECI_ARM)
    XXX
#endif
}
// returns true if at least one value pair is unequal
template<> inline bool veci_i64x2_t::neq_one(packed_t v2) const
{
#if defined(PVECI_INTEL)
# if defined(SSE4)
    return _mm_movemask_epi8(_mm_cmpeq_epi64(p, v2)) != 0xFFFF;
# else
    return _mm_movemask_epi8(_mm_cmpeq_epi32(p, v2)) != 0xFFFF;
# endif
#elif defined(PVECI_ARM)
    XXX
#endif
}

// vector comparisons
template<> inline bool veci_i64x2_t::operator==(const veci_i64x2_t & v2) const
{ return operator==(v2.p); }
template<> inline bool veci_i64x2_t::operator!=(const veci_i64x2_t & v2) const
{ return operator!=(v2.p); }
template<> inline bool veci_i64x2_t::operator<(const veci_i64x2_t & v2) const
{ return operator<(v2.p); }
template<> inline bool veci_i64x2_t::operator<=(const veci_i64x2_t & v2) const
{ return operator<=(v2.p); }
template<> inline bool veci_i64x2_t::operator>(const veci_i64x2_t & v2) const
{ return operator>(v2.p); }
template<> inline bool veci_i64x2_t::operator>=(const veci_i64x2_t & v2) const
{ return operator>=(v2.p); }
template<> inline bool veci_i64x2_t::neq_one(const veci_i64x2_t & v2) const
{ return neq_one(v2.p); }

// (impl postponed for ARM NEON)
#endif // defined(PVECI_INTEL)


// SSE2 intrinsic support for int16_t*8 and uint8_t*16 only
// AVX2: ???
// TODO: either disable these per enable_if<> for all other combos
//       or provide a scalar implementation (emit a performance warning
//       message, too)
template<> inline veci_i64x2_t veci_i64x2_t::min_(packed_t p1, packed_t p2)
{
#if defined(PVECI_INTEL)
# pragma message("performance warning: SSE2/SSE3/SSSE3/SSE4/SSE4.1 do not provide min() for packed int64_t x 2")
    // destroy the larger value and or that with the other
    //unsigned mask2 = ((val1 - val2) >> 63) - 1, mask1 = ~mask2;
    //return (val1 & mask1) | (val2 & mask2);
    __m128i mask =
        _mm_shuffle_epi32(
            _mm_cmpeq_epi32(
                _mm_srli_epi64(
                    _mm_sub_epi64(p1, p2),  // v1 - v2
                    63
                ),                          // shift sign bit
                _mm_setzero_si128()         // convert to mask (cmpeq_epi64() would be better)
            ),
            0xA0
        );                                  // get rid of high dword comparison results
    return
        veci_i64x2_t(
            _mm_or_si128(
                _mm_andnot_si128(mask, p1), // destroys larger value or keeps it
                _mm_and_si128(mask, p2)     // dito
            )
        );
#elif defined(PVECI_ARM)
#if 1
    // implemented in scalar code for now
# pragma message("performance warning: NEON does not provide min() for packed int64_t x 2, scalar code is used")
    veci_i64x2_t ret, v1(p1), v2(p2);
    ret[0] = (std::min)(v1[0], v2[0]);
    ret[1] = (std::min)(v1[1], v2[1]);
    return ret;
#else
    // there's no vceqq_u64()
    uint32x4_t tmp =
        vceqq_u32(
            vreinterpretq_u32_u64(
                vshrq_n_u64(
                    vreinterpretq_u64_s64(vsubq_s64(p1, p2)/*v1 - v2*/),    // [(v1h-v2h):64,(v1l-v2l):64]
                    63
                )/*shift sign bit*/                                         // [sign(v1h-v2h),sign(v1l-v2l)]
            ),
            vdupq_n_u32(0)
        );

    // S=0: v1{h,l} >= v2{h,l} -> goal: v2
    // S=1: v1{h,l}  < v2{h,l} -> goal: v1

    // [0,sign(v1h-v2h),0,sign(v1l-v2l)] (u32 elems)
    // tmp = S==0: S=0 -> tmp = [...,0xFFFFFFFF,...]
    //       S==1: S=1 -> tmp = [...,0xFFFFFFFF,...]
    // overall: [0xFFFFFFFF,tmph,0xFFFFFFFF,tmpl] = [-,tmph,-,tmpl]

    uint32x4_t mask =
        vcombine_u32(
            vdup_lane_u32(vget_high_u32(tmp), 1/*0*/),
            vdup_lane_u32(vget_low_u32(tmp), 1/*0*/)
        );
    // [tmph,tmph,tmpl,tmpl]
    return
        veci_i64x2_t(
            vreinterpretq_s64_u32(
                vorrq_u32(
                    vbicq_u32(vreinterpretq_u32_s64(p1), mask), // destroys larger value or keeps it
                    vandq_u32(vreinterpretq_u32_s64(p2), mask)  // dito
                )
            )
        );

    // vminq_u32(a,b): a=[a0,a1,a2,a3], b=[b0,b1,b2,b3] -> [min(a0,b0), min(a1,b1), min(a2,b2), min(a3,b3)]
    // here: a=[va1h,va1l,va2h,va2l]  (conjecture, could be the other way around)
    //       b=[vb1h,vb1l,vb2h,vb2l]
    // vminq_u32(a,b): [min(va1h,vb1h), min(va1l,vb1l), min(va2h,vb2h), min(va2l,vb2l)]
    // if va >= 0 && vb >= 0:
    //   if va.h == 0 && va.l >= 0 && vb.h == 0 && vb.l >= 0 (both 32b positive):
    //     res.h = 0, res.l = min(va.l, vb.l)
#endif
#endif
}
template<> inline veci_i64x2_t veci_i64x2_t::min_(const veci_i64x2_t & v1, const veci_i64x2_t & v2)
{ return min_(v1.p, v2.p); }

template<> inline veci_i64x2_t veci_i64x2_t::max_(packed_t p1, packed_t p2)
{
#if defined(PVECI_INTEL)
# pragma message("performance warning: SSE2/SSE3/SSSE3/SSE4/SSE4.1 do not provide max() for packed int64_t x 2")
    // destroy the smaller value and or that with the other
    __m128i mask =
        _mm_shuffle_epi32(
            _mm_cmpeq_epi32(
                _mm_srli_epi64(
                    _mm_sub_epi64(p1, p2), // v1 - v2
                    63
                ),                         // shift sign bit
                _mm_setzero_si128()        // convert to mask (cmpeq_epi64() would be better)
            ),
            0xA0
        );                                 // get rid of high dword comparison results
    return
        veci_i64x2_t(
            _mm_or_si128(
                _mm_and_si128(mask, p1),   // destroys smaller value or keeps it
                _mm_andnot_si128(mask, p2) // dito
            )
        );
#elif defined(PVECI_ARM)
#if 1
    // implemented in scalar code for now
# pragma message("performance warning: NEON does not provide max() for packed int64_t x 2, scalar code is used")
    /**/
    veci_i64x2_t ret, v1(p1), v2(p2);
    ret[0] = (std::max)(v1[0], v2[0]);
    ret[1] = (std::max)(v1[1], v2[1]);
    return ret;
#else

    uint32x4_t tmp =
        vceqq_u32(
            vreinterpretq_u32_u64(
                vshrq_n_u64(
                    vreinterpretq_u64_s64(vsubq_s64(p1, p2)/*v1 - v2*/),
                    63
                )/*shift sign bit*/
            ),
            vdupq_n_u32(0)
        );
    // (2,2,0,0)
    uint32x4_t mask =
        vcombine_u32(
            vdup_lane_u32(vget_high_u32(tmp), 0),
            vdup_lane_u32(vget_low_u32(tmp), 0)
        );
    return
        veci_i64x2_t(
            vreinterpretq_s64_u32(
                vorrq_u32(
                    vandq_u32(vreinterpretq_u32_s64(p1), mask),  // destroys smaller value or keeps it
                    vbicq_u32(vreinterpretq_u32_s64(p2), mask)  // dito
                )
            )
        );
#endif
#endif
}
template<> inline veci_i64x2_t veci_i64x2_t::max_(const veci_i64x2_t & v1, const veci_i64x2_t & v2)
{ return max_(v1.p, v2.p); }



template<> template<> inline void veci_i64x2_t::abs_()
{
#if defined(PVECI_INTEL)
# pragma message("performance warning: SSE2 does not provide abs() for packed int64_t x 2")
    __m128i tmp = _mm_cmplt_epi32(p, math_t::zeroes());
    __m128i mask = _mm_unpackhi_epi64(_mm_unpacklo_epi32(tmp, tmp), _mm_unpackhi_epi32(tmp, tmp));
    p = _mm_sub_epi64(_mm_xor_si128(p, mask), mask);
#elif defined(PVECI_ARM)
# pragma message("performance warning: NEON does not provide abs() for packed int64_t x 2")
    uint32x4_t tmp =
        vceqq_u32(
            vreinterpretq_u32_s64(p),
            vreinterpretq_u32_s64(math_t::zeroes())
        );
    // [{0,-1},{0,-1},{0,-1},{0,-1}]
    uint32x4_t mask =
        vcombine_u32(
            vdup_lane_u32(vget_high_u32(tmp), 1),
            vdup_lane_u32(vget_low_u32(tmp), 1)
        );
    p = vsubq_s64(
            vreinterpretq_s64_u32(veorq_u32(vreinterpretq_u32_s64(p), mask)),
            vreinterpretq_s64_u32(mask)
        );
#endif
}

template<> template<> inline veci_i64x2_t veci_i64x2_t::abs_() const
{ veci_i64x2_t ret(p); ret.abs_(); return ret; }


// unary minus
inline veci_i64x2_t operator-(const veci_i64x2_t & v)
{
#if defined(PVECI_INTEL)
    return veci_i64x2_t(math::imath_t<int64_t,__m128i>::zeroes()) -= v.p;
#elif defined(PVECI_ARM)
#ifndef PVECI_ARM_GCC
    return veci_i64x2_t(math::imath_t<int64_t,__n128>::zeroes()) -= v.p;
#else
    return veci_i64x2_t(math::imath_t<int64_t,int64x2_t>::zeroes()) -= v.p;
#endif
#endif
}


// load aligned
template<> inline void veci_i64x2_t::loada(const int64_t * ptr)
{
#if defined(PVECI_INTEL)
    p = _mm_load_si128(reinterpret_cast<const packed_t *>(ptr));
#elif defined(PVECI_ARM)
    p = *reinterpret_cast<const packed_t *>(ptr);
#endif
}

// load unaligned
template<> inline void veci_i64x2_t::loadu(const int64_t * ptr)
{
#if defined(PVECI_INTEL)
    p = _mm_loadu_si128(reinterpret_cast<const packed_t *>(ptr));
#elif defined(PVECI_ARM)
    p = *reinterpret_cast<const packed_t *>(ptr);
#endif
}

// store unaligned
template<> inline void veci_i64x2_t::storeu(int64_t * ptr)
{
#if defined(PVECI_INTEL)
    _mm_storeu_si128(reinterpret_cast<packed_t *>(ptr), p);
#elif defined(PVECI_ARM)
    // this may not even be supported (or maybe it is supported per VLDx)
    p = *reinterpret_cast<const packed_t *>(ptr);
#endif
}


#ifdef PVECI_ARM
template<> inline veci_i64x2_t veci_i64x2_t::operator~() const
{ return veci_i64x2_t(veorq_s64(p, math_t::onebits())); }
#endif


// free-standing arithmetic operations (element-wise)
inline veci_i64x2_t operator+(const veci_i64x2_t & v1, const veci_i64x2_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_i64x2_t(_mm_add_epi64(v1.p, v2.p));
#elif defined(PVECI_ARM)
    return veci_i64x2_t(vaddq_s64(v1.p, v2.p));
#endif
}

inline veci_i64x2_t operator+(const veci_i64x2_t & v, int64_t s)
{
#if defined(PVECI_INTEL)
    return veci_i64x2_t(_mm_add_epi64(v.p, _mm_set1_epi64x(s)));
#elif defined(PVECI_ARM)
    return veci_i64x2_t(vaddq_s64(v.p, vmovq_n_s64(s)));
#endif
}

inline veci_i64x2_t operator+(int64_t s, const veci_i64x2_t & v)
{
#if defined(PVECI_INTEL)
    return veci_i64x2_t(_mm_add_epi64(_mm_set1_epi64x(s), v.p));
#elif defined(PVECI_ARM)
    return veci_i64x2_t(vaddq_s64(vmovq_n_s64(s), v.p));
#endif
}

inline veci_i64x2_t operator+(const veci_i64x2_t & v1, veci_i64x2_t::packed_t v2)
{
#if defined(PVECI_INTEL)
    return veci_i64x2_t(_mm_add_epi64(v1.p, v2));
#elif defined(PVECI_ARM)
    return veci_i64x2_t(vaddq_s64(v1.p, v2));
#endif
}

inline veci_i64x2_t operator+(veci_i64x2_t::packed_t v1, const veci_i64x2_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_i64x2_t(_mm_add_epi64(v1, v2.p));
#elif defined(PVECI_ARM)
    return veci_i64x2_t(vaddq_s64(v1, v2.p));
#endif
}

inline veci_i64x2_t operator-(const veci_i64x2_t & v1, const veci_i64x2_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_i64x2_t(_mm_sub_epi64(v1.p, v2.p));
#elif defined(PVECI_ARM)
    return veci_i64x2_t(vsubq_s64(v1.p, v2.p));
#endif
}

inline veci_i64x2_t operator-(const veci_i64x2_t & v, int64_t s)
{
#if defined(PVECI_INTEL)
    return veci_i64x2_t(_mm_sub_epi64(v.p, _mm_set1_epi64x(s)));
#elif defined(PVECI_ARM)
    return veci_i64x2_t(vsubq_s64(v.p, vmovq_n_s64(s)));
#endif
}

inline veci_i64x2_t operator-(int64_t s, const veci_i64x2_t & v)
{
#if defined(PVECI_INTEL)
    return veci_i64x2_t(_mm_sub_epi64(_mm_set1_epi64x(s), v.p));
#elif defined(PVECI_ARM)
    return veci_i64x2_t(vsubq_s64(vmovq_n_s64(s), v.p));
#endif
}

inline veci_i64x2_t operator-(const veci_i64x2_t & v1, veci_i64x2_t::packed_t v2)
{
#if defined(PVECI_INTEL)
    return veci_i64x2_t(_mm_sub_epi64(v1.p, v2));
#elif defined(PVECI_ARM)
    return veci_i64x2_t(vsubq_s64(v1.p, v2));
#endif
}

inline veci_i64x2_t operator-(veci_i64x2_t::packed_t v1, const veci_i64x2_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_i64x2_t(_mm_sub_epi64(v1, v2.p));
#elif defined(PVECI_ARM)
    return veci_i64x2_t(vsubq_s64(v1, v2.p));
#endif
}

// free-standing bit-wise logical operations
inline veci_i64x2_t operator&(veci_i64x2_t op1, veci_i64x2_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i64x2_t(_mm_and_si128(op1.p, op2.p));
#elif defined(PVECI_ARM)
    return veci_i64x2_t(vandq_s64(op1.p, op2.p));
#endif
}

inline veci_i64x2_t operator&(veci_i64x2_t::packed_t op1, veci_i64x2_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i64x2_t(_mm_and_si128(op1, op2.p));
#elif defined(PVECI_ARM)
    return veci_i64x2_t(vandq_s64(op1, op2.p));
#endif
}

inline veci_i64x2_t operator&(veci_i64x2_t op1, veci_i64x2_t::packed_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i64x2_t(_mm_and_si128(op1.p, op2));
#elif defined(PVECI_ARM)
    return veci_i64x2_t(vandq_s64(op1.p, op2));
#endif
}

inline veci_i64x2_t operator|(veci_i64x2_t op1, veci_i64x2_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i64x2_t(_mm_or_si128(op1.p, op2.p));
#elif defined(PVECI_ARM)
    return veci_i64x2_t(vorrq_s64(op1.p, op2.p));
#endif
}

inline veci_i64x2_t operator|(veci_i64x2_t::packed_t op1, veci_i64x2_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i64x2_t(_mm_or_si128(op1, op2.p));
#elif defined(PVECI_ARM)
    return veci_i64x2_t(vorrq_s64(op1, op2.p));
#endif
}

inline veci_i64x2_t operator|(veci_i64x2_t op1, veci_i64x2_t::packed_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i64x2_t(_mm_or_si128(op1.p, op2));
#elif defined(PVECI_ARM)
    return veci_i64x2_t(vorrq_s64(op1.p, op2));
#endif
}

inline veci_i64x2_t operator^(veci_i64x2_t op1, veci_i64x2_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i64x2_t(_mm_xor_si128(op1.p, op2.p));
#elif defined(PVECI_ARM)
    return veci_i64x2_t(veorq_s64(op1.p, op2.p));
#endif
}

inline veci_i64x2_t operator^(veci_i64x2_t::packed_t op1, veci_i64x2_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i64x2_t(_mm_xor_si128(op1, op2.p));
#elif defined(PVECI_ARM)
    return veci_i64x2_t(veorq_s64(op1, op2.p));
#endif
}

inline veci_i64x2_t operator^(veci_i64x2_t op1, veci_i64x2_t::packed_t op2)
{
#if defined(PVECI_INTEL)
    return veci_i64x2_t(_mm_xor_si128(op1.p, op2));
#elif defined(PVECI_ARM)
    return veci_i64x2_t(veorq_s64(op1.p, op2));
#endif
}


#if defined(PVECI_INTEL)

inline veci_i64x2_t xx(const veci_i64x2_t & v) { return veci_i64x2_t(_mm_shuffle_epi32(v.p, _MM_SHUFFLE(1, 0, 1, 0))); }
inline veci_i64x2_t xx(const veci_i64x2_t & v1, const veci_i64x2_t & v2)
{ return veci_i64x2_t(_mm_unpacklo_epi64(v1.p, v2.p)); }
inline veci_i64x2_t xy(const veci_i64x2_t & v) { return v; }
inline veci_i64x2_t xy(const veci_i64x2_t & v1, const veci_i64x2_t & v2)
{ return veci_i64x2_t(_mm_unpackhi_epi64(_mm_shuffle_epi32(v1.p, _MM_SHUFFLE(1, 0, 1, 0)), _mm_shuffle_epi32(v2.p, _MM_SHUFFLE(3, 2, 3, 2)))); }

inline veci_i64x2_t yx(const veci_i64x2_t & v) { return veci_i64x2_t(_mm_shuffle_epi32(v.p, _MM_SHUFFLE(1, 0, 3, 2))); }
inline veci_i64x2_t yx(const veci_i64x2_t & v1, const veci_i64x2_t & v2)
{ return veci_i64x2_t(_mm_unpackhi_epi64(_mm_shuffle_epi32(v1.p, _MM_SHUFFLE(3, 2, 3, 2)), _mm_shuffle_epi32(v2.p, _MM_SHUFFLE(1, 0, 1, 0)))); }

inline veci_i64x2_t yy(const veci_i64x2_t & v) { return veci_i64x2_t(_mm_shuffle_epi32(v.p, _MM_SHUFFLE(3, 2, 3, 2))); }
inline veci_i64x2_t yy(const veci_i64x2_t & v1, const veci_i64x2_t & v2)
{ return veci_i64x2_t(_mm_unpackhi_epi64(v1.p, v2.p)); }

#elif defined(PVECI_ARM)

inline veci_i64x2_t xx(const veci_i64x2_t & v)
{ return veci_i64x2_t(vcombine_s64(vget_low_s64(v.p), vget_low_s64(v.p))); }
inline veci_i64x2_t xx(const veci_i64x2_t & v1, const veci_i64x2_t & v2)
{ return veci_i64x2_t(vcombine_s64(vget_low_s64(v1.p), vget_low_s64(v2.p))); }

inline veci_i64x2_t xy(const veci_i64x2_t & v)
{ return veci_i64x2_t(v); }
inline veci_i64x2_t xy(const veci_i64x2_t & v1, const veci_i64x2_t & v2)
{ return veci_i64x2_t(vcombine_s64(vget_low_s64(v1.p), vget_high_s64(v2.p))); }

inline veci_i64x2_t yx(const veci_i64x2_t & v)
{ return veci_i64x2_t(vcombine_s64(vget_high_s64(v.p), vget_low_s64(v.p))); }
inline veci_i64x2_t yx(const veci_i64x2_t & v1, const veci_i64x2_t & v2)
{ return veci_i64x2_t(vcombine_s64(vget_high_s64(v1.p), vget_low_s64(v2.p))); }

inline veci_i64x2_t yy(const veci_i64x2_t & v)
{ return veci_i64x2_t(vcombine_s64(vget_high_s64(v.p), vget_high_s64(v.p))); }
inline veci_i64x2_t yy(const veci_i64x2_t & v1, const veci_i64x2_t & v2)
{ return veci_i64x2_t(vcombine_s64(vget_high_s64(v1.p), vget_high_s64(v2.p))); }


#endif // defined(PVECI_INTEL)


#undef veci_i64x2_t


/*****************************************************************************
 *                                                                           *
 * veci_ui64x2_t implementation                                              *
 *                                                                           *
 *****************************************************************************/
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
// x86/x64 with MSVC
#  define veci_ui64x2_t veci_t<uint64_t,2,__m128i>
#elif defined(_MSC_VER) && defined(_M_ARM)
// ARM NEON with MSVC
#  define veci_ui64x2_t veci_t<uint64_t,2,__n128>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
// x86/x64 with GCC / MinGW
#  define veci_ui64x2_t veci_t<uint64_t,2,__m128i>
#elif defined(__GNUC__) && defined(__ARM_ARCH) && defined(__ARM_NEON)
// ARM NEON with GCC
#  define veci_ui64x2_t veci_t<uint64_t,2,uint64x2_t>
#endif

template<> inline veci_ui64x2_t::veci_t()
{ p = math_t::zeroes(); }
template<> inline veci_ui64x2_t::veci_t(uint64_t v0)
{ p = math_t::zeroes();
  v[0] = v0; }
template<> inline veci_ui64x2_t::veci_t(uint64_t v0, uint64_t v1)
{ v[0] = v0; v[1] = v1; }
template<> inline veci_ui64x2_t::veci_t(std::initializer_list<type_t> l)
{
    p = math_t::zeroes();
    int count = int(l.size() <= 2 ? l.size() : 2);
    int i = 0; for(auto it = l.begin(); i < count; ++i, ++it) v[i] = *it;
}


// vector addition, subtraction (not saturated)
// and logical operations
template<> inline veci_ui64x2_t & veci_ui64x2_t::operator+=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_add_epi64(p, v2);
#elif defined(PVECI_ARM)
    p = vaddq_u64(p, v2);
#endif
    return *this;
}

template<> inline veci_ui64x2_t & veci_ui64x2_t::operator-=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_sub_epi64(p, v2);
#elif defined(PVECI_ARM)
    p = vsubq_u64(p, v2);
#endif
    return *this;
}

template<> inline veci_ui64x2_t & veci_ui64x2_t::operator&=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_and_si128(p, v2);
#elif defined(PVECI_ARM)
    p = vandq_u64(p, v2);
#endif
    return *this;
}

template<> inline veci_ui64x2_t & veci_ui64x2_t::operator|=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_or_si128(p, v2);
#elif defined(PVECI_ARM)
    p = vorrq_u64(p, v2);
#endif
    return *this;
}

template<> inline veci_ui64x2_t & veci_ui64x2_t::operator^=(packed_t v2)
{
#if defined(PVECI_INTEL)
    p = _mm_xor_si128(p, v2);
#elif defined(PVECI_ARM)
    p = veorq_u64(p, v2);
#endif
    return *this;
}

template<> inline veci_ui64x2_t & veci_ui64x2_t::operator+=(const veci_ui64x2_t & v2)
{ return operator+=(v2.p); }

template<> inline veci_ui64x2_t & veci_ui64x2_t::operator-=(const veci_ui64x2_t & v2)
{ return operator-=(v2.p); }

template<> inline veci_ui64x2_t & veci_ui64x2_t::operator&=(const veci_ui64x2_t & v2)
{ return operator&=(v2.p); }

template<> inline veci_ui64x2_t & veci_ui64x2_t::operator|=(const veci_ui64x2_t & v2)
{ return operator|=(v2.p); }

template<> inline veci_ui64x2_t & veci_ui64x2_t::operator^=(const veci_ui64x2_t & v2)
{ return operator^=(v2.p); }


// packed vector comparisons
// (impl postponed for ARM NEON)
#ifdef PVECI_INTEL
template<> inline bool veci_ui64x2_t::operator==(packed_t v2) const
{
#if defined(PVECI_INTEL)
# if defined(SSE4)
    return _mm_movemask_epi8(_mm_cmpeq_epi64(p, v2)) == 0xFFFF;
# else
    return _mm_movemask_epi8(_mm_cmpeq_epi32(p, v2)) == 0xFFFF;
# endif
#elif defined(PVECI_ARM)
    XXX
#endif
}

template<> inline bool veci_ui64x2_t::operator!=(packed_t v2) const
{
#if defined(PVECI_INTEL)
# if defined(SSE4)
    return _mm_movemask_epi8(_mm_cmpeq_epi64(p, v2)) == 0x0000;
# else
    return _mm_movemask_epi8(_mm_cmpeq_epi32(p, v2)) == 0x0000;
# endif
#elif defined(PVECI_ARM)

#endif
}

template<> inline bool veci_ui64x2_t::operator<(packed_t v2) const
{
#if defined(PVECI_INTEL)
# pragma message("performance warning: SSE2/SSE3/SSSE3/SSE4.1/SSE4.2 do not provide this comparison operation for packed uint64_t x 2")
# if defined(SSE4_2)
    return
        _mm_movemask_epi8(
            _mm_or_si128(
                _mm_cmpeq_epi64(p, v2),
                _mm_cmpgt_epi64(
                    _mm_xor_si128(p, math_t::sign_mask()),
                    _mm_xor_si128(v2, math_t::sign_mask())
                )
            )
        ) == 0x0000;
# else
    __m128i mask = imath_t<uint32_t,__m128i>::sign_mask();
    int lt = _mm_movemask_epi8(_mm_cmplt_epi32(_mm_xor_si128(p, mask), _mm_xor_si128(v2, mask)));
    int eq = _mm_movemask_epi8(_mm_cmpeq_epi32(p, v2));
    // (upper(a) < upper(b)) | ((upper(a) == upper(b)) && (lower(a) < lower(b))
    return (lt & 0xFF00) || ((eq & 0xFF00) && (lt & 0x00FF));
# endif
#elif defined(PVECI_ARM)
    XXX
#endif
}

template<> inline bool veci_ui64x2_t::operator<=(packed_t v2) const
{
#if defined(PVECI_INTEL)
# pragma message("performance warning: SSE2/SSE3/SSSE3/SSE4.1/SSE4.2 do not provide this comparison operation for packed uint64_t x 2")
# if defined(SSE4_2)
    return
        _mm_movemask_epi8(
            _mm_cmpgt_epi64(
                _mm_xor_si128(p, math_t::sign_mask()),
                _mm_xor_si128(v2, math_t::sign_mask())
            )
        ) == 0x0000;
# else
    __m128i mask = imath_t<uint32_t,__m128i>::sign_mask();
    int lt = _mm_movemask_epi8(_mm_cmplt_epi32(_mm_xor_si128(p, mask), _mm_xor_si128(v2, mask)));
    int eq = _mm_movemask_epi8(_mm_cmpeq_epi32(p, v2));
    return (lt & 0xF0F0) || ((eq & 0xF0F0) && ((lt & 0x0F0F) || (eq & 0x0F0F)));
# endif
#elif defined(PVECI_ARM)
    XXX
#endif
}

template<> inline bool veci_ui64x2_t::operator>(packed_t v2) const
{
#if defined(PVECI_INTEL)
# pragma message("performance warning: SSE2/SSE3/SSSE3/SSE4.1/SSE4.2 do not provide this comparison operation for packed uint64_t x 2")
# if defined(SSE4_2)
    return
        _mm_movemask_epi8(
            _mm_cmpgt_epi64(
                _mm_xor_si128(p, math_t::sign_mask()),
                _mm_xor_si128(v2, math_t::sign_mask())
            )
        ) == 0xFFFF;
# else
    __m128i mask = imath_t<uint32_t,__m128i>::sign_mask();
    int gt = _mm_movemask_epi8(_mm_cmpgt_epi32(_mm_xor_si128(p, mask), _mm_xor_si128(v2, mask)));
    int eq = _mm_movemask_epi8(_mm_cmpeq_epi32(p, v2));
    // (upper(a) > upper(b)) | ((upper(a) == upper(b)) && (lower(a) > lower(b))
    return (gt & 0xF0F0) || ((eq & 0xF0F0) && (gt & 0x0F0F));
# endif
#elif defined(PVECI_ARM)
    XXX
#endif
}

template<> inline bool veci_ui64x2_t::operator>=(packed_t v2) const
{
#if defined(PVECI_INTEL)
# pragma message("performance warning: SSE2/SSE3/SSSE3/SSE4.1/SSE4.2 do not provide this comparison operation for packed uint64_t x 2")
# if defined(SSE4_2)
    return
        _mm_movemask_epi8(
            _mm_or_si128(
                _mm_cmpeq_epi64(p, v2),
                _mm_cmpgt_epi64(
                    _mm_xor_si128(p, math_t::sign_mask()),
                    _mm_xor_si128(v2, math_t::sign_mask())
                )
            )
        ) == 0xFFFF;
# else
    __m128i mask = imath_t<uint32_t,__m128i>::sign_mask();
    int gt = _mm_movemask_epi8(_mm_cmpgt_epi32(_mm_xor_si128(p, mask), _mm_xor_si128(v2, mask)));
    int eq = _mm_movemask_epi8(_mm_cmpeq_epi32(p, v2));
    return (gt & 0xF0F0) || ((eq & 0xF0F0) && ((gt & 0x0F0F) || (eq & 0x0F0F)));
# endif
#elif defined(PVECI_ARM)
    XXX
#endif
}

// returns true if at least one value pair is unequal
template<> inline bool veci_ui64x2_t::neq_one(packed_t v2) const
{
#if defined(PVECI_INTEL)
# if defined(SSE4)
    return _mm_movemask_epi8(_mm_cmpeq_epi64(p, v2)) != 0xFFFF;
# else
    return _mm_movemask_epi8(_mm_cmpeq_epi32(p, v2)) != 0xFFFF;
# endif
#elif defined(PVECI_ARM)
    XXX
#endif
}

// vector comparisons
// SSE2 doesn't provide 64bit comparisons; operators ==,!=,neq_one() is
// implemented per other sized comparisons as only (un-)identical bit patterns
// are relevant
template<> inline bool veci_ui64x2_t::operator==(const veci_ui64x2_t & v2) const
{ return operator==(v2.p); }

template<> inline bool veci_ui64x2_t::operator!=(const veci_ui64x2_t & v2) const
{ return operator!=(v2.p); }

template<> inline bool veci_ui64x2_t::operator<(const veci_ui64x2_t & v2) const
{ return operator<(v2.p); }

template<> inline bool veci_ui64x2_t::operator<=(const veci_ui64x2_t & v2) const
{ return operator<=(v2.p); }

template<> inline bool veci_ui64x2_t::operator>(const veci_ui64x2_t & v2) const
{ return operator>(v2.p); }

template<> inline bool veci_ui64x2_t::operator>=(const veci_ui64x2_t & v2) const
{ return operator>=(v2.p); }

template<> inline bool veci_ui64x2_t::neq_one(const veci_ui64x2_t & v2) const
{ return neq_one(v2.p); }

// (impl postponed for ARM NEON)
#endif // defined(PVECI_INTEL)


// SSE2 intrinsic support for int16_t*8 and uint8_t*16 only
// AVX2: ???
// TODO: either disable these per enable_if<> for all other combos
//       or provide a scalar implementation (emit a performance warning
//       message, too)
template<> inline veci_ui64x2_t veci_ui64x2_t::min_(packed_t p1, packed_t p2)
{
#if defined(PVECI_INTEL)
#   pragma message("performance warning: SSE2/SSE3/SSSE3/SSE4.1/SSE4.2 do not provide min() for packed uint64_t x 2")
    __m128i mask =_mm_shuffle_epi32(_mm_cmpeq_epi32(_mm_srli_epi64(_mm_sub_epi64(p1, p2), 63), _mm_setzero_si128()), 0xA0);
    return veci_ui64x2_t(_mm_or_si128(_mm_andnot_si128(mask, p1), _mm_and_si128(mask, p2)));
#elif defined(PVECI_ARM)
#if 1
# pragma message("performance warning: NEON does not provide min() for packed uint64_t x 2, scalar code is used")
    veci_ui64x2_t ret, v1(p1), v2(p2);
    ret[0] = (std::min)(v1[0], v2[0]);
    ret[1] = (std::min)(v1[1], v2[1]);
    return ret;
#else
#   pragma message("performance warning: NEON does not provide min() for packed uint64_t x 2")
    uint32x4_t tmp =
        vceqq_u32(
            vreinterpretq_u32_u64(
                vshrq_n_u64(
                    vsubq_u64(p1, p2)/*v1 - v2*/,
                    63
                )/*shift sign bit*/
            ),
            vdupq_n_u32(0)
        );
    // (2,2,0,0)
    uint32x4_t mask =
        vcombine_u32(
            vdup_lane_u32(vget_high_u32(tmp), 0),
            vdup_lane_u32(vget_low_u32(tmp), 0)
        );
    return
        veci_ui64x2_t(
            vreinterpretq_u64_u32(
                vorrq_u32(
                    vbicq_u32(vreinterpretq_u32_u64(p1), mask), // destroys larger value or keeps it
                    vandq_u32(vreinterpretq_u32_u64(p2), mask)  // dito
                )
            )
        );
#endif
#endif
}

template<> inline veci_ui64x2_t veci_ui64x2_t::min_(const veci_ui64x2_t & v1, const veci_ui64x2_t & v2)
{ return min_(v1.p, v2.p); }

template<> inline veci_ui64x2_t veci_ui64x2_t::max_(packed_t p1, packed_t p2)
{
#if defined(PVECI_INTEL)
#   pragma message("performance warning: SSE2/SSE3/SSSE3/SSE4.1/SSE4.2 do not provide max() for packed uint64_t x 2")
    __m128i mask =_mm_shuffle_epi32(_mm_cmpeq_epi32(_mm_srli_epi64(_mm_sub_epi64(p1, p2), 63), _mm_setzero_si128()), 0xA0);
    return veci_ui64x2_t(_mm_or_si128(_mm_and_si128(mask, p1), _mm_andnot_si128(mask, p2)));
#elif defined(PVECI_ARM)
#if 1
# pragma message("performance warning: NEON does not provide max() for packed uint64_t x 2, scalar code is used")
    veci_ui64x2_t ret, v1(p1), v2(p2);
    ret[0] = (std::max)(v1[0], v2[0]);
    ret[1] = (std::max)(v1[1], v2[1]);
    return ret;
#else
# pragma message("performance warning: NEON does not provide max() for packed uint64_t x 2")
    uint32x4_t tmp =
        vceqq_u32(
            vreinterpretq_u32_u64(
                vshrq_n_u64(
                    vsubq_u64(p1, p2)/*v1 - v2*/,
                    63
                )/*shift sign bit*/
            ),
            vdupq_n_u32(0)
        );
    // (2,2,0,0)
    uint32x4_t mask =
        vcombine_u32(
            vdup_lane_u32(vget_high_u32(tmp), 0),
            vdup_lane_u32(vget_low_u32(tmp), 0)
        );
    return
        veci_ui64x2_t(
            vreinterpretq_u64_u32(
                vorrq_u32(
                    vandq_u32(vreinterpretq_u32_u64(p1), mask), // destroys smaller value or keeps it
                    vbicq_u32(vreinterpretq_u32_u64(p2), mask)  // dito
                )
            )
        );
#endif
#endif
}

template<> inline veci_ui64x2_t veci_ui64x2_t::max_(const veci_ui64x2_t & v1, const veci_ui64x2_t & v2)
{ return max_(v1.p, v2.p); }



// load aligned
template<> inline void veci_ui64x2_t::loada(const uint64_t * ptr)
{
#if defined(PVECI_INTEL)
    p = _mm_load_si128(reinterpret_cast<const packed_t *>(ptr));
#elif defined(PVECI_ARM)
    p = *reinterpret_cast<const packed_t *>(ptr);
#endif
}

// load unaligned
template<> inline void veci_ui64x2_t::loadu(const uint64_t * ptr)
{
#if defined(PVECI_INTEL)
    p = _mm_loadu_si128(reinterpret_cast<const packed_t *>(ptr));
#elif defined(PVECI_ARM)
    p = *reinterpret_cast<const packed_t *>(ptr);
#endif
}

// store unaligned
template<> inline void veci_ui64x2_t::storeu(uint64_t * ptr)
{
#if defined(PVECI_INTEL)
    _mm_storeu_si128(reinterpret_cast<packed_t *>(ptr), p);
#elif defined(PVECI_ARM)
    // this may not even be supported (or maybe it is supported per VLDx)
    p = *reinterpret_cast<const packed_t *>(ptr);
#endif
}


#ifdef PVECI_ARM
template<> inline veci_ui64x2_t veci_ui64x2_t::operator~() const
{ return veci_ui64x2_t(veorq_u64(p, math_t::onebits())); }
#endif


// free-standing arithmetic operations (element-wise)
inline veci_ui64x2_t operator+(const veci_ui64x2_t & v1, const veci_ui64x2_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_ui64x2_t(_mm_add_epi64(v1.p, v2.p));
#elif defined(PVECI_ARM)
    return veci_ui64x2_t(vaddq_u64(v1.p, v2.p));
#endif
}

inline veci_ui64x2_t operator+(const veci_ui64x2_t & v, uint64_t s)
{
#if defined(PVECI_INTEL)
    return veci_ui64x2_t(_mm_add_epi64(v.p, _mm_set1_epi64x(s)));
#elif defined(PVECI_ARM)
    return veci_ui64x2_t(vaddq_u64(v.p, vmovq_n_u64(s)));
#endif
}

inline veci_ui64x2_t operator+(int64_t s, const veci_ui64x2_t & v)
{
#if defined(PVECI_INTEL)
    return veci_ui64x2_t(_mm_add_epi64(_mm_set1_epi64x(s), v.p));
#elif defined(PVECI_ARM)
    return veci_ui64x2_t(vaddq_u64(vmovq_n_u64(s), v.p));
#endif
}

inline veci_ui64x2_t operator+(const veci_ui64x2_t & v1, veci_ui64x2_t::packed_t v2)
{
#if defined(PVECI_INTEL)
    return veci_ui64x2_t(_mm_add_epi64(v1.p, v2));
#elif defined(PVECI_ARM)
    return veci_ui64x2_t(vaddq_u64(v1.p, v2));
#endif
}

inline veci_ui64x2_t operator+(veci_ui64x2_t::packed_t v1, const veci_ui64x2_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_ui64x2_t(_mm_add_epi64(v1, v2.p));
#elif defined(PVECI_ARM)
    return veci_ui64x2_t(vaddq_u64(v1, v2.p));
#endif
}

inline veci_ui64x2_t operator-(const veci_ui64x2_t & v1, const veci_ui64x2_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_ui64x2_t(_mm_sub_epi64(v1.p, v2.p));
#elif defined(PVECI_ARM)
    return veci_ui64x2_t(vsubq_u64(v1.p, v2.p));
#endif
}

inline veci_ui64x2_t operator-(const veci_ui64x2_t & v, int64_t s)
{
#if defined(PVECI_INTEL)
    return veci_ui64x2_t(_mm_sub_epi64(v.p, _mm_set1_epi64x(s)));
#elif defined(PVECI_ARM)
    return veci_ui64x2_t(vsubq_u64(v.p, vmovq_n_u64(s)));
#endif
}

inline veci_ui64x2_t operator-(int64_t s, const veci_ui64x2_t & v)
{
#if defined(PVECI_INTEL)
    return veci_ui64x2_t(_mm_sub_epi64(_mm_set1_epi64x(s), v.p));
#elif defined(PVECI_ARM)
    return veci_ui64x2_t(vsubq_u64(vmovq_n_u64(s), v.p));
#endif
}

inline veci_ui64x2_t operator-(const veci_ui64x2_t & v1, veci_ui64x2_t::packed_t v2)
{
#if defined(PVECI_INTEL)
    return veci_ui64x2_t(_mm_sub_epi64(v1.p, v2));
#elif defined(PVECI_ARM)
    return veci_ui64x2_t(vsubq_u64(v1.p, v2));
#endif
}

inline veci_ui64x2_t operator-(veci_ui64x2_t::packed_t v1, const veci_ui64x2_t & v2)
{
#if defined(PVECI_INTEL)
    return veci_ui64x2_t(_mm_sub_epi64(v1, v2.p));
#elif defined(PVECI_ARM)
    return veci_ui64x2_t(vsubq_u64(v1, v2.p));
#endif
}


// free-standing bit-wise logical operations
inline veci_ui64x2_t operator&(veci_ui64x2_t op1, veci_ui64x2_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui64x2_t(_mm_and_si128(op1.p, op2.p));
#elif defined(PVECI_ARM)
    return veci_ui64x2_t(vandq_u64(op1.p, op2.p));
#endif
}

inline veci_ui64x2_t operator&(veci_ui64x2_t::packed_t op1, veci_ui64x2_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui64x2_t(_mm_and_si128(op1, op2.p));
#elif defined(PVECI_ARM)
    return veci_ui64x2_t(vandq_u64(op1, op2.p));
#endif
}

inline veci_ui64x2_t operator&(veci_ui64x2_t op1, veci_ui64x2_t::packed_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui64x2_t(_mm_and_si128(op1.p, op2));
#elif defined(PVECI_ARM)
    return veci_ui64x2_t(vandq_u64(op1.p, op2));
#endif
}

inline veci_ui64x2_t operator|(veci_ui64x2_t op1, veci_ui64x2_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui64x2_t(_mm_or_si128(op1.p, op2.p));
#elif defined(PVECI_ARM)
    return veci_ui64x2_t(vorrq_u64(op1.p, op2.p));
#endif
}

inline veci_ui64x2_t operator|(veci_ui64x2_t::packed_t op1, veci_ui64x2_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui64x2_t(_mm_or_si128(op1, op2.p));
#elif defined(PVECI_ARM)
    return veci_ui64x2_t(vorrq_u64(op1, op2.p));
#endif
}

inline veci_ui64x2_t operator|(veci_ui64x2_t op1, veci_ui64x2_t::packed_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui64x2_t(_mm_or_si128(op1.p, op2));
#elif defined(PVECI_ARM)
    return veci_ui64x2_t(vorrq_u64(op1.p, op2));
#endif
}

inline veci_ui64x2_t operator^(veci_ui64x2_t op1, veci_ui64x2_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui64x2_t(_mm_xor_si128(op1.p, op2.p));
#elif defined(PVECI_ARM)
    return veci_ui64x2_t(veorq_u64(op1.p, op2.p));
#endif
}

inline veci_ui64x2_t operator^(veci_ui64x2_t::packed_t op1, veci_ui64x2_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui64x2_t(_mm_xor_si128(op1, op2.p));
#elif defined(PVECI_ARM)
    return veci_ui64x2_t(veorq_u64(op1, op2.p));
#endif
}

inline veci_ui64x2_t operator^(veci_ui64x2_t op1, veci_ui64x2_t::packed_t op2)
{
#if defined(PVECI_INTEL)
    return veci_ui64x2_t(_mm_xor_si128(op1.p, op2));
#elif defined(PVECI_ARM)
    return veci_ui64x2_t(veorq_u64(op1.p, op2));
#endif
}


#if defined(PVECI_INTEL)

inline veci_ui64x2_t xx(const veci_ui64x2_t & v) { return veci_ui64x2_t(_mm_shuffle_epi32(v.p, _MM_SHUFFLE(1, 0, 1, 0))); }
inline veci_ui64x2_t xx(const veci_ui64x2_t & v1, const veci_ui64x2_t & v2)
{ return veci_ui64x2_t(_mm_unpacklo_epi64(v1.p, v2.p)); }

inline veci_ui64x2_t xy(const veci_ui64x2_t & v) { return v; }
inline veci_ui64x2_t xy(const veci_ui64x2_t & v1, const veci_ui64x2_t & v2)
{ return veci_ui64x2_t(_mm_unpackhi_epi64(_mm_shuffle_epi32(v1.p, _MM_SHUFFLE(1, 0, 1, 0)), _mm_shuffle_epi32(v2.p, _MM_SHUFFLE(3, 2, 3, 2)))); }

inline veci_ui64x2_t yx(const veci_ui64x2_t & v) { return veci_ui64x2_t(_mm_shuffle_epi32(v.p, _MM_SHUFFLE(1, 0, 3, 2))); }
inline veci_ui64x2_t yx(const veci_ui64x2_t & v1, const veci_ui64x2_t & v2)
{ return veci_ui64x2_t(_mm_unpackhi_epi64(_mm_shuffle_epi32(v1.p, _MM_SHUFFLE(3, 2, 3, 2)), _mm_shuffle_epi32(v2.p, _MM_SHUFFLE(1, 0, 1, 0)))); }

inline veci_ui64x2_t yy(const veci_ui64x2_t & v) { return veci_ui64x2_t(_mm_shuffle_epi32(v.p, _MM_SHUFFLE(3, 2, 3, 2))); }
inline veci_ui64x2_t yy(const veci_ui64x2_t & v1, const veci_ui64x2_t & v2)
{ return veci_ui64x2_t(_mm_unpackhi_epi64(v1.p, v2.p)); }

#elif defined(PVECI_ARM)

inline veci_ui64x2_t xx(const veci_ui64x2_t & v)
{ return veci_ui64x2_t(vcombine_u64(vget_low_u64(v.p), vget_low_u64(v.p))); }
inline veci_ui64x2_t xx(const veci_ui64x2_t & v1, const veci_ui64x2_t & v2)
{ return veci_ui64x2_t(vcombine_u64(vget_low_u64(v1.p), vget_low_u64(v2.p))); }

inline veci_ui64x2_t xy(const veci_ui64x2_t & v)
{ return veci_ui64x2_t(v); }
inline veci_ui64x2_t xy(const veci_ui64x2_t & v1, const veci_ui64x2_t & v2)
{ return veci_ui64x2_t(vcombine_u64(vget_low_u64(v1.p), vget_high_u64(v2.p))); }

inline veci_ui64x2_t yx(const veci_ui64x2_t & v)
{ return veci_ui64x2_t(vcombine_u64(vget_high_u64(v.p), vget_low_u64(v.p))); }
inline veci_ui64x2_t yx(const veci_ui64x2_t & v1, const veci_ui64x2_t & v2)
{ return veci_ui64x2_t(vcombine_u64(vget_high_u64(v1.p), vget_low_u64(v2.p))); }

inline veci_ui64x2_t yy(const veci_ui64x2_t & v)
{ return veci_ui64x2_t(vcombine_u64(vget_high_u64(v.p), vget_high_u64(v.p))); }
inline veci_ui64x2_t yy(const veci_ui64x2_t & v1, const veci_ui64x2_t & v2)
{ return veci_ui64x2_t(vcombine_u64(vget_high_u64(v1.p), vget_high_u64(v2.p))); }

#endif


#undef veci_ui64x2_t



#if defined(_MSC_VER)
#define PVECI_ALIGN(x) __declspec(align(x))
#elif defined(__GNUC__)
#define PVECI_ALIGN(x) __attribute__((aligned (x)))
#endif

#if defined(PVECI_INTEL)

typedef veci_t<int8_t,16,__m128i> PVECI_ALIGN(16) veci_i8x16_t;
typedef veci_t<int16_t,8,__m128i> PVECI_ALIGN(16) veci_i16x8_t;
typedef veci_t<int32_t,4,__m128i> PVECI_ALIGN(16) veci_i32x4_t;
typedef veci_t<int64_t,2,__m128i> PVECI_ALIGN(16) veci_i64x2_t;

typedef veci_t<uint8_t,16,__m128i> PVECI_ALIGN(16) veci_ui8x16_t;
typedef veci_t<uint16_t,8,__m128i> PVECI_ALIGN(16) veci_ui16x8_t;
typedef veci_t<uint32_t,4,__m128i> PVECI_ALIGN(16) veci_ui32x4_t;
typedef veci_t<uint64_t,2,__m128i> PVECI_ALIGN(16) veci_ui64x2_t;

#elif defined(PVECI_ARM)

#ifndef PVECI_ARM_GCC

typedef veci_t<int8_t,16,__n128> PVECI_ALIGN(8) veci_i8x16_t;
typedef veci_t<int16_t,8,__n128> PVECI_ALIGN(8) veci_i16x8_t;
typedef veci_t<int32_t,4,__n128> PVECI_ALIGN(8) veci_i32x4_t;
typedef veci_t<int64_t,2,__n128> PVECI_ALIGN(8) veci_i64x2_t;

typedef veci_t<uint8_t,16,__n128> PVECI_ALIGN(8) veci_ui8x16_t;
typedef veci_t<uint16_t,8,__n128> PVECI_ALIGN(8) veci_ui16x8_t;
typedef veci_t<uint32_t,4,__n128> PVECI_ALIGN(8) veci_ui32x4_t;
typedef veci_t<uint64_t,2,__n128> PVECI_ALIGN(8) veci_ui64x2_t;

#else

typedef veci_t<int8_t,16,int8x16_t> PVECI_ALIGN(8) veci_i8x16_t;
typedef veci_t<int16_t,8,int16x8_t> PVECI_ALIGN(8) veci_i16x8_t;
typedef veci_t<int32_t,4,int32x4_t> PVECI_ALIGN(8) veci_i32x4_t;
typedef veci_t<int64_t,2,int64x2_t> PVECI_ALIGN(8) veci_i64x2_t;

typedef veci_t<uint8_t,16,uint8x16_t> PVECI_ALIGN(8) veci_ui8x16_t;
typedef veci_t<uint16_t,8,uint16x8_t> PVECI_ALIGN(8) veci_ui16x8_t;
typedef veci_t<uint32_t,4,uint32x4_t> PVECI_ALIGN(8) veci_ui32x4_t;
typedef veci_t<uint64_t,2,uint64x2_t> PVECI_ALIGN(8) veci_ui64x2_t;

#endif

#endif


} // namespace math


#endif // !defined(PVECI_H)

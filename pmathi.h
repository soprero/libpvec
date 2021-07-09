/*******************************************************************************
 * pmathi.h                                                                    *
 *                                                                             *
 * Copyright (c) 2015-2017 Ronny Press                                         *
 *                                                                             *
 * rpress@soprero.de                                                           *
 *                                                                             *
 * All rights reserved.                                                        *
 *******************************************************************************/
#ifndef PMATHI_H
#define PMATHI_H


//#include <limits>
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


// preprocessor shortcuts

#if (defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
#define PMATHI_INTEL
#elif (defined(_MSC_VER) && defined(_M_ARM))
#define PMATHI_ARM
#elif (defined(__GNUC__) && defined(__ARM_ARCH) && defined(__ARM_NEON))
#define PMATHI_ARM
#define PMATHI_ARM_GCC
#endif



namespace math {

//
// math_t general template
//
template<typename scalar_t,typename packed_t>
class imath_t
{
public:
    static_assert(
        std::is_integral<scalar_t>::value,
        "imath_t: supports integer types only"
    );

    static inline packed_t zeroes();
    static inline packed_t onebits();
    static inline packed_t sign_mask();
    static inline packed_t largest_val();
    static inline packed_t smallest_val();

    // returns vector with elements containing masks with Z bits cleared starting from MSB
    // and W-Z bits set starting from LSB with W being the width on bits of the elements
    template<unsigned Z> static inline packed_t mask_zupper();
    // returns vector with elements containing masks with Z bits set starting from MSB
    // and W-Z bits cleared starting from LSB with W being the width on bits of the elements
    template<unsigned Z> static inline packed_t mask_zlower();
    // returns vector with elements containing bit number B set and all other bits cleared
    template<unsigned B> static inline packed_t mask_1bit();
    
    imath_t() = delete;
    imath_t(const imath_t &) = delete;
    imath_t & operator=(const imath_t &) = delete;
};


#if defined(PMATHI_INTEL)

template<>
class imath_t<int8_t, __m128i>
{
public:
    typedef int8_t int_t;
    typedef __m128i packed_t;

    static inline __m128i zeroes() { return _mm_setzero_si128(); }
    static inline __m128i onebits() { return _mm_cmpeq_epi32(zeroes(), zeroes()); }
    static inline __m128i sign_mask() {
        __m128i tmp = _mm_slli_epi16(onebits(), 15); // {0x8000}*8
        return _mm_or_si128(tmp, _mm_srli_epi16(tmp, 8)/*{0x0080}*8*/);
    }
    static inline __m128i largest_val() { return _mm_xor_si128(sign_mask(), onebits()); }
    static inline __m128i smallest_val() { return sign_mask(); }

    template<unsigned Z> static inline packed_t mask_zupper() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        __m128i tmp = _mm_srli_epi16(onebits(), Z+8);
        return _mm_or_si128(tmp, _mm_slli_epi16(tmp, 8));
    }
    template<unsigned Z> static inline packed_t mask_zlower() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        __m128i tmp = _mm_slli_epi16(onebits(), Z+8);
        return _mm_or_si128(tmp, _mm_srli_epi16(tmp, 8));
    }
    template<unsigned B> static inline packed_t mask_1bit() {
        static_assert(B < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        const unsigned bits = sizeof(int_t)*8;
        __m128i tmp = _mm_add_epi16(_mm_srli_epi16(onebits(), bits - B), _mm_srli_epi16(onebits(), 15));
        return _mm_or_si128(tmp, _mm_srli_epi16(tmp, 8));
    }

};

template<>
class imath_t<uint8_t, __m128i>
{
public:
    typedef uint8_t int_t;
    typedef __m128i packed_t;

    static inline __m128i zeroes() { return _mm_setzero_si128(); }
    static inline __m128i onebits() { return _mm_cmpeq_epi32(zeroes(), zeroes()); }
    static inline __m128i sign_mask() {
        __m128i tmp = _mm_slli_epi16(onebits(), 15); // {0x8000}*8
        return _mm_or_si128(tmp, _mm_srli_epi16(tmp, 8)/*{0x0080}*8*/);
    }
    static inline __m128i largest_val() { return onebits(); }
    static inline __m128i smallest_val() { return zeroes(); }
    
    template<unsigned Z> static inline packed_t mask_zupper() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        __m128i tmp = _mm_srli_epi16(onebits(), Z+8);
        return _mm_or_si128(tmp, _mm_slli_epi16(tmp, 8));
    }
    template<unsigned Z> static inline packed_t mask_zlower() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        __m128i tmp = _mm_slli_epi16(onebits(), Z+8);
        return _mm_or_si128(tmp, _mm_srli_epi16(tmp, 8));
    }
    template<unsigned B> static inline packed_t mask_1bit() {
        static_assert(B < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        const unsigned bits = sizeof(int_t)*8;
        __m128i tmp = _mm_add_epi16(_mm_srli_epi16(onebits(), bits - B), _mm_srli_epi16(onebits(), 15));
        return _mm_or_si128(tmp, _mm_srli_epi16(tmp, 8));
    }


};

template<>
class imath_t<int16_t, __m128i>
{
public:
    typedef int16_t int_t;
    typedef __m128i packed_t;

    static inline __m128i zeroes() { return _mm_setzero_si128(); }
    static inline __m128i onebits() { return _mm_cmpeq_epi32(zeroes(), zeroes()); }
    static inline __m128i sign_mask() { return _mm_slli_epi16(onebits(), 15); /*{0x8000}*8*/ }
    static inline __m128i largest_val() { return _mm_srli_epi16(onebits(), 1); }
    static inline __m128i smallest_val() { return sign_mask(); }

    template<unsigned Z> static inline packed_t mask_zupper() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        return _mm_srli_epi16(onebits(), Z);
    }
    template<unsigned Z> static inline packed_t mask_zlower() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        return _mm_slli_epi16(onebits(), Z);
    }
    template<unsigned B> static inline packed_t mask_1bit() {
        static_assert(B < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        const unsigned bits = sizeof(int_t)*8;
        return
            _mm_add_epi16(
                B > 0 ? _mm_srli_epi16(onebits(), bits - B) : zeroes(),
                _mm_srli_epi16(onebits(), 15) //(1)*8
            );
    }

};

template<>
class imath_t<uint16_t, __m128i>
{
public:
    typedef uint16_t int_t;
    typedef __m128i packed_t;

    static inline __m128i zeroes() { return _mm_setzero_si128(); }
    static inline __m128i onebits() { return _mm_cmpeq_epi32(zeroes(), zeroes()); }
    static inline __m128i sign_mask() { return _mm_slli_epi16(onebits(), 15); /*{0x8000}*8*/ }
    
    static inline __m128i largest_val() { return onebits(); }
    static inline __m128i smallest_val() { return zeroes(); }

    template<unsigned Z> static inline packed_t mask_zupper() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        return _mm_srli_epi16(onebits(), Z);
    }
    template<unsigned Z> static inline packed_t mask_zlower() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        return _mm_slli_epi16(onebits(), Z);
    }
    template<unsigned B> static inline packed_t mask_1bit() {
        static_assert(B < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        const unsigned bits = sizeof(int_t)*8;
        return
            _mm_add_epi16(
                B > 0 ? _mm_srli_epi16(onebits(), bits - B) : zeroes(),
                _mm_srli_epi16(onebits(), 15) //(1)*8
            );
    }

};

template<>
class imath_t<int32_t, __m128i>
{
public:
    typedef int32_t int_t;
    typedef __m128i packed_t;

    static inline __m128i zeroes() { return _mm_setzero_si128(); }
    static inline __m128i onebits() { return _mm_cmpeq_epi32(zeroes(), zeroes()); }
    static inline __m128i sign_mask() { return _mm_slli_epi32(onebits(), 31); /*{0x80000000}*4*/ }
    static inline __m128i largest_val() { return _mm_srli_epi32(onebits(), 1); }
    static inline __m128i smallest_val() { return sign_mask(); }

    template<unsigned Z> static inline packed_t mask_zupper() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        return _mm_srli_epi32(onebits(), Z);
    }
    template<unsigned Z> static inline packed_t mask_zlower() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        return _mm_slli_epi32(onebits(), Z);
    }
    template<unsigned B> static inline packed_t mask_1bit() {
        static_assert(B < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        const unsigned bits = sizeof(int_t)*8;
        return
            _mm_add_epi32(
                B > 0 ? _mm_srli_epi32(onebits(), bits - B) : zeroes(),
                _mm_srli_epi32(onebits(), 31) //(1)*8
            );
    }

};



template<>
class imath_t<uint32_t, __m128i>
{
public:
    typedef uint32_t int_t;
    typedef __m128i packed_t;

    static inline __m128i zeroes() { return _mm_setzero_si128(); }
    static inline __m128i onebits() { return _mm_cmpeq_epi32(zeroes(), zeroes()); }
    static inline __m128i sign_mask() { return _mm_slli_epi32(onebits(), 31); /*{0x80000000}*4*/ }
    static inline __m128i largest_val() { return onebits(); }
    static inline __m128i smallest_val() { return zeroes(); }

    template<unsigned Z> static inline packed_t mask_zupper() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        return _mm_srli_epi32(onebits(), Z);
    }
    template<unsigned Z> static inline packed_t mask_zlower() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        return _mm_slli_epi32(onebits(), Z);
    }
    template<unsigned B> static inline packed_t mask_1bit() {
        static_assert(B < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        const unsigned bits = sizeof(int_t)*8;
        return
            _mm_add_epi32(
                B > 0 ? _mm_srli_epi32(onebits(), bits - B) : zeroes(),
                _mm_srli_epi32(onebits(), 31) //(1)*8
            );
    }


};

template<>
class imath_t<int64_t, __m128i>
{
public:
    typedef int64_t int_t;
    typedef __m128i packed_t;

    static inline __m128i zeroes() { return _mm_setzero_si128(); }
    static inline __m128i onebits() { return _mm_cmpeq_epi32(zeroes(), zeroes()); }
    static inline __m128i sign_mask() { return _mm_slli_epi64(onebits(), 63); /*{0x8000000000000000}*2*/ }
    static inline __m128i largest_val() { return _mm_srli_epi64(onebits(), 1); }
    static inline __m128i smallest_val() { return sign_mask(); }

    template<unsigned Z> static inline packed_t mask_zupper() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        return _mm_srli_epi64(onebits(), Z);
    }
    template<unsigned Z> static inline packed_t mask_zlower() {
        static_assert(Z > 0, "Zmust be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        return _mm_slli_epi64(onebits(), Z);
    }
    template<unsigned B> static inline packed_t mask_1bit() {
        static_assert(B < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        const unsigned bits = sizeof(int_t)*8;
        return
            _mm_add_epi64(
                B > 0 ? _mm_srli_epi64(onebits(), bits - B) : zeroes(),
                _mm_srli_epi64(onebits(), 63) //(1)*8
            );
    }


};

template<>
class imath_t<uint64_t, __m128i>
{
public:
    typedef uint64_t int_t;
    typedef __m128i packed_t;

    static inline __m128i zeroes() { return _mm_setzero_si128(); }
    static inline __m128i onebits() { return _mm_cmpeq_epi32(zeroes(), zeroes()); }
    static inline __m128i sign_mask() { return _mm_slli_epi64(onebits(), 63); /*{0x8000000000000000}*2*/ }
    static inline __m128i largest_val() { return onebits(); }
    static inline __m128i smallest_val() { return zeroes(); }

    template<unsigned Z> static inline packed_t mask_zupper() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        return _mm_srli_epi64(onebits(), Z);
    }
    template<unsigned Z> static inline packed_t mask_zlower() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        return _mm_slli_epi64(onebits(), Z);
    }
    template<unsigned B> static inline packed_t mask_1bit() {
        static_assert(B < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        const unsigned bits = sizeof(int_t)*8;
        return
            _mm_add_epi64(
                B > 0 ? _mm_srli_epi64(onebits(), bits - B) : zeroes(),
                _mm_srli_epi64(onebits(), 63) //(1)*8
            );
    }
    
};


#elif defined(PMATHI_ARM)

#ifdef PMATHI_ARM_GCC
#define __n128 int8x16_t
#endif

template<>
class imath_t<int8_t, __n128>
{
public:
    typedef int8_t int_t;
    typedef __n128 packed_t;

    static inline __n128 zeroes() { return vdupq_n_s8(0); }
    static inline __n128 onebits()
    {
        return
            vreinterpretq_s8_u32(
                vceqq_u32(
                    vreinterpretq_u32_s8(zeroes()),
                    vreinterpretq_u32_s8(zeroes())
                )
            );
    }
    static inline __n128 sign_mask()
    {
        return
            vreinterpretq_s8_u8(
                vshlq_n_u8(
                    vreinterpretq_u8_s8(onebits()),
                    7
                )
            );
    }

    static inline __n128 largest_val() { return veorq_s8(sign_mask(), onebits()); }
    static inline __n128 smallest_val() { return sign_mask(); }

    template<unsigned Z> static inline packed_t mask_zupper() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        return
            vreinterpretq_s8_u8(
                vshrq_n_u8(
                    vreinterpretq_u8_s8(onebits()),
                    Z
                )
            );
    }
    template<unsigned Z> static inline packed_t mask_zlower() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        return
            vreinterpretq_s8_u8(
                vshlq_n_u8(
                    vreinterpretq_u8_s8(onebits()),
                    Z
                )
            );
    }
    template<unsigned B> static inline packed_t mask_1bit() {
        static_assert(B < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        const unsigned bits = sizeof(int_t)*8;
        return
            vreinterpretq_s8_u8(
                vaddq_u8(
                    B > 0
                        ? vshrq_n_u8(vreinterpretq_u8_s8(onebits()), bits - B)
                        : vreinterpretq_u8_s8(zeroes()),
                    vshrq_n_u8(vreinterpretq_u8_s8(onebits()), 7) //(1)*16
                )
            );
    }
};


#ifdef PMATHI_ARM_GCC
#undef __n128
#define __n128 uint8x16_t
#endif

template<>
class imath_t<uint8_t, __n128>
{
public:
    typedef uint8_t int_t;
    typedef __n128 packed_t;

    static inline __n128 zeroes() { return vdupq_n_u8(0); }

    static inline __n128 onebits()
    {
        return
            vreinterpretq_u8_u32(
                vceqq_u32(
                    vreinterpretq_u32_u8(zeroes()),
                    vreinterpretq_u32_u8(zeroes())
                )
            );
    }

    static inline __n128 sign_mask() { return vshlq_n_u8(onebits(), 7); }

    static inline __n128 largest_val() { return onebits(); }
    static inline __n128 smallest_val() { return zeroes(); }

    template<unsigned Z> static inline packed_t mask_zupper() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        return vshrq_n_u8(onebits(), Z);
    }
    template<unsigned Z> static inline packed_t mask_zlower() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        return vshlq_n_u8(onebits(), Z);
    }
    template<unsigned B> static inline packed_t mask_1bit() {
        static_assert(B < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        const unsigned bits = sizeof(int_t)*8;
        return
            vaddq_u8(
                B > 0 ? vshrq_n_u8(onebits(), bits - B) : zeroes(),
                vshrq_n_u8(onebits(), 7) //(1)*16
            );
    }
};


#ifdef PMATHI_ARM_GCC
#undef __n128
#define __n128 int16x8_t
#endif

template<>
class imath_t<int16_t, __n128>
{
public:
    typedef int16_t int_t;
    typedef __n128 packed_t;

    static inline __n128 zeroes() { return vdupq_n_s16(0); }
    static inline __n128 onebits()
    {
        return
            vreinterpretq_s16_u32(
                vceqq_u32(
                    vreinterpretq_u32_s16(zeroes()),
                    vreinterpretq_u32_s16(zeroes())
                )
            );
    }

    static inline __n128 sign_mask()
    {
        return
            vreinterpretq_s16_u16(
                vshlq_n_u16(
                    vreinterpretq_u16_s16(onebits()),
                    15
                )
            );
    }

    static inline __n128 largest_val() { return veorq_s16(sign_mask(), onebits()); }
    static inline __n128 smallest_val() { return sign_mask(); }

    template<unsigned Z> static inline packed_t mask_zupper() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        return
            vreinterpretq_s16_u16(
                vshrq_n_u16(
                    vreinterpretq_u16_s16(onebits()),
                    Z
                )
            );
    }
    template<unsigned Z> static inline packed_t mask_zlower() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        return
            vreinterpretq_s16_u16(
                vshlq_n_u16(
                    vreinterpretq_u16_s16(onebits()),
                    Z
                )
            );
    }
    template<unsigned B> static inline packed_t mask_1bit() {
        static_assert(B < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        const unsigned bits = sizeof(int_t)*8;
        return
            vreinterpretq_s16_u16(
                vaddq_u16(
                    B > 0
                        ? vshrq_n_u16(vreinterpretq_u16_s16(onebits()), bits - B)
                        : vreinterpretq_u16_s16(zeroes()),
                    vshrq_n_u16(vreinterpretq_u16_s16(onebits()), 15) //(1)*8
                )
            );
    }
};


#ifdef PMATHI_ARM_GCC
#undef __n128
#define __n128 uint16x8_t
#endif

template<>
class imath_t<uint16_t, __n128>
{
public:
    typedef uint16_t int_t;
    typedef __n128 packed_t;

    static inline __n128 zeroes() { return vdupq_n_u16(0); }
    static inline __n128 onebits()
    {
        return
            vreinterpretq_u16_u32(
                vceqq_u32(
                    vreinterpretq_u32_u16(zeroes()),
                    vreinterpretq_u32_u16(zeroes())
                )
            );
    }
    static inline __n128 sign_mask() { return vshlq_n_u16(onebits(), 15); }

    static inline __n128 largest_val() { return onebits(); }
    static inline __n128 smallest_val() { return zeroes(); }

    template<unsigned Z> static inline packed_t mask_zupper() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        return vshrq_n_u16(onebits(), Z);
    }
    template<unsigned Z> static inline packed_t mask_zlower() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        return vshlq_n_u16(onebits(), Z);
    }
    template<unsigned B> static inline packed_t mask_1bit() {
        static_assert(B < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        const unsigned bits = sizeof(int_t)*8;
        return
            vaddq_u16(
                B > 0 ? vshrq_n_u16(onebits(), bits - B) : zeroes(),
                vshrq_n_u16(onebits(), 15) //(1)*8
            );
    }
};


#ifdef PMATHI_ARM_GCC
#undef __n128
#define __n128 int32x4_t
#endif

template<>
class imath_t<int32_t, __n128>
{
public:
    typedef int32_t int_t;
    typedef __n128 packed_t;

    static inline __n128 zeroes() { return vdupq_n_s32(0); }
    static inline __n128 onebits() { return vreinterpretq_s32_u32(vceqq_s32(zeroes(), zeroes())); }
    static inline __n128 sign_mask()
    {
        return
            vreinterpretq_s32_u32(
                vshlq_n_u32(
                    vreinterpretq_u32_s32(onebits()),
                    31
                )
            );
    }
    
    static inline __n128 largest_val() { return veorq_s32(sign_mask(), onebits()); }
    static inline __n128 smallest_val() { return sign_mask(); }

    template<unsigned Z> static inline packed_t mask_zupper() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        return
            vreinterpretq_s32_u32(
                vshrq_n_u32(
                    vreinterpretq_u32_s32(onebits()),
                    Z
                )
            );
    }
    template<unsigned Z> static inline packed_t mask_zlower() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        return
            vreinterpretq_s32_u32(
                vshlq_n_u32(
                    vreinterpretq_u32_s32(onebits()),
                    Z
                )
            );
    }
    template<unsigned B> static inline packed_t mask_1bit() {
        static_assert(B < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        const unsigned bits = sizeof(int_t)*8;
        return
            vreinterpretq_s32_u32(
                vaddq_u32(
                    B > 0
                        ? vshrq_n_u32(vreinterpretq_u32_s32(onebits()), bits - B)
                        : vreinterpretq_u32_s32(zeroes()),
                    vshrq_n_u32(vreinterpretq_u32_s32(onebits()), 31) //(1)*4
                )
            );
    }
};


#ifdef PMATHI_ARM_GCC
#undef __n128
#define __n128 uint32x4_t
#endif

template<>
class imath_t<uint32_t, __n128>
{
public:
    typedef uint32_t int_t;
    typedef __n128 packed_t;

    static inline __n128 zeroes() { return vdupq_n_u32(0); }
    static inline __n128 onebits() { return vceqq_u32(zeroes(), zeroes()); }
    static inline __n128 sign_mask() { return vshlq_n_u32(onebits(), 31); }

    static inline __n128 largest_val() { return onebits(); }
    static inline __n128 smallest_val() { return zeroes(); }

    template<unsigned Z> static inline packed_t mask_zupper() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        return vshrq_n_u32(onebits(), Z);
    }
    template<unsigned Z> static inline packed_t mask_zlower() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        return vshlq_n_u32(onebits(), Z);
    }
    template<unsigned B> static inline packed_t mask_1bit() {
        static_assert(B < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        const unsigned bits = sizeof(int_t)*8;
        return
            vaddq_u32(
                B > 0 ? vshrq_n_u32(onebits(), bits - B) : zeroes(),
                vshrq_n_u32(onebits(), 31) //(1)*4
            );
    }
};


#ifdef PMATHI_ARM_GCC
#undef __n128
#define __n128 int64x2_t
#endif

template<>
class imath_t<int64_t, __n128>
{
public:
    typedef int64_t int_t;
    typedef __n128 packed_t;

    static inline __n128 zeroes() { return vdupq_n_s64(0); }
    static inline __n128 onebits()
    {
        return
            vreinterpretq_s64_u32(
                vceqq_u32(
                    vreinterpretq_u32_s64(zeroes()),
                    vreinterpretq_u32_s64(zeroes())
                )
            );
    }
    static inline __n128 sign_mask()
    {
        return
            vreinterpretq_s64_u64(
                vshlq_n_u64(
                    vreinterpretq_u64_s64(onebits()),
                    63
                )
            );
    }

    static inline __n128 largest_val() { return veorq_s64(sign_mask(), onebits()); }
    static inline __n128 smallest_val() { return sign_mask(); }

    template<unsigned Z> static inline packed_t mask_zupper() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        return
            vreinterpretq_s64_u64(
                vshrq_n_u64(
                    vreinterpretq_u64_s64(onebits()),
                    Z
                )
            );
    }
    template<unsigned Z> static inline packed_t mask_zlower() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        return
            vreinterpretq_s64_u64(
                vshlq_n_u64(
                    vreinterpretq_u64_s64(onebits()),
                    Z
                )
            );
    }
    template<unsigned B> static inline packed_t mask_1bit() {
        static_assert(B < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        const unsigned bits = sizeof(int_t)*8;
        return
            vreinterpretq_s64_u64(
                vaddq_u64(
                    B > 0
                        ? vshrq_n_u64(vreinterpretq_u64_s64(onebits()), bits - B)
                        : vreinterpretq_u64_s64(zeroes()),
                    vshrq_n_u64(vreinterpretq_u64_s64(onebits()), 63) //(1)*2
                )
            );
    }
};


#ifdef PMATHI_ARM_GCC
#undef __n128
#define __n128 uint64x2_t
#endif

template<>
class imath_t<uint64_t, __n128>
{
public:
    typedef uint64_t int_t;
    typedef __n128 packed_t;

    static inline __n128 zeroes() { return vdupq_n_u64(0); }
    static inline __n128 onebits()
    {
        return
            vreinterpretq_u64_u32(
                vceqq_u32(
                    vreinterpretq_u32_u64(zeroes()),
                    vreinterpretq_u32_u64(zeroes())
                )
            );
    }
    static inline __n128 sign_mask() { return vshlq_n_u64(onebits(), 63); }

    static inline __n128 largest_val() { return onebits(); }
    static inline __n128 smallest_val() { return zeroes(); }

    template<unsigned Z> static inline packed_t mask_zupper() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        return vshrq_n_u64(onebits(), Z);
    }
    template<unsigned Z> static inline packed_t mask_zlower() {
        static_assert(Z > 0, "Z must be > 0, use onebits() for a value with all bits set");
        static_assert(Z < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        return vshlq_n_u64(onebits(), Z);
    }
    template<unsigned B> static inline packed_t mask_1bit() {
        static_assert(B < sizeof(int_t)*8, "Z must be less than the number of bits per element");
        const unsigned bits = sizeof(int_t)*8;
        return
            vaddq_u64(
                B > 0 ? vshrq_n_u64(onebits(), bits - B) : zeroes(),
                vshrq_n_u64(onebits(), 63) //(1)*2
            );
    }

};


#endif // defined(PMATHI_ARM)

} // namespace math


#endif // !defined(PMATHI_H)

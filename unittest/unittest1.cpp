#include "stdafx.h"

#include <memory>


//#define ALGO_PERFORMANCE_TESTS


//#ifndef ANDROID
#if 1
// only in one compilation unit, tests can be split over
// multiple CUs by just including the Catch header
#define CATCH_CONFIG_MAIN
#include "3rdparty/catch.hpp"
#else
#include <android/log.h>
class Assert
{
public:
#define Q2(x) #x
#define Q(x) Q2(x)
#define IsTrue(cond) IsTrue_((cond), __LINE__, Q(cond))
    static void IsTrue_(bool b, int line, const char * expr)
    {
        if(!b) {
            __android_log_print(
                ANDROID_LOG_INFO,
                "pvecmath"/*tag*/,
                "Assertion failed: line %d, expression: %s",
                line,
                expr
            );
        }
    }
};
#endif

#include <pvecf.h>
#include <pveci.h>


// no std::make_unique<> in gnustl lib on android
// (this comment might be not up-to-date anymore)
#ifdef ANDROID
//
// make_unique()
//
namespace impl {

    // non-arrays
    template<typename T, typename ...Args>
    //inline
    typename
        std::enable_if<
            !std::is_array<T>::value,
            std::unique_ptr<T>
        >::type
        make_unique(Args && ...args)
    {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }

    // arrays with unknown size
    template<typename T>
    //inline
    typename
        std::enable_if<
            std::is_array<T>::value && std::extent<T>::value == 0,
            std::unique_ptr<T>
        >::type
        make_unique(std::size_t size)
    {
        using elem_t = typename std::remove_extent<T>::type;
        //return (unique_ptr<_Ty>(new _Elem[_Size]()));
        return std::unique_ptr<T>(new elem_t[size]());
    }

    // arrays with known size (not allowed)
    template<typename T, typename ...Args>
    //inline
    typename
        std::enable_if<
            std::is_array<T>::value && std::extent<T>::value != 0
        >::type
        make_unique(Args && ...args) = delete;

} // namespace impl

using impl::make_unique;

#else // >= C++14

using std::make_unique;

#endif



TEST_CASE("Mat4f identity inverse == identity", "TestMat4fIdentityInverse")
{
    using namespace math;
    mat4f_t mat;
    mat.set_identity();

    // TODO: prob with NEON: NaNs are generated

    mat.inverse();
    
    for(int i = 0; i < 16; ++i) {
        int pos = (i >> 2) & 0x3;
        REQUIRE(mat.v[i] == ((i & 0x3) == ((i >> 2) & 0x3) ? 1.0f : 0.0f));
    }
}

TEST_CASE("Math4f fast trig", "TestMathV4fFastTrig")
{
    using vec4f_t = math::vec4f_t;
    using math4f_t = vec4f_t::math_t;

    vec4f_t angles(
         0.0f*math::deg2rad<float>(), 30.0f*math::deg2rad<float>(),
        60.0f*math::deg2rad<float>(), 90.0f*math::deg2rad<float>()
    );
    vec4f_t vsines(math4f_t::fast_sin_0(angles));
    
    REQUIRE(math::almost_equal(vsines[0], 0.0f, 0.0001f));
    REQUIRE(math::almost_equal(vsines[1], 0.5f, 0.0001f));
    REQUIRE(math::almost_equal(vsines[2], 0.8660254f, 0.0001f));
    REQUIRE(math::almost_equal(vsines[3], 1.0f, 0.0001f));

    vec4f_t vcosines(math4f_t::fast_cos_0(angles));

    REQUIRE(math::almost_equal(vcosines[0], 1.0f, 0.001f));
    REQUIRE(math::almost_equal(vcosines[1], 0.8660254f, 0.001f));
    REQUIRE(math::almost_equal(vcosines[2], 0.5f, 0.001f));
    REQUIRE(math::almost_equal(vcosines[3], 0.0f, 0.001f));
}

TEST_CASE("Vec4f initialization", "TestVec4fInit")
{
    using vec4f_t = math::vec4f_t;

    vec4f_t v1;

    for(int i = 0; i < 4; ++i)
        REQUIRE(math::almost_equal(v1[i], 0.0f));

    vec4f_t v2(1.0f, 2.0f, 3.0f, 4.0f);
    REQUIRE(math::almost_equal(v2[0], 1.0f));
    REQUIRE(math::almost_equal(v2[1], 2.0f));
    REQUIRE(math::almost_equal(v2[2], 3.0f));
    REQUIRE(math::almost_equal(v2[3], 4.0f));

    vec4f_t v3(1.0f);
    REQUIRE(math::almost_equal(v3[0], 1.0f));
    for(int i = 1; i < 4; ++i)
        REQUIRE(math::almost_equal(v3[i], 0.0f));

}

TEST_CASE("Vec4f free-standing operations", "TestVec4fFreeStandingOps")
{
    using vec4f_t = math::vec4f_t;
    //using namespace math;

    vec4f_t v2(1.0f, 2.0f, 3.0f, 4.0f);
    vec4f_t v4;
    v4 = v2 + v2;
    REQUIRE(math::almost_equal(v4[0], 2.0f));
    REQUIRE(math::almost_equal(v4[1], 4.0f));
    REQUIRE(math::almost_equal(v4[2], 6.0f));
    REQUIRE(math::almost_equal(v4[3], 8.0f));
    v4 = add(v2, v2);
    REQUIRE(math::almost_equal(v4[0], 2.0f));
    REQUIRE(math::almost_equal(v4[1], 4.0f));
    REQUIRE(math::almost_equal(v4[2], 6.0f));
    REQUIRE(math::almost_equal(v4[3], 8.0f));
    v4 = add(v2, v2.p);
    REQUIRE(math::almost_equal(v4[0], 2.0f));
    REQUIRE(math::almost_equal(v4[1], 4.0f));
    REQUIRE(math::almost_equal(v4[2], 6.0f));
    REQUIRE(math::almost_equal(v4[3], 8.0f));
    v4 = add(v2.p, v2);
    REQUIRE(math::almost_equal(v4[0], 2.0f));
    REQUIRE(math::almost_equal(v4[1], 4.0f));
    REQUIRE(math::almost_equal(v4[2], 6.0f));
    REQUIRE(math::almost_equal(v4[3], 8.0f));


    v4 = v2 - v2;
    for(int i = 0; i < 4; ++i)
        REQUIRE(math::almost_equal(v4[i], 0.0f));
    v4 = sub(v2, v2);
    for(int i = 0; i < 4; ++i)
        REQUIRE(math::almost_equal(v4[i], 0.0f));
    v4 = sub(v2, v2.p);
    for(int i = 0; i < 4; ++i)
        REQUIRE(math::almost_equal(v4[i], 0.0f));
    v4 = sub(v2.p, v2);
    for(int i = 0; i < 4; ++i)
        REQUIRE(math::almost_equal(v4[i], 0.0f));


    v4 = v2 * v2;
    REQUIRE(math::almost_equal(v4[0], 1.0f));
    REQUIRE(math::almost_equal(v4[1], 4.0f));
    REQUIRE(math::almost_equal(v4[2], 9.0f));
    REQUIRE(math::almost_equal(v4[3], 16.0f));
    v4 = mul(v2, v2);
    REQUIRE(math::almost_equal(v4[0], 1.0f));
    REQUIRE(math::almost_equal(v4[1], 4.0f));
    REQUIRE(math::almost_equal(v4[2], 9.0f));
    REQUIRE(math::almost_equal(v4[3], 16.0f));
    v4 = mul(v2, v2.p);
    REQUIRE(math::almost_equal(v4[0], 1.0f));
    REQUIRE(math::almost_equal(v4[1], 4.0f));
    REQUIRE(math::almost_equal(v4[2], 9.0f));
    REQUIRE(math::almost_equal(v4[3], 16.0f));
    v4 = mul(v2.p, v2);
    REQUIRE(math::almost_equal(v4[0], 1.0f));
    REQUIRE(math::almost_equal(v4[1], 4.0f));
    REQUIRE(math::almost_equal(v4[2], 9.0f));
    REQUIRE(math::almost_equal(v4[3], 16.0f));


    v4 = v2 / v2;
    for(int i = 0; i < 4; ++i)
        REQUIRE(math::almost_equal(v4[i], 1.0f));
    v4 = div(v2, v2);
    for(int i = 0; i < 4; ++i)
        REQUIRE(math::almost_equal(v4[i], 1.0f));
    v4 = div(v2, v2.p);
    for(int i = 0; i < 4; ++i)
        REQUIRE(math::almost_equal(v4[i], 1.0f));
    v4 = div(v2.p, v2);
    for(int i = 0; i < 4; ++i)
        REQUIRE(math::almost_equal(v4[i], 1.0f));


    math::vec4f_t v_negzeroes(-0.0f, -0.0f, -0.0f, -0.0f);
    v4[0] = 1.0f; v4[1] = -1.0f; v4[2] = 2.0f; v4[3] = -2.0f;

    v4 = v4 & v_negzeroes;
    REQUIRE(math::almost_equal(v4[0], 0.0f));
    REQUIRE(math::almost_equal(v4[1], -0.0f));
    REQUIRE(math::almost_equal(v4[2], 0.0f));
    REQUIRE(math::almost_equal(v4[3], -0.0f));
    v4[0] = 1.0f; v4[1] = -1.0f; v4[2] = 2.0f; v4[3] = -2.0f;
    v4 = and_(v4, v_negzeroes);
    REQUIRE(math::almost_equal(v4[0], 0.0f));
    REQUIRE(math::almost_equal(v4[1], -0.0f));
    REQUIRE(math::almost_equal(v4[2], 0.0f));
    REQUIRE(math::almost_equal(v4[3], -0.0f));
    v4[0] = 1.0f; v4[1] = -1.0f; v4[2] = 2.0f; v4[3] = -2.0f;
    v4 = and_(v4, v_negzeroes.p);
    REQUIRE(math::almost_equal(v4[0], 0.0f));
    REQUIRE(math::almost_equal(v4[1], -0.0f));
    REQUIRE(math::almost_equal(v4[2], 0.0f));
    REQUIRE(math::almost_equal(v4[3], -0.0f));
    v4[0] = 1.0f; v4[1] = -1.0f; v4[2] = 2.0f; v4[3] = -2.0f;
    v4 = and_(v4.p, v_negzeroes);
    REQUIRE(math::almost_equal(v4[0], 0.0f));
    REQUIRE(math::almost_equal(v4[1], -0.0f));
    REQUIRE(math::almost_equal(v4[2], 0.0f));
    REQUIRE(math::almost_equal(v4[3], -0.0f));
    v4[0] = 1.0f; v4[1] = -1.0f; v4[2] = 2.0f; v4[3] = -2.0f;
    v4 = and_(v4, -0.0f);
    REQUIRE(math::almost_equal(v4[0], 0.0f));
    REQUIRE(math::almost_equal(v4[1], -0.0f));
    REQUIRE(math::almost_equal(v4[2], 0.0f));
    REQUIRE(math::almost_equal(v4[3], -0.0f));
    v4[0] = 1.0f; v4[1] = -1.0f; v4[2] = 2.0f; v4[3] = -2.0f;
    v4 = and_(-0.0f, v4);
    REQUIRE(math::almost_equal(v4[0], 0.0f));
    REQUIRE(math::almost_equal(v4[1], -0.0f));
    REQUIRE(math::almost_equal(v4[2], 0.0f));
    REQUIRE(math::almost_equal(v4[3], -0.0f));


    v4[0] = 1.0f; v4[1] = 2.0f; v4[2] = 3.0f; v4[3] = 4.0f;
    v4 = v4 | v_negzeroes;
    REQUIRE(math::almost_equal(v4[0], -1.0f));
    REQUIRE(math::almost_equal(v4[1], -2.0f));
    REQUIRE(math::almost_equal(v4[2], -3.0f));
    REQUIRE(math::almost_equal(v4[3], -4.0f));
    v4[0] = 1.0f; v4[1] = 2.0f; v4[2] = 3.0f; v4[3] = 4.0f;
    v4 = or_(v4, v_negzeroes);
    REQUIRE(math::almost_equal(v4[0], -1.0f));
    REQUIRE(math::almost_equal(v4[1], -2.0f));
    REQUIRE(math::almost_equal(v4[2], -3.0f));
    REQUIRE(math::almost_equal(v4[3], -4.0f));
    v4[0] = 1.0f; v4[1] = 2.0f; v4[2] = 3.0f; v4[3] = 4.0f;
    v4 = or_(v4, v_negzeroes.p);
    REQUIRE(math::almost_equal(v4[0], -1.0f));
    REQUIRE(math::almost_equal(v4[1], -2.0f));
    REQUIRE(math::almost_equal(v4[2], -3.0f));
    REQUIRE(math::almost_equal(v4[3], -4.0f));
    v4[0] = 1.0f; v4[1] = 2.0f; v4[2] = 3.0f; v4[3] = 4.0f;
    v4 = or_(v4.p, v_negzeroes);
    REQUIRE(math::almost_equal(v4[0], -1.0f));
    REQUIRE(math::almost_equal(v4[1], -2.0f));
    REQUIRE(math::almost_equal(v4[2], -3.0f));
    REQUIRE(math::almost_equal(v4[3], -4.0f));
    v4[0] = 1.0f; v4[1] = 2.0f; v4[2] = 3.0f; v4[3] = 4.0f;
    v4 = or_(v4, -0.0f);
    REQUIRE(math::almost_equal(v4[0], -1.0f));
    REQUIRE(math::almost_equal(v4[1], -2.0f));
    REQUIRE(math::almost_equal(v4[2], -3.0f));
    REQUIRE(math::almost_equal(v4[3], -4.0f));
    v4[0] = 1.0f; v4[1] = 2.0f; v4[2] = 3.0f; v4[3] = 4.0f;
    v4 = or_(-0.0f, v4);
    REQUIRE(math::almost_equal(v4[0], -1.0f));
    REQUIRE(math::almost_equal(v4[1], -2.0f));
    REQUIRE(math::almost_equal(v4[2], -3.0f));
    REQUIRE(math::almost_equal(v4[3], -4.0f));

    v4[0] = 1.0f; v4[1] = 2.0f; v4[2] = -1.0f; v4[3] = -2.0f;
    v4 = v4 ^ v_negzeroes;
    REQUIRE(math::almost_equal(v4[0], -1.0f));
    REQUIRE(math::almost_equal(v4[1], -2.0f));
    REQUIRE(math::almost_equal(v4[2], 1.0f));
    REQUIRE(math::almost_equal(v4[3], 2.0f));
    v4[0] = 1.0f; v4[1] = 2.0f; v4[2] = -1.0f; v4[3] = -2.0f;
    v4 = xor_(v4, v_negzeroes);
    REQUIRE(math::almost_equal(v4[0], -1.0f));
    REQUIRE(math::almost_equal(v4[1], -2.0f));
    REQUIRE(math::almost_equal(v4[2], 1.0f));
    REQUIRE(math::almost_equal(v4[3], 2.0f));
    v4[0] = 1.0f; v4[1] = 2.0f; v4[2] = -1.0f; v4[3] = -2.0f;
    v4 = xor_(v4, v_negzeroes.p);
    REQUIRE(math::almost_equal(v4[0], -1.0f));
    REQUIRE(math::almost_equal(v4[1], -2.0f));
    REQUIRE(math::almost_equal(v4[2], 1.0f));
    REQUIRE(math::almost_equal(v4[3], 2.0f));
    v4[0] = 1.0f; v4[1] = 2.0f; v4[2] = -1.0f; v4[3] = -2.0f;
    v4 = xor_(v4.p, v_negzeroes);
    REQUIRE(math::almost_equal(v4[0], -1.0f));
    REQUIRE(math::almost_equal(v4[1], -2.0f));
    REQUIRE(math::almost_equal(v4[2], 1.0f));
    REQUIRE(math::almost_equal(v4[3], 2.0f));
    v4[0] = 1.0f; v4[1] = 2.0f; v4[2] = -1.0f; v4[3] = -2.0f;
    v4 = xor_(v4, -0.0f);
    REQUIRE(math::almost_equal(v4[0], -1.0f));
    REQUIRE(math::almost_equal(v4[1], -2.0f));
    REQUIRE(math::almost_equal(v4[2], 1.0f));
    REQUIRE(math::almost_equal(v4[3], 2.0f));
    v4[0] = 1.0f; v4[1] = 2.0f; v4[2] = -1.0f; v4[3] = -2.0f;
    v4 = xor_(-0.0f, v4);
    REQUIRE(math::almost_equal(v4[0], -1.0f));
    REQUIRE(math::almost_equal(v4[1], -2.0f));
    REQUIRE(math::almost_equal(v4[2], 1.0f));
    REQUIRE(math::almost_equal(v4[3], 2.0f));


    v4[0] = 1.0f; v4[1] = 2.0f; v4[2] = 3.0f; v4[3] = 4.0f;
    v4 += 2.0f;
    REQUIRE(math::almost_equal(v4[0], 3.0f));
    REQUIRE(math::almost_equal(v4[1], 4.0f));
    REQUIRE(math::almost_equal(v4[2], 5.0f));
    REQUIRE(math::almost_equal(v4[3], 6.0f));

    v4[0] = 2.0f; v4[1] = 3.0f; v4[2] = 4.0f; v4[3] = 5.0f;
    v4 -= 2.0f;
    REQUIRE(math::almost_equal(v4[0], 0.0f));
    REQUIRE(math::almost_equal(v4[1], 1.0f));
    REQUIRE(math::almost_equal(v4[2], 2.0f));
    REQUIRE(math::almost_equal(v4[3], 3.0f));

    v4[0] = 1.0f; v4[1] = 2.0f; v4[2] = 3.0f; v4[3] = 4.0f;
    v4 *= 2.0f;
    REQUIRE(math::almost_equal(v4[0], 2.0f));
    REQUIRE(math::almost_equal(v4[1], 4.0f));
    REQUIRE(math::almost_equal(v4[2], 6.0f));
    REQUIRE(math::almost_equal(v4[3], 8.0f));

    v4[0] = 1.0f; v4[1] = 2.0f; v4[2] = 4.0f; v4[3] = 8.0f;
    v4 /= 2.0f;
    REQUIRE(math::almost_equal(v4[0], 0.5f));
    REQUIRE(math::almost_equal(v4[1], 1.0f));
    REQUIRE(math::almost_equal(v4[2], 2.0f));
    REQUIRE(math::almost_equal(v4[3], 4.0f));


    v4[0] = 1.0f; v4[1] = 2.0f; v4[2] = 3.0f; v4[3] = 4.0f;
    math::vec4f_t v5;
    v5 = add(v4, 2.0f);
    REQUIRE(math::almost_equal(v5[0], 3.0f));
    REQUIRE(math::almost_equal(v5[1], 4.0f));
    REQUIRE(math::almost_equal(v5[2], 5.0f));
    REQUIRE(math::almost_equal(v5[3], 6.0f));
    v5 = add(2.0f, v4);
    REQUIRE(math::almost_equal(v5[0], 3.0f));
    REQUIRE(math::almost_equal(v5[1], 4.0f));
    REQUIRE(math::almost_equal(v5[2], 5.0f));
    REQUIRE(math::almost_equal(v5[3], 6.0f));

    v4[0] = 2.0f; v4[1] = 3.0f; v4[2] = 4.0f; v4[3] = 5.0f;
    v5 = sub(v4, 2.0f);
    REQUIRE(math::almost_equal(v5[0], 0.0f));
    REQUIRE(math::almost_equal(v5[1], 1.0f));
    REQUIRE(math::almost_equal(v5[2], 2.0f));
    REQUIRE(math::almost_equal(v5[3], 3.0f));
    v5 = sub(2.0f, v4);
    REQUIRE(math::almost_equal(v5[0], 0.0f));
    REQUIRE(math::almost_equal(v5[1], -1.0f));
    REQUIRE(math::almost_equal(v5[2], -2.0f));
    REQUIRE(math::almost_equal(v5[3], -3.0f));

    v4[0] = 1.0f; v4[1] = 2.0f; v4[2] = 3.0f; v4[3] = 4.0f;
    v5 = mul(v4, 2.0f);
    REQUIRE(math::almost_equal(v5[0], 2.0f));
    REQUIRE(math::almost_equal(v5[1], 4.0f));
    REQUIRE(math::almost_equal(v5[2], 6.0f));
    REQUIRE(math::almost_equal(v5[3], 8.0f));
    v5 = mul(2.0f, v4);
    REQUIRE(math::almost_equal(v5[0], 2.0f));
    REQUIRE(math::almost_equal(v5[1], 4.0f));
    REQUIRE(math::almost_equal(v5[2], 6.0f));
    REQUIRE(math::almost_equal(v5[3], 8.0f));

    v4[0] = 1.0f; v4[1] = 2.0f; v4[2] = 4.0f; v4[3] = 8.0f;
    v5 = div(v4, 2.0f);
    REQUIRE(math::almost_equal(v5[0], 0.5f));
    REQUIRE(math::almost_equal(v5[1], 1.0f));
    REQUIRE(math::almost_equal(v5[2], 2.0f));
    REQUIRE(math::almost_equal(v5[3], 4.0f));
    v5 = div(8.0f, v4);
    REQUIRE(math::almost_equal(v5[0], 8.0f));
    REQUIRE(math::almost_equal(v5[1], 4.0f));
    REQUIRE(math::almost_equal(v5[2], 2.0f));
    REQUIRE(math::almost_equal(v5[3], 1.0f));

    v4[0] = 1.0f; v4[1] = 2.0f; v4[2] = 4.0f; v4[3] = 8.0f;
    v5 = v4;
    v5 = sub(mul(v4, v4), v4);

    REQUIRE(math::almost_equal(v5[0],  0.0f));
    REQUIRE(math::almost_equal(v5[1],  2.0f));
    REQUIRE(math::almost_equal(v5[2], 12.0f));
    REQUIRE(math::almost_equal(v5[3], 56.0f));

    v4[0] = 1.0f; v4[1] = 2.0f; v4[2] = 4.0f; v4[3] = 8.0f;
    v5 = v4;
    v5 = mul(add(v4, v4), v4);

    REQUIRE(math::almost_equal(v5[0],   2.0f));
    REQUIRE(math::almost_equal(v5[1],   8.0f));
    REQUIRE(math::almost_equal(v5[2],  32.0f));
    REQUIRE(math::almost_equal(v5[3], 128.0f));


    // test operator~ and not() (identical ops)

    vec4f_t v6;
    v6[0] = 1.0f; v6[1] = -1.0f; v6[2] = 2.0f; v6[3] = -2.0f;

    union uf32u32 {
        float f;
        uint32_t u32;
    };

    uf32u32 u0[4], u1[4];

    for(int i = 0; i < 4; ++i) {
        u0[i].f = v6[i];
        u0[i].u32 = ~u0[i].u32;
    }

    vec4f_t v7 = v6;
    v7 = ~v7; // operation under test

    for(int i = 0; i < 4; ++i) {
        u1[i].f = v7[i];
        REQUIRE(u1[i].u32 == u0[i].u32);
    }

    // test of not() (identical to operator~())

    v7 = v6;
    v7 = not(v7);

    for(int i = 0; i < 4; ++i) {
        u1[i].f = v7[i];
        REQUIRE(u1[i].u32 == u0[i].u32);
    }
}


TEST_CASE("Vec4f member operations", "TestVec4fMemberOps")
{
    using vec4f_t = math::vec4f_t;

    vec4f_t v4(1.0f, 2.0f, 3.0f, 4.0f);
    vec4f_t v1(v4);
    float fval = v4.dot(v1);
    REQUIRE(math::almost_equal(fval, 30.0f));

    fval = v4.sqlen();
    REQUIRE(math::almost_equal(fval, 30.0f));

    fval = v4.len();
    REQUIRE(math::almost_equal(fval, sqrtf(30.0f), 0.001f));

    v4[0] = 1.0f; v4[1] = v4[2] = v4[3] = 0.0f;
    v4.normalize();
    REQUIRE(math::almost_equal(v4[0], 1.0f));
    for(int i = 1; i < 4; ++i)
        REQUIRE(math::almost_equal(v4[i], 0.0f));

    v4[0] = -1.0f; v4[1] = 0.25f; v4[2] = 0.75f; v4[3] = 2.0f;
    v4.clamp_0_1();
    REQUIRE(math::almost_equal(v4[0], 0.00f));
    REQUIRE(math::almost_equal(v4[1], 0.25f));
    REQUIRE(math::almost_equal(v4[2], 0.75f));
    REQUIRE(math::almost_equal(v4[3], 1.00f));

    v1[0] = 1.0f; v1[1] = v1[2] = v1[3] = 0.0f;
    v4[1] = 1.0f; v4[0] = v4[2] = v4[3] = 0.0f;
    v4 = v1.cross(v4);
    REQUIRE(math::almost_equal(v4[0], 0.0f));
    REQUIRE(math::almost_equal(v4[1], 0.0f));
    REQUIRE(math::almost_equal(v4[2], 1.0f));
    REQUIRE(math::almost_equal(v4[3], 0.0f));

    v1[0] = 2.0f; v1[1] = v1[2] = v1[3] = 0.0f;
    v4[1] = 2.0f; v4[0] = v4[2] = v4[3] = 0.0f;
    v4 = v1.unit_cross(v4);
    REQUIRE(math::almost_equal(v4[0], 0.0f));
    REQUIRE(math::almost_equal(v4[1], 0.0f));
    REQUIRE(math::almost_equal(v4[2], 1.0f));
    REQUIRE(math::almost_equal(v4[3], 0.0f));

    v1[0] = 2.0f; v1[1] = 1.0f;  v1[2] = 0.0f; v1[3] = -1.0f;
    v4[0] = 3.0f; v4[1] = 0.0f;  v4[2] = 1.0f; v4[3] = -2.0f;
    v4 = vec4f_t::min_(v1, v4);
    REQUIRE(math::almost_equal(v4[0], 2.0f));
    REQUIRE(math::almost_equal(v4[1], 0.0f));
    REQUIRE(math::almost_equal(v4[2], 0.0f));
    REQUIRE(math::almost_equal(v4[3], -2.0f));

    v4[0] = 1.0f; v4[1] = 2.0f;  v4[2] = -1.0f; v4[3] = 0.0f;
    v4 = vec4f_t::max_(v1, v4);
    REQUIRE(math::almost_equal(v4[0], 2.0f));
    REQUIRE(math::almost_equal(v4[1], 2.0f));
    REQUIRE(math::almost_equal(v4[2], 0.0f));
    REQUIRE(math::almost_equal(v4[3], 0.0f));
}

#ifndef PVECF_ARM

TEST_CASE("Vec4f comparison operations", "TestVec4fCmpOps")
{
    using vec4f_t = math::vec4f_t;

    vec4f_t v1(1.0f, 2.0f, 3.0f, 4.0f);
    vec4f_t v2(1.0f, 0.0f, 3.0f, 0.0f);

    REQUIRE((v1 == v2) == false);
    v2[1] = 2.0f; v2[3] = 4.0f;
    REQUIRE((v1 == v2) == true);
    v2 += math::vec4f_t::math_t::ones();
    REQUIRE((v1  < v2) == true);
    REQUIRE((v1 <= v2) == true);
    REQUIRE((v1 != v2) == true);
    REQUIRE((v2 != v1) == true);
    REQUIRE((v2  < v1) == false);
    REQUIRE((v2 <= v1) == false);
    v2 = v1 - math::vec4f_t::math_t::ones();
    REQUIRE((v1 == v2) == false);
    REQUIRE((v1  > v2) == true);
    REQUIRE((v1 >= v2) == true);
    REQUIRE((v1 != v2) == true);
    REQUIRE((v2  < v1) == true);
    REQUIRE((v2 <= v1) == true);
    REQUIRE((v2 != v1) == true);
}

TEST_CASE("Vec4f free-standing comparison operations", "TestVec4fFrStCmpOps")
{
    using vec4f_t = math::vec4f_t;

    vec4f_t v1(1.0f, 2.0f, 3.0f, 4.0f);
    vec4f_t v2(v1);

    vec4f_t vres;
    vres = cmp_eq(v1, v2);
    for(int i = 0; i < 4; ++i)
        REQUIRE(*(const uint32_t *)(&vres[i]) == 0xFFFFFFFF);

    vres = cmp_lte(v1, v2);
    for(int i = 0; i < 4; ++i)
        REQUIRE(*(const uint32_t *) (&vres[i]) == 0xFFFFFFFF);

    vres = cmp_gt(v1, v2);
    for(int i = 0; i < 4; ++i)
        REQUIRE(*(const uint32_t *) (&vres[i]) == 0);

    vres = cmp_lt(v1, v2);
    for(int i = 0; i < 4; ++i)
        REQUIRE(*(const uint32_t *) (&vres[i]) == 0);

    vres = cmp_neq(v1, v2);
    for(int i = 0; i < 4; ++i)
        REQUIRE(*(const uint32_t *) (&vres[i]) == 0);

    v2[0] = 0.0f;
    v2[2] = 0.0f;
    vres = cmp_eq(v1, v2);
    REQUIRE(*(const uint32_t *) (&vres[0]) == 0);
    REQUIRE(*(const uint32_t *) (&vres[1]) == 0xFFFFFFFF);
    REQUIRE(*(const uint32_t *) (&vres[2]) == 0);
    REQUIRE(*(const uint32_t *) (&vres[3]) == 0xFFFFFFFF);

    vres = cmp_lte(v1, v2);
    REQUIRE(*(const uint32_t *) (&vres[0]) == 0);
    REQUIRE(*(const uint32_t *) (&vres[1]) == 0xFFFFFFFF);
    REQUIRE(*(const uint32_t *) (&vres[2]) == 0);
    REQUIRE(*(const uint32_t *) (&vres[3]) == 0xFFFFFFFF);

    vres = cmp_lt(v1, v2);
    for(int i = 0; i < 4; ++i)
        REQUIRE(*(const uint32_t *) (&vres[i]) == 0);


    v1[0] = 1.0f; v1[1] = 2.0f; v1[2] = 3.0f; v1[3] = 4.0f;
    v2 = v1;
    vec4f_t masks, vals1(10.0f, 20.0f, 30.0f, 40.0f), vals2(50.0f, 60.0f, 70.0f, 80.0f);
    masks = cmp_eq(v1, v2);
    vres = select(masks, vals1, vals2);
    REQUIRE(math::almost_equal(vres[0], 10.0f));
    REQUIRE(math::almost_equal(vres[1], 20.0f));
    REQUIRE(math::almost_equal(vres[2], 30.0f));
    REQUIRE(math::almost_equal(vres[3], 40.0f));
    masks = cmp_neq(v1, v2);
    vres = select(masks, vals1, vals2);
    REQUIRE(math::almost_equal(vres[0], 50.0f));
    REQUIRE(math::almost_equal(vres[1], 60.0f));
    REQUIRE(math::almost_equal(vres[2], 70.0f));
    REQUIRE(math::almost_equal(vres[3], 80.0f));
    masks = cmp_lt(v1, v2);
    vres = select(masks, vals1, vals2);
    REQUIRE(math::almost_equal(vres[0], 50.0f));
    REQUIRE(math::almost_equal(vres[1], 60.0f));
    REQUIRE(math::almost_equal(vres[2], 70.0f));
    REQUIRE(math::almost_equal(vres[3], 80.0f));
    masks = cmp_gt(v1, v2);
    vres = select(masks, vals1, vals2);
    REQUIRE(math::almost_equal(vres[0], 50.0f));
    REQUIRE(math::almost_equal(vres[1], 60.0f));
    REQUIRE(math::almost_equal(vres[2], 70.0f));
    REQUIRE(math::almost_equal(vres[3], 80.0f));
    masks = cmp_lte(v1, v2);
    vres = select(masks, vals1, vals2);
    REQUIRE(math::almost_equal(vres[0], 10.0f));
    REQUIRE(math::almost_equal(vres[1], 20.0f));
    REQUIRE(math::almost_equal(vres[2], 30.0f));
    REQUIRE(math::almost_equal(vres[3], 40.0f));
    masks = cmp_gte(v1, v2);
    vres = select(masks, vals1, vals2);
    REQUIRE(math::almost_equal(vres[0], 10.0f));
    REQUIRE(math::almost_equal(vres[1], 20.0f));
    REQUIRE(math::almost_equal(vres[2], 30.0f));
    REQUIRE(math::almost_equal(vres[3], 40.0f));


    vres = select_eq(v1, v2, vals1, vals2);
    REQUIRE(math::almost_equal(vres[0], 10.0f));
    REQUIRE(math::almost_equal(vres[1], 20.0f));
    REQUIRE(math::almost_equal(vres[2], 30.0f));
    REQUIRE(math::almost_equal(vres[3], 40.0f));
    vres = select_neq(v1, v2, vals1, vals2);
    REQUIRE(math::almost_equal(vres[0], 50.0f));
    REQUIRE(math::almost_equal(vres[1], 60.0f));
    REQUIRE(math::almost_equal(vres[2], 70.0f));
    REQUIRE(math::almost_equal(vres[3], 80.0f));
    vres = select_lt(v1, v2, vals1, vals2);
    REQUIRE(math::almost_equal(vres[0], 50.0f));
    REQUIRE(math::almost_equal(vres[1], 60.0f));
    REQUIRE(math::almost_equal(vres[2], 70.0f));
    REQUIRE(math::almost_equal(vres[3], 80.0f));
    vres = select_gt(v1, v2, vals1, vals2);
    REQUIRE(math::almost_equal(vres[0], 50.0f));
    REQUIRE(math::almost_equal(vres[1], 60.0f));
    REQUIRE(math::almost_equal(vres[2], 70.0f));
    REQUIRE(math::almost_equal(vres[3], 80.0f));
    vres = select_lte(v1, v2, vals1, vals2);
    REQUIRE(math::almost_equal(vres[0], 10.0f));
    REQUIRE(math::almost_equal(vres[1], 20.0f));
    REQUIRE(math::almost_equal(vres[2], 30.0f));
    REQUIRE(math::almost_equal(vres[3], 40.0f));
    vres = select_gte(v1, v2, vals1, vals2);
    REQUIRE(math::almost_equal(vres[0], 10.0f));
    REQUIRE(math::almost_equal(vres[1], 20.0f));
    REQUIRE(math::almost_equal(vres[2], 30.0f));
    REQUIRE(math::almost_equal(vres[3], 40.0f));


    v1[0] = 0.0f; v1[1] =  0.0f; v1[2] = 0.0f; v1[3] = 0.0f;
    v2[0] = 1.0f; v2[1] = -1.0f; v2[2] = 2.0f; v2[3] = 0.0f;
    vres = select_eq(v1, v2, vals1, vals2);
    REQUIRE(math::almost_equal(vres[0], 50.0f));
    REQUIRE(math::almost_equal(vres[1], 60.0f));
    REQUIRE(math::almost_equal(vres[2], 70.0f));
    REQUIRE(math::almost_equal(vres[3], 40.0f));
    vres = select_neq(v1, v2, vals1, vals2);
    REQUIRE(math::almost_equal(vres[0], 10.0f));
    REQUIRE(math::almost_equal(vres[1], 20.0f));
    REQUIRE(math::almost_equal(vres[2], 30.0f));
    REQUIRE(math::almost_equal(vres[3], 80.0f));
    vres = select_lt(v1, v2, vals1, vals2);
    REQUIRE(math::almost_equal(vres[0], 10.0f));
    REQUIRE(math::almost_equal(vres[1], 60.0f));
    REQUIRE(math::almost_equal(vres[2], 30.0f));
    REQUIRE(math::almost_equal(vres[3], 80.0f));
    vres = select_lte(v1, v2, vals1, vals2);
    REQUIRE(math::almost_equal(vres[0], 10.0f));
    REQUIRE(math::almost_equal(vres[1], 60.0f));
    REQUIRE(math::almost_equal(vres[2], 30.0f));
    REQUIRE(math::almost_equal(vres[3], 40.0f));
    vres = select_gt(v1, v2, vals1, vals2);
    REQUIRE(math::almost_equal(vres[0], 50.0f));
    REQUIRE(math::almost_equal(vres[1], 20.0f));
    REQUIRE(math::almost_equal(vres[2], 70.0f));
    REQUIRE(math::almost_equal(vres[3], 80.0f));
    vres = select_gte(v1, v2, vals1, vals2);
    REQUIRE(math::almost_equal(vres[0], 50.0f));
    REQUIRE(math::almost_equal(vres[1], 20.0f));
    REQUIRE(math::almost_equal(vres[2], 70.0f));
    REQUIRE(math::almost_equal(vres[3], 40.0f));



#if 0
    REQUIRE((v1 == v2) == false);
    v2[1] = 2.0f; v2[3] = 4.0f;
    REQUIRE((v1 == v2) == true);
    v2 += math::vec4f_t::math_t::ones();
    REQUIRE((v1  < v2) == true);
    REQUIRE((v1 <= v2) == true);
    REQUIRE((v1 != v2) == true);
    REQUIRE((v2 != v1) == true);
    REQUIRE((v2  < v1) == false);
    REQUIRE((v2 <= v1) == false);
    v2 = v1 - math::vec4f_t::math_t::ones();
    REQUIRE((v1 == v2) == false);
    REQUIRE((v1  > v2) == true);
    REQUIRE((v1 >= v2) == true);
    REQUIRE((v1 != v2) == true);
    REQUIRE((v2  < v1) == true);
    REQUIRE((v2 <= v1) == true);
    REQUIRE((v2 != v1) == true);
#endif
}

#endif // !PVECF_ARM


TEST_CASE("Vec4f swizzles", "TestVec4fSwizzles")
{
    using vec4f_t = math::vec4f_t;
    
    vec4f_t v1(1.0f, 2.0f, 3.0f, 4.0f);
    vec4f_t v4;

#define TEST_SWIZZLE4_(a,b,c,d,a_,b_,c_,d_)     \
        v4 = a##b##c##d(v1);                    \
        REQUIRE(math::almost_equal(v4[0], a_)); \
        REQUIRE(math::almost_equal(v4[1], b_)); \
        REQUIRE(math::almost_equal(v4[2], c_)); \
        REQUIRE(math::almost_equal(v4[3], d_));
//#ifndef ANDROID
#if 1
#define TEST_SWIZZLE4(a,b,c,d,a_,b_,c_,d_) TEST_SWIZZLE4_(a,b,c,d,a_,b_,c_,d_)
#else
#define TEST_SWIZZLE4(a,b,c,d,a_,b_,c_,d_) { \
    __android_log_print(                     \
        ANDROID_LOG_INFO,                    \
        "pvecmath"/*tag*/,                   \
        "testing: " Q(a##b##c##d()) "..."    \
    );                                       \
TEST_SWIZZLE4_(a,b,c,d,a_,b_,c_,d_)          \
    __android_log_print(                     \
        ANDROID_LOG_INFO,                    \
        "pvecmath"/*tag*/,                   \
        "  v4(0,1,2,3)=(%f,%f,%f,%f)",       \
        v4[0], v4[1], v4[2], v4[3]           \
    );                                       \
}
#endif

#define TEST_SWIZZLE3(a,b,c,a_,b_,c_)    \
    TEST_SWIZZLE4(a,b,c,x,a_,b_,c_,1.0f) \
    TEST_SWIZZLE4(a,b,c,y,a_,b_,c_,2.0f) \
    TEST_SWIZZLE4(a,b,c,z,a_,b_,c_,3.0f) \
    TEST_SWIZZLE4(a,b,c,w,a_,b_,c_,4.0f)
#define TEST_SWIZZLE2(a,b,a_,b_)    \
    TEST_SWIZZLE3(a,b,x,a_,b_,1.0f) \
    TEST_SWIZZLE3(a,b,y,a_,b_,2.0f) \
    TEST_SWIZZLE3(a,b,z,a_,b_,3.0f) \
    TEST_SWIZZLE3(a,b,w,a_,b_,4.0f)
#define TEST_SWIZZLE1(a,a_)    \
    TEST_SWIZZLE2(a,x,a_,1.0f) \
    TEST_SWIZZLE2(a,y,a_,2.0f) \
    TEST_SWIZZLE2(a,z,a_,3.0f) \
    TEST_SWIZZLE2(a,w,a_,4.0f)

    TEST_SWIZZLE1(x, 1.0f)
    TEST_SWIZZLE1(y, 2.0f)
    TEST_SWIZZLE1(z, 3.0f)
    TEST_SWIZZLE1(w, 4.0f)

#undef TEST_SWIZZLE4_
#ifdef ANDROID
#undef TEST_SWIZZLE4
#endif
#undef TEST_SWIZZLE1
#undef TEST_SWIZZLE2
#undef TEST_SWIZZLE3
#undef TEST_SWIZZLE4

    //
    // test shuffle operations with 2 operands
    //

    v1[0] = 1.0f; v1[1] = 2.0f; v1[2] = 3.0f; v1[3] = 4.0f;
    vec4f_t v2(5.0f, 6.0f, 7.0f, 8.0f);

#define TEST_SWIZZLE4_(a,b,c,d,a_,b_,c_,d_) \
    v4 = a##b##c##d(v1, v2);                \
    REQUIRE(math::almost_equal(v4[0], a_)); \
    REQUIRE(math::almost_equal(v4[1], b_)); \
    REQUIRE(math::almost_equal(v4[2], c_)); \
    REQUIRE(math::almost_equal(v4[3], d_));
//#ifndef ANDROID
#if 1
#define TEST_SWIZZLE4(a,b,c,d,a_,b_,c_,d_) TEST_SWIZZLE4_(a,b,c,d,a_,b_,c_,d_)
#else
#define TEST_SWIZZLE4(a,b,c,d,a_,b_,c_,d_) {   \
    __android_log_print(                       \
        ANDROID_LOG_INFO,                      \
        "pvecmath"/*tag*/,                     \
        "testing: " Q(a##b##c##d(v1,v2)) "..." \
    );                                         \
TEST_SWIZZLE4_(a,b,c,d,a_,b_,c_,d_)            \
    __android_log_print(                       \
        ANDROID_LOG_INFO,                      \
        "pvecmath"/*tag*/,                     \
        "  v4(0,1,2,3)=(%f,%f,%f,%f)",         \
        v4[0], v4[1], v4[2], v4[3]             \
    );                                         \
}
#endif

#define TEST_SWIZZLE3(a,b,c,a_,b_,c_)    \
    TEST_SWIZZLE4(a,b,c,x,a_,b_,c_,5.0f) \
    TEST_SWIZZLE4(a,b,c,y,a_,b_,c_,6.0f) \
    TEST_SWIZZLE4(a,b,c,z,a_,b_,c_,7.0f) \
    TEST_SWIZZLE4(a,b,c,w,a_,b_,c_,8.0f)
#define TEST_SWIZZLE2(a,b,a_,b_)    \
    TEST_SWIZZLE3(a,b,x,a_,b_,5.0f) \
    TEST_SWIZZLE3(a,b,y,a_,b_,6.0f) \
    TEST_SWIZZLE3(a,b,z,a_,b_,7.0f) \
    TEST_SWIZZLE3(a,b,w,a_,b_,8.0f)
#define TEST_SWIZZLE1(a,a_)    \
    TEST_SWIZZLE2(a,x,a_,1.0f) \
    TEST_SWIZZLE2(a,y,a_,2.0f) \
    TEST_SWIZZLE2(a,z,a_,3.0f) \
    TEST_SWIZZLE2(a,w,a_,4.0f)

    TEST_SWIZZLE1(x, 1.0f)
    TEST_SWIZZLE1(y, 2.0f)
    TEST_SWIZZLE1(z, 3.0f)
    TEST_SWIZZLE1(w, 4.0f)

#undef TEST_SWIZZLE4_
#ifdef ANDROID
#undef TEST_SWIZZLE4
#endif
#undef TEST_SWIZZLE1
#undef TEST_SWIZZLE2
#undef TEST_SWIZZLE3
#undef TEST_SWIZZLE4
    
    
    v1[0] = 1.0f; v1[1] = -2.0f; v1[2] = 3.0f; v1[3] = -4.0f;
    v1.abs_();
    REQUIRE(math::almost_equal(v1[0], 1.0f));
    REQUIRE(math::almost_equal(v1[1], 2.0f));
    REQUIRE(math::almost_equal(v1[2], 3.0f));
    REQUIRE(math::almost_equal(v1[3], 4.0f));
}


TEST_CASE("TestVec4fInitializerList")
{
    using vec4f_t = math::vec4f_t;
    vec4f_t v1{1.0f, 2.0f, 3.0f, 4.0f};
    REQUIRE(math::almost_equal(v1[0], 1.0f));
    REQUIRE(math::almost_equal(v1[1], 2.0f));
    REQUIRE(math::almost_equal(v1[2], 3.0f));
    REQUIRE(math::almost_equal(v1[3], 4.0f));

    vec4f_t v2{1.0f, 2.0f};
    REQUIRE(math::almost_equal(v2[0], 1.0f));
    REQUIRE(math::almost_equal(v2[1], 2.0f));
    REQUIRE(math::almost_equal(v2[2], 0.0f));
    REQUIRE(math::almost_equal(v2[3], 0.0f));

    vec4f_t v3{1.0f, 2.0f, 3.0f};
    REQUIRE(math::almost_equal(v3[0], 1.0f));
    REQUIRE(math::almost_equal(v3[1], 2.0f));
    REQUIRE(math::almost_equal(v3[2], 3.0f));
    REQUIRE(math::almost_equal(v3[3], 0.0f));

    vec4f_t v4{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    REQUIRE(math::almost_equal(v4[0], 1.0f));
    REQUIRE(math::almost_equal(v4[1], 2.0f));
    REQUIRE(math::almost_equal(v4[2], 3.0f));
    REQUIRE(math::almost_equal(v4[3], 4.0f));
}

TEST_CASE("TestVec4fMiscOps")
{
    using vec4f_t = math::vec4f_t;

    vec4f_t v1(1.6f, -1.6f, 1.0f, -1.0f);
    v1.trunc();
    REQUIRE(math::almost_equal(v1[0],  1.0f));
    REQUIRE(math::almost_equal(v1[1], -1.0f));
    REQUIRE(math::almost_equal(v1[2],  1.0f));
    REQUIRE(math::almost_equal(v1[3], -1.0f));

    vec4f_t v2(1.6f, -1.6f, 1.0f, -1.0f);
    v2.floor();
    REQUIRE(math::almost_equal(v2[0],  1.0f));
    REQUIRE(math::almost_equal(v2[1], -2.0f));
    REQUIRE(math::almost_equal(v2[2],  1.0f));
    REQUIRE(math::almost_equal(v2[3], -1.0f));

    vec4f_t v3(1.6f, -1.6f, 1.0f, -1.0f);
    v3.ceil();
    REQUIRE(math::almost_equal(v3[0],  2.0f));
    REQUIRE(math::almost_equal(v3[1], -1.0f));
    REQUIRE(math::almost_equal(v3[2],  1.0f));
    REQUIRE(math::almost_equal(v3[3], -1.0f));

    vec4f_t v4(1.6f, -1.6f, 1.0f, -1.0f);
    v4.frac();
    REQUIRE(math::almost_equal(v4[0],  0.6f));
    REQUIRE(math::almost_equal(v4[1],  0.4f));
    REQUIRE(math::almost_equal(v4[2],  0.0f));
    REQUIRE(math::almost_equal(v4[3],  0.0f));


    vec4f_t v5(-1.0f, -2.0f, -3.0f, -4.0f);
    vec4f_t v6(v5);
    v6.elemAbs<0>();
    REQUIRE(math::almost_equal(v6[0],  1.0f));
    REQUIRE(math::almost_equal(v6[1], -2.0f));
    REQUIRE(math::almost_equal(v6[2], -3.0f));
    REQUIRE(math::almost_equal(v6[3], -4.0f));

    v6 = v5;
    v6.elemAbs<1>();
    REQUIRE(math::almost_equal(v6[0], -1.0f));
    REQUIRE(math::almost_equal(v6[1],  2.0f));
    REQUIRE(math::almost_equal(v6[2], -3.0f));
    REQUIRE(math::almost_equal(v6[3], -4.0f));

    v6 = v5;
    v6.elemAbs<2>();
    REQUIRE(math::almost_equal(v6[0], -1.0f));
    REQUIRE(math::almost_equal(v6[1], -2.0f));
    REQUIRE(math::almost_equal(v6[2],  3.0f));
    REQUIRE(math::almost_equal(v6[3], -4.0f));

    v6 = v5;
    v6.elemAbs<3>();
    REQUIRE(math::almost_equal(v6[0], -1.0f));
    REQUIRE(math::almost_equal(v6[1], -2.0f));
    REQUIRE(math::almost_equal(v6[2], -3.0f));
    REQUIRE(math::almost_equal(v6[3],  4.0f));

}

#include <pvecf-algos.h>

// very coarse measuring
class CoarseMeasuring
{
public:
    CoarseMeasuring()
        : start_(0)
        , stop_(0)
        , duration_(0)
    {}
    CoarseMeasuring(const CoarseMeasuring &) = delete;
    CoarseMeasuring & operator=(const CoarseMeasuring &) = delete;
    ~CoarseMeasuring() = default;

    void start()
    {
#ifdef WIN32
        start_ = ::GetTickCount64();
#elif defined(ANDROID)
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        start_ = uint64_t(now.tv_sec * 1000000000LL + now.tv_nsec);
#endif
    }

    void stop()
    {
#ifdef WIN32
        stop_ = ::GetTickCount64();
#elif defined(ANDROID)
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        stop_ = uint64_t(now.tv_sec * 1000000000LL + now.tv_nsec);
#endif

        duration_ = stop_ - start_;
    }

    uint64_t duration() const
    {
        return duration_;
    }

    void reset()
    {
        start_ = stop_ = duration_ = 0;
    }

private:
    uint64_t start_, stop_, duration_;
};


#ifdef ALGO_PERFORMANCE_TESTS
TEST_CASE("TestVec4fAlgos1")
{
    const int runs = 100;
    const int elemCount = 10000000;
    auto buf = make_unique<float[]>(elemCount);
    for(int i = 0; i < elemCount; ++i)
        buf[i] = 1.0f;

    std::printf("algo: sum_all_{scalar,packed}()\n");
    std::printf("===============================\n");
    std::printf("elems, runs: %d, %d\n", elemCount, runs);
    float sum;
    CoarseMeasuring measure;

    measure.start();
    for(int i = 0; i < runs; ++i) {
        sum = math::sum_all_scalar(buf.get(), elemCount);
    }
    measure.stop();
    std::printf("scalar duration: %u ms\n", unsigned(measure.duration()));
    std::printf("  sum          : %f\n", sum);
    measure.reset();

    measure.start();
    for(int i = 0; i < runs; ++i) {
        sum = math::sum_all_packed(buf.get(), elemCount);
    }
    measure.stop();
    std::printf("packed vector duration: %u ms\n", unsigned(measure.duration()));
    std::printf("  sum                 : %f\n\n", sum);
    measure.reset();

    buf.release();
    

    std::printf("algo: add_mulpairs_{scalar,packed}()\n");
    std::printf("====================================\n");
    std::printf("elems, runs: %d, %d\n", elemCount, runs);
    auto inbuf = make_unique<float[]>(elemCount);
    // 1,2,3,4
    for(int i = 0; i < elemCount; ++i) {
        switch(i & 3) {
        case 0: inbuf[i] = 1.0f; break;
        case 1: inbuf[i] = 2.0f; break;
        case 2: inbuf[i] = 3.0f; break;
        case 3: inbuf[i] = 4.0f; break;
        }
    }
    auto outbuf = make_unique<float[]>(elemCount / 4);
    measure.start();
    for(int i = 0; i < runs; ++i) {
        math::add_mulpairs_scalar(outbuf.get(), inbuf.get(), elemCount / 4);
    }
    measure.stop();

    std::printf("scalar duration: %u ms\n", unsigned(measure.duration()));
    REQUIRE(math::almost_equal(outbuf[0], 14.0f));

    ::memset(outbuf.get(), 0, (elemCount / 4) * sizeof(float));
    measure.start();
    for(int i = 0; i < runs; ++i) {
        math::add_mulpairs_packed(outbuf.get(), inbuf.get(), elemCount / 4);
    }
    measure.stop();
    std::printf("packed vector duration: %u ms\n\n", unsigned(measure.duration()));
    REQUIRE(math::almost_equal(outbuf[0], 14.0f));


    std::printf("algo: min_arr_all_{scalar,packed}()\n");
    std::printf("===================================\n");
    float minval_scalar;
    measure.start();
    for(int i = 0; i < runs; ++i) {
        minval_scalar = math::min_arr_all_scalar(inbuf.get(), elemCount);
    }
    measure.stop();
    std::printf("scalar duration: %u ms\n", unsigned(measure.duration()));

    float minval_packed;
    measure.start();
    for(int i = 0; i < runs; ++i) {
        minval_packed = math::min_arr_all_packed(inbuf.get(), elemCount / 4);
    }
    measure.stop();
    std::printf("packed vector duration: %u ms\n", unsigned(measure.duration()));
    REQUIRE(math::almost_equal(minval_scalar, minval_packed));


    std::printf("algo: max_arr_all_{scalar,packed}()\n");
    std::printf("===================================\n");
    float maxval_scalar;
    measure.start();
    for(int i = 0; i < runs; ++i) {
        maxval_scalar = math::max_arr_all_scalar(inbuf.get(), elemCount);
    }
    measure.stop();
    std::printf("scalar duration: %u ms\n", unsigned(measure.duration()));

    float maxval_packed;
    measure.start();
    for(int i = 0; i < runs; ++i) {
        maxval_packed = math::max_arr_all_packed(inbuf.get(), elemCount / 4);
    }
    measure.stop();
    std::printf("packed vector duration: %u ms\n", unsigned(measure.duration()));
    REQUIRE(math::almost_equal(maxval_scalar, maxval_packed));

}

TEST_CASE("TestVec4fAlgos2")
{
    const int runs = 100;
    const int elemCount = 10000000;

    std::printf("algo: min_arr4channels_all_{scalar,packed}()\n");
    std::printf("============================================\n");
    auto inbuf = make_unique<float[]>(elemCount);
    // 1,2,3,4
    for(int i = 0; i < elemCount; ++i) {
        switch(i & 3) {
        case 0: inbuf[i] = 1.0f; break;
        case 1: inbuf[i] = 2.0f; break;
        case 2: inbuf[i] = 3.0f; break;
        case 3: inbuf[i] = 4.0f; break;
        }
    }

    float minvals_scalar[4];
    CoarseMeasuring measure;
    measure.start();
    for(int i = 0; i < runs; ++i) {
        math::min_arr4channels_all_scalar(minvals_scalar, inbuf.get(), elemCount / 4);
    }
    measure.stop();
    std::printf("scalar duration: %u ms\n", unsigned(measure.duration()));
    measure.reset();

    float minvals_packed[4];
    measure.start();
    for(int i = 0; i < runs; ++i) {
        math::min_arr4channels_all_packed(minvals_packed, inbuf.get(), elemCount / 4);
    }
    measure.stop();
    std::printf("packed vector duration: %u ms\n", unsigned(measure.duration()));
    measure.reset();
    REQUIRE(math::almost_equal(minvals_scalar[0], minvals_packed[0]));
    REQUIRE(math::almost_equal(minvals_scalar[1], minvals_packed[1]));
    REQUIRE(math::almost_equal(minvals_scalar[2], minvals_packed[2]));
    REQUIRE(math::almost_equal(minvals_scalar[3], minvals_packed[3]));
    

    std::printf("algo: max_arr4channels_all_{scalar,packed}()\n");
    std::printf("============================================\n");
    float maxvals_scalar[4];
    measure.start();
    for(int i = 0; i < runs; ++i) {
        math::max_arr4channels_all_scalar(maxvals_scalar, inbuf.get(), elemCount / 4);
    }
    measure.stop();
    std::printf("scalar duration: %u ms\n", unsigned(measure.duration()));
    measure.reset();

    float maxvals_packed[4];
    measure.start();
    for(int i = 0; i < runs; ++i) {
        math::max_arr4channels_all_packed(maxvals_packed, inbuf.get(), elemCount / 4);
    }
    measure.stop();
    std::printf("packed vector duration: %u ms\n", unsigned(measure.duration()));
    measure.reset();
    REQUIRE(math::almost_equal(maxvals_scalar[0], maxvals_packed[0]));
    REQUIRE(math::almost_equal(maxvals_scalar[1], maxvals_packed[1]));
    REQUIRE(math::almost_equal(maxvals_scalar[2], maxvals_packed[2]));
    REQUIRE(math::almost_equal(maxvals_scalar[3], maxvals_packed[3]));

}
#endif


// no double vector support in NEON
#ifndef ANDROID
TEST_CASE("TestVec2d")
{
    using vec2d_t = math::vec2d_t;

    vec2d_t v1;

    REQUIRE(math::almost_equal(v1[0], 0.0));
    REQUIRE(math::almost_equal(v1[1], 0.0));

    vec2d_t v2(1.0, 2.0);
    REQUIRE(math::almost_equal(v2[0], 1.0));
    REQUIRE(math::almost_equal(v2[1], 2.0));

    vec2d_t v3(1.0);
    REQUIRE(math::almost_equal(v3[0], 1.0));
    REQUIRE(math::almost_equal(v3[1], 0.0));

    // free-standing operators
    vec2d_t v4;
    v4 = v2 + v2;
    REQUIRE(math::almost_equal(v4[0], 2.0));
    REQUIRE(math::almost_equal(v4[1], 4.0));
    v4 = add(v2, v2);
    REQUIRE(math::almost_equal(v4[0], 2.0));
    REQUIRE(math::almost_equal(v4[1], 4.0));
    v4 = add(v2, v2.p);
    REQUIRE(math::almost_equal(v4[0], 2.0));
    REQUIRE(math::almost_equal(v4[1], 4.0));
    v4 = add(v2.p, v2);
    REQUIRE(math::almost_equal(v4[0], 2.0));
    REQUIRE(math::almost_equal(v4[1], 4.0));
    v4 = add(v2, 1.0);
    REQUIRE(math::almost_equal(v4[0], 2.0));
    REQUIRE(math::almost_equal(v4[1], 3.0));
    v4 = add(1.0, v2);
    REQUIRE(math::almost_equal(v4[0], 2.0));
    REQUIRE(math::almost_equal(v4[1], 3.0));


    v4 = v2 - v2;
    REQUIRE(math::almost_equal(v4[0], 0.0));
    REQUIRE(math::almost_equal(v4[1], 0.0));
    v4 = sub(v2, v2);
    REQUIRE(math::almost_equal(v4[0], 0.0));
    REQUIRE(math::almost_equal(v4[1], 0.0));
    v4 = sub(v2, v2.p);
    REQUIRE(math::almost_equal(v4[0], 0.0));
    REQUIRE(math::almost_equal(v4[1], 0.0));
    v4 = sub(v2.p, v2);
    REQUIRE(math::almost_equal(v4[0], 0.0));
    REQUIRE(math::almost_equal(v4[1], 0.0));
    v4 = sub(v2, 1.0);
    REQUIRE(math::almost_equal(v4[0], 0.0));
    REQUIRE(math::almost_equal(v4[1], 1.0));
    v4 = sub(2.0, v2);
    REQUIRE(math::almost_equal(v4[0], 1.0));
    REQUIRE(math::almost_equal(v4[1], 0.0));


    v4 = v2 * v2;
    REQUIRE(math::almost_equal(v4[0], 1.0));
    REQUIRE(math::almost_equal(v4[1], 4.0));
    v4 = mul(v2, v2);
    REQUIRE(math::almost_equal(v4[0], 1.0));
    REQUIRE(math::almost_equal(v4[1], 4.0));
    v4 = mul(v2, v2.p);
    REQUIRE(math::almost_equal(v4[0], 1.0));
    REQUIRE(math::almost_equal(v4[1], 4.0));
    v4 = mul(v2.p, v2);
    REQUIRE(math::almost_equal(v4[0], 1.0));
    REQUIRE(math::almost_equal(v4[1], 4.0));
    v4 = mul(v2, 2.0);
    REQUIRE(math::almost_equal(v4[0], 2.0));
    REQUIRE(math::almost_equal(v4[1], 4.0));
    v4 = mul(2.0, v2);
    REQUIRE(math::almost_equal(v4[0], 2.0));
    REQUIRE(math::almost_equal(v4[1], 4.0));


    v4 = v2 / v2;
    REQUIRE(math::almost_equal(v4[0], 1.0));
    REQUIRE(math::almost_equal(v4[1], 1.0));
    v4 = div(v2, v2);
    REQUIRE(math::almost_equal(v4[0], 1.0));
    REQUIRE(math::almost_equal(v4[1], 1.0));
    v4 = div(v2, v2.p);
    REQUIRE(math::almost_equal(v4[0], 1.0));
    REQUIRE(math::almost_equal(v4[1], 1.0));
    v4 = div(v2.p, v2);
    REQUIRE(math::almost_equal(v4[0], 1.0));
    REQUIRE(math::almost_equal(v4[1], 1.0));
    v4 = div(v2, 2.0);
    REQUIRE(math::almost_equal(v4[0], 0.5));
    REQUIRE(math::almost_equal(v4[1], 1.0));
    v4 = div(2.0, v2);
    REQUIRE(math::almost_equal(v4[0], 2.0));
    REQUIRE(math::almost_equal(v4[1], 1.0));


    vec2d_t v_negzeroes(-0.0, -0.0);
    v4[0] = 1.0; v4[1] = -1.0;
    v4 = v4 & v_negzeroes;
    REQUIRE(math::almost_equal(v4[0],  0.0));
    REQUIRE(math::almost_equal(v4[1], -0.0));
    v4[0] = 1.0; v4[1] = -1.0;
    v4 = and_(v4, v_negzeroes);
    REQUIRE(math::almost_equal(v4[0],  0.0));
    REQUIRE(math::almost_equal(v4[1], -0.0));
    v4[0] = 1.0; v4[1] = -1.0;
    v4 = and_(v4, v_negzeroes.p);
    REQUIRE(math::almost_equal(v4[0],  0.0));
    REQUIRE(math::almost_equal(v4[1], -0.0));
    v4[0] = 1.0; v4[1] = -1.0;
    v4 = and_(v4.p, v_negzeroes);
    REQUIRE(math::almost_equal(v4[0],  0.0));
    REQUIRE(math::almost_equal(v4[1], -0.0));
    v4[0] = 1.0; v4[1] = -1.0;
    v4 = and_(v4, -0.0);
    REQUIRE(math::almost_equal(v4[0],  0.0));
    REQUIRE(math::almost_equal(v4[1], -0.0));
    v4[0] = 1.0; v4[1] = -1.0;
    v4 = and_(-0.0, v4);
    REQUIRE(math::almost_equal(v4[0],  0.0));
    REQUIRE(math::almost_equal(v4[1], -0.0));


    v4[0] = 1.0; v4[1] = 2.0;
    v4 = v4 | v_negzeroes;
    REQUIRE(math::almost_equal(v4[0], -1.0));
    REQUIRE(math::almost_equal(v4[1], -2.0));
    v4[0] = 1.0; v4[1] = 2.0;
    v4 = or_(v4, v_negzeroes);
    REQUIRE(math::almost_equal(v4[0], -1.0));
    REQUIRE(math::almost_equal(v4[1], -2.0));
    v4[0] = 1.0; v4[1] = 2.0;
    v4 = or_(v4, v_negzeroes.p);
    REQUIRE(math::almost_equal(v4[0], -1.0));
    REQUIRE(math::almost_equal(v4[1], -2.0));
    v4[0] = 1.0; v4[1] = 2.0;
    v4 = or_(v4.p, v_negzeroes);
    REQUIRE(math::almost_equal(v4[0], -1.0));
    REQUIRE(math::almost_equal(v4[1], -2.0));
    v4[0] = 1.0; v4[1] = 2.0;
    v4 = or_(v4, -0.0);
    REQUIRE(math::almost_equal(v4[0], -1.0));
    REQUIRE(math::almost_equal(v4[1], -2.0));
    v4[0] = 1.0; v4[1] = 2.0;
    v4 = or_(-0.0, v4);
    REQUIRE(math::almost_equal(v4[0], -1.0));
    REQUIRE(math::almost_equal(v4[1], -2.0));


    v4[0] = 1.0; v4[1] = -1.0;
    v4 = v4 ^ v_negzeroes;
    REQUIRE(math::almost_equal(v4[0], -1.0));
    REQUIRE(math::almost_equal(v4[1],  1.0));
    v4[0] = 1.0; v4[1] = -1.0;
    v4 = xor_(v4, v_negzeroes);
    REQUIRE(math::almost_equal(v4[0], -1.0));
    REQUIRE(math::almost_equal(v4[1],  1.0));
    v4[0] = 1.0; v4[1] = -1.0;
    v4 = xor_(v4, v_negzeroes.p);
    REQUIRE(math::almost_equal(v4[0], -1.0));
    REQUIRE(math::almost_equal(v4[1],  1.0));
    v4[0] = 1.0; v4[1] = -1.0;
    v4 = xor_(v4.p, v_negzeroes);
    REQUIRE(math::almost_equal(v4[0], -1.0));
    REQUIRE(math::almost_equal(v4[1],  1.0));
    v4[0] = 1.0; v4[1] = -1.0;
    v4 = xor_(v4, -0.0);
    REQUIRE(math::almost_equal(v4[0], -1.0));
    REQUIRE(math::almost_equal(v4[1],  1.0));
    v4[0] = 1.0; v4[1] = -1.0;
    v4 = xor_(-0.0, v4);
    REQUIRE(math::almost_equal(v4[0], -1.0));
    REQUIRE(math::almost_equal(v4[1],  1.0));


    v4[0] = 1.0; v4[1] = 2.0;
    v4 += 2.0;
    REQUIRE(math::almost_equal(v4[0], 3.0));
    REQUIRE(math::almost_equal(v4[1], 4.0));
    v4[0] = 1.0; v4[1] = 2.0;
    v4 = add(v4, 2.0);
    REQUIRE(math::almost_equal(v4[0], 3.0));
    REQUIRE(math::almost_equal(v4[1], 4.0));
    v4[0] = 1.0; v4[1] = 2.0;
    v4 = add(2.0, v4);
    REQUIRE(math::almost_equal(v4[0], 3.0));
    REQUIRE(math::almost_equal(v4[1], 4.0));

    v4[0] = 2.0; v4[1] = 3.0;
    v4 -= 2.0;
    REQUIRE(math::almost_equal(v4[0], 0.0));
    REQUIRE(math::almost_equal(v4[1], 1.0));

    v4[0] = 1.0; v4[1] = 2.0;
    v4 *= 2.0;
    REQUIRE(math::almost_equal(v4[0], 2.0));
    REQUIRE(math::almost_equal(v4[1], 4.0));

    v4[0] = 1.0; v4[1] = 2.0;
    v4 /= 2.0;
    REQUIRE(math::almost_equal(v4[0], 0.5));
    REQUIRE(math::almost_equal(v4[1], 1.0));

    v4[0] = 1.0; v4[1] = 2.0;
    v1 = v4;
    double fval = v4.dot(v1);
    REQUIRE(math::almost_equal(fval, 5.0));

    fval = v4.sqlen();
    REQUIRE(math::almost_equal(fval, 5.0));

    fval = v4.len();
    REQUIRE(math::almost_equal(fval, sqrt(5.0), 0.0001));


    v4[0] = 1.0; v4[1] = 0.0;
    v4.normalize();
    REQUIRE(math::almost_equal(v4[0], 1.0));
    REQUIRE(math::almost_equal(v4[1], 0.0));

    v4[0] = -1.0; v4[1] = 0.25;
    v4.clamp_0_1();
    REQUIRE(math::almost_equal(v4[0], 0.00));
    REQUIRE(math::almost_equal(v4[1], 0.25));

    v4[0] = 0.75; v4[1] = 2.0;
    v4.clamp_0_1();
    REQUIRE(math::almost_equal(v4[0], 0.75));
    REQUIRE(math::almost_equal(v4[1], 1.00));
            
    
#if 0
    v1[0] = 1.0; v1[1] = 0.0;
    v4[0] = 0.0; v4[1] = 1.0; // dummy (2D)
    v4 = v1.cross(v4/*dummy (2D) */);
    REQUIRE(math::almost_equal(v4[0],  0.0));
    REQUIRE(math::almost_equal(v4[1], -1.0));

    v1[0] = 2.0; v1[1] = 0.0;
    v4[0] = 0.0; v4[1] = 2.0;
    v4 = v1.unit_cross(v4);
    REQUIRE(math::almost_equal(v4[0], 0.0));
    REQUIRE(math::almost_equal(v4[1], 0.0));
#endif

    v1[0] = 2.0; v1[1] = -1.0;
    v4[0] = 3.0; v4[1] = -2.0;
    v4 = vec2d_t::min_(v1, v4);
    REQUIRE(math::almost_equal(v4[0],  2.0));
    REQUIRE(math::almost_equal(v4[1], -2.0));

    v4[0] = 3.0; v4[1] = -2.0;
    v4 = vec2d_t::max_(v1, v4);
    REQUIRE(math::almost_equal(v4[0],  3.0));
    REQUIRE(math::almost_equal(v4[1], -1.0));

    v1[0] = 1.0; v1[1] = 2.0;

    v4 = xx(v1);
    REQUIRE(math::almost_equal(v4[0], 1.0));
    REQUIRE(math::almost_equal(v4[1], 1.0));

    v4 = xy(v1);
    REQUIRE(math::almost_equal(v4[0], 1.0));
    REQUIRE(math::almost_equal(v4[1], 2.0));

    v4 = yx(v1);
    REQUIRE(math::almost_equal(v4[0], 2.0));
    REQUIRE(math::almost_equal(v4[1], 1.0));

    v4 = yy(v1);
    REQUIRE(math::almost_equal(v4[0], 2.0));
    REQUIRE(math::almost_equal(v4[1], 2.0));


    v1[0] = 1.0; v1[1] = -2.0;
    v1.abs_();
    REQUIRE(math::almost_equal(v1[0], 1.0));
    REQUIRE(math::almost_equal(v1[1], 2.0));

}

TEST_CASE("TestVec2dInitializerList")
{
    using vec2d_t = math::vec2d_t;

    vec2d_t v1{1.0, 2.0};
    REQUIRE(math::almost_equal(v1[0], 1.0));
    REQUIRE(math::almost_equal(v1[1], 2.0));

    vec2d_t v2{1.0};
    REQUIRE(math::almost_equal(v2[0], 1.0));
    REQUIRE(math::almost_equal(v2[1], 0.0));

    vec2d_t v3{1.0, 2.0, 3.0, 4.0};
    REQUIRE(math::almost_equal(v3[0], 1.0));
    REQUIRE(math::almost_equal(v3[1], 2.0));
}


TEST_CASE("TestVec2dMiscOps")
{
    using vec2d_t = math::vec2d_t;

    vec2d_t v1(-1.0, -2.0);
    vec2d_t v2(v1);
    v2.elemAbs<0>();
    REQUIRE(math::almost_equal(v2[0],  1.0));
    REQUIRE(math::almost_equal(v2[1], -2.0));

    v2 = v1;
    v2.elemAbs<1>();
    REQUIRE(math::almost_equal(v2[0], -1.0));
    REQUIRE(math::almost_equal(v2[1],  2.0));
}


#endif // !defined(ANDROID)


TEST_CASE("TestVeci8x16")
{
    using veci_i8x16_t = math::veci_i8x16_t;

    veci_i8x16_t v1(
        0, -1,  2,  -3,  4,  -5,  6, -7,
        8, -9, 10, -11, 12, -13, 14, -15
    );
    veci_i8x16_t v2;

    v2 = v1;
    v2.abs_();
    REQUIRE(v2[ 0] ==  0); REQUIRE(v2[ 1] ==  1);
    REQUIRE(v2[ 2] ==  2); REQUIRE(v2[ 3] ==  3);
    REQUIRE(v2[ 4] ==  4); REQUIRE(v2[ 5] ==  5);
    REQUIRE(v2[ 6] ==  6); REQUIRE(v2[ 7] ==  7);
    REQUIRE(v2[ 8] ==  8); REQUIRE(v2[ 9] ==  9);
    REQUIRE(v2[10] == 10); REQUIRE(v2[11] == 11);
    REQUIRE(v2[12] == 12); REQUIRE(v2[13] == 13);
    REQUIRE(v2[14] == 14); REQUIRE(v2[15] == 15);

    // member operators
            
    v2 = v1;
    veci_i8x16_t v3(
          3,  4,  5,  6,  7,  8,  9, 10,
        -10, -9, -8, -7, -6, -5, -4, -3
    );
    v2 += v3;
    for(int i = 0; i < 8; ++i)
        REQUIRE(v2[i] == v1[i]+v3[i]);

    v2 = v1;
    v2 -= v3;
    for(int i = 0; i < 8; ++i)
        REQUIRE(v2[i] == v1[i]-v3[i]);

    v2 = v1;
    v2 &= v3;
    for(int i = 0; i < 8; ++i)
        REQUIRE(v2[i] == (v1[i]&v3[i]));

    v2 = v1;
    v2 |= v3;
    for(int i = 0; i < 8; ++i)
        REQUIRE(v2[i] == (v1[i]|v3[i]));
            
    v2 = v1;
    v2 ^= v3;
    for(int i = 0; i < 8; ++i)
        REQUIRE(v2[i] == (v1[i]^v3[i]));

    v2 = ~v1;
    for(int i = 0; i < 8; ++i)
        REQUIRE(v2[i] == ~v1[i]);


    veci_i8x16_t v4(
          3,  4,  1, -4,  2, -5,  8, -3,
        -10, 10, -5,  5,  7, -7,  3,  1
    );
    v2 = veci_i8x16_t::min_(v1, v4);
    REQUIRE(v2[ 0] ==   0); REQUIRE(v2[ 1] ==  -1);
    REQUIRE(v2[ 2] ==   1); REQUIRE(v2[ 3] ==  -4);
    REQUIRE(v2[ 4] ==   2); REQUIRE(v2[ 5] ==  -5);
    REQUIRE(v2[ 6] ==   6); REQUIRE(v2[ 7] ==  -7);

    REQUIRE(v2[ 8] == -10); REQUIRE(v2[ 9] ==  -9);
    REQUIRE(v2[10] ==  -5); REQUIRE(v2[11] == -11);
    REQUIRE(v2[12] ==   7); REQUIRE(v2[13] == -13);
    REQUIRE(v2[14] ==   3); REQUIRE(v2[15] == -15);


    veci_i8x16_t v5(
        -4,  -3,   4,   4,   2,  -2,   8,  -1,
        15,  -1,  -7,   8,  13,  -5,  -2,   0
    );
    v2 = veci_i8x16_t::max_(v1, v5);
    REQUIRE(v2[ 0] ==   0); REQUIRE(v2[ 1] ==  -1);
    REQUIRE(v2[ 2] ==   4); REQUIRE(v2[ 3] ==   4);
    REQUIRE(v2[ 4] ==   4); REQUIRE(v2[ 5] ==  -2);
    REQUIRE(v2[ 6] ==   8); REQUIRE(v2[ 7] ==  -1);

    REQUIRE(v2[ 8] ==  15); REQUIRE(v2[ 9] ==  -1);
    REQUIRE(v2[10] ==  10); REQUIRE(v2[11] ==   8);
    REQUIRE(v2[12] ==  13); REQUIRE(v2[13] ==  -5);
    REQUIRE(v2[14] ==  14); REQUIRE(v2[15] ==   0);

    veci_i8x16_t mask;
    uint8_t allbits_elem = 0xFF;
    uint8_t val;
    int16_t val_s;

#define TEST_MASK(shift)                                    \
    mask = veci_i8x16_t::math_t::mask_zupper<(shift)>();    \
    val = allbits_elem >> (shift);                          \
    val_s = int8_t(val);                                    \
    REQUIRE(mask[ 0] == val_s); REQUIRE(mask[ 1] == val_s); \
    REQUIRE(mask[ 2] == val_s); REQUIRE(mask[ 3] == val_s); \
    REQUIRE(mask[ 4] == val_s); REQUIRE(mask[ 5] == val_s); \
    REQUIRE(mask[ 6] == val_s); REQUIRE(mask[ 7] == val_s); \
    REQUIRE(mask[ 8] == val_s); REQUIRE(mask[ 9] == val_s); \
    REQUIRE(mask[10] == val_s); REQUIRE(mask[11] == val_s); \
    REQUIRE(mask[12] == val_s); REQUIRE(mask[13] == val_s); \
    REQUIRE(mask[14] == val_s); REQUIRE(mask[15] == val_s); \
    mask = veci_i8x16_t::math_t::mask_zlower<(shift)>();    \
    val = allbits_elem << (shift);                          \
    val_s = int8_t(val);                                    \
    REQUIRE(mask[ 0] == val_s); REQUIRE(mask[ 1] == val_s); \
    REQUIRE(mask[ 2] == val_s); REQUIRE(mask[ 3] == val_s); \
    REQUIRE(mask[ 4] == val_s); REQUIRE(mask[ 5] == val_s); \
    REQUIRE(mask[ 6] == val_s); REQUIRE(mask[ 7] == val_s); \
    REQUIRE(mask[ 8] == val_s); REQUIRE(mask[ 9] == val_s); \
    REQUIRE(mask[10] == val_s); REQUIRE(mask[11] == val_s); \
    REQUIRE(mask[12] == val_s); REQUIRE(mask[13] == val_s); \
    REQUIRE(mask[14] == val_s); REQUIRE(mask[15] == val_s);

    TEST_MASK(1)
    TEST_MASK(2)
    TEST_MASK(3)
    TEST_MASK(4)
    TEST_MASK(5)
    TEST_MASK(6)
    TEST_MASK(7)

#undef TEST_MASK


#define TEST_MASK(bitno)                                    \
    mask = veci_i8x16_t::math_t::mask_1bit<bitno>();        \
    val = 1 << (bitno);                                     \
    val_s = int8_t(val);                                    \
    REQUIRE(mask[ 0] == val_s); REQUIRE(mask[ 1] == val_s); \
    REQUIRE(mask[ 2] == val_s); REQUIRE(mask[ 3] == val_s); \
    REQUIRE(mask[ 4] == val_s); REQUIRE(mask[ 5] == val_s); \
    REQUIRE(mask[ 6] == val_s); REQUIRE(mask[ 7] == val_s); \
    REQUIRE(mask[ 8] == val_s); REQUIRE(mask[ 9] == val_s); \
    REQUIRE(mask[10] == val_s); REQUIRE(mask[11] == val_s); \
    REQUIRE(mask[12] == val_s); REQUIRE(mask[13] == val_s); \
    REQUIRE(mask[14] == val_s); REQUIRE(mask[15] == val_s);

    TEST_MASK(0)
    TEST_MASK(1)
    TEST_MASK(2)
    TEST_MASK(3)
    TEST_MASK(4)
    TEST_MASK(5)
    TEST_MASK(6)
    TEST_MASK(7)

#undef TEST_MASK

}

TEST_CASE("TestVeci8x16InitializerList")
{
    using veci_i8x16_t = math::veci_i8x16_t;

    veci_i8x16_t v1{
        0, -1,  2,  -3,  4,  -5,  6,  -7,
        8, -9, 10, -11, 12, -13, 14, -15
    };
    REQUIRE(v1[ 0] ==  0); REQUIRE(v1[ 1] ==  -1);
    REQUIRE(v1[ 2] ==  2); REQUIRE(v1[ 3] ==  -3);
    REQUIRE(v1[ 4] ==  4); REQUIRE(v1[ 5] ==  -5);
    REQUIRE(v1[ 6] ==  6); REQUIRE(v1[ 7] ==  -7);
    REQUIRE(v1[ 8] ==  8); REQUIRE(v1[ 9] ==  -9);
    REQUIRE(v1[10] == 10); REQUIRE(v1[11] == -11);
    REQUIRE(v1[12] == 12); REQUIRE(v1[13] == -13);
    REQUIRE(v1[14] == 14); REQUIRE(v1[15] == -15);

    veci_i8x16_t v2{
        2, 4, 6, 8, 10, 12, 14, 16
    };
    REQUIRE(v2[ 0] ==  2); REQUIRE(v2[ 1] ==  4);
    REQUIRE(v2[ 2] ==  6); REQUIRE(v2[ 3] ==  8);
    REQUIRE(v2[ 4] == 10); REQUIRE(v2[ 5] == 12);
    REQUIRE(v2[ 6] == 14); REQUIRE(v2[ 7] == 16);
    REQUIRE(v2[ 8] ==  0); REQUIRE(v2[ 9] ==  0);
    REQUIRE(v2[10] ==  0); REQUIRE(v2[11] ==  0);
    REQUIRE(v2[12] ==  0); REQUIRE(v2[13] ==  0);
    REQUIRE(v2[14] ==  0); REQUIRE(v2[15] ==  0);

    veci_i8x16_t v3{
        2, 4, 6, 8, 10, 12, 14, 16,
        18, 20, 22, 24, 26, 28, 30,
        32, 34, 36, 38
    };
    REQUIRE(v3[ 0] ==  2); REQUIRE(v3[ 1] ==  4);
    REQUIRE(v3[ 2] ==  6); REQUIRE(v3[ 3] ==  8);
    REQUIRE(v3[ 4] == 10); REQUIRE(v3[ 5] == 12);
    REQUIRE(v3[ 6] == 14); REQUIRE(v3[ 7] == 16);
    REQUIRE(v3[ 8] == 18); REQUIRE(v3[ 9] == 20);
    REQUIRE(v3[10] == 22); REQUIRE(v3[11] == 24);
    REQUIRE(v3[12] == 26); REQUIRE(v3[13] == 28);
    REQUIRE(v3[14] == 30); REQUIRE(v3[15] == 32);

}



TEST_CASE("TestVecui8x16")
{
    using veci_ui8x16_t = math::veci_ui8x16_t;

    veci_ui8x16_t v1(
        0,  1,  2,   3,  4,   5,  6,  7,
        8,  9, 10,  11, 12,  13, 14, 15
    );
    veci_ui8x16_t v2;

            
    // member operators
            
    v2 = v1;
    veci_ui8x16_t v3(
            3,  4,  5,  6,  7,  8,  9, 10,
            10,  9,  8,  7,  6,  5,  4,  3
    );
    v2 += v3;
    for(int i = 0; i < 8; ++i)
        REQUIRE(v2[i] == v1[i]+v3[i]);

    v2 = v1;
    v2 -= v3;
    for(int i = 0; i < 8; ++i)
        REQUIRE(uint8_t(v2[i]) == uint8_t(v1[i]-v3[i]));

    v2 = v1;
    v2 &= v3;
    for(int i = 0; i < 8; ++i)
        REQUIRE(v2[i] == (v1[i]&v3[i]));

    v2 = v1;
    v2 |= v3;
    for(int i = 0; i < 8; ++i)
        REQUIRE(v2[i] == (v1[i]|v3[i]));
            
    v2 = v1;
    v2 ^= v3;
    for(int i = 0; i < 8; ++i)
        REQUIRE(v2[i] == (v1[i]^v3[i]));

    v2 = ~v1;
    for(int i = 0; i < 8; ++i)
        REQUIRE(uint8_t(v2[i]) == uint8_t(~v1[i]));



    veci_ui8x16_t v4(
            3,  4,  1,  4,  2,  5,  8,  3,
            10, 10,  5,  5,  7,  7,  3,  1
    );
    v2 = veci_ui8x16_t::min_(v1, v4);
    REQUIRE(v2[ 0] ==  0); REQUIRE(v2[ 1] ==  1);
    REQUIRE(v2[ 2] ==  1); REQUIRE(v2[ 3] ==  3);
    REQUIRE(v2[ 4] ==  2); REQUIRE(v2[ 5] ==  5);
    REQUIRE(v2[ 6] ==  6); REQUIRE(v2[ 7] ==  3);

    REQUIRE(v2[ 8] ==  8); REQUIRE(v2[ 9] ==  9);
    REQUIRE(v2[10] ==  5); REQUIRE(v2[11] ==  5);
    REQUIRE(v2[12] ==  7); REQUIRE(v2[13] ==  7);
    REQUIRE(v2[14] ==  3); REQUIRE(v2[15] ==  1);


    veci_ui8x16_t v5(
         4,  3,  4,  4,  2,  2,  8,  1,
        15,  1,  7,  8, 13,  5,  2,  0
    );
    v2 = veci_ui8x16_t::max_(v1, v5);
    REQUIRE(v2[ 0] ==  4); REQUIRE(v2[ 1] ==  3);
    REQUIRE(v2[ 2] ==  4); REQUIRE(v2[ 3] ==  4);
    REQUIRE(v2[ 4] ==  4); REQUIRE(v2[ 5] ==  5);
    REQUIRE(v2[ 6] ==  8); REQUIRE(v2[ 7] ==  7);

    REQUIRE(v2[ 8] == 15); REQUIRE(v2[ 9] ==  9);
    REQUIRE(v2[10] == 10); REQUIRE(v2[11] == 11);
    REQUIRE(v2[12] == 13); REQUIRE(v2[13] == 13);
    REQUIRE(v2[14] == 14); REQUIRE(v2[15] == 15);

    veci_ui8x16_t mask;
    uint8_t allbits_elem = 0xFF;
    uint8_t val;

#define TEST_MASK(shift)                                  \
    mask = veci_ui8x16_t::math_t::mask_zupper<(shift)>(); \
    val = allbits_elem >> (shift);                        \
    REQUIRE(mask[ 0] == val); REQUIRE(mask[ 1] == val);   \
    REQUIRE(mask[ 2] == val); REQUIRE(mask[ 3] == val);   \
    REQUIRE(mask[ 4] == val); REQUIRE(mask[ 5] == val);   \
    REQUIRE(mask[ 6] == val); REQUIRE(mask[ 7] == val);   \
    REQUIRE(mask[ 8] == val); REQUIRE(mask[ 9] == val);   \
    REQUIRE(mask[10] == val); REQUIRE(mask[11] == val);   \
    REQUIRE(mask[12] == val); REQUIRE(mask[13] == val);   \
    REQUIRE(mask[14] == val); REQUIRE(mask[15] == val);   \
    mask = veci_ui8x16_t::math_t::mask_zlower<(shift)>(); \
    val = allbits_elem << (shift);                        \
    REQUIRE(mask[ 0] == val); REQUIRE(mask[ 1] == val);   \
    REQUIRE(mask[ 2] == val); REQUIRE(mask[ 3] == val);   \
    REQUIRE(mask[ 4] == val); REQUIRE(mask[ 5] == val);   \
    REQUIRE(mask[ 6] == val); REQUIRE(mask[ 7] == val);   \
    REQUIRE(mask[ 8] == val); REQUIRE(mask[ 9] == val);   \
    REQUIRE(mask[10] == val); REQUIRE(mask[11] == val);   \
    REQUIRE(mask[12] == val); REQUIRE(mask[13] == val);   \
    REQUIRE(mask[14] == val); REQUIRE(mask[15] == val);

    TEST_MASK(1)
    TEST_MASK(2)
    TEST_MASK(3)
    TEST_MASK(4)
    TEST_MASK(5)
    TEST_MASK(6)
    TEST_MASK(7)

#undef TEST_MASK


#define TEST_MASK(bitno)                                \
    mask = veci_ui8x16_t::math_t::mask_1bit<bitno>();   \
    val = 1 << (bitno);                                 \
    REQUIRE(mask[ 0] == val); REQUIRE(mask[ 1] == val); \
    REQUIRE(mask[ 2] == val); REQUIRE(mask[ 3] == val); \
    REQUIRE(mask[ 4] == val); REQUIRE(mask[ 5] == val); \
    REQUIRE(mask[ 6] == val); REQUIRE(mask[ 7] == val); \
    REQUIRE(mask[ 8] == val); REQUIRE(mask[ 9] == val); \
    REQUIRE(mask[10] == val); REQUIRE(mask[11] == val); \
    REQUIRE(mask[12] == val); REQUIRE(mask[13] == val); \
    REQUIRE(mask[14] == val); REQUIRE(mask[15] == val);

    TEST_MASK(0)
    TEST_MASK(1)
    TEST_MASK(2)
    TEST_MASK(3)
    TEST_MASK(4)
    TEST_MASK(5)
    TEST_MASK(6)
    TEST_MASK(7)

#undef TEST_MASK

}


TEST_CASE("TestVecui8x16InitializerList")
{
    using veci_ui8x16_t = math::veci_ui8x16_t;

    veci_ui8x16_t v1{
        0,  1,  2,  3,  4,  5,  6,  7,
        8,  9, 10, 11, 12, 13, 14, 15
    };
    REQUIRE(v1[ 0] ==  0); REQUIRE(v1[ 1] ==  1);
    REQUIRE(v1[ 2] ==  2); REQUIRE(v1[ 3] ==  3);
    REQUIRE(v1[ 4] ==  4); REQUIRE(v1[ 5] ==  5);
    REQUIRE(v1[ 6] ==  6); REQUIRE(v1[ 7] ==  7);
    REQUIRE(v1[ 8] ==  8); REQUIRE(v1[ 9] ==  9);
    REQUIRE(v1[10] == 10); REQUIRE(v1[11] == 11);
    REQUIRE(v1[12] == 12); REQUIRE(v1[13] == 13);
    REQUIRE(v1[14] == 14); REQUIRE(v1[15] == 15);

    veci_ui8x16_t v2{
        2, 4, 6, 8, 10, 12, 14, 16
    };
    REQUIRE(v2[ 0] ==  2); REQUIRE(v2[ 1] ==  4);
    REQUIRE(v2[ 2] ==  6); REQUIRE(v2[ 3] ==  8);
    REQUIRE(v2[ 4] == 10); REQUIRE(v2[ 5] == 12);
    REQUIRE(v2[ 6] == 14); REQUIRE(v2[ 7] == 16);
    REQUIRE(v2[ 8] ==  0); REQUIRE(v2[ 9] ==  0);
    REQUIRE(v2[10] ==  0); REQUIRE(v2[11] ==  0);
    REQUIRE(v2[12] ==  0); REQUIRE(v2[13] ==  0);
    REQUIRE(v2[14] ==  0); REQUIRE(v2[15] ==  0);

    veci_ui8x16_t v3{
         2,  4,  6,  8, 10, 12, 14, 16,
        18, 20, 22, 24, 26, 28, 30,
        32, 34, 36, 38
    };
    REQUIRE(v3[ 0] ==  2); REQUIRE(v3[ 1] ==  4);
    REQUIRE(v3[ 2] ==  6); REQUIRE(v3[ 3] ==  8);
    REQUIRE(v3[ 4] == 10); REQUIRE(v3[ 5] == 12);
    REQUIRE(v3[ 6] == 14); REQUIRE(v3[ 7] == 16);
    REQUIRE(v3[ 8] == 18); REQUIRE(v3[ 9] == 20);
    REQUIRE(v3[10] == 22); REQUIRE(v3[11] == 24);
    REQUIRE(v3[12] == 26); REQUIRE(v3[13] == 28);
    REQUIRE(v3[14] == 30); REQUIRE(v3[15] == 32);

}



TEST_CASE("TestVeci16x8")
{
    using veci_i16x8_t = math::veci_i16x8_t;

    veci_i16x8_t v1(0, -1, 2, -3, 4, -5, 6, -7);
    veci_i16x8_t v2;

    v2 = v1;
    v2.abs_();
    REQUIRE(v2[0] == 0);
    REQUIRE(v2[1] == 1);
    REQUIRE(v2[2] == 2);
    REQUIRE(v2[3] == 3);
    REQUIRE(v2[4] == 4);
    REQUIRE(v2[5] == 5);
    REQUIRE(v2[6] == 6);
    REQUIRE(v2[7] == 7);

            
    // member operators
            
    v2 = v1;
    veci_i16x8_t v3(3, 4, 5, 6, 7, 8, 9, 10);
    v2 += v3;
    for(int i = 0; i < 8; ++i)
        REQUIRE(v2[i] == v1[i]+v3[i]);

    v2 = v1;
    v2 -= v3;
    for(int i = 0; i < 8; ++i)
        REQUIRE(v2[i] == v1[i]-v3[i]);

    v2 = v1;
    v2 &= v3;
    for(int i = 0; i < 8; ++i)
        REQUIRE(v2[i] == (v1[i]&v3[i]));

    v2 = v1;
    v2 |= v3;
    for(int i = 0; i < 8; ++i)
        REQUIRE(v2[i] == (v1[i]|v3[i]));
            
    v2 = v1;
    v2 ^= v3;
    for(int i = 0; i < 8; ++i)
        REQUIRE(v2[i] == (v1[i]^v3[i]));

    v2 = ~v1;
    for(int i = 0; i < 8; ++i)
        REQUIRE(v2[i] == ~v1[i]);

    veci_i16x8_t v4(3, 4, 1, -4, 2, -5, 8, -3);
    v2 = veci_i16x8_t::min_(v1, v4);
    REQUIRE(v2[0] ==  0);
    REQUIRE(v2[1] == -1);
    REQUIRE(v2[2] ==  1);
    REQUIRE(v2[3] == -4);
    REQUIRE(v2[4] ==  2);
    REQUIRE(v2[5] == -5);
    REQUIRE(v2[6] ==  6);
    REQUIRE(v2[7] == -7);

    veci_i16x8_t v5(-4, -3, 4, 4, 2, -2, 8, -1);
    v2 = veci_i16x8_t::max_(v1, v5);
    REQUIRE(v2[0] ==  0);
    REQUIRE(v2[1] == -1);
    REQUIRE(v2[2] ==  4);
    REQUIRE(v2[3] ==  4);
    REQUIRE(v2[4] ==  4);
    REQUIRE(v2[5] == -2);
    REQUIRE(v2[6] ==  8);
    REQUIRE(v2[7] == -1);


    veci_i16x8_t mask;
    uint16_t allbits_elem = 0xFFFF;
    uint16_t val;
    int16_t val_s;

#define TEST_MASK(shift)                                  \
    mask = veci_i16x8_t::math_t::mask_zupper<(shift)>();  \
    val = allbits_elem >> (shift);                        \
    val_s = int16_t(val);                                 \
    REQUIRE(mask[0] == val_s); REQUIRE(mask[1] == val_s); \
    REQUIRE(mask[2] == val_s); REQUIRE(mask[3] == val_s); \
    REQUIRE(mask[4] == val_s); REQUIRE(mask[5] == val_s); \
    REQUIRE(mask[6] == val_s); REQUIRE(mask[7] == val_s); \
    mask = veci_i16x8_t::math_t::mask_zlower<(shift)>();  \
    val = allbits_elem << (shift);                        \
    val_s = int16_t(val);                                 \
    REQUIRE(mask[0] == val_s); REQUIRE(mask[1] == val_s); \
    REQUIRE(mask[2] == val_s); REQUIRE(mask[3] == val_s); \
    REQUIRE(mask[4] == val_s); REQUIRE(mask[5] == val_s); \
    REQUIRE(mask[6] == val_s); REQUIRE(mask[7] == val_s);
#define TEST_MASK4(base_shift)    \
        TEST_MASK(base_shift + 0) \
        TEST_MASK(base_shift + 1) \
        TEST_MASK(base_shift + 2) \
        TEST_MASK(base_shift + 3)
            
    TEST_MASK4( 1)
    TEST_MASK4( 5)
    TEST_MASK4( 9)
    TEST_MASK(13)
    TEST_MASK(14)
    TEST_MASK(15)

#undef TEST_MASK4
#undef TEST_MASK


#define TEST_MASK(bitno)                                  \
    mask = veci_i16x8_t::math_t::mask_1bit<bitno>();      \
    val = 1 << (bitno);                                   \
    val_s = int16_t(val);                                 \
    REQUIRE(mask[0] == val_s); REQUIRE(mask[1] == val_s); \
    REQUIRE(mask[2] == val_s); REQUIRE(mask[3] == val_s); \
    REQUIRE(mask[4] == val_s); REQUIRE(mask[5] == val_s); \
    REQUIRE(mask[6] == val_s); REQUIRE(mask[7] == val_s);
#define TEST_MASK4(base_bitno)    \
        TEST_MASK(base_bitno + 0) \
        TEST_MASK(base_bitno + 1) \
        TEST_MASK(base_bitno + 2) \
        TEST_MASK(base_bitno + 3)
            
    TEST_MASK4( 0)
    TEST_MASK4( 4)
    TEST_MASK4( 8)
    TEST_MASK4(12)

#undef TEST_MASK4
#undef TEST_MASK

}


TEST_CASE("TestVeci16x8InitializerList")
{
    using veci_i16x8_t = math::veci_i16x8_t;

    veci_i16x8_t v1{0, -1, 2, -3, 4, -5, 6, -7};
    REQUIRE(v1[0] ==  0);
    REQUIRE(v1[1] == -1);
    REQUIRE(v1[2] ==  2);
    REQUIRE(v1[3] == -3);
    REQUIRE(v1[4] ==  4);
    REQUIRE(v1[5] == -5);
    REQUIRE(v1[6] ==  6);
    REQUIRE(v1[7] == -7);

    veci_i16x8_t v2{0, -1, 2, -3};
    REQUIRE(v2[0] ==  0);
    REQUIRE(v2[1] == -1);
    REQUIRE(v2[2] ==  2);
    REQUIRE(v2[3] == -3);
    REQUIRE(v2[4] ==  0);
    REQUIRE(v2[5] ==  0);
    REQUIRE(v2[6] ==  0);
    REQUIRE(v2[7] ==  0);

    veci_i16x8_t v3{0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11};
    REQUIRE(v3[0] ==  0);
    REQUIRE(v3[1] == -1);
    REQUIRE(v3[2] ==  2);
    REQUIRE(v3[3] == -3);
    REQUIRE(v3[4] ==  4);
    REQUIRE(v3[5] == -5);
    REQUIRE(v3[6] ==  6);
    REQUIRE(v3[7] == -7);

}


TEST_CASE("TestVecui16x8")
{
    using veci_ui16x8_t = math::veci_ui16x8_t;

    veci_ui16x8_t v1(0, 1, 2, 3, 4, 5, 6, 7);
    veci_ui16x8_t v2;

    // member operators
            
    v2 = v1;
    veci_ui16x8_t v3(1, 2, 3, 4, 5, 6, 7, 8);
    v2 += v3;
    for(int i = 0; i < 8; ++i)
        REQUIRE(v2[i] == v1[i]+v3[i]);

    v2 = v1;
    v2 -= v3;
    for(int i = 0; i < 8; ++i)
        REQUIRE(uint16_t(v2[i]) == uint16_t(v1[i]-v3[i]));

    v2 = v1;
    v2 &= v3;
    for(int i = 0; i < 8; ++i)
        REQUIRE(v2[i] == (v1[i]&v3[i]));

    v2 = v1;
    v2 |= v3;
    for(int i = 0; i < 8; ++i)
        REQUIRE(v2[i] == (v1[i]|v3[i]));
            
    v2 = v1;
    v2 ^= v3;
    for(int i = 0; i < 8; ++i)
        REQUIRE(v2[i] == (v1[i]^v3[i]));

    v2 = ~v1;
    for(int i = 0; i < 8; ++i)
        REQUIRE(uint16_t(v2[i]) == uint16_t(~v1[i]));

    veci_ui16x8_t v4(3, 4, 1, 4, 2, 5, 8, 3);
    v2 = veci_ui16x8_t::min_(v1, v4);
    REQUIRE(v2[0] == 0);
    REQUIRE(v2[1] == 1);
    REQUIRE(v2[2] == 1);
    REQUIRE(v2[3] == 3);
    REQUIRE(v2[4] == 2);
    REQUIRE(v2[5] == 5);
    REQUIRE(v2[6] == 6);
    REQUIRE(v2[7] == 3);

    veci_ui16x8_t v5(4, 3, 4, 4, 2, 2, 8, 1);
    v2 = veci_ui16x8_t::max_(v1, v5);
    REQUIRE(v2[0] == 4);
    REQUIRE(v2[1] == 3);
    REQUIRE(v2[2] == 4);
    REQUIRE(v2[3] == 4);
    REQUIRE(v2[4] == 4);
    REQUIRE(v2[5] == 5);
    REQUIRE(v2[6] == 8);
    REQUIRE(v2[7] == 7);


    veci_ui16x8_t mask;
    uint16_t allbits_elem = 0xFFFF;
    uint16_t val;

#define TEST_MASK(shift)                                  \
    mask = veci_ui16x8_t::math_t::mask_zupper<(shift)>(); \
    val = allbits_elem >> (shift);                        \
    REQUIRE(mask[0] == val); REQUIRE(mask[1] == val);     \
    REQUIRE(mask[2] == val); REQUIRE(mask[3] == val);     \
    REQUIRE(mask[4] == val); REQUIRE(mask[5] == val);     \
    REQUIRE(mask[6] == val); REQUIRE(mask[7] == val);     \
    mask = veci_ui16x8_t::math_t::mask_zlower<(shift)>(); \
    val = allbits_elem << (shift);                        \
    REQUIRE(mask[0] == val); REQUIRE(mask[1] == val);     \
    REQUIRE(mask[2] == val); REQUIRE(mask[3] == val);     \
    REQUIRE(mask[4] == val); REQUIRE(mask[5] == val);     \
    REQUIRE(mask[6] == val); REQUIRE(mask[7] == val);
#define TEST_MASK4(base_shift) \
    TEST_MASK(base_shift + 0)  \
    TEST_MASK(base_shift + 1)  \
    TEST_MASK(base_shift + 2)  \
    TEST_MASK(base_shift + 3)
            
    TEST_MASK4( 1)
    TEST_MASK4( 5)
    TEST_MASK4( 9)
    TEST_MASK(13)
    TEST_MASK(14)
    TEST_MASK(15)

#undef TEST_MASK4
#undef TEST_MASK


#define TEST_MASK(bitno)                                  \
        mask = veci_ui16x8_t::math_t::mask_1bit<bitno>(); \
        val = 1 << (bitno);                               \
        REQUIRE(mask[0] == val); REQUIRE(mask[1] == val); \
        REQUIRE(mask[2] == val); REQUIRE(mask[3] == val); \
        REQUIRE(mask[4] == val); REQUIRE(mask[5] == val); \
        REQUIRE(mask[6] == val); REQUIRE(mask[7] == val);
#define TEST_MASK4(base_bitno)    \
        TEST_MASK(base_bitno + 0) \
        TEST_MASK(base_bitno + 1) \
        TEST_MASK(base_bitno + 2) \
        TEST_MASK(base_bitno + 3)
            
    TEST_MASK4( 0)
    TEST_MASK4( 4)
    TEST_MASK4( 8)
    TEST_MASK4(12)

#undef TEST_MASK4
#undef TEST_MASK
    
}


TEST_CASE("TestVecui16x8InitializerList")
{
    using veci_ui16x8_t = math::veci_ui16x8_t;

    veci_ui16x8_t v1{0, 1, 2, 3, 4, 5, 6, 7};
    REQUIRE(v1[0] == 0);
    REQUIRE(v1[1] == 1);
    REQUIRE(v1[2] == 2);
    REQUIRE(v1[3] == 3);
    REQUIRE(v1[4] == 4);
    REQUIRE(v1[5] == 5);
    REQUIRE(v1[6] == 6);
    REQUIRE(v1[7] == 7);

    veci_ui16x8_t v2{0, 1, 2, 3};
    REQUIRE(v2[0] == 0);
    REQUIRE(v2[1] == 1);
    REQUIRE(v2[2] == 2);
    REQUIRE(v2[3] == 3);
    REQUIRE(v2[4] == 0);
    REQUIRE(v2[5] == 0);
    REQUIRE(v2[6] == 0);
    REQUIRE(v2[7] == 0);

    veci_ui16x8_t v3{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    REQUIRE(v3[0] == 0);
    REQUIRE(v3[1] == 1);
    REQUIRE(v3[2] == 2);
    REQUIRE(v3[3] == 3);
    REQUIRE(v3[4] == 4);
    REQUIRE(v3[5] == 5);
    REQUIRE(v3[6] == 6);
    REQUIRE(v3[7] == 7);

}


TEST_CASE("TestVeci32x4")
{
    using veci_i32x4_t = math::veci_i32x4_t;

    veci_i32x4_t v1(0, -1, 2, -3);
    veci_i32x4_t v2;

#define TEST_SWIZZLE4(a,b,c,d,a_,b_,c_,d_) \
    v2 = a##b##c##d(v1);                   \
    REQUIRE(v2[0] == a_);                  \
    REQUIRE(v2[1] == b_);                  \
    REQUIRE(v2[2] == c_);                  \
    REQUIRE(v2[3] == d_);
#define TEST_SWIZZLE3(a,b,c,a_,b_,c_)  \
    TEST_SWIZZLE4(a,b,c,x,a_,b_,c_, 0) \
    TEST_SWIZZLE4(a,b,c,y,a_,b_,c_,-1) \
    TEST_SWIZZLE4(a,b,c,z,a_,b_,c_, 2) \
    TEST_SWIZZLE4(a,b,c,w,a_,b_,c_,-3)
#define TEST_SWIZZLE2(a,b,a_,b_)  \
    TEST_SWIZZLE3(a,b,x,a_,b_, 0) \
    TEST_SWIZZLE3(a,b,y,a_,b_,-1) \
    TEST_SWIZZLE3(a,b,z,a_,b_, 2) \
    TEST_SWIZZLE3(a,b,w,a_,b_,-3)
#define TEST_SWIZZLE1(a,a_)  \
    TEST_SWIZZLE2(a,x,a_, 0) \
    TEST_SWIZZLE2(a,y,a_,-1) \
    TEST_SWIZZLE2(a,z,a_, 2) \
    TEST_SWIZZLE2(a,w,a_,-3)

    // TODO: prob with NEON here, but the first test only (v2[0] == 0)
    //       expansion: -3 == 0
    //       for v2 = xxxx(v1)
    // first test only: Catch does not run further test if a REQUIRE() fails

    TEST_SWIZZLE1(x, 0)
    TEST_SWIZZLE1(y, -1)
    TEST_SWIZZLE1(z, 2)
    TEST_SWIZZLE1(w, -3)

#undef TEST_SWIZZLE1
#undef TEST_SWIZZLE2
#undef TEST_SWIZZLE3
#undef TEST_SWIZZLE4

    
    
    v2 = v1;
    v2.abs_();
    
    REQUIRE(v2[0] == 0);
    REQUIRE(v2[1] == 1);
    REQUIRE(v2[2] == 2);
    REQUIRE(v2[3] == 3);
    
    // member operators
    
    v2 = v1;
    veci_i32x4_t v3(3, 4, 5, 6);
    v2 += v3;
    REQUIRE(v2[0] == 3);
    REQUIRE(v2[1] == 3);
    REQUIRE(v2[2] == 7);
    REQUIRE(v2[3] == 3);

    v2 = v1;
    v2 -= v3;
    REQUIRE(v2[0] == -3);
    REQUIRE(v2[1] == -5);
    REQUIRE(v2[2] == -3);
    REQUIRE(v2[3] == -9);

    v2 = v1;
    v2 &= v3;
    REQUIRE(v2[0] ==  0);
    REQUIRE(v2[1] ==  4);
    REQUIRE(v2[2] ==  0);
    REQUIRE(v2[3] ==  4);

    v2 = v1;
    v2 |= v3;
    REQUIRE(v2[0] ==  3);
    REQUIRE(v2[1] == -1);
    REQUIRE(v2[2] ==  7);
    REQUIRE(v2[3] == -1);

    v2 = v1;
    v2 ^= v3;
    REQUIRE(v2[0] ==  3);
    REQUIRE(v2[1] == -5);
    REQUIRE(v2[2] ==  7);
    REQUIRE(v2[3] == -5);


    v2 = ~v1;
    REQUIRE(v2[0] == -1);
    REQUIRE(v2[1] ==  0);
    REQUIRE(v2[2] == -3);
    REQUIRE(v2[3] ==  2);


    veci_i32x4_t v4(3, 4, 1, -4);
    v2 = veci_i32x4_t::min_(v1, v4);
    REQUIRE(v2[0] ==  0);
    REQUIRE(v2[1] == -1);
    REQUIRE(v2[2] ==  1);
    REQUIRE(v2[3] == -4);

    veci_i32x4_t v5(-4, -3, 4, 4);
    v2 = veci_i32x4_t::max_(v1, v5);
    REQUIRE(v2[0] ==  0);
    REQUIRE(v2[1] == -1);
    REQUIRE(v2[2] ==  4);
    REQUIRE(v2[3] ==  4);


    veci_i32x4_t mask;
    uint32_t allbits_elem = 0xFFFFFFFF;
    uint32_t val;
    int32_t val_s;

#define TEST_MASK(shift)                                  \
    mask = veci_i32x4_t::math_t::mask_zupper<(shift)>();  \
    val = allbits_elem >> (shift);                        \
    val_s = int32_t(val);                                 \
    REQUIRE(mask[0] == val_s); REQUIRE(mask[1] == val_s); \
    REQUIRE(mask[2] == val_s); REQUIRE(mask[3] == val_s); \
    mask = veci_i32x4_t::math_t::mask_zlower<(shift)>();  \
    val = allbits_elem << (shift);                        \
    val_s = int32_t(val);                                 \
    REQUIRE(mask[0] == val_s); REQUIRE(mask[1] == val_s); \
    REQUIRE(mask[2] == val_s); REQUIRE(mask[3] == val_s);
#define TEST_MASK4(base_shift) \
    TEST_MASK(base_shift + 0)  \
    TEST_MASK(base_shift + 1)  \
    TEST_MASK(base_shift + 2)  \
    TEST_MASK(base_shift + 3)
            
    TEST_MASK4( 1)
    TEST_MASK4( 5)
    TEST_MASK4( 9)
    TEST_MASK4(13)
    TEST_MASK4(17)
    TEST_MASK4(21)
    TEST_MASK4(25)
    TEST_MASK(29)
    TEST_MASK(30)
    TEST_MASK(31)

#undef TEST_MASK4
#undef TEST_MASK


#define TEST_MASK(bitno)                                  \
    mask = veci_i32x4_t::math_t::mask_1bit<bitno>();      \
    val = 1 << (bitno);                                   \
    val_s = int32_t(val);                                 \
    REQUIRE(mask[0] == val_s); REQUIRE(mask[1] == val_s); \
    REQUIRE(mask[2] == val_s); REQUIRE(mask[3] == val_s);
#define TEST_MASK4(base_bitno) \
    TEST_MASK(base_bitno + 0)  \
    TEST_MASK(base_bitno + 1)  \
    TEST_MASK(base_bitno + 2)  \
    TEST_MASK(base_bitno + 3)
            
    TEST_MASK4( 0)
    TEST_MASK4( 4)
    TEST_MASK4( 8)
    TEST_MASK4(12)
    TEST_MASK4(16)
    TEST_MASK4(20)
    TEST_MASK4(24)
    TEST_MASK4(28)

#undef TEST_MASK4
#undef TEST_MASK

}


TEST_CASE("TestVeci32x4InitializerList")
{
    using veci_i32x4_t = math::veci_i32x4_t;

    veci_i32x4_t v1{0, -1, 2, -3};
    REQUIRE(v1[0] ==  0);
    REQUIRE(v1[1] == -1);
    REQUIRE(v1[2] ==  2);
    REQUIRE(v1[3] == -3);
    
    veci_i32x4_t v2{0, -1};
    REQUIRE(v2[0] ==  0);
    REQUIRE(v2[1] == -1);
    REQUIRE(v2[2] ==  0);
    REQUIRE(v2[3] ==  0);

    veci_i32x4_t v3{0, -1, 2, -3, 4, -5};
    REQUIRE(v3[0] ==  0);
    REQUIRE(v3[1] == -1);
    REQUIRE(v3[2] ==  2);
    REQUIRE(v3[3] == -3);
}


TEST_CASE("TestVecui32x4")
{
    using veci_ui32x4_t = math::veci_ui32x4_t;

    veci_ui32x4_t v1(0, 1, 2, 3);
    veci_ui32x4_t v2;

#define TEST_SWIZZLE4(a,b,c,d,a_,b_,c_,d_) \
    v2 = a##b##c##d(v1);                   \
    REQUIRE(v2[0] == a_);                  \
    REQUIRE(v2[1] == b_);                  \
    REQUIRE(v2[2] == c_);                  \
    REQUIRE(v2[3] == d_);
#define TEST_SWIZZLE3(a,b,c,a_,b_,c_) \
    TEST_SWIZZLE4(a,b,c,x,a_,b_,c_,0) \
    TEST_SWIZZLE4(a,b,c,y,a_,b_,c_,1) \
    TEST_SWIZZLE4(a,b,c,z,a_,b_,c_,2) \
    TEST_SWIZZLE4(a,b,c,w,a_,b_,c_,3)
#define TEST_SWIZZLE2(a,b,a_,b_) \
    TEST_SWIZZLE3(a,b,x,a_,b_,0) \
    TEST_SWIZZLE3(a,b,y,a_,b_,1) \
    TEST_SWIZZLE3(a,b,z,a_,b_,2) \
    TEST_SWIZZLE3(a,b,w,a_,b_,3)
#define TEST_SWIZZLE1(a,a_) \
    TEST_SWIZZLE2(a,x,a_,0) \
    TEST_SWIZZLE2(a,y,a_,1) \
    TEST_SWIZZLE2(a,z,a_,2) \
    TEST_SWIZZLE2(a,w,a_,3)

    TEST_SWIZZLE1(x, 0)
    TEST_SWIZZLE1(y, 1)
    TEST_SWIZZLE1(z, 2)
    TEST_SWIZZLE1(w, 3)

#undef TEST_SWIZZLE1
#undef TEST_SWIZZLE2
#undef TEST_SWIZZLE3
#undef TEST_SWIZZLE4

    
    // member operators
    
    v2 = v1;
    veci_ui32x4_t v3(3, 4, 5, 6);

    v2 += v3;
    REQUIRE(v2[0] == 3);
    REQUIRE(v2[1] == 5);
    REQUIRE(v2[2] == 7);
    REQUIRE(v2[3] == 9);

    v2 = v1;
    v2 -= v3;
    REQUIRE(v2[0] == 0xFFFFFFFD);
    REQUIRE(v2[1] == 0xFFFFFFFD);
    REQUIRE(v2[2] == 0xFFFFFFFD);
    REQUIRE(v2[3] == 0xFFFFFFFD);

    v2 = v1;
    v2 &= v3;
    REQUIRE(v2[0] ==  0);
    REQUIRE(v2[1] ==  0);
    REQUIRE(v2[2] ==  0);
    REQUIRE(v2[3] ==  2);

    v2 = v1;
    v2 |= v3;
    REQUIRE(v2[0] ==  3);
    REQUIRE(v2[1] ==  5);
    REQUIRE(v2[2] ==  7);
    REQUIRE(v2[3] ==  7);

    v2 = v1;
    v2 ^= v3;
    REQUIRE(v2[0] ==  3);
    REQUIRE(v2[1] ==  5);
    REQUIRE(v2[2] ==  7);
    REQUIRE(v2[3] ==  5);

    v2 = ~v1;
    REQUIRE(v2[0] == 0xFFFFFFFF);
    REQUIRE(v2[1] == 0xFFFFFFFE);
    REQUIRE(v2[2] == 0xFFFFFFFD);
    REQUIRE(v2[3] == 0xFFFFFFFC);
    //veci_ui32x4_t v1(0, 1, 2, 3);
    //veci_ui32x4_t v3(3, 4, 5, 6);
            

    veci_ui32x4_t mask;
    uint32_t allbits_elem = 0xFFFFFFFF;
    uint32_t val;

#define TEST_MASK(shift)                                  \
    mask = veci_ui32x4_t::math_t::mask_zupper<(shift)>(); \
    val = allbits_elem >> (shift);                        \
    REQUIRE(mask[0] == val); REQUIRE(mask[1] == val);     \
    REQUIRE(mask[2] == val); REQUIRE(mask[3] == val);     \
    mask = veci_ui32x4_t::math_t::mask_zlower<(shift)>(); \
    val = allbits_elem << (shift);                        \
    REQUIRE(mask[0] == val); REQUIRE(mask[1] == val);     \
    REQUIRE(mask[2] == val); REQUIRE(mask[3] == val);
#define TEST_MASK4(base_shift) \
    TEST_MASK(base_shift + 0)  \
    TEST_MASK(base_shift + 1)  \
    TEST_MASK(base_shift + 2)  \
    TEST_MASK(base_shift + 3)
            
    TEST_MASK4( 1)
    TEST_MASK4( 5)
    TEST_MASK4( 9)
    TEST_MASK4(13)
    TEST_MASK4(17)
    TEST_MASK4(21)
    TEST_MASK4(25)
    TEST_MASK(29)
    TEST_MASK(30)
    TEST_MASK(31)

#undef TEST_MASK4
#undef TEST_MASK


#define TEST_MASK(bitno)                              \
    mask = veci_ui32x4_t::math_t::mask_1bit<bitno>(); \
    val = 1 << (bitno);                               \
    REQUIRE(mask[0] == val); REQUIRE(mask[1] == val); \
    REQUIRE(mask[2] == val); REQUIRE(mask[3] == val);
#define TEST_MASK4(base_bitno)    \
        TEST_MASK(base_bitno + 0) \
        TEST_MASK(base_bitno + 1) \
        TEST_MASK(base_bitno + 2) \
        TEST_MASK(base_bitno + 3)
            
    TEST_MASK4( 0)
    TEST_MASK4( 4)
    TEST_MASK4( 8)
    TEST_MASK4(12)
    TEST_MASK4(16)
    TEST_MASK4(20)
    TEST_MASK4(24)
    TEST_MASK4(28)

#undef TEST_MASK4
#undef TEST_MASK

}


TEST_CASE("TestVecui32x4InitializerList")
{
    using veci_ui32x4_t = math::veci_ui32x4_t;

    veci_ui32x4_t v1{0, 1, 2, 3};
    REQUIRE(v1[0] == 0);
    REQUIRE(v1[1] == 1);
    REQUIRE(v1[2] == 2);
    REQUIRE(v1[3] == 3);

    veci_ui32x4_t v2{1, 2};
    REQUIRE(v2[0] == 1);
    REQUIRE(v2[1] == 2);
    REQUIRE(v2[2] == 0);
    REQUIRE(v2[3] == 0);

    veci_ui32x4_t v3{1, 2, 3, 4, 5, 6};
    REQUIRE(v3[0] == 1);
    REQUIRE(v3[1] == 2);
    REQUIRE(v3[2] == 3);
    REQUIRE(v3[3] == 4);
}


TEST_CASE("TestVeci64x2")
{
    using veci_i64x2_t = math::veci_i64x2_t;

    veci_i64x2_t v1(0x0A0A0A0A0A0A0A0ALL, 0x5050505050505050LL);
    veci_i64x2_t v2(xx(v1));

    REQUIRE(v2[0] == 0x0A0A0A0A0A0A0A0ALL);
    REQUIRE(v2[1] == 0x0A0A0A0A0A0A0A0ALL);

    v2 = xy(v1);
    REQUIRE(v2[0] == 0x0A0A0A0A0A0A0A0ALL);
    REQUIRE(v2[1] == 0x5050505050505050LL);

    v2 = yx(v1);
    REQUIRE(v2[0] == 0x5050505050505050LL);
    REQUIRE(v2[1] == 0x0A0A0A0A0A0A0A0ALL);

    v2 = yy(v1);
    REQUIRE(v2[0] == 0x5050505050505050LL);
    REQUIRE(v2[1] == 0x5050505050505050LL);

    veci_i64x2_t v3(0x0303030303030303LL, 0x7070707070707070LL);

    v2 = xx(v1, v3);
    REQUIRE(v2[0] == 0x0A0A0A0A0A0A0A0ALL);
    REQUIRE(v2[1] == 0x0303030303030303LL);

    v2 = xy(v1, v3);
    REQUIRE(v2[0] == 0x0A0A0A0A0A0A0A0ALL);
    REQUIRE(v2[1] == 0x7070707070707070LL);

    v2 = yx(v1, v3);
    REQUIRE(v2[0] == 0x5050505050505050LL);
    REQUIRE(v2[1] == 0x0303030303030303LL);

    v2 = yy(v1, v3);
    REQUIRE(v2[0] == 0x5050505050505050LL);
    REQUIRE(v2[1] == 0x7070707070707070LL);


    veci_i64x2_t v4(0x0102030405060708LL, 0x090A0B0C0D0E0F10LL);
    veci_i64x2_t v5;

    v5 = xx(v4);
    REQUIRE(v5[0] == 0x0102030405060708LL);
    REQUIRE(v5[1] == 0x0102030405060708LL);

    v5 = xy(v4);
    REQUIRE(v5[0] == 0x0102030405060708LL);
    REQUIRE(v5[1] == 0x090A0B0C0D0E0F10LL);

    v5 = yx(v4);
    REQUIRE(v5[0] == 0x090A0B0C0D0E0F10LL);
    REQUIRE(v5[1] == 0x0102030405060708LL);

    v5 = yy(v4);
    REQUIRE(v5[0] == 0x090A0B0C0D0E0F10LL);
    REQUIRE(v5[1] == 0x090A0B0C0D0E0F10LL);


    veci_i64x2_t v6(0x1020304050607080LL, 0x00A0B0C0D0E0F000LL);

    v5 = xx(v4, v6);
    REQUIRE(v5[0] == 0x0102030405060708LL);
    REQUIRE(v5[1] == 0x1020304050607080LL);

    v5 = xy(v4, v6);
    REQUIRE(v5[0] == 0x0102030405060708LL);
    REQUIRE(v5[1] == 0x00A0B0C0D0E0F000LL);

    v5 = yx(v4, v6);
    REQUIRE(v5[0] == 0x090A0B0C0D0E0F10LL);
    REQUIRE(v5[1] == 0x1020304050607080LL);

    v5 = yy(v4, v6);
    REQUIRE(v5[0] == 0x090A0B0C0D0E0F10LL);
    REQUIRE(v5[1] == 0x00A0B0C0D0E0F000LL);


    veci_i64x2_t v7(-3LL, 4LL);
    veci_i64x2_t v8;

    v8 = v7;
    v8.abs_();
    REQUIRE(v8[0] == 3LL);
    REQUIRE(v8[1] == 4LL);

    // member operators
    veci_i64x2_t v9(1LL, 2LL);
    v8 = v7;
    v8 += v9;
    REQUIRE(v8[0] == -2LL);
    REQUIRE(v8[1] ==  6LL);

    v8 = v7;
    v8 -= v9;
    REQUIRE(v8[0] == -4LL);
    REQUIRE(v8[1] ==  2LL);

    v8 = v7;
    v8 &= v9;
    REQUIRE(v8[0] ==  1LL);
    REQUIRE(v8[1] ==  0LL);

    v8 = v7;
    v8 |= v9;
    REQUIRE(v8[0] == -3LL);
    REQUIRE(v8[1] ==  6LL);

    v8 = v7;
    v8 ^= v9;
    REQUIRE(v8[0] == -4LL);
    REQUIRE(v8[1] ==  6LL);

    v8 = ~v7;
    REQUIRE(v8[0] ==  2LL);
    REQUIRE(v8[1] == -5LL);

    v8 = veci_i64x2_t::min_(v7, v9);
    REQUIRE(v8[0] == -3LL);
    REQUIRE(v8[1] ==  2LL);

    v8 = veci_i64x2_t::max_(v7, v9);
    REQUIRE(v8[0] ==  1LL);
    REQUIRE(v8[1] ==  4LL);

    veci_i64x2_t mask;
    uint64_t allbits_elem = 0xFFFFFFFFFFFFFFFF;
    uint64_t val;
    int64_t val_s;

#define TEST_MASK(shift)                                  \
    mask = veci_i64x2_t::math_t::mask_zupper<(shift)>();  \
    val = allbits_elem >> (shift);                        \
    val_s = int64_t(val);                                 \
    REQUIRE(mask[0] == val_s); REQUIRE(mask[1] == val_s); \
    mask = veci_i64x2_t::math_t::mask_zlower<(shift)>();  \
    val = allbits_elem << (shift);                        \
    val_s = int64_t(val);                                 \
    REQUIRE(mask[0] == val_s); REQUIRE(mask[1] == val_s);
#define TEST_MASK8(base_shift) \
    TEST_MASK(base_shift + 0)  \
    TEST_MASK(base_shift + 1)  \
    TEST_MASK(base_shift + 2)  \
    TEST_MASK(base_shift + 3)  \
    TEST_MASK(base_shift + 4)  \
    TEST_MASK(base_shift + 5)  \
    TEST_MASK(base_shift + 6)  \
    TEST_MASK(base_shift + 7)
            
    TEST_MASK8( 1)
    TEST_MASK8( 9)
    TEST_MASK8(17)
    TEST_MASK8(25)
    TEST_MASK8(33)
    TEST_MASK8(41)
    TEST_MASK8(49)
    TEST_MASK(57)
    TEST_MASK(58)
    TEST_MASK(59)
    TEST_MASK(60)
    TEST_MASK(61)
    TEST_MASK(62)
    TEST_MASK(63)

#undef TEST_MASK8
#undef TEST_MASK


#define TEST_MASK(bitno)                                  \
    mask = veci_i64x2_t::math_t::mask_1bit<bitno>();      \
    val = 1ULL << (bitno);                                \
    val_s = int64_t(val);                                 \
    REQUIRE(mask[0] == val_s); REQUIRE(mask[1] == val_s);
#define TEST_MASK8(base_bitno) \
    TEST_MASK(base_bitno + 0)  \
    TEST_MASK(base_bitno + 1)  \
    TEST_MASK(base_bitno + 2)  \
    TEST_MASK(base_bitno + 3)  \
    TEST_MASK(base_bitno + 4)  \
    TEST_MASK(base_bitno + 5)  \
    TEST_MASK(base_bitno + 6)  \
    TEST_MASK(base_bitno + 7)
            
    TEST_MASK8( 0)
    TEST_MASK8( 8)
    TEST_MASK8(16)
    TEST_MASK8(24)
    TEST_MASK8(32)
    TEST_MASK8(40)
    TEST_MASK8(48)
    TEST_MASK8(56)

#undef TEST_MASK8
#undef TEST_MASK


    veci_i64x2_t vs1(0LL, 1LL);
    veci_i64x2_t vs2;

    vs2 = xx(vs1);
    REQUIRE(vs2[0] == 0LL);
    REQUIRE(vs2[1] == 0LL);
    vs2 = xy(vs1);
    REQUIRE(vs2[0] == 0LL);
    REQUIRE(vs2[1] == 1LL);
    vs2 = yx(vs1);
    REQUIRE(vs2[0] == 1LL);
    REQUIRE(vs2[1] == 0LL);
    vs2 = yy(vs1);
    REQUIRE(vs2[0] == 1LL);
    REQUIRE(vs2[1] == 1LL);
}

TEST_CASE("TestVeci64x2InitializerList")
{
    using veci_i64x2_t = math::veci_i64x2_t;

    veci_i64x2_t v1{1LL, -2LL};
    REQUIRE(v1[0] == 1LL);
    REQUIRE(v1[1] == -2LL);

    veci_i64x2_t v2{1LL};
    REQUIRE(v2[0] == 1LL);
    REQUIRE(v2[1] == 0LL);

    veci_i64x2_t v3{1LL, -2LL, 3LL};
    REQUIRE(v3[0] == 1LL);
    REQUIRE(v3[1] == -2LL);
}



TEST_CASE("TestVecui64x2")
{
    using veci_ui64x2_t = math::veci_ui64x2_t;

    veci_ui64x2_t v1(0x0A0A0A0A0A0A0A0AULL, 0x5050505050505050ULL);
    veci_ui64x2_t v2(xx(v1));

    REQUIRE(v2[0] == 0x0A0A0A0A0A0A0A0AULL);
    REQUIRE(v2[1] == 0x0A0A0A0A0A0A0A0AULL);

    v2 = xy(v1);
    REQUIRE(v2[0] == 0x0A0A0A0A0A0A0A0AULL);
    REQUIRE(v2[1] == 0x5050505050505050ULL);

    v2 = yx(v1);
    REQUIRE(v2[0] == 0x5050505050505050ULL);
    REQUIRE(v2[1] == 0x0A0A0A0A0A0A0A0AULL);

    v2 = yy(v1);
    REQUIRE(v2[0] == 0x5050505050505050ULL);
    REQUIRE(v2[1] == 0x5050505050505050ULL);

    veci_ui64x2_t v3(0x0303030303030303ULL, 0x7070707070707070ULL);

    v2 = xx(v1, v3);
    REQUIRE(v2[0] == 0x0A0A0A0A0A0A0A0AULL);
    REQUIRE(v2[1] == 0x0303030303030303ULL);

    v2 = xy(v1, v3);
    REQUIRE(v2[0] == 0x0A0A0A0A0A0A0A0AULL);
    REQUIRE(v2[1] == 0x7070707070707070ULL);

    v2 = yx(v1, v3);
    REQUIRE(v2[0] == 0x5050505050505050ULL);
    REQUIRE(v2[1] == 0x0303030303030303ULL);

    v2 = yy(v1, v3);
    REQUIRE(v2[0] == 0x5050505050505050ULL);
    REQUIRE(v2[1] == 0x7070707070707070ULL);


    veci_ui64x2_t v4(0x0102030405060708ULL, 0x090A0B0C0D0E0F10ULL);
    veci_ui64x2_t v5;

    v5 = xx(v4);
    REQUIRE(v5[0] == 0x0102030405060708ULL);
    REQUIRE(v5[1] == 0x0102030405060708ULL);

    v5 = xy(v4);
    REQUIRE(v5[0] == 0x0102030405060708ULL);
    REQUIRE(v5[1] == 0x090A0B0C0D0E0F10ULL);

    v5 = yx(v4);
    REQUIRE(v5[0] == 0x090A0B0C0D0E0F10ULL);
    REQUIRE(v5[1] == 0x0102030405060708ULL);

    v5 = yy(v4);
    REQUIRE(v5[0] == 0x090A0B0C0D0E0F10ULL);
    REQUIRE(v5[1] == 0x090A0B0C0D0E0F10ULL);


    veci_ui64x2_t v6(0x1020304050607080ULL, 0x90A0B0C0D0E0F000ULL);

    v5 = xx(v4, v6);
    REQUIRE(v5[0] == 0x0102030405060708ULL);
    REQUIRE(v5[1] == 0x1020304050607080ULL);

    v5 = xy(v4, v6);
    REQUIRE(v5[0] == 0x0102030405060708ULL);
    REQUIRE(v5[1] == 0x90A0B0C0D0E0F000ULL);

    v5 = yx(v4, v6);
    REQUIRE(v5[0] == 0x090A0B0C0D0E0F10ULL);
    REQUIRE(v5[1] == 0x1020304050607080ULL);

    v5 = yy(v4, v6);
    REQUIRE(v5[0] == 0x090A0B0C0D0E0F10ULL);
    REQUIRE(v5[1] == 0x90A0B0C0D0E0F000ULL);


    // member operators
    veci_ui64x2_t v7(1ULL, 4ULL);
    veci_ui64x2_t v9(2ULL, 3ULL);

    veci_ui64x2_t v8;
    v8 = v7;
    v8 += v9;
    REQUIRE(v8[0] == v7[0]+v9[0]);
    REQUIRE(v8[1] == v7[1]+v9[1]);

    v8 = v7;
    v8 -= v9;
    REQUIRE(v8[0] == v7[0]-v9[0]);
    REQUIRE(v8[1] == v7[1]-v9[1]);

    v8 = v7;
    v8 &= v9;
    REQUIRE(v8[0] == (v7[0]&v9[0]));
    REQUIRE(v8[1] == (v7[1]&v9[1]));

    v8 = v7;
    v8 |= v9;
    REQUIRE(v8[0] == (v7[0]|v9[0]));
    REQUIRE(v8[1] == (v7[1]|v9[1]));

    v8 = v7;
    v8 ^= v9;
    REQUIRE(v8[0] == (v7[0]^v9[0]));
    REQUIRE(v8[1] == (v7[1]^v9[1]));

    v8 = ~v7;
    REQUIRE(v8[0] == ~v7[0]);
    REQUIRE(v8[1] == ~v7[1]);


    v8 = veci_ui64x2_t::min_(v7, v9);
    REQUIRE(v8[0] ==  1ULL);
    REQUIRE(v8[1] ==  3ULL);

    v8 = veci_ui64x2_t::max_(v7, v9);
    REQUIRE(v8[0] ==  2ULL);
    REQUIRE(v8[1] ==  4ULL);


    veci_ui64x2_t mask;
    uint64_t allbits_elem = 0xFFFFFFFFFFFFFFFF;
    uint64_t val;

#define TEST_MASK(shift)                                  \
    mask = veci_ui64x2_t::math_t::mask_zupper<(shift)>(); \
    val = allbits_elem >> (shift);                        \
    REQUIRE(mask[0] == val); REQUIRE(mask[1] == val);     \
    mask = veci_ui64x2_t::math_t::mask_zlower<(shift)>(); \
    val = allbits_elem << (shift);                        \
    REQUIRE(mask[0] == val); REQUIRE(mask[1] == val);
#define TEST_MASK8(base_shift) \
    TEST_MASK(base_shift + 0)  \
    TEST_MASK(base_shift + 1)  \
    TEST_MASK(base_shift + 2)  \
    TEST_MASK(base_shift + 3)  \
    TEST_MASK(base_shift + 4)  \
    TEST_MASK(base_shift + 5)  \
    TEST_MASK(base_shift + 6)  \
    TEST_MASK(base_shift + 7)
    
    TEST_MASK8( 1)
    TEST_MASK8( 9)
    TEST_MASK8(17)
    TEST_MASK8(25)
    TEST_MASK8(33)
    TEST_MASK8(41)
    TEST_MASK8(49)
    TEST_MASK(57)
    TEST_MASK(58)
    TEST_MASK(59)
    TEST_MASK(60)
    TEST_MASK(61)
    TEST_MASK(62)
    TEST_MASK(63)

#undef TEST_MASK8
#undef TEST_MASK


#define TEST_MASK(bitno)                              \
    mask = veci_ui64x2_t::math_t::mask_1bit<bitno>(); \
    val = 1ULL << (bitno);                            \
    REQUIRE(mask[0] == val); REQUIRE(mask[1] == val);
#define TEST_MASK8(base_bitno) \
    TEST_MASK(base_bitno + 0)  \
    TEST_MASK(base_bitno + 1)  \
    TEST_MASK(base_bitno + 2)  \
    TEST_MASK(base_bitno + 3)  \
    TEST_MASK(base_bitno + 4)  \
    TEST_MASK(base_bitno + 5)  \
    TEST_MASK(base_bitno + 6)  \
    TEST_MASK(base_bitno + 7)
            
    TEST_MASK8( 0)
    TEST_MASK8( 8)
    TEST_MASK8(16)
    TEST_MASK8(24)
    TEST_MASK8(32)
    TEST_MASK8(40)
    TEST_MASK8(48)
    TEST_MASK8(56)

#undef TEST_MASK8
#undef TEST_MASK


    veci_ui64x2_t vs1(0ULL, 1ULL);
    veci_ui64x2_t vs2;

    vs2 = xx(vs1);
    REQUIRE(vs2[0] == 0ULL);
    REQUIRE(vs2[1] == 0ULL);
    vs2 = xy(vs1);
    REQUIRE(vs2[0] == 0ULL);
    REQUIRE(vs2[1] == 1ULL);
    vs2 = yx(vs1);
    REQUIRE(vs2[0] == 1ULL);
    REQUIRE(vs2[1] == 0ULL);
    vs2 = yy(vs1);
    REQUIRE(vs2[0] == 1ULL);
    REQUIRE(vs2[1] == 1ULL);

}

TEST_CASE("TestVecui64x2InitializerList")
{
    using veci_ui64x2_t = math::veci_ui64x2_t;

    veci_ui64x2_t v1{1ULL, 2ULL};
    REQUIRE(v1[0] == 1ULL);
    REQUIRE(v1[1] == 2ULL);

    veci_ui64x2_t v2{1ULL};
    REQUIRE(v2[0] == 1ULL);
    REQUIRE(v2[1] == 0ULL);

    veci_ui64x2_t v3{1ULL, 2ULL, 3ULL};
    REQUIRE(v3[0] == 1ULL);
    REQUIRE(v3[1] == 2ULL);
}



// tests for scalar-i-tools.h

#include "scalar-i-tools.h"

TEST_CASE("TestScalarITools1")
{
    // XXXX XXXX XXXX XXXX XXXX XXXX AAAA XXXX
    uint32_t val = 0x00000050; // XXXX XXXX XXXX XXXX XXXX XXXX 0101 XXXX
    REQUIRE((scalari::extrbf_c_normal<4, 4>(val)) == 0x5);
    REQUIRE((scalari::extrbf_c_normal<0, 1>(val)) == 0);
    REQUIRE((scalari::extrbf_c_normal<0, 2>(val)) == 0);
    REQUIRE((scalari::extrbf_c_normal<0, 3>(val)) == 0);
    REQUIRE((scalari::extrbf_c_normal<0, 4>(val)) == 0);
    REQUIRE((scalari::extrbf_c_normal<1, 4>(val)) == 8);
    REQUIRE((scalari::extrbf_c_normal<2, 4>(val)) == 4);
    REQUIRE((scalari::extrbf_c_normal<3, 4>(val)) == 0xA);

    REQUIRE((scalari::extrbf_c_acc<4, 4>(val)) == 0x5);
    REQUIRE((scalari::extrbf_c_acc<0, 1>(val)) == 0);
    REQUIRE((scalari::extrbf_c_acc<0, 2>(val)) == 0);
    REQUIRE((scalari::extrbf_c_acc<0, 3>(val)) == 0);
    REQUIRE((scalari::extrbf_c_acc<0, 4>(val)) == 0);
    REQUIRE((scalari::extrbf_c_acc<1, 4>(val)) == 8);
    REQUIRE((scalari::extrbf_c_acc<2, 4>(val)) == 4);
    REQUIRE((scalari::extrbf_c_acc<3, 4>(val)) == 0xA);

}


TEST_CASE("exprTemplatesFMA")
{
    // we want to achieve:
    // 1. user code uses operators for expressions on packed vectors
    //    e.g.
    //      vec4f_t a, b, c, res;
    //      a = ...;
    //      b = ...;
    //      c = ...;
    //      res = a*b + c;
    //
    // 2. this would (approximately) result in the following intrinsics code:
    //      ...
    //      res = paddps(pmulps(a, b), c);
    //    so 2 instructions (and a truncated intermediate value)
    // 3. FMA instructions do a multiplication and add operation in a single
    //    instruction:
    //      - better code density (this is relevant wrt performance)
    //      - no truncation of intermediate value (this is relevant wrt to semantic/precision of the calculation)
    // 4. expression templates allow to postpone the "generation of the code" to a later time by building
    //    class/struct-based represenations of the expressions:
    //      res = a*b + c;
    //    would result in sth like:
    //      assign(res, add(mul(a, b), c);
    // 5. the programmer could simply write:
    //      res = fma(a, b, c);
    //    and be done with it, so what's the point?


    // note: we don't do the runtime check here for now (support of FMA3 needs to be checked
    //       per CPUID instruction)

    using vec4f_t = math::vec4f_t;
    using math4f_t = vec4f_t::math_t;

    vec4f_t a, b, c, res;

    c = math4f_t::ones(); // [1 1 1 1]
    b = c + c;            // [2 2 2 2]
    a = b + b;            // [4 4 4 4]

    res = _mm_fmadd_ps(a, b, c); // ab+c

    REQUIRE(res[0] == 9.0f);
    REQUIRE(res[1] == 9.0f);
    REQUIRE(res[2] == 9.0f);
    REQUIRE(res[3] == 9.0f);


    res = _mm_fmsub_ps(a, b, c); // ab-c

    REQUIRE(res[0] == 7.0f);
    REQUIRE(res[1] == 7.0f);
    REQUIRE(res[2] == 7.0f);
    REQUIRE(res[3] == 7.0f);


    res = _mm_fnmadd_ps(a, b, c); // -(ab)+c ~= c-ab

    REQUIRE(res[0] == -7.0f);
    REQUIRE(res[1] == -7.0f);
    REQUIRE(res[2] == -7.0f);
    REQUIRE(res[3] == -7.0f);

    res = _mm_fnmsub_ps(a, b, c); // -(ab)-c

    REQUIRE(res[0] == -9.0f);
    REQUIRE(res[1] == -9.0f);
    REQUIRE(res[2] == -9.0f);
    REQUIRE(res[3] == -9.0f);

}


// TODO: test_main() -> main() for Android?
//#ifdef ANDROID
#if 0
void test_main()
{
    pmathunittest::UnitTest1 test;

    test.TestMat4fIdentityInverse();
    test.TestMathV4fFastTrig();
    test.TestVec4f();
    test.TestVec4fMiscOps();
    test.TestVeci8x16();
    test.TestVecui8x16();
    test.TestVeci16x8();
    test.TestVecui16x8();
    test.TestVeci32x4();
    test.TestVecui32x4();
    test.TestVeci64x2();
    test.TestVecui64x2();
}
#endif


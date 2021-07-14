# libpvec

libpvec is a header-only C++ library for using the SIMD instruction sets
of processors without having to directly deal with compiler intrinsics or inline
assembly for normal use cases. There are quite a few special use cases with the
need to use intrinsics directly and libpvec allows to freely interchange operations
provided by the library and intrinsics.

There's a price to pay for the convenience, of course: the generated code
is not as good as hand-crafted use of compiler intrinsics or inline assembly.

Some operations are focused on the use as 3D vectors.

The library provides two basic elements to work with:

1. special types for holding the contents of a SIMD register with operations
   directly applied to these types similar to OOP:
   
       vec4f_t v1 = ...;
       float len = v1.length();
       
2. some operations are meant to be called directly, similar to shading languages
   used in OpenGL/Direct3D:
   
       vec4f_t v1 = ...;
       float len = length(v1);
       
Goals:

- basic operations must be inlined by the compiler
- except for some utility functionality (relatively large functions)
  for which inlining would be counterproductive
- it should be possible to generate code for various instruction
  sets for the same architecture, e.g. have a code path for SSE2 and
  another one specifically for machines supporting SSE3 on top of SSE2,
  there's no need for an automatic dispatching functionality
  
  
Supported architectures and instruction sets

- Intel x86 and x64/amd64: SSE, SSE2 primary
                           SSE3, SSSE3 partially
                           AVX und AVX2 partially (currently deactivated)
- ARM NEON -- partially
- strong focus on floating point with some operations implemented for
  packed integer formats


Usage Details

- provides special types and operator overloading for basic operations:
  - vec4f_t
  - ...
  
- shuffling of elements is a little unintuitive for the user: there's
  two basic function patterns:
    * xyzw(vec) returns the elements in the order given with the layout of a vec4_t
      interpreted as {x, y, z, w} (from high-order element to low-order one), so
      xyzw(vec) is the identity function and xxxx(vec) returns {x, x, x, x}
    * xyzw(v1,v2) returns a vector with the upper half extracted from v1 and
      the lower half extracted from v2, so xyzw(v1,v2) returns {v1.x, v1.y, v2.z, v2.w}
      and xxxx(v1,v2) returns {v1.x, v1.x, v2.x, v2.x}
  
  The ARM NEON implementation of the shuffling operations is currently suboptimal.
  
See the provided unit tests for examples of using the library.

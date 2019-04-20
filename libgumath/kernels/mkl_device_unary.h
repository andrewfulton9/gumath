/*
* BSD 3-Clause License
*
* Copyright (c) 2017-2018, plures
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice,
*    this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
*
* 3. Neither the name of the copyright holder nor the names of its
*    contributors may be used to endorse or promote products derived from
*    this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#ifndef MKL_DEVICE_UNARY_H
#define MKL_DEVICE_UNARY_H


#ifdef __cplusplus
#include <cinttypes>
#include <mkl.h>
#include <complex>
#include "contrib/bfloat16.h"
typedef tf::bfloat16 bfloat16_t;
typedef std::complex<float> complex64_t;
typedef std::complex<double> complex128_t;
#else
#include <stdint.h>
#endif


typedef bool bool_t;
typedef float float32_t;
typedef double float64_t;


/*****************************************************************************/
/*                        Cuda device kernel signature                       */
/*****************************************************************************/

#ifdef __cplusplus
  #define MKL_DEVICE_UNARY_DECL(name, t0, t1) \
  extern "C" void gm_mkl_device_fixed_1D_C_##name##_##t0##_##t1(const char *a0, char *a1,           \
                                                                const int64_t N);                   \
  extern "C" void gm_mkl_device_fixed_1D_S_##name##_##t0##_##t1(const char *a0, char *a1,           \
                                                                const int64_t s0, const int64_t s1, \
                                                                const int64_t N);                   \
  extern "C" void gm_mkl_device_0D_##name##_##t0##_##t1(const char *a0, char *a1);
#else
  #define MKL_DEVICE_UNARY_DECL(name, t0, t1) \
  void gm_mkl_device_fixed_1D_C_##name##_##t0##_##t1(const char *a0, char *a1,           \
                                                     const int64_t N);                   \
  void gm_mkl_device_fixed_1D_S_##name##_##t0##_##t1(const char *a0, char *a1,           \
                                                     const int64_t s0, const int64_t s1, \
                                                     const int64_t N);                   \
  void gm_mkl_device_0D_##name##_##t0##_##t1(const char *a0, char *a1);
#endif

#define MKL_DEVICE_UNARY_NOIMPL_DECL(name, t0, t1)


/*****************************************************************************/
/*                                   Copy                                    */
/*****************************************************************************/

#define MKL_DEVICE_ALL_UNARY_ALL_DECL(name) \
    MKL_DEVICE_UNARY_DECL(name, bool, bool)                   \
    MKL_DEVICE_UNARY_DECL(name, bool, uint8)                  \
    MKL_DEVICE_UNARY_DECL(name, bool, uint16)                 \
    MKL_DEVICE_UNARY_DECL(name, bool, uint32)                 \
    MKL_DEVICE_UNARY_DECL(name, bool, uint64)                 \
    MKL_DEVICE_UNARY_DECL(name, bool, int8)                   \
    MKL_DEVICE_UNARY_DECL(name, bool, int16)                  \
    MKL_DEVICE_UNARY_DECL(name, bool, int32)                  \
    MKL_DEVICE_UNARY_DECL(name, bool, int64)                  \
    MKL_DEVICE_UNARY_DECL(name, bool, bfloat16)               \
    MKL_DEVICE_UNARY_NOIMPL_DECL(name, bool, float16)         \
    MKL_DEVICE_UNARY_DECL(name, bool, float32)                \
    MKL_DEVICE_UNARY_DECL(name, bool, float64)                \
    MKL_DEVICE_UNARY_NOIMPL_DECL(name, bool, complex32)       \
    MKL_DEVICE_UNARY_DECL(name, bool, complex64)              \
    MKL_DEVICE_UNARY_DECL(name, bool, complex128)             \
    MKL_DEVICE_UNARY_DECL(name, uint8, uint8)                 \
    MKL_DEVICE_UNARY_DECL(name, uint8, uint16)                \
    MKL_DEVICE_UNARY_DECL(name, uint8, uint32)                \
    MKL_DEVICE_UNARY_DECL(name, uint8, uint64)                \
    MKL_DEVICE_UNARY_DECL(name, uint8, int16)                 \
    MKL_DEVICE_UNARY_DECL(name, uint8, int32)                 \
    MKL_DEVICE_UNARY_DECL(name, uint8, int64)                 \
    MKL_DEVICE_UNARY_DECL(name, uint8, bfloat16)              \
    MKL_DEVICE_UNARY_NOIMPL_DECL(name, uint8, float16)        \
    MKL_DEVICE_UNARY_DECL(name, uint8, float32)               \
    MKL_DEVICE_UNARY_DECL(name, uint8, float64)               \
    MKL_DEVICE_UNARY_NOIMPL_DECL(name, uint8, complex32)      \
    MKL_DEVICE_UNARY_DECL(name, uint8, complex64)             \
    MKL_DEVICE_UNARY_DECL(name, uint8, complex128)            \
    MKL_DEVICE_UNARY_DECL(name, uint16, uint16)               \
    MKL_DEVICE_UNARY_DECL(name, uint16, uint32)               \
    MKL_DEVICE_UNARY_DECL(name, uint16, uint64)               \
    MKL_DEVICE_UNARY_DECL(name, uint16, int32)                \
    MKL_DEVICE_UNARY_DECL(name, uint16, int64)                \
    MKL_DEVICE_UNARY_DECL(name, uint16, float32)              \
    MKL_DEVICE_UNARY_DECL(name, uint16, float64)              \
    MKL_DEVICE_UNARY_DECL(name, uint16, complex64)            \
    MKL_DEVICE_UNARY_DECL(name, uint16, complex128)           \
    MKL_DEVICE_UNARY_DECL(name, uint32, uint32)               \
    MKL_DEVICE_UNARY_DECL(name, uint32, uint64)               \
    MKL_DEVICE_UNARY_DECL(name, uint32, int64)                \
    MKL_DEVICE_UNARY_DECL(name, uint32, float64)              \
    MKL_DEVICE_UNARY_DECL(name, uint32, complex128)           \
    MKL_DEVICE_UNARY_DECL(name, uint64, uint64)               \
    MKL_DEVICE_UNARY_DECL(name, int8, int8)                   \
    MKL_DEVICE_UNARY_DECL(name, int8, int16)                  \
    MKL_DEVICE_UNARY_DECL(name, int8, int32)                  \
    MKL_DEVICE_UNARY_DECL(name, int8, int64)                  \
    MKL_DEVICE_UNARY_DECL(name, int8, bfloat16)               \
    MKL_DEVICE_UNARY_NOIMPL_DECL(name, int8, float16)         \
    MKL_DEVICE_UNARY_DECL(name, int8, float32)                \
    MKL_DEVICE_UNARY_DECL(name, int8, float64)                \
    MKL_DEVICE_UNARY_NOIMPL_DECL(name, int8, complex32)       \
    MKL_DEVICE_UNARY_DECL(name, int8, complex64)              \
    MKL_DEVICE_UNARY_DECL(name, int8, complex128)             \
    MKL_DEVICE_UNARY_DECL(name, int16, int16)                 \
    MKL_DEVICE_UNARY_DECL(name, int16, int32)                 \
    MKL_DEVICE_UNARY_DECL(name, int16, int64)                 \
    MKL_DEVICE_UNARY_DECL(name, int16, float32)               \
    MKL_DEVICE_UNARY_DECL(name, int16, float64)               \
    MKL_DEVICE_UNARY_DECL(name, int16, complex64)             \
    MKL_DEVICE_UNARY_DECL(name, int16, complex128)            \
    MKL_DEVICE_UNARY_DECL(name, int32, int32)                 \
    MKL_DEVICE_UNARY_DECL(name, int32, int64)                 \
    MKL_DEVICE_UNARY_DECL(name, int32, float64)               \
    MKL_DEVICE_UNARY_DECL(name, int32, complex128)            \
    MKL_DEVICE_UNARY_DECL(name, int64, int64)                 \
    MKL_DEVICE_UNARY_DECL(name, bfloat16, bfloat16)           \
    MKL_DEVICE_UNARY_DECL(name, bfloat16, float32)            \
    MKL_DEVICE_UNARY_DECL(name, bfloat16, float64)            \
    MKL_DEVICE_UNARY_DECL(name, bfloat16, complex64)          \
    MKL_DEVICE_UNARY_DECL(name, bfloat16, complex128)         \
    MKL_DEVICE_UNARY_NOIMPL_DECL(name, float16, float16)      \
    MKL_DEVICE_UNARY_NOIMPL_DECL(name, float16, float32)      \
    MKL_DEVICE_UNARY_NOIMPL_DECL(name, float16, float64)      \
    MKL_DEVICE_UNARY_NOIMPL_DECL(name, float16, complex32)    \
    MKL_DEVICE_UNARY_NOIMPL_DECL(name, float16, complex64)    \
    MKL_DEVICE_UNARY_NOIMPL_DECL(name, float16, complex128)   \
    MKL_DEVICE_UNARY_DECL(name, float32, float32)             \
    MKL_DEVICE_UNARY_DECL(name, float32, float64)             \
    MKL_DEVICE_UNARY_DECL(name, float32, complex64)           \
    MKL_DEVICE_UNARY_DECL(name, float32, complex128)          \
    MKL_DEVICE_UNARY_DECL(name, float64, float64)             \
    MKL_DEVICE_UNARY_DECL(name, float64, complex128)          \
    MKL_DEVICE_UNARY_NOIMPL_DECL(name, complex32, complex32)  \
    MKL_DEVICE_UNARY_NOIMPL_DECL(name, complex32, complex64)  \
    MKL_DEVICE_UNARY_NOIMPL_DECL(name, complex32, complex128) \
    MKL_DEVICE_UNARY_DECL(name, complex64, complex64)         \
    MKL_DEVICE_UNARY_DECL(name, complex64, complex128)        \
    MKL_DEVICE_UNARY_DECL(name, complex128, complex128)


MKL_DEVICE_ALL_UNARY_ALL_DECL(copy)
MKL_DEVICE_ALL_UNARY_ALL_DECL(abs)


/*****************************************************************************/
/*                               Bitwise NOT                                 */
/*****************************************************************************/

MKL_DEVICE_UNARY_DECL(invert, bool, bool)

MKL_DEVICE_UNARY_DECL(invert, uint8, uint8)
MKL_DEVICE_UNARY_DECL(invert, uint16, uint16)
MKL_DEVICE_UNARY_DECL(invert, uint32, uint32)
MKL_DEVICE_UNARY_DECL(invert, uint64, uint64)

MKL_DEVICE_UNARY_DECL(invert, int8, int8)
MKL_DEVICE_UNARY_DECL(invert, int16, int16)
MKL_DEVICE_UNARY_DECL(invert, int32, int32)
MKL_DEVICE_UNARY_DECL(invert, int64, int64)


/*****************************************************************************/
/*                                 Negative                                  */
/*****************************************************************************/

MKL_DEVICE_UNARY_DECL(negative, uint8, int16)
MKL_DEVICE_UNARY_DECL(negative, uint16, int32)
MKL_DEVICE_UNARY_DECL(negative, uint32, int64)

MKL_DEVICE_UNARY_DECL(negative, int8, int8)
MKL_DEVICE_UNARY_DECL(negative, int16, int16)
MKL_DEVICE_UNARY_DECL(negative, int32, int32)
MKL_DEVICE_UNARY_DECL(negative, int64, int64)

MKL_DEVICE_UNARY_DECL(negative, bfloat16, bfloat16)
MKL_DEVICE_UNARY_NOIMPL_DECL(negative, float16, float16)
MKL_DEVICE_UNARY_DECL(negative, float32, float32)
MKL_DEVICE_UNARY_DECL(negative, float64, float64)

MKL_DEVICE_UNARY_NOIMPL_DECL(negative, complex32, complex32)
MKL_DEVICE_UNARY_DECL(negative, complex64, complex64)
MKL_DEVICE_UNARY_DECL(negative, complex128, complex128)


/*****************************************************************************/
/*                                    Math                                   */
/*****************************************************************************/

#define MKL_DEVICE_UNARY_ALL_REAL_MATH_DECL(name) \
    MKL_DEVICE_UNARY_DECL(name##f, uint16, float32)      \
    MKL_DEVICE_UNARY_DECL(name##f, int16, float32)       \
    MKL_DEVICE_UNARY_DECL(name##b16, bfloat16, bfloat16) \
    MKL_DEVICE_UNARY_DECL(name##f, float32, float32)     \
    MKL_DEVICE_UNARY_DECL(name, uint32, float64)         \
    MKL_DEVICE_UNARY_DECL(name, int32, float64)          \
    MKL_DEVICE_UNARY_DECL(name, float64, float64)

#define MKL_DEVICE_UNARY_ALL_COMPLEX_MATH_DECL(name) \
    MKL_DEVICE_UNARY_ALL_REAL_MATH_DECL(name)           \
    MKL_DEVICE_UNARY_DECL(name, complex32, complex32)   \
    MKL_DEVICE_UNARY_DECL(name, complex64, complex64)   \
    MKL_DEVICE_UNARY_DECL(name, complex128, complex128)

#define MKL_DEVICE_UNARY_ALL_HALF_MATH_DECL(name) \
    MKL_DEVICE_UNARY_DECL(name##f16, uint8, float16)   \
    MKL_DEVICE_UNARY_DECL(name##f16, int8, float16)    \
    MKL_DEVICE_UNARY_DECL(name##f16, float16, float16)

#define MKL_DEVICE_UNARY_ALL_REAL_MATH_WITH_HALF_DECL(name) \
    MKL_DEVICE_UNARY_ALL_HALF_MATH_DECL(name)               \
    MKL_DEVICE_UNARY_ALL_REAL_MATH_DECL(name)               \

#define MKL_DEVICE_UNARY_ALL_COMPLEX_MATH_WITH_HALF_DECL(name) \
    MKL_DEVICE_UNARY_ALL_HALF_MATH_DECL(name)                  \
    MKL_DEVICE_UNARY_ALL_COMPLEX_MATH_DECL(name)               \


/*****************************************************************************/
/*                                Abs functions                              */
/*****************************************************************************/

MKL_DEVICE_UNARY_ALL_REAL_MATH_DECL(fabs)


/*****************************************************************************/
/*                             Exponential functions                         */
/*****************************************************************************/

MKL_DEVICE_UNARY_ALL_COMPLEX_MATH_DECL(exp)
MKL_DEVICE_UNARY_ALL_REAL_MATH_DECL(exp2)
MKL_DEVICE_UNARY_ALL_REAL_MATH_DECL(expm1)


/*****************************************************************************/
/*                              Logarithm functions                          */
/*****************************************************************************/

MKL_DEVICE_UNARY_ALL_COMPLEX_MATH_DECL(log)
MKL_DEVICE_UNARY_ALL_COMPLEX_MATH_DECL(log10)
MKL_DEVICE_UNARY_ALL_REAL_MATH_DECL(log2)
MKL_DEVICE_UNARY_ALL_REAL_MATH_DECL(log1p)
MKL_DEVICE_UNARY_ALL_REAL_MATH_DECL(logb)


/*****************************************************************************/
/*                              Power functions                              */
/*****************************************************************************/

MKL_DEVICE_UNARY_ALL_COMPLEX_MATH_DECL(sqrt)
MKL_DEVICE_UNARY_ALL_REAL_MATH_DECL(cbrt)


/*****************************************************************************/
/*                           Trigonometric functions                         */
/*****************************************************************************/

MKL_DEVICE_UNARY_ALL_COMPLEX_MATH_DECL(sin)
MKL_DEVICE_UNARY_ALL_COMPLEX_MATH_DECL(cos)
MKL_DEVICE_UNARY_ALL_COMPLEX_MATH_DECL(tan)
MKL_DEVICE_UNARY_ALL_COMPLEX_MATH_DECL(asin)
MKL_DEVICE_UNARY_ALL_COMPLEX_MATH_DECL(acos)
MKL_DEVICE_UNARY_ALL_COMPLEX_MATH_DECL(atan)


/*****************************************************************************/
/*                             Hyperbolic functions                          */
/*****************************************************************************/

MKL_DEVICE_UNARY_ALL_COMPLEX_MATH_DECL(sinh)
MKL_DEVICE_UNARY_ALL_COMPLEX_MATH_DECL(cosh)
MKL_DEVICE_UNARY_ALL_COMPLEX_MATH_DECL(tanh)
MKL_DEVICE_UNARY_ALL_COMPLEX_MATH_DECL(asinh)
MKL_DEVICE_UNARY_ALL_COMPLEX_MATH_DECL(acosh)
MKL_DEVICE_UNARY_ALL_COMPLEX_MATH_DECL(atanh)


/*****************************************************************************/
/*                            Error and gamma functions                      */
/*****************************************************************************/

MKL_DEVICE_UNARY_ALL_REAL_MATH_DECL(erf)
MKL_DEVICE_UNARY_ALL_REAL_MATH_DECL(erfc)
MKL_DEVICE_UNARY_ALL_REAL_MATH_DECL(lgamma)
MKL_DEVICE_UNARY_ALL_REAL_MATH_DECL(tgamma)


/*****************************************************************************/
/*                              Ceiling, floor, trunc                        */
/*****************************************************************************/

MKL_DEVICE_UNARY_ALL_REAL_MATH_DECL(ceil)
MKL_DEVICE_UNARY_ALL_REAL_MATH_DECL(floor)
MKL_DEVICE_UNARY_ALL_REAL_MATH_DECL(trunc)
MKL_DEVICE_UNARY_ALL_REAL_MATH_DECL(round)
MKL_DEVICE_UNARY_ALL_REAL_MATH_DECL(nearbyint)


#endif /* MKL_DEVICE_UNARY_H */

#ifndef CARPETX_DERIVS_SPACETIMEX_DERIVS_HXX
#define CARPETX_DERIVS_SPACETIMEX_DERIVS_HXX

#include <loop_device.hxx>
#include <mat.hxx>
#include <simd.hxx>
#include <vec.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <type_traits>
// Unified derivative header

namespace derivs_spacetimex {

// TODO: Make this a runtime parameter
constexpr int deriv_order = 4;

////////////////////////////////////////////////////////////////////////////////

using Arith::pow2;

template <typename T>
inline ARITH_INLINE ARITH_DEVICE ARITH_HOST T pow3(const T &x) {
  return pow2(x) * x;
}

////////////////////////////////////////////////////////////////////////////////

using Arith::simd;
using Arith::simdl;

template <typename T>
inline ARITH_INLINE ARITH_DEVICE ARITH_HOST simd<T>
deriv1d(const simdl<T> &mask, const T *restrict const var, const ptrdiff_t di,
        const T dx) {
  const auto load = [&](const int n) {
    return maskz_loadu(mask, &var[n * di]) - maskz_loadu(mask, &var[-n * di]);
  };
  if constexpr (deriv_order == 2)
    return 1 / T(2) * load(1) / dx;
  if constexpr (deriv_order == 4)
    return (-1 / T(12) * load(2) + 2 / T(3) * load(1)) / dx;
  if constexpr (deriv_order == 6)
    return (1 / T(60) * load(3) - 3 / T(20) * load(2) + 3 / T(4) * load(1)) /
           dx;
}

using Loop::GF3D2;
using Arith::vect;
using Loop::dim;
using Arith::vec;

template <int dir, typename T, int D>
inline ARITH_INLINE ARITH_DEVICE ARITH_HOST simd<T>
deriv(const simdl<T> &mask, const GF3D2<const T> &gf_, const vect<int, dim> &I,
      const vec<T, D> &dx) {
  static_assert(dir >= 0 && dir < D, "");
  const auto &DI = vect<int, dim>::unit;
  const ptrdiff_t di = gf_.delta(DI(dir));
  return deriv1d(mask, &gf_(I), di, dx(dir));
}
template <typename T>
inline ARITH_INLINE ARITH_DEVICE ARITH_HOST vec<simd<T>, dim>
deriv(const simdl<T> &mask, const GF3D2<const T> &gf_, const vect<int, dim> &I,
      const vec<T, dim> &dx) {
  return {deriv<0>(mask, gf_, I, dx), deriv<1>(mask, gf_, I, dx),
          deriv<2>(mask, gf_, I, dx)};
}

template <typename T>
inline ARITH_INLINE ARITH_DEVICE ARITH_HOST simd<T>
deriv2_1d(const simdl<T> &mask, const T *restrict const var, const ptrdiff_t di,
          const T dx) {
  const auto load = [&](const int n) {
    return maskz_loadu(mask, &var[n * di]) + maskz_loadu(mask, &var[-n * di]);
  };
  const auto load0 = [&]() { return maskz_loadu(mask, &var[0]); };
  if constexpr (deriv_order == 2)
    return (load(1) - 2 * load0()) / pow2(dx);
  if constexpr (deriv_order == 4)
    return (1 / T(12) * load(2) - 4 / T(3) * load(1) + 5 / T(2) * load0()) /
           pow2(dx);
  if constexpr (deriv_order == 6)
    return (1 / T(90) * load(3) - 3 / T(20) * load(2) + 3 / T(2) * load(1) -
            49 / T(18) * load0()) /
           pow2(dx);
}

using std::tuple_size_v;
using Arith::min;
using std::array;
using Arith::div_floor;
using Arith::div_ceil;
using Arith::align_ceil;
using Arith::mask_for_loop_tail;

template <typename T>
inline ARITH_INLINE ARITH_DEVICE ARITH_HOST simd<T>
deriv2_2d(const int vavail, const simdl<T> &mask, const T *restrict const var,
          const ptrdiff_t di, const ptrdiff_t dj, const T dx, const T dy) {
  constexpr size_t vsize = tuple_size_v<simd<T> >;
  if (di == 1) {
    assert(vavail > 0);
    constexpr int maxnpoints = deriv_order + 1 + vsize - 1;
    const int npoints = deriv_order + 1 + min(int(vsize), vavail) - 1;
    array<simd<T>, div_ceil(maxnpoints, int(vsize))> arrx;
    for (int i = 0; i < maxnpoints; i += vsize) {
      if (i < npoints) {
        const simdl<T> mask1 = Arith::mask_for_loop_tail<simdl<T> >(i, npoints);
        arrx[div_floor(i, int(vsize))] =
            deriv1d(mask1, &var[i - deriv_order / 2], dj, dy);
      }
    }
#ifdef CCTK_DEBUG
    for (int i = npoints; i < align_ceil(maxnpoints, int(vsize)); ++i)
      ((T *)&arrx[0])[i] = Arith::nan<T>()(); // unused
#endif
    const T *const varx = (T *)&arrx[0] + deriv_order / 2;
    return deriv1d(mask, varx, 1, dx);
  } else {
    assert(dj != 1);
    array<simd<T>, deriv_order + 1> arrx;
    for (int j = -deriv_order / 2; j <= deriv_order / 2; ++j)
      if (j == 0) {
#ifdef CCTK_DEBUG
        arrx[deriv_order / 2 + j] = Arith::nan<simd<T> >()(); // unused
#endif
      } else {
        arrx[deriv_order / 2 + j] = deriv1d(mask, &var[j * dj], di, dx);
      }
    const T *const varx = (T *)(&arrx[deriv_order / 2]);
    return deriv1d(mask, varx, vsize, dy);
  }
}

using Loop::GF3D5;
using Loop::GF3D5layout;
using Loop::GF3D5index;
using Arith::smat;
using Loop::PointDesc;


template <typename T>
CCTK_ATTRIBUTE_NOINLINE void
calc_derivs(const cGH *restrict const cctkGH, const GF3D2<const T> &gf1,
            const GF3D5<T> &gf0, const vec<GF3D5<T>, dim> &dgf0,
            const GF3D5layout &layout0) {
  DECLARE_CCTK_ARGUMENTS;

  typedef simd<CCTK_REAL> vreal;
  typedef simdl<CCTK_REAL> vbool;
  constexpr size_t vsize = tuple_size_v<vreal>;

  const vec<CCTK_REAL, dim> dx([&](int a) { return CCTK_DELTA_SPACE(a); });

  const Loop::GridDescBaseDevice grid(cctkGH);
  grid.loop_int_device<0, 0, 0, vsize>(
      grid.nghostzones, [=] ARITH_DEVICE(const PointDesc &p) ARITH_INLINE {
        const vbool mask = Arith::mask_for_loop_tail<vbool>(p.i, p.imax);
        const GF3D5index index0(layout0, p.I);
        const auto val = gf1(mask, p.I);
        gf0.store(mask, index0, val);
        const auto dval = deriv(mask, gf1, p.I, dx);
        dgf0.store(mask, index0, dval);
      });
}

using std::enable_if_t;

template <int dir1, int dir2, typename T, int D>
inline ARITH_INLINE
    ARITH_DEVICE ARITH_HOST enable_if_t<(dir1 == dir2), simd<T> >
    deriv2(const int vavail, const simdl<T> &mask, const GF3D2<const T> &gf_,
           const vect<int, dim> &I, const vec<T, D> &dx) {
  static_assert(dir1 >= 0 && dir1 < D, "");
  static_assert(dir2 >= 0 && dir2 < D, "");
  const auto &DI = vect<int, dim>::unit;
  const ptrdiff_t di = gf_.delta(DI(dir1));
  return deriv2_1d(mask, &gf_(I), di, dx(dir1));
}

template <int dir1, int dir2, typename T, int D>
inline ARITH_INLINE
    ARITH_DEVICE ARITH_HOST enable_if_t<(dir1 != dir2), simd<T> >
    deriv2(const int vavail, const simdl<T> &mask, const GF3D2<const T> &gf_,
           const vect<int, dim> &I, const vec<T, D> &dx) {
  static_assert(dir1 >= 0 && dir1 < D, "");
  static_assert(dir2 >= 0 && dir2 < D, "");
  const auto &DI = vect<int, dim>::unit;
  const ptrdiff_t di = gf_.delta(DI(dir1));
  const ptrdiff_t dj = gf_.delta(DI(dir2));
  return deriv2_2d(vavail, mask, &gf_(I), di, dj, dx(dir1), dx(dir2));
}

template <typename T>
inline ARITH_INLINE ARITH_DEVICE ARITH_HOST smat<simd<T>, dim>
deriv2(const int vavail, const simdl<T> &mask, const GF3D2<const T> &gf_,
       const vect<int, dim> &I, const vec<T, dim> &dx) {
  return {deriv2<0, 0>(vavail, mask, gf_, I, dx),
          deriv2<0, 1>(vavail, mask, gf_, I, dx),
          deriv2<0, 2>(vavail, mask, gf_, I, dx),
          deriv2<1, 1>(vavail, mask, gf_, I, dx),
          deriv2<1, 2>(vavail, mask, gf_, I, dx),
          deriv2<2, 2>(vavail, mask, gf_, I, dx)};
}

template <typename T>
CCTK_ATTRIBUTE_NOINLINE void
calc_derivs2(const cGH *restrict const cctkGH, const GF3D2<const T> &gf1,
             const GF3D5<T> &gf0, const vec<GF3D5<T>, dim> &dgf0,
             const smat<GF3D5<T>, dim> &ddgf0, const GF3D5layout &layout0) {
  DECLARE_CCTK_ARGUMENTS;

  typedef simd<CCTK_REAL> vreal;
  typedef simdl<CCTK_REAL> vbool;
  constexpr size_t vsize = tuple_size_v<vreal>;

  const vec<CCTK_REAL, dim> dx([&](int a) { return CCTK_DELTA_SPACE(a); });

  const Loop::GridDescBaseDevice grid(cctkGH);
  grid.loop_int_device<0, 0, 0, vsize>(
      grid.nghostzones, [=] ARITH_DEVICE(const PointDesc &p) ARITH_INLINE {
        const vbool mask = Arith::mask_for_loop_tail<vbool>(p.i, p.imax);
        const int vavail = p.imax - p.i;
        const GF3D5index index0(layout0, p.I);
        const auto val = gf1(mask, p.I);
        gf0.store(mask, index0, val);
        const auto dval = deriv(mask, gf1, p.I, dx);
        dgf0.store(mask, index0, dval);
        const auto ddval = deriv2(vavail, mask, gf1, p.I, dx);
        ddgf0.store(mask, index0, ddval);
      });
}

template <typename T>
void CCTK_ATTRIBUTE_NOINLINE calc_derivs(
    const cGH *restrict const cctkGH, const vec<GF3D2<const T>, dim> &gf0_,
    const vec<GF3D5<T>, dim> &gf_, const vec<vec<GF3D5<T>, dim>, dim> &dgf_,
    const GF3D5layout &layout) {
  for (int a = 0; a < 3; ++a)
    calc_derivs(cctkGH, gf0_(a), gf_(a), dgf_(a), layout);
}

template <typename T>
void CCTK_ATTRIBUTE_NOINLINE calc_derivs2(
    const cGH *restrict const cctkGH, const vec<GF3D2<const T>, dim> &gf0_,
    const vec<GF3D5<T>, dim> &gf_, const vec<vec<GF3D5<T>, dim>, dim> &dgf_,
    const vec<smat<GF3D5<T>, dim>, dim> &ddgf_, const GF3D5layout &layout) {
  for (int a = 0; a < 3; ++a)
    calc_derivs2(cctkGH, gf0_(a), gf_(a), dgf_(a), ddgf_(a), layout);
}

template <typename T>
CCTK_ATTRIBUTE_NOINLINE void
calc_copy(const cGH *restrict const cctkGH, const GF3D2<const T> &gf1,
          const GF3D5<T> &gf0, const GF3D5layout &layout0) {
  DECLARE_CCTK_ARGUMENTS;

  typedef simd<CCTK_REAL> vreal;
  typedef simdl<CCTK_REAL> vbool;
  constexpr size_t vsize = tuple_size_v<vreal>;

  const vec<CCTK_REAL, dim> dx([&](int a) { return CCTK_DELTA_SPACE(a); });

  const Loop::GridDescBaseDevice grid(cctkGH);
  grid.loop_int_device<0, 0, 0, vsize>(
      grid.nghostzones,
      [=] ARITH_DEVICE ARITH_HOST(const PointDesc &p) ARITH_INLINE {
        const vbool mask = Arith::mask_for_loop_tail<vbool>(p.i, p.imax);
        const GF3D5index index0(layout0, p.I);
        const auto val = gf1(mask, p.I);
        gf0.store(mask, index0, val);
      });
}

template <typename T>
void CCTK_ATTRIBUTE_NOINLINE calc_copy(const cGH *restrict const cctkGH,
                                       const vec<GF3D2<const T>, dim> &gf0_,
                                       const vec<GF3D5<T>, dim> &gf_,
                                       const GF3D5layout &layout) {
  for (int a = 0; a < 3; ++a)
    calc_copy(cctkGH, gf0_(a), gf_(a), layout);
}

template <typename T>
CCTK_ATTRIBUTE_NOINLINE void calc_derivs(
    const cGH *restrict const cctkGH, const smat<GF3D2<const T>, dim> &gf0_,
    const smat<GF3D5<T>, dim> &gf_, const smat<vec<GF3D5<T>, dim>, dim> &dgf_,
    const GF3D5layout &layout) {
  for (int a = 0; a < 3; ++a)
    for (int b = a; b < 3; ++b)
      calc_derivs(cctkGH, gf0_(a, b), gf_(a, b), dgf_(a, b), layout);
}

template <typename T>
CCTK_ATTRIBUTE_NOINLINE void calc_derivs2(
    const cGH *restrict const cctkGH, const smat<GF3D2<const T>, dim> &gf0_,
    const smat<GF3D5<T>, dim> &gf_, const smat<vec<GF3D5<T>, dim>, dim> &dgf_,
    const smat<smat<GF3D5<T>, dim>, dim> &ddgf_, const GF3D5layout &layout) {
  for (int a = 0; a < 3; ++a)
    for (int b = a; b < 3; ++b)
      calc_derivs2(cctkGH, gf0_(a, b), gf_(a, b), dgf_(a, b), ddgf_(a, b),
                   layout);
}

template <typename T>
inline ARITH_INLINE ARITH_DEVICE ARITH_HOST simd<T>
deriv1d_upwind(const simdl<T> &mask, const T *restrict const var,
               const ptrdiff_t di, const simd<T> &vel, const T dx) {
  // arXiv:1111.2177 [gr-qc], (71)
  if constexpr (deriv_order == 2) {
    // if (sign)
    //   // +     [ 0   -1   +1    0    0]
    //   // + 1/2 [+1   -2   +1    0    0]
    //   //       [+1/2 -2   +3/2  0    0]
    //   return (1 / T(2) * var[-2 * di] //
    //           - 2 * var[-di]          //
    //           + 3 / T(2) * var[0]) /
    //          dx;
    // else
    //   // +     [ 0    0   -1   +1    0  ]
    //   // - 1/2 [ 0    0   +1   -2   +1  ]
    //   //       [ 0    0   -3/2 +2   -1/2]
    //   return (-3 / T(2) * var[0] //
    //           + 2 * var[+di]     //
    //           - 1 / T(2) * var[+2 * di]) /
    //          dx;

    // + 1/2 [+1/2 -2   +3/2  0    0  ]
    // + 1/2 [ 0    0   -3/2 +2   -1/2]
    //       [+1/4 -1    0   +1   -1/4]
    const simd<T> symm =
        1 / T(4) *
            (maskz_loadu(mask, &var[-2 * di]) -
             maskz_loadu(mask, &var[2 * di])) //
        - (maskz_loadu(mask, &var[-di]) - maskz_loadu(mask, &var[di]));
    // + 1/2 [+1/2 -2   +3/2  0    0  ]
    // - 1/2 [ 0    0   -3/2 +2   -1/2]
    //       [+1/4 -1   +3/2 -1   +1/4]
    const simd<T> anti =
        1 / T(4) *
            (maskz_loadu(mask, &var[-2 * di]) +
             maskz_loadu(mask, &var[2 * di]))                          //
        - (maskz_loadu(mask, &var[-di]) + maskz_loadu(mask, &var[di])) //
        + 3 / T(2) * maskz_loadu(mask, &var[0]);
    return (vel * symm - fabs(vel) * anti) / dx;
  }
  if constexpr (deriv_order == 4) {
    // A fourth order stencil for a first derivative, shifted by one grid point

    // if (sign)
    //   return (-1 / T(12) * var[-3 * di] //
    //           + 1 / T(2) * var[-2 * di] //
    //           - 3 / T(2) * var[-di]     //
    //           + 5 / T(6) * var[0]       //
    //           + 1 / T(4) * var[+di]) /
    //          dx;
    // else
    //   return (-1 / T(4) * var[-di]      //
    //           - 5 / T(6) * var[0]       //
    //           + 3 / T(2) * var[+di]     //
    //           - 1 / T(2) * var[+2 * di] //
    //           + 1 / T(12) * var[+3 * di]) /
    //          dx;

    const simd<T> symm =
        -1 / T(24) *
            (maskz_loadu(mask, &var[-3 * di]) -
             maskz_loadu(mask, &var[3 * di])) //
        + 1 / T(4) *
              (maskz_loadu(mask, &var[-2 * di]) -
               maskz_loadu(mask, &var[2 * di])) //
        -
        7 / T(8) * (maskz_loadu(mask, &var[-di]) - maskz_loadu(mask, &var[di]));
    const simd<T> anti =
        -1 / T(24) *
            (maskz_loadu(mask, &var[-3 * di]) +
             maskz_loadu(mask, &var[3 * di])) //
        + 1 / T(4) *
              (maskz_loadu(mask, &var[-2 * di]) +
               maskz_loadu(mask, &var[2 * di])) //
        - 5 / T(8) *
              (maskz_loadu(mask, &var[-di]) + maskz_loadu(mask, &var[di])) //
        + 5 / T(6) * maskz_loadu(mask, &var[0]);
    return (vel * symm - fabs(vel) * anti) / dx;
  }
}

template <int dir, typename T, int D>
inline ARITH_INLINE ARITH_DEVICE ARITH_HOST simd<T>
deriv_upwind(const simdl<T> &mask, const GF3D2<const T> &gf_,
             const vect<int, dim> &I, const vec<simd<T>, D> &vel,
             const vec<T, D> &dx) {
  static_assert(dir >= 0 && dir < D, "");
  const auto &DI = vect<int, dim>::unit;
  const ptrdiff_t di = gf_.delta(DI(dir));
  return deriv1d_upwind(mask, &gf_(I), di, vel(dir), dx(dir));
}

template <typename T>
inline ARITH_INLINE ARITH_DEVICE ARITH_HOST simd<T>
deriv_upwind(const simdl<T> &mask, const GF3D2<const T> &gf_,
             const vect<int, dim> &I, const vec<simd<T>, dim> &vel,
             const vec<T, dim> &dx) {
  return deriv_upwind<0>(mask, gf_, I, vel, dx) +
         deriv_upwind<1>(mask, gf_, I, vel, dx) +
         deriv_upwind<2>(mask, gf_, I, vel, dx);
}

template <typename T>
inline ARITH_INLINE ARITH_DEVICE ARITH_HOST simd<T>
deriv1d_diss(const simdl<T> &mask, const T *restrict const var,
             const ptrdiff_t di, const T dx) {
  if constexpr (deriv_order == 2)
    return ((maskz_loadu(mask, &var[-2 * di]) +
             maskz_loadu(mask, &var[+2 * di])) //
            -
            4 * (maskz_loadu(mask, &var[-di]) + maskz_loadu(mask, &var[+di])) //
            + 6 * maskz_loadu(mask, &var[0])) /
           dx;
  if constexpr (deriv_order == 4)
    return ((maskz_loadu(mask, &var[-3 * di]) +
             maskz_loadu(mask, &var[+3 * di])) //
            - 6 * (maskz_loadu(mask, &var[-2 * di]) +
                   maskz_loadu(mask, &var[+2 * di])) //
            + 15 * (maskz_loadu(mask, &var[-di]) +
                    maskz_loadu(mask, &var[+di])) //
            - 20 * maskz_loadu(mask, &var[0])) /
           dx;
}

using Arith::pown;

template <int dir, typename T, int D>
inline ARITH_INLINE ARITH_DEVICE ARITH_HOST simd<T>
deriv_diss(const simdl<T> &mask, const GF3D2<const T> &gf_,
           const vect<int, dim> &I, const vec<T, D> &dx) {
  static_assert(dir >= 0 && dir < D, "");
  const auto &DI = vect<int, dim>::unit;
  const ptrdiff_t di = gf_.delta(DI(dir));
  return deriv1d_diss(mask, &gf_(I), di, dx(dir));
}

template <typename T>
inline ARITH_INLINE ARITH_DEVICE ARITH_HOST simd<T>
diss(const simdl<T> &mask, const GF3D2<const T> &gf_, const vect<int, dim> &I,
     const vec<T, dim> &dx) {
  // arXiv:gr-qc/0610128, (63), with r=2
  constexpr int diss_order = deriv_order + 2;
  constexpr int sign = diss_order % 4 == 0 ? -1 : +1;
  return sign / T(pown(2, deriv_order + 2)) *
         (deriv_diss<0>(mask, gf_, I, dx)   //
          + deriv_diss<1>(mask, gf_, I, dx) //
          + deriv_diss<2>(mask, gf_, I, dx));
}

template <typename T>
CCTK_ATTRIBUTE_NOINLINE void
apply_upwind_diss(const cGH *restrict const cctkGH, const GF3D2<const T> &gf_,
                  const vec<GF3D2<const T>, dim> &gf_betaG_,
                  const GF3D2<T> &gf_rhs_) {
  DECLARE_CCTK_ARGUMENTS;
  //DECLARE_CCTK_PARAMETERS;
  const CCTK_REAL& epsdiss = *(CCTK_REAL*)CCTK_ParameterGet("epsdiss", "Derivs", nullptr);

  const vec<CCTK_REAL, dim> dx([&](int a) { return CCTK_DELTA_SPACE(a); });

  typedef simd<CCTK_REAL> vreal;
  typedef simdl<CCTK_REAL> vbool;
  constexpr size_t vsize = tuple_size_v<vreal>;

  if (epsdiss == 0) {

    const Loop::GridDescBaseDevice grid(cctkGH);
    grid.loop_int_device<0, 0, 0, vsize>(
        grid.nghostzones, [=] ARITH_DEVICE(const PointDesc &p) ARITH_INLINE {
          const vbool mask = mask_for_loop_tail<vbool>(p.i, p.imax);
          const vec<vreal, dim> betaG = gf_betaG_(mask, p.I);
          const vreal rhs_old = gf_rhs_(mask, p.I);
          const vreal rhs_new =
              rhs_old + deriv_upwind(mask, gf_, p.I, betaG, dx);
          gf_rhs_.store(mask, p.I, rhs_new);
        });

  } else {

    const Loop::GridDescBaseDevice grid(cctkGH);
    grid.loop_int_device<0, 0, 0, vsize>(
        grid.nghostzones, [=] ARITH_DEVICE(const PointDesc &p) ARITH_INLINE {
          const vbool mask = mask_for_loop_tail<vbool>(p.i, p.imax);
          const vec<vreal, dim> betaG = gf_betaG_(mask, p.I);
          const vreal rhs_old = gf_rhs_(mask, p.I);
          const vreal rhs_new = rhs_old +
                                deriv_upwind(mask, gf_, p.I, betaG, dx) +
                                epsdiss * diss(mask, gf_, p.I, dx);
          gf_rhs_.store(mask, p.I, rhs_new);
        });
  }
}

template <typename T>
void CCTK_ATTRIBUTE_NOINLINE calc_copy(const cGH *restrict const cctkGH,
                                       const smat<GF3D2<const T>, dim> &gf0_,
                                       const smat<GF3D5<T>, dim> &gf_,
                                       const GF3D5layout &layout) {
  for (int a = 0; a < 3; ++a)
    for (int b = a; b < 3; ++b)
      calc_copy(cctkGH, gf0_(a, b), gf_(a, b), layout);
}
 
}

#endif // #ifndef CARPETX_DERIVS_DERIVS_HXX

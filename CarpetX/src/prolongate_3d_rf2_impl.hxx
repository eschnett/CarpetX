#ifndef CARPETX_CARPETX_PROLONGATE_3D_RF2_IMPL_HXX
#define CARPETX_CARPETX_PROLONGATE_3D_RF2_IMPL_HXX

#include "prolongate_3d_rf2.hxx"

#include "timer.hxx"

#include <vect.hxx>

#include <cctk.h>
#include <cctk_Parameters.h>

#include <AMReX_Gpu.H>

#if defined _OPENMP
#include <omp.h>
#elif defined __HIPCC__
#define omp_get_num_threads() 1
#define omp_get_thread_num() 0
#else
static inline int omp_get_num_threads() { return 1; }
static inline int omp_get_thread_num() { return 0; }
#endif

#include <array>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace CarpetX {

namespace {
// TODO: Move this function to loop.hxx
template <typename F>
CCTK_ATTRIBUTE_NOINLINE __attribute__((__flatten__, __hot__)) void
loop_region(const F &f, const Arith::vect<int, dim> &imin,
            const Arith::vect<int, dim> &imax) {
  assert(!any(imax <= imin));
  if (any(imax <= imin))
    return;

  const amrex::Box box(amrex::IntVect(imin[0], imin[1], imin[2]),
                       amrex::IntVect(imax[0] - 1, imax[1] - 1, imax[2] - 1));
  amrex::ParallelFor(box, [=] CCTK_DEVICE(const int i, const int j, const int k)
                              CCTK_ATTRIBUTE_ALWAYS_INLINE {
                                const Arith::vect<int, dim> p{i, j, k};
                                f(p);
                              });
}
} // namespace

// 1D interpolation coefficients

template <centering_t CENT, interpolation_t INTP, int ORDER, typename T>
struct coeffs1d;

template <typename T> struct coeffs1d<VC, POLY, /*order*/ 1, T> {
  static constexpr std::array<T, 2> coeffs = {
      +1 / T(2),
      +1 / T(2),
  };
};
template <typename T> struct coeffs1d<VC, POLY, /*order*/ 3, T> {
  static constexpr std::array<T, 4> coeffs = {
      -1 / T(16),
      +9 / T(16),
      +9 / T(16),
      -1 / T(16),
  };
};
template <typename T> struct coeffs1d<VC, POLY, /*order*/ 5, T> {
  static constexpr std::array<T, 6> coeffs = {
      +3 / T(256),  -25 / T(256), +75 / T(128),
      +75 / T(128), -25 / T(256), +3 / T(256),
  };
};
template <typename T> struct coeffs1d<VC, POLY, /*order*/ 7, T> {
  static constexpr std::array<T, 8> coeffs = {
      -5 / T(2048),    +49 / T(2048),  -245 / T(2048), +1225 / T(2048),
      +1225 / T(2048), -245 / T(2048), +49 / T(2048),  -5 / T(2048),
  };
};

template <typename T> struct coeffs1d<CC, POLY, /*order*/ 0, T> {
  static constexpr std::array<T, 1> coeffs = {
      +1 / T(1),
  };
};
template <typename T> struct coeffs1d<CC, POLY, /*order*/ 1, T> {
  static constexpr std::array<T, 2> coeffs = {
      +1 / T(4),
      +3 / T(4),
  };
};
template <typename T> struct coeffs1d<CC, POLY, /*order*/ 2, T> {
  static constexpr std::array<T, 3> coeffs = {
      +5 / T(32),
      +15 / T(16),
      -3 / T(32),
  };
};
template <typename T> struct coeffs1d<CC, POLY, /*order*/ 3, T> {
  static constexpr std::array<T, 4> coeffs = {
      -5 / T(128),
      +35 / T(128),
      +105 / T(128),
      -7 / T(128),
  };
};
template <typename T> struct coeffs1d<CC, POLY, /*order*/ 4, T> {
  static constexpr std::array<T, 5> coeffs = {
      -45 / T(2048), +105 / T(512), +945 / T(1024), -63 / T(512), +35 / T(2048),
  };
};
template <typename T> struct coeffs1d<CC, POLY, /*order*/ 5, T> {
  static constexpr std::array<T, 6> coeffs = {
      +63 / T(8192),   -495 / T(8192), +1155 / T(4096),
      +3465 / T(4096), -693 / T(8192), +77 / T(8192),
  };
};
template <typename T> struct coeffs1d<CC, POLY, /*order*/ 6, T> {
  static constexpr std::array<T, 7> coeffs = {
      +273 / T(65536),  -1287 / T(32768), +15015 / T(65536), +15015 / T(16384),
      -9009 / T(65536), +1001 / T(32768), -231 / T(65536),
  };
};
template <typename T> struct coeffs1d<CC, POLY, /*order*/ 7, T> {
  static constexpr std::array<T, 8> coeffs = {
      -429 / T(262144),   +4095 / T(262144),   -19305 / T(262144),
      +75075 / T(262144), +225225 / T(262144), -27027 / T(262144),
      +5005 / T(262144),  -495 / T(262144),
  };
};

// Hermite interpolation (with matched first derivatives)

// Linear Hermite interpolation is the same as linear Lagrange interpolation
template <typename T> struct coeffs1d<VC, HERMITE, /*order*/ 1, T> {
  static constexpr std::array<T, 4> coeffs = {
      +1 / T(2),
      +1 / T(2),
  };
};
// Cubic Hermite interpolation is the same as cubic Lagrange interpolation
template <typename T> struct coeffs1d<VC, HERMITE, /*order*/ 3, T> {
  static constexpr std::array<T, 4> coeffs = {
      -1 / T(16),
      +9 / T(16),
      +9 / T(16),
      -1 / T(16),
  };
};
template <typename T> struct coeffs1d<VC, HERMITE, /*order*/ 5, T> {
  static constexpr std::array<T, 6> coeffs = {
      +121 / T(8192),  -875 / T(8192), +2425 / T(4096),
      +2425 / T(4096), -875 / T(8192), +121 / T(8192),
  };
};
template <typename T> struct coeffs1d<VC, HERMITE, /*order*/ 7, T> {
  static constexpr std::array<T, 6> coeffs = {
      -129 / T(32768),     +1127 / T(36864),    -6419 / T(49152),
      +178115 / T(294912), +178115 / T(294912), -6419 / T(49152),
      +1127 / T(36864),    -129 / T(32768),
  };
};

// Deprecated
template <typename T> struct coeffs1d<VC, CONS, /*order*/ 0, T> {
  static constexpr std::array<T, 1> coeffs0 = {
      +1 / T(1),
  };
  static constexpr std::array<T, 0> coeffs1 = {};
};
template <typename T> struct coeffs1d<VC, CONS, /*order*/ 1, T> {
  static constexpr std::array<T, 1> coeffs0 = {
      +1 / T(1),
  };
  static constexpr std::array<T, 2> coeffs1 = {
      +1 / T(2),
      +1 / T(2),
  };
};
// template <typename T> struct coeffs1d<VC, CONS, /*order*/ 2, T> {
//   static constexpr std::array<T, 3> coeffs0 = {
//       -1 / T(32),
//       +17 / T(16),
//       -1 / T(32),
//   };
//   static constexpr std::array<T, 3> coeffs1 = {
//       +13 / T(16),
//       -5 / T(32),
//       +11 / T(32),
//   };
// };
// template <typename T> struct coeffs1d<VC, CONS, /*order*/ 4, T> {
//   static constexpr std::array<T, 5> coeffs0 = {
//       +7 / T(2048), -23 / T(512), +1109 / T(1024), -23 / T(512), +7 /
//       T(2048),
//   };
//   static constexpr std::array<T, 5> coeffs1 = {
//       +63 / T(2048), -103 / T(512), +781 / T(1024),
//       +233 / T(512), -97 / T(2048),
//   };
// };

template <typename T> struct coeffs1d<CC, CONS, /*order*/ 0, T> {
  static constexpr std::array<T, 1> coeffs = {
      +1 / T(1),
  };
};
template <typename T> struct coeffs1d<CC, CONS, /*order*/ 1, T> {
  static constexpr std::array<T, 2> coeffs = {
      +1 / T(4),
      +3 / T(4),
  };
};
template <typename T> struct coeffs1d<CC, CONS, /*order*/ 2, T> {
  static constexpr std::array<T, 3> coeffs = {
      +1 / T(8),
      +1 / T(1),
      -1 / T(8),
  };
};
template <typename T> struct coeffs1d<CC, CONS, /*order*/ 3, T> {
  static constexpr std::array<T, 4> coeffs = {
      -3 / T(64),
      +17 / T(64),
      +55 / T(64),
      -5 / T(64),
  };
};
template <typename T> struct coeffs1d<CC, CONS, /*order*/ 4, T> {
  static constexpr std::array<T, 5> coeffs = {
      -3 / T(128), +11 / T(64), +1 / T(1), -11 / T(64), +3 / T(128),
  };
};
template <typename T> struct coeffs1d<CC, CONS, /*order*/ 5, T> {
  static constexpr std::array<T, 6> coeffs = {
      +5 / T(512),   -37 / T(512), +69 / T(256),
      +231 / T(256), -63 / T(512), +7 / T(512),
  };
};
template <typename T> struct coeffs1d<CC, CONS, /*order*/ 6, T> {
  static constexpr std::array<T, 7> coeffs = {
      +5 / T(1024),   -11 / T(256), +201 / T(1024), +1 / T(1),
      -201 / T(1024), +11 / T(256), -5 / T(1024),
  };
};
template <typename T> struct coeffs1d<CC, CONS, /*order*/ 7, T> {
  static constexpr std::array<T, 8> coeffs = {
      -35 / T(16384),    +325 / T(16384),  -1439 / T(16384), +4441 / T(16384),
      +15159 / T(16384), -2481 / T(16384), +459 / T(16384),  -45 / T(16384),
  };
};
template <typename T> struct coeffs1d<CC, CONS, /*order*/ 8, T> {
  static constexpr std::array<T, 9> coeffs = {
      -35 / T(32768),   +185 / T(16384), -949 / T(16384),
      +3461 / T(16384), +1 / T(1),       -3461 / T(16384),
      +949 / T(16384),  -185 / T(16384), +35 / T(32768),
  };
};

template <typename T> struct coeffs1d<CC, ENO, /*order*/ 0, T> {
  static constexpr std::array<std::array<T, 1>, 1> coeffs = {{
      // centred
      {
          +1 / T(1),
      },
  }};
};
template <typename T> struct coeffs1d<CC, ENO, /*order*/ 2, T> {
  static constexpr std::array<std::array<T, 3>, 3> coeffs = {{
      // left
      {
          -1 / T(8),
          +1 / T(2),
          +5 / T(8),
      },
      // centred
      {
          +1 / T(8),
          +1 / T(1),
          -1 / T(8),
      },
      // right
      {
          +11 / T(8),
          -1 / T(2),
          +1 / T(8),
      },
  }};
};
template <typename T> struct coeffs1d<CC, ENO, /*order*/ 4, T> {
  static constexpr std::array<std::array<T, 5>, 5> coeffs = {{
      // left 2 cells
      {
          -7 / T(128),
          +19 / T(64),
          -11 / T(16),
          +61 / T(64),
          +63 / T(128),
      },
      // left 1 cell
      {
          +3 / T(128),
          -9 / T(64),
          +13 / T(32),
          +49 / T(64),
          -7 / T(128),
      },
      // centred
      {
          -3 / T(128),
          +11 / T(64),
          +1 / T(1),
          -11 / T(64),
          +3 / T(128),
      },
      // right 1 cell
      {
          +7 / T(128),
          +79 / T(64),
          -13 / T(32),
          +9 / T(64),
          -3 / T(128),
      },
      // right 2 cells
      {
          +193 / T(128),
          -61 / T(64),
          +11 / T(16),
          -19 / T(64),
          +7 / T(128),
      },
  }};
};

// 1D interpolation operators

template <centering_t CENT, interpolation_t INTP, int ORDER> struct interp1d;
// static constexpr int required_ghosts;
// constexpr std::array<int, 2> stencil_radius(int shift, int off) const;
// template <typename F>
// T operator()(const F& crse, int shift, int off) const;

// off=0: on coarse point
// off=1: between coarse points
template <int ORDER> struct interp1d<VC, POLY, ORDER> {
  static_assert(ORDER % 2 == 1);
  static constexpr int required_ghosts = (ORDER + 1) / 2;
  CCTK_DEVICE
  CCTK_HOST constexpr CCTK_ATTRIBUTE_ALWAYS_INLINE std::array<int, 2>
  stencil_radius(const int shift, const int off) const {
    if (off == 0)
      return {0, 0};
    constexpr int N = ORDER + 1;
    const int i0 = N / 2 - off;
    return {0 - i0, ORDER - i0};
  }
  template <typename F, typename T = std::invoke_result_t<F, int> >
  CCTK_DEVICE CCTK_HOST inline CCTK_ATTRIBUTE_ALWAYS_INLINE T
  operator()(const F &crse, const int shift, const int off) const {
#ifdef CCTK_DEBUG
    assert(off == 0 || off == 1);
#endif

    if (off == 0)
      return crse(0);

    constexpr int N = ORDER + 1;
    constexpr std::array<T, N> cs = coeffs1d<VC, POLY, ORDER, T>::coeffs;
    const int i0 = N / 2 - off;

    // nvcc doesn't accept the constexpr terms below
#ifndef __CUDACC__
    constexpr int i0min = N / 2 - 1;
    constexpr int i0max = N / 2;
    constexpr int imin = 0;
    constexpr int imax = N / 2 - 1;
    constexpr int i1min = N - 1 - imax;
    constexpr int i1max = N - 1 - imin;
    const auto abs0 = [](auto x) { return x >= 0 ? x : -x; };
    static_assert(abs0(imin - i0min) <= required_ghosts);
    static_assert(abs0(imin - i0max) <= required_ghosts);
    static_assert(abs0(imax - i0min) <= required_ghosts);
    static_assert(abs0(imax - i0max) <= required_ghosts);
    static_assert(abs0(i1min - i0min) <= required_ghosts);
    static_assert(abs0(i1min - i0max) <= required_ghosts);
    static_assert(abs0(i1max - i0min) <= required_ghosts);
    static_assert(abs0(i1max - i0max) <= required_ghosts);
#endif

    T y = 0;
    // Make use of symmetry in coefficients
    for (int i = 0; i < N / 2; ++i) {
      const int i1 = ORDER - i;
#ifdef CCTK_DEBUG
      assert(cs[i1] == cs[i]);
#endif
      y += cs[i] * (crse(i - i0) + crse(i1 - i0));
    }
#ifdef CCTK_DEBUG
    assert(isfinite(y));
#endif
    return y;
  }
};

// off=0: left sub-cell
// off=1: right sub-cell
template <int ORDER> struct interp1d<CC, POLY, ORDER> {
  static constexpr int required_ghosts = (ORDER + 1) / 2;
  CCTK_DEVICE
  CCTK_HOST constexpr CCTK_ATTRIBUTE_ALWAYS_INLINE std::array<int, 2>
  stencil_radius(const int shift, const int off) const {
#ifdef CCTK_DEBUG
    assert(off == 0 || off == 1);
#endif
    constexpr int N = ORDER + 1;
    constexpr int i0 = N / 2;
    if (off == 0)
      return {0 - i0, N - 1 - i0};
    else
      return {0 - i0 + ORDER % 2, N - 1 - i0 + ORDER % 2};
  }
  template <typename F, typename T = std::invoke_result_t<F, int> >
  CCTK_DEVICE CCTK_HOST inline CCTK_ATTRIBUTE_ALWAYS_INLINE T
  operator()(const F &crse, const int shift, const int off) const {
#ifdef CCTK_DEBUG
    assert(off == 0 || off == 1);
#endif
    constexpr int N = ORDER + 1;
    constexpr std::array<T, N> cs = coeffs1d<CC, POLY, ORDER, T>::coeffs;
    constexpr int i0 = N / 2;

    // nvcc doesn't accept the constexpr terms below
#ifndef __CUDACC__
    constexpr int imin = 0;
    constexpr int imax = ORDER;
    const auto abs0 = [](auto x) { return x >= 0 ? x : -x; };
    static_assert(abs0(imin - i0) <= required_ghosts);
    static_assert(abs0(imax - i0) <= required_ghosts);
#endif

    T y = 0;
    if (off == 0)
      for (int i = 0; i < N; ++i)
        y += cs[i] * crse(i - i0);
    else
      for (int i = 0; i < N; ++i)
        // For odd orders, the stencil has an even number of points and is
        // thus offset. This offset moves the stencil right by one point
        // when it is reversed.
        y += cs[ORDER - i] * crse(i - i0 + ORDER % 2);
#ifdef CCTK_DEBUG
    assert(isfinite(y));
#endif
    return y;
  }
};

// off=0: on coarse point
// off=1: between coarse points
template <int ORDER> struct interp1d<VC, HERMITE, ORDER> {
  static_assert(ORDER % 2 == 1);
  static constexpr int required_ghosts = (ORDER + 1) / 2;
  CCTK_DEVICE
  CCTK_HOST constexpr CCTK_ATTRIBUTE_ALWAYS_INLINE std::array<int, 2>
  stencil_radius(const int shift, const int off) const {
#ifdef CCTK_DEBUG
    assert(off == 0 || off == 1);
#endif
    if (off == 0)
      return {0, 0};
    constexpr int N = ORDER + 1;
    const int i0 = N / 2 - off;
    return {0 - i0, N - 1 - i0};
  }
  template <typename F, typename T = std::invoke_result_t<F, int> >
  CCTK_DEVICE CCTK_HOST inline CCTK_ATTRIBUTE_ALWAYS_INLINE T
  operator()(const F &crse, const int shift, const int off) const {
#ifdef CCTK_DEBUG
    assert(off == 0 || off == 1);
#endif

    if (off == 0)
      return crse(0);

    constexpr int N = ORDER + 1;
    constexpr std::array<T, N> cs = coeffs1d<VC, POLY, N - 1, T>::coeffs;
    const int i0 = N / 2 - off;
#ifndef __CUDACC__
    constexpr int i0min = N / 2 - 1;
    constexpr int i0max = N / 2;
    constexpr int imin = 0;
    constexpr int imax = N / 2 - 1;
    constexpr int i1min = N - 1 - imax;
    constexpr int i1max = N - 1 - imin;
    // nvcc doesn't accept the constexpr terms below
    const auto abs0 = [](auto x) { return x >= 0 ? x : -x; };
    static_assert(abs0(imin - i0min) <= required_ghosts);
    static_assert(abs0(imin - i0max) <= required_ghosts);
    static_assert(abs0(imax - i0min) <= required_ghosts);
    static_assert(abs0(imax - i0max) <= required_ghosts);
    static_assert(abs0(i1min - i0min) <= required_ghosts);
    static_assert(abs0(i1min - i0max) <= required_ghosts);
    static_assert(abs0(i1max - i0min) <= required_ghosts);
    static_assert(abs0(i1max - i0max) <= required_ghosts);
#endif

    T y = 0;
    // Make use of symmetry in coefficients
    for (int i = 0; i < N / 2; ++i) {
      const int i1 = N - 1 - i;
#ifdef CCTK_DEBUG
      assert(cs[i1] == cs[i]);
#endif
      y += cs[i] * (crse(i - i0) + crse(i1 - i0));
    }
#ifdef CCTK_DEBUG
    assert(isfinite(y));
#endif
    return y;
  }
};

// Deprecated
// off=0: on coarse point
// off=1: between coarse points
template <int ORDER> struct interp1d<VC, CONS, ORDER> {
  static constexpr int required_ghosts = (ORDER + 1) / 2;
  CCTK_DEVICE
  CCTK_HOST constexpr CCTK_ATTRIBUTE_ALWAYS_INLINE std::array<int, 2>
  stencil_radius(const int shift, const int off) const {
#ifdef CCTK_DEBUG
    assert(shift == 0);
    assert(off == 0 || off == 1);
#endif
    if (off == 0) {
      constexpr int i0 = ORDER / 2;
      return {0 - i0, ORDER / 2 * 2 - i0};
    } else {
      constexpr int i0 = (ORDER + 1) / 2 - 1;
      return {0 - i0, (ORDER + 1) / 2 * 2 - 1 - i0};
    }
  }
  template <typename F, typename T = std::invoke_result_t<F, int> >
  CCTK_DEVICE CCTK_HOST inline CCTK_ATTRIBUTE_ALWAYS_INLINE T
  operator()(const F &crse, const int shift, const int off) const {
#ifdef CCTK_DEBUG
    assert(shift == 0);
    assert(off == 0 || off == 1);
#endif

    T y = 0;
    // TODO: use symmetry
    if (off == 0) {
      constexpr int i0 = ORDER / 2;
      constexpr std::array<T, ORDER / 2 * 2 + 1> cs =
          coeffs1d<VC, CONS, ORDER, T>::coeffs0;
      for (int i = 0; i < ORDER / 2 * 2 + 1; ++i)
        y += cs[i] * crse(i - i0);
    } else {
      constexpr int i0 = (ORDER + 1) / 2 - 1;
      constexpr std::array<T, (ORDER + 1) / 2 * 2> cs =
          coeffs1d<VC, CONS, ORDER, T>::coeffs1;
      for (int i = 0; i < (ORDER + 1) / 2 * 2; ++i)
        y += cs[i] * crse(i - i0);
    }
    return y;
  }
};

// off=0: left sub-cell
// off=1: right sub-cell
template <int ORDER> struct interp1d<CC, CONS, ORDER> {
  static constexpr int required_ghosts = (ORDER + 1) / 2;
  CCTK_DEVICE
  CCTK_HOST constexpr CCTK_ATTRIBUTE_ALWAYS_INLINE std::array<int, 2>
  stencil_radius(const int shift, const int off) const {
#ifdef CCTK_DEBUG
    assert(shift == 0);
    assert(off == 0 || off == 1);
#endif
    if (off == 0)
      return {-((ORDER + 1) / 2), +(ORDER / 2)};
    else
      return {-(ORDER / 2), +((ORDER + 1) / 2)};
  }
  template <typename F, typename T = std::invoke_result_t<F, int> >
  CCTK_DEVICE CCTK_HOST inline CCTK_ATTRIBUTE_ALWAYS_INLINE T
  operator()(const F &crse, const int shift, const int off) const {
#ifdef CCTK_DEBUG
    assert(shift == 0);
    assert(off == 0 || off == 1);
#endif

    constexpr int N = ORDER + 1;
    constexpr std::array<T, N> cs = coeffs1d<CC, CONS, N - 1, T>::coeffs;
    constexpr int i0 = N / 2;
#ifndef __CUDACC__
    constexpr int imin = 0;
    constexpr int imax = N - 1;
    // nvcc doesn't accept the constexpr terms below
    const auto abs0 = [](auto x) { return x >= 0 ? x : -x; };
    static_assert(abs0(imin - i0) <= required_ghosts);
    static_assert(abs0(imax - i0) <= required_ghosts);
#endif

    T y;
    if (ORDER % 2 == 0) {
      if (off == 0) {
        y = cs[ORDER / 2] * crse(ORDER / 2 - i0);
        // Make use of symmetry in coefficients
        for (int i = 0; i < ORDER / 2; ++i) {
          const int i1 = ORDER - i;
#ifdef CCTK_DEBUG
          assert(cs[i1] == -cs[i]);
#endif
          y += cs[i] * (crse(i - i0) - crse(i1 - i0));
        }
      } else {
        y = cs[ORDER / 2] * crse(ORDER / 2 - i0);
        // Make use of symmetry in coefficients
        for (int i = 0; i < ORDER / 2; ++i) {
          const int i1 = ORDER - i;
#ifdef CCTK_DEBUG
          assert(cs[i1] == -cs[i]);
#endif
          y += cs[i] * (crse(i1 - i0) - crse(i - i0));
        }
      }
    } else {
      if (off == 0) {
        y = 0;
        for (int i = 0; i < N; ++i) {
          assert(i - i0 >= -((ORDER + 1) / 2));
          assert(i - i0 <= +(ORDER / 2));
          y += cs[i] * crse(i - i0);
        }
      } else {
        y = 0;
        for (int i = 0; i < N; ++i) {
          // TODO y += cs[(N - 1) - i] * crse(i - (i0 - 1));
          static_assert((+((ORDER + 1) / 2)) - (-(ORDER / 2)) + 1 == N);
          assert((ORDER + 1) / 2 - i >= -(ORDER / 2));
          assert((ORDER + 1) / 2 - i <= +((ORDER + 1) / 2));
          y += cs[i] * crse((ORDER + 1) / 2 - i);
          // TODO y += cs[(N - 1) - i] * crse(i + i0 + 1);
        }
      }
    }

    return y;
  }
};

template <int N> struct undivided_difference_weights;
template <> struct undivided_difference_weights<1> {
  static constexpr std::array<int, 1> weights = {+1};
};
template <> struct undivided_difference_weights<2> {
  static constexpr std::array<int, 2> weights = {-1, +1};
};
template <> struct undivided_difference_weights<3> {
  static constexpr std::array<int, 3> weights = {+1, -2, +1};
};
template <> struct undivided_difference_weights<4> {
  static constexpr std::array<int, 4> weights = {-1, +3, -3, +1};
};
template <> struct undivided_difference_weights<5> {
  static constexpr std::array<int, 5> weights = {+1, -4, +6, -4, +1};
};
template <> struct undivided_difference_weights<6> {
  static constexpr std::array<int, 6> weights = {-1, +5, -10, +10, -5, +1};
};
template <> struct undivided_difference_weights<7> {
  static constexpr std::array<int, 7> weights = {+1, -6, +15, -20, +15, -6, +1};
};

template <centering_t CENT, interpolation_t INTP, int ORDER>
struct undivided_difference_1d {
  static constexpr int required_ghosts = 0;
  template <typename F, typename T = std::invoke_result_t<F, int> >
  CCTK_DEVICE CCTK_HOST inline CCTK_ATTRIBUTE_ALWAYS_INLINE T
  operator()(const F &crse) const {
    return 0;
  }
};

template <int ORDER> struct undivided_difference_1d<CC, ENO, ORDER> {
  static_assert(ORDER % 2 == 0);
  // The stencil size is ORDER / 2, and the stencil shift can be ORDER / 2
  // static constexpr int required_ghosts = ORDER;
  template <typename F, typename T = std::invoke_result_t<F, int> >
  CCTK_DEVICE CCTK_HOST inline CCTK_ATTRIBUTE_ALWAYS_INLINE T
  operator()(const F &crse) const {
    constexpr int N = ORDER + 1;
    constexpr std::array<int, N> ws = undivided_difference_weights<N>::weights;
    constexpr std::ptrdiff_t i0 = ORDER / 2;

    T dd;
    if constexpr (ORDER % 2 == 0) {
      // Make use of symmetry in coefficients
      dd = ws[ORDER / 2] * crse(0);
      for (int i = 0; i < ORDER / 2; ++i) {
        const int i1 = ORDER - i;
#ifdef CCTK_DEBUG
        assert(ws[i1] == ws[i]);
#endif
        dd += ws[i] * (crse(i - i0) + crse(i1 - i0));
      }
    } else {
      // Make use of antisymmetry in coefficients
      // TODO: implement this
      assert(false);
      dd = 0;
      for (int i = 0; i < ORDER / 2; ++i) {
        const int i1 = ORDER - i;
#ifdef CCTK_DEBUG
        assert(ws[i1] == -ws[i]);
#endif
        dd += ws[i] * (crse(i - i0) - crse(i1 - i0));
      }
    }
    return dd;
  }
};

// off=0: left sub-cell
// off=1: right sub-cell
template <int ORDER> struct interp1d<CC, ENO, ORDER> {
  static_assert(ORDER % 2 == 0);
  static constexpr int required_ghosts = ORDER;
  CCTK_DEVICE
  CCTK_HOST constexpr CCTK_ATTRIBUTE_ALWAYS_INLINE std::array<int, 2>
  stencil_radius(const int shift, const int off) const {
#ifdef CCTK_DEBUG
    assert(-ORDER / 2 <= shift && shift <= +ORDER / 2);
    assert(off == 0 || off == 1);
#endif
    constexpr int N = ORDER + 1;
    const int i0 = N / 2 - shift;
    return {0 - i0, ORDER - i0};
  }
  template <typename F, typename T = std::invoke_result_t<F, int> >
  CCTK_DEVICE CCTK_HOST inline CCTK_ATTRIBUTE_ALWAYS_INLINE T
  operator()(const F &crse, const int shift, const int off) const {
#ifdef CCTK_DEBUG
    assert(-ORDER / 2 <= shift && shift <= +ORDER / 2);
    assert(off == 0 || off == 1);
#endif
    // For off=1, use the reversed stencil for the opposite shift
    // const std::array<T, ORDER + 1> cs =
    //     coeffs1d<CC, ENO, ORDER, T>::coeffs[ORDER / 2 +
    //                                         (off == 0 ? +1 : -1) * shift];
    constexpr int N = ORDER + 1;
    const std::array<std::array<T, N>, N> css =
        coeffs1d<CC, ENO, ORDER, T>::coeffs;
    const std::array<T, N> &cs = css[ORDER / 2 + (off == 0 ? +1 : -1) * shift];
    const int i0 = N / 2 - shift;

#ifdef CCTK_DEBUG
    constexpr int imin = 0;
    constexpr int imax = ORDER;
    assert(abs(imin - i0) <= required_ghosts);
    assert(abs(imax - i0) <= required_ghosts);
#endif

    T y = 0;
    if (off == 0)
      for (int i = 0; i <= ORDER; ++i)
        y += cs[i] * crse(i - i0);
    else
      for (int i = 0; i <= ORDER; ++i)
        y += cs[ORDER - i] * crse(i - i0);
    return y;
  }
};

// Test 1d interpolators

template <centering_t CENT, interpolation_t INTP, int ORDER, typename T>
struct test_interp1d;

template <centering_t CENT, int ORDER, typename T>
struct test_interp1d<CENT, POLY, ORDER, T> {
  test_interp1d() {
    constexpr interp1d<CENT, POLY, ORDER> stencil1d;
    constexpr int nghosts = stencil1d.required_ghosts;
    static_assert(nghosts >= 0);
    constexpr int n = 1 + 2 * (nghosts + 1);
    constexpr int i0 = n / 2;
    std::array<T, n> ysarr;
    T *restrict const ys = &ysarr[i0];

    for (int order = 0; order <= ORDER; ++order) {
      auto f = [&](T x) { return pown(x, order); };
      for (int off = 0; off < 2; ++off) {
        const T rmin = stencil1d.stencil_radius(0, off)[0];
        const T rmax = stencil1d.stencil_radius(0, off)[1];
        assert(rmin <= 0 && rmin >= -nghosts);
        assert(rmax >= 0 && rmax <= +nghosts);
        for (int i = -(nghosts + 1); i <= +(nghosts + 1); ++i) {
          if (i < rmin || i > rmax) {
            ys[i] = 0 / T(0);
          } else {
            T x = i + int(CENT) / T(2);
            T y = f(x);
            ys[i] = y;
          }
        }

        T x = int(CENT) / T(4) + off / T(2);
        T y = f(x);
        T y1 = stencil1d(
            [&](int i) {
              assert(i >= rmin);
              assert(i <= rmax);
              return ys[i];
            },
            0, off);
        // We carefully choose the test problem so that round-off
        // cannot be a problem here
        assert(isfinite(y1));
        assert(y1 == y);
      }
    }
  }
};

template <int ORDER, typename T> struct test_interp1d<VC, HERMITE, ORDER, T> {
  test_interp1d() {
    constexpr interp1d<VC, HERMITE, ORDER> stencil1d;
    constexpr int nghosts = stencil1d.required_ghosts;
    static_assert(nghosts >= 0);
    constexpr int n = 1 + 2 * nghosts;
    constexpr int i0 = n / 2;
    std::array<T, n + 2> ys;

    for (int order = 0; order <= ORDER; ++order) {
      auto f = [&](T x) { return pown(x, order); };
      for (int off = 0; off < 2; ++off) {
        const auto [rmin, rmax] = stencil1d.stencil_radius(0, off);
        assert(rmin >= -nghosts && rmax <= +nghosts);
        for (int i = -1; i < n + 1; ++i) {
          if (i - i0 < rmin || i - i0 > rmax) {
            ys[i + 1] = 0 / T(0);
          } else {
            T x = (i - i0) + int(VC) / T(2);
            T y = f(x);
            ys[i + 1] = y;
          }
        }

        T x = int(VC) / T(4) + off / T(2);
        T y = f(x);
        T y1 = stencil1d([&](int i) { return ys[i0 + 1 + i]; }, 0, off);
        // We carefully choose the test problem so that round-off
        // cannot be a problem here
        assert(isfinite(y1));
        assert(y1 == y);
      }
    }
  }
};

template <int ORDER, typename T> struct test_interp1d<CC, CONS, ORDER, T> {
  test_interp1d() {
    constexpr interp1d<CC, CONS, ORDER> stencil1d;
    constexpr int nghosts = stencil1d.required_ghosts;
    static_assert(nghosts >= 0);
    constexpr int n = 1 + 2 * (nghosts + 1);
    constexpr int i0 = n / 2;
    std::array<T, n> xsarr, ysarr;
    T *restrict const xs = &xsarr[i0];
    T *restrict const ys = &ysarr[i0];

    for (int order = 0; order <= ORDER; ++order) {
      // Function f, a polynomial
      // const auto f{[&](T x) { return (order + 1) * pown(x, order); }};
      // Integral of f (antiderivative)
      const auto fint{[&](T x) { return pown(x, order + 1); }};
      std::array<T, 2> x1;
      std::array<T, 2> y1;
      for (int off = 0; off < 2; ++off) {
        const T rmin = stencil1d.stencil_radius(0, off)[0];
        const T rmax = stencil1d.stencil_radius(0, off)[1];
        assert(rmin <= 0 && rmin >= -nghosts);
        assert(rmax >= 0 && rmax <= +nghosts);
        for (int i = -(nghosts + 1); i <= +(nghosts + 1); ++i) {
          if (i < rmin || i > rmax) {
            xs[i] = 0 / T(0);
            ys[i] = 0 / T(0);
          } else {
            T x = i + int(CC) / T(2);
            // T y = f(x);
            const T dx = 1;
            const T xlo = x - dx / 2;
            const T xhi = x + dx / 2;
            const T y = fint(xhi) - fint(xlo); // average of f over cell
            xs[i] = x;
            ys[i] = y;
          }
        }

        x1[off] = int(CC) / T(4) + off / T(2);
        y1[off] = stencil1d(
            [&](const int i) {
              assert(i >= rmin);
              assert(i <= rmax);
              return ys[i];
            },
            0, off);
        assert(isfinite(y1[off]));
      } // for off
      assert(y1[0] / 2 + y1[1] / 2 == ys[0]);
      const T dx = x1[1] - x1[0];
      const T xlo = x1[0] - dx / 2;
      const T xhi = x1[1] + dx / 2;
      const T yint = fint(xhi) - fint(xlo);
      assert(y1[0] * dx + y1[1] * dx == yint);
    }
  }
};

template <int ORDER, typename T> struct test_interp1d<VC, CONS, ORDER, T> {
  test_interp1d() {
    constexpr interp1d<VC, CONS, ORDER> stencil1d;
    constexpr int nghosts = stencil1d.required_ghosts;
    static_assert(nghosts >= 0);

    // Don't test this, the case (VC,CONS) should not be used
  }
};

template <int ORDER, typename T> struct test_interp1d<CC, ENO, ORDER, T> {
  test_interp1d() {
    constexpr interp1d<CC, ENO, ORDER> stencil1d;
    constexpr int nghosts = stencil1d.required_ghosts;
    static_assert(nghosts >= 0);
    constexpr int n = 1 + 2 * nghosts;
    constexpr int i0 = n / 2;
    std::array<T, n + 2> xs, ys;

    static_assert(ORDER % 2 == 0);
    for (int shift = -ORDER / 2; shift <= +ORDER / 2; ++shift) {
      for (int order = 0; order <= ORDER; ++order) {
        // Function f, a polynomial
        // const auto f{[&](T x) { return (order + 1) * pown(x, order); }};
        // Integral of f (antiderivative)
        const auto fint{[&](T x) { return pown(x, order + 1); }};
        std::array<T, 2> x1;
        std::array<T, 2> y1;
        for (int off = 0; off < 2; ++off) {
          const T rmin = stencil1d.stencil_radius(0, off)[0];
          const T rmax = stencil1d.stencil_radius(0, off)[1];
          assert(rmin >= -nghosts && rmax <= +nghosts);
          for (int i = -1; i < n + 1; ++i) {
            if (i - i0 < rmin || i - i0 > rmax) {
              xs[i + 1] = 0 / T(0);
              ys[i + 1] = 0 / T(0);
            } else {
              T x = (i - i0) + int(CC) / T(2);
              // T y = f(x);
              const T dx = 1;
              const T xlo = x - dx / 2;
              const T xhi = x + dx / 2;
              const T y = fint(xhi) - fint(xlo); // average of f over cell
              xs[i + 1] = x;
              ys[i + 1] = y;
            }
          }

          x1[off] = int(CC) / T(4) + off / T(2);
          y1[off] = stencil1d([&](int i) { return ys[i0 + 1 + i]; }, 0, off);
          assert(isfinite(y1[off]));
        } // for off
        assert(y1[0] / 2 + y1[1] / 2 == ys[i0 + 1]);
        const T dx = x1[1] - x1[0];
        const T xlo = x1[0] - dx / 2;
        const T xhi = x1[1] + dx / 2;
        const T yint = fint(xhi) - fint(xlo);
        assert(y1[0] * dx + y1[1] * dx == yint);
      }
    }
  }
};

////////////////////////////////////////////////////////////////////////////////

template <typename F, typename T = std::invoke_result_t<F> >
CCTK_DEVICE CCTK_HOST inline CCTK_ATTRIBUTE_ALWAYS_INLINE T
call_stencil_0d(const F &crse) {
  return crse();
}

template <typename F, typename Si, typename T = std::invoke_result_t<F, int> >
CCTK_DEVICE CCTK_HOST inline CCTK_ATTRIBUTE_ALWAYS_INLINE T
call_stencil_1d(const F &crse, const Si &si) {
  return si(
      [&](const int i) { return call_stencil_0d([&]() { return crse(i); }); });
}

template <typename F, typename Si, typename Sj,
          typename T = std::invoke_result_t<F, int, int> >
CCTK_DEVICE CCTK_HOST inline CCTK_ATTRIBUTE_ALWAYS_INLINE T
call_stencil_2d(const F &crse, const Si &si, const Sj &sj) {
  return sj([&](const int j) {
    return call_stencil_1d([&](const int i) { return crse(i, j); }, si);
  });
}

template <typename F, typename Si, typename Sj, typename Sk,
          typename T = std::invoke_result_t<F, int, int, int> >
CCTK_DEVICE CCTK_HOST inline CCTK_ATTRIBUTE_ALWAYS_INLINE T
call_stencil_3d(const F &crse, const Si &si, const Sj &sj, const Sk &sk) {
  return sk([&](const int k) {
    return call_stencil_2d(
        [&](const int i, const int j) { return crse(i, j, k); }, si, sj);
  });
}

////////////////////////////////////////////////////////////////////////////////

template <centering_t CENTI, centering_t CENTJ, centering_t CENTK,
          interpolation_t INTPI, interpolation_t INTPJ, interpolation_t INTPK,
          int ORDERI, int ORDERJ, int ORDERK, fallback_t FB>
prolongate_3d_rf2<CENTI, CENTJ, CENTK, INTPI, INTPJ, INTPK, ORDERI, ORDERJ,
                  ORDERK, FB>::~prolongate_3d_rf2() {}

template <centering_t CENTI, centering_t CENTJ, centering_t CENTK,
          interpolation_t INTPI, interpolation_t INTPJ, interpolation_t INTPK,
          int ORDERI, int ORDERJ, int ORDERK, fallback_t FB>
amrex::Box
prolongate_3d_rf2<CENTI, CENTJ, CENTK, INTPI, INTPJ, INTPK, ORDERI, ORDERJ,
                  ORDERK, FB>::CoarseBox(const amrex::Box &fine,
                                         const int ratio) {
  return CoarseBox(fine, amrex::IntVect(ratio));
}

template <centering_t CENTI, centering_t CENTJ, centering_t CENTK,
          interpolation_t INTPI, interpolation_t INTPJ, interpolation_t INTPK,
          int ORDERI, int ORDERJ, int ORDERK, fallback_t FB>
amrex::Box
prolongate_3d_rf2<CENTI, CENTJ, CENTK, INTPI, INTPJ, INTPK, ORDERI, ORDERJ,
                  ORDERK, FB>::CoarseBox(const amrex::Box &fine,
                                         const amrex::IntVect &ratio) {
  constexpr vect<int, dim> required_ghosts = {
      interp1d<CENTI, INTPI, ORDERI>::required_ghosts,
      interp1d<CENTJ, INTPJ, ORDERJ>::required_ghosts,
      interp1d<CENTK, INTPK, ORDERK>::required_ghosts,
  };
  for (int d = 0; d < dim; ++d)
    assert(ratio.getVect()[d] == 2);
  amrex::Box crse = amrex::coarsen(fine, 2);
  for (int d = 0; d < dim; ++d)
    crse.grow(d, required_ghosts[d]);
  return crse;
}

template <centering_t CENTI, centering_t CENTJ, centering_t CENTK,
          interpolation_t INTPI, interpolation_t INTPJ, interpolation_t INTPK,
          int ORDERI, int ORDERJ, int ORDERK, fallback_t FB>
void prolongate_3d_rf2<
    CENTI, CENTJ, CENTK, INTPI, INTPJ, INTPK, ORDERI, ORDERJ, ORDERK,
    FB>::interp_per_var(const amrex::FArrayBox &crse, const int crse_comp,
                        amrex::FArrayBox &fine, const int fine_comp,
                        const int ncomp, const amrex::Box &fine_region,
                        const amrex::IntVect &ratio,
                        const amrex::Geometry &crse_geom,
                        const amrex::Geometry &fine_geom,
                        amrex::Vector<amrex::BCRec> const &bcr,
                        const int actual_comp, const int actual_state,
                        const amrex::RunOn gpu_or_cpu) {
  DECLARE_CCTK_PARAMETERS;

  static std::once_flag have_timers;
  static std::vector<Timer> timers;

  const int thread_num = omp_get_thread_num();

  call_once(have_timers, [&]() {
    const int num_threads = omp_get_num_threads();
    timers.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
      std::ostringstream buf;
      buf << "prolongate_3d_rf2<[" << CENTI << "," << CENTJ << "," << CENTK
          << "],[" << INTPI << "," << INTPJ << "," << INTPK << "],[" << ORDERI
          << "," << ORDERJ << "," << ORDERK << "," << FB << "]>[thread=" << i
          << "]";
      timers.emplace_back(buf.str());
    }
  });

  Timer &timer = timers.at(thread_num);
  Interval interval(timer);

  for (int d = 0; d < dim; ++d)
    assert(ratio.getVect()[d] == 2);
  // ??? assert(gpu_or_cpu == RunOn::Cpu);

  assert(actual_comp == 0);  // ???
  assert(actual_state == 0); // ???

  // Target box is intersection of fine_region and domain of fine
  const amrex::Box target_region = fine_region & fine.box();
  assert(target_region == fine_region);

  using STENCILI = interp1d<CENTI, INTPI, ORDERI>;
  using STENCILJ = interp1d<CENTJ, INTPJ, ORDERJ>;
  using STENCILK = interp1d<CENTK, INTPK, ORDERK>;

  constexpr vect<centering_t, dim> centering{CENTI, CENTJ, CENTK};
  constexpr vect<interpolation_t, dim> interpolation{INTPI, INTPJ, INTPK};
  constexpr vect<int, dim> order{ORDERI, ORDERJ, ORDERK};

  {
    static test_interp1d<CENTI, INTPI, ORDERI, CCTK_REAL> testi;
    static test_interp1d<CENTJ, INTPJ, ORDERJ, CCTK_REAL> testj;
    static test_interp1d<CENTK, INTPK, ORDERK, CCTK_REAL> testk;
  }

#ifdef CCTK_DEBUG
  // The points we will access, i.e. the coarsened fine region, with ghosts
  // added
  const amrex::Box source_region = CoarseBox(target_region, /*reffact*/ 2);
#endif

  const auto crsebox = crse.box();
  const auto finebox = fine.box();

  constexpr vect<int, dim> required_ghosts{
      STENCILI::required_ghosts,
      STENCILJ::required_ghosts,
      STENCILK::required_ghosts,
  };

  // Do we need shifted stencils?
  constexpr vect<bool, dim> use_shift{interpolation[0] == ENO && order[0] > 0,
                                      interpolation[1] == ENO && order[1] > 0,
                                      interpolation[2] == ENO && order[2] > 0};

  for (int comp = 0; comp < ncomp; ++comp) {
    const CCTK_REAL *restrict const crseptr = crse.dataPtr(crse_comp + comp);
    CCTK_REAL *restrict fineptr = fine.dataPtr(fine_comp + comp);

    const auto crse = [=] CCTK_DEVICE(const int i, const int j, const int k)
                          CCTK_ATTRIBUTE_ALWAYS_INLINE {
                            const amrex::IntVect vcrse(i, j, k);
#ifdef CCTK_DEBUG
                            assert(crsebox.contains(vcrse));
#endif
                            return crseptr[crsebox.index(vcrse)];
                          };
    const auto fine = [=] CCTK_DEVICE(const int i, const int j, const int k)
                          CCTK_ATTRIBUTE_ALWAYS_INLINE {
                            const amrex::IntVect vfine(i, j, k);
#ifdef CCTK_DEBUG
                            assert(finebox.contains(vfine));
#endif
                            return fineptr[finebox.index(vfine)];
                          };
    const auto setfine = [=] CCTK_DEVICE(const int i, const int j, const int k,
                                         const CCTK_REAL val)
                             CCTK_ATTRIBUTE_ALWAYS_INLINE {
                               const amrex::IntVect vfine(i, j, k);
#ifdef CCTK_DEBUG
                               assert(finebox.contains(vfine));
#endif
                               fineptr[finebox.index(vfine)] = val;
                             };

#ifdef CCTK_DEBUG
    // Check that the input values are finite
    amrex::ParallelFor(
        source_region,
        [=] CCTK_DEVICE(const int i, const int j, const int k)
            CCTK_ATTRIBUTE_ALWAYS_INLINE { assert(isfinite(crse(i, j, k))); });
#endif

    // Undivided differences
    // Maximum ENO shift
    constexpr vect<int, dim> maxshift{use_shift[0] ? required_ghosts[0] / 2 : 0,
                                      use_shift[1] ? required_ghosts[1] / 2 : 0,
                                      use_shift[2] ? required_ghosts[2] / 2
                                                   : 0};

    // We don't pre-calculate the undivided differences

    const vect<int, dim> imin{target_region.loVect()[0],
                              target_region.loVect()[1],
                              target_region.loVect()[2]};
    const vect<int, dim> imax{target_region.hiVect()[0] + 1,
                              target_region.hiVect()[1] + 1,
                              target_region.hiVect()[2] + 1};
    loop_region(
        [=] CCTK_DEVICE(
            const vect<int, dim> &ifine) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          // Redefine `constexpr` values since they are only captured as
          // `const` by nvcc
          constexpr vect<interpolation_t, dim> interpolation{INTPI, INTPJ,
                                                             INTPK};
          constexpr vect<int, dim> order{ORDERI, ORDERJ, ORDERK};
          // Do we need shifted stencils?
          constexpr vect<bool, dim> use_shift{
              interpolation[0] == ENO && order[0] > 0,
              interpolation[1] == ENO && order[1] > 0,
              interpolation[2] == ENO && order[2] > 0};
          // Maximum ENO shift
          constexpr vect<int, dim> maxshift{
              use_shift[0] ? required_ghosts[0] / 2 : 0,
              use_shift[1] ? required_ghosts[1] / 2 : 0,
              use_shift[2] ? required_ghosts[2] / 2 : 0};

          const vect<int, dim> icrse = ifine >> 1;
          const vect<int, dim> off = ifine & 0x1;

          const auto pcrse = [&](const int di, const int dj, const int dk) {
            return crse(icrse[0] + di, icrse[1] + dj, icrse[2] + dk);
          };

          // Choose stencil shift
          vect<int, dim> shift{0, 0, 0};
          constexpr bool any_use_shift = any(use_shift);
          if (any_use_shift) {
            CCTK_REAL min_dd = 1 / CCTK_REAL(0);
            // Loop over all possible shifts
            for (int sk = -maxshift[2]; sk <= +maxshift[2]; ++sk) {
              for (int sj = -maxshift[1]; sj <= +maxshift[1]; ++sj) {
                for (int si = -maxshift[0]; si <= +maxshift[0]; ++si) {

                  // Stencil radius
                  // This stencil radius already takes the shift into account,
                  // but only for stencils that use a shift. Undo this; we add
                  // the shift back later manually.
                  vect<std::array<int, 2>, dim> stencil_radius{
                      STENCILI().stencil_radius(si, off[0]),
                      STENCILJ().stencil_radius(sj, off[1]),
                      STENCILK().stencil_radius(sk, off[2])};
                  if (use_shift[0]) {
                    stencil_radius[0][0] -= si;
                    stencil_radius[0][1] -= si;
                  }
                  if (use_shift[1]) {
                    stencil_radius[1][0] -= sj;
                    stencil_radius[1][1] -= sj;
                  }
                  if (use_shift[2]) {
                    stencil_radius[2][0] -= sk;
                    stencil_radius[2][1] -= sk;
                  }

                  // Calculate all undivided differences in the x-direction,
                  // looping over the y- and z-directions and finding the
                  // maximum undivided difference there
                  CCTK_REAL ddx = 0;
                  if (use_shift[0]) {
                    for (int dk = stencil_radius[2][0];
                         dk <= stencil_radius[2][1]; ++dk) {
                      for (int dj = stencil_radius[1][0];
                           dj <= stencil_radius[1][1]; ++dj) {
                        const CCTK_REAL dd =
                            undivided_difference_1d<CENTI, INTPI, ORDERI>()(
                                [&](const int di) {
                                  return pcrse(si + di, sj + dj, sk + dk);
                                });
                        ddx = fmax(ddx, fabs(dd));
                      }
                    }
                  }

                  // Same with y-undivided differences
                  CCTK_REAL ddy = 0;
                  if (use_shift[1]) {
                    for (int dk = stencil_radius[2][0];
                         dk <= stencil_radius[2][1]; ++dk) {
                      for (int di = stencil_radius[0][0];
                           di <= stencil_radius[0][1]; ++di) {
                        const CCTK_REAL dd =
                            undivided_difference_1d<CENTJ, INTPJ, ORDERJ>()(
                                [&](const int dj) {
                                  return pcrse(si + di, sj + dj, sk + dk);
                                });
                        ddy = fmax(ddy, fabs(dd));
                      }
                    }
                  }

                  // Same with z-undivided differences
                  CCTK_REAL ddz = 0;
                  if (use_shift[2]) {
                    for (int dj = stencil_radius[1][0];
                         dj <= stencil_radius[1][1]; ++dj) {
                      for (int di = stencil_radius[0][0];
                           di <= stencil_radius[0][1]; ++di) {
                        const CCTK_REAL dd =
                            undivided_difference_1d<CENTK, INTPK, ORDERK>()(
                                [&](const int dk) {
                                  return pcrse(si + di, sj + dj, sk + dk);
                                });
                        ddz = fmax(ddz, fabs(dd));
                      }
                    }
                  }

                  // Prefer centred stencils
                  const CCTK_REAL penalty =
                      1 + sqrt(std::numeric_limits<CCTK_REAL>::epsilon()) *
                              (abs(si) + abs(sj) + abs(sk));
                  const CCTK_REAL dd = penalty * fmax(fmax(ddx, ddy), ddz);
                  if (dd < min_dd) {
                    min_dd = dd;
                    shift = {si, sj, sk};
                  }
                }
              }
            }
          } // if any_use_shift

          const CCTK_REAL val = call_stencil_3d(
              pcrse,
              [&](const auto &crse) {
                return interp1d<CENTI, INTPI, ORDERI>()(crse, shift[0], off[0]);
              },
              [&](const auto &crse) {
                return interp1d<CENTJ, INTPJ, ORDERJ>()(crse, shift[1], off[1]);
              },
              [&](const auto &crse) {
                return interp1d<CENTK, INTPK, ORDERK>()(crse, shift[2], off[2]);
              });

          CCTK_REAL res;

          if constexpr (FB == FB_NONE ||
                        (((INTPI != CONS && INTPI != ENO) || ORDERI <= 1) &&
                         ((INTPJ != CONS && INTPJ != ENO) || ORDERJ <= 1) &&
                         ((INTPK != CONS && INTPK != ENO) || ORDERK <= 1))) {
            // We are not falling back to linear interpolation

            res = val;

          } else {
            // We might want to fall back to linear interpolation

            // Calculate the linearly interpolated value
            constexpr int LINORDERI =
                INTPI == CONS || INTPI == ENO ? 1 : ORDERI;
            constexpr int LINORDERJ =
                INTPJ == CONS || INTPJ == ENO ? 1 : ORDERJ;
            constexpr int LINORDERK =
                INTPK == CONS || INTPK == ENO ? 1 : ORDERK;
            const CCTK_REAL val_lin = call_stencil_3d(
                pcrse,
                [&](const auto &crse) {
                  return interp1d<CENTI, INTPI, LINORDERI>()(crse, 0, off[0]);
                },
                [&](const auto &crse) {
                  return interp1d<CENTJ, INTPJ, LINORDERJ>()(crse, 0, off[1]);
                },
                [&](const auto &crse) {
                  return interp1d<CENTK, INTPK, LINORDERK>()(crse, 0, off[2]);
                });

            // Check whether we need to fall back
            bool need_fallback = false;
            {
              const std::array<int, 2> sradi =
                  interp1d<CENTI, INTPI, ORDERI>().stencil_radius(shift[0],
                                                                  off[0]);
              const std::array<int, 2> sradj =
                  interp1d<CENTJ, INTPJ, ORDERJ>().stencil_radius(shift[1],
                                                                  off[1]);
              const std::array<int, 2> sradk =
                  interp1d<CENTK, INTPK, ORDERK>().stencil_radius(shift[2],
                                                                  off[2]);
              // Fallback condition 1: The interpolated value introduces a new
              // extremum
              CCTK_REAL minval = +1 / CCTK_REAL(0), maxval = -1 / CCTK_REAL(0);
              for (int dk = sradk[0]; dk <= sradk[1]; ++dk) {
                for (int dj = sradj[0]; dj <= sradj[1]; ++dj) {
                  for (int di = sradi[0]; di <= sradi[1]; ++di) {
                    using std::fmax, std::fmin;
                    minval = fmin(minval, pcrse(di, dj, dk));
                    maxval = fmax(maxval, pcrse(di, dj, dk));
                  }
                }
              }
              need_fallback |= val < minval || val > maxval;

              // Fallback condition 2 (checked for every direction in
              // which the operator is conservative): The slope or
              // curvature of the values at the stencil points changes
              // sign anywhere in the stencil

              if constexpr (INTPI == CONS || INTPI == ENO) {
                if (false && sradi[1] - sradi[0] >= 2) {
                  // Check whether slopes change sign
                  for (int dk = sradk[0]; dk <= sradk[1]; ++dk) {
                    for (int dj = sradj[0]; dj <= sradj[1]; ++dj) {
                      const int di = sradi[0];
                      const CCTK_REAL s0 =
                          pcrse(di + 1, dj, dk) - pcrse(di + 0, dj, dk);
                      for (int di = sradi[0] + 1; di <= sradi[1] - 1; ++di) {
                        const CCTK_REAL s =
                            pcrse(di + 1, dj, dk) - pcrse(di + 0, dj, dk);
                        need_fallback |= s * s0 < 0;
                      }
                    }
                  }
                }
                if (sradi[1] - sradi[0] >= 3) {
                  // Check whether curvatures change sign
                  for (int dk = sradk[0]; dk <= sradk[1]; ++dk) {
                    for (int dj = sradj[0]; dj <= sradj[1]; ++dj) {
                      const int di = sradi[0];
                      const CCTK_REAL c0 = pcrse(di + 0, dj, dk) -
                                           2 * pcrse(di + 1, dj, dk) +
                                           pcrse(di + 2, dj, dk);
                      for (int di = sradi[0] + 1; di <= sradi[1] - 2; ++di) {
                        const CCTK_REAL c = pcrse(di + 0, dj, dk) -
                                            2 * pcrse(di + 1, dj, dk) +
                                            pcrse(di + 2, dj, dk);
                        need_fallback |= c * c0 < 0;
                      }
                    }
                  }
                }
              }

              if constexpr (INTPJ == CONS || INTPJ == ENO) {
                if (false && sradj[1] - sradj[0] >= 2) {
                  // Check whether slopes change sign
                  for (int dk = sradk[0]; dk <= sradk[1]; ++dk) {
                    for (int di = sradi[0]; di <= sradi[1]; ++di) {
                      const int dj = sradj[0];
                      const CCTK_REAL s0 =
                          pcrse(di, dj + 1, dk) - pcrse(di, dj + 0, dk);
                      for (int dj = sradj[0] + 1; dj <= sradj[1] - 1; ++dj) {
                        const CCTK_REAL s =
                            pcrse(di, dj + 1, dk) - pcrse(di, dj + 0, dk);
                        need_fallback |= s * s0 < 0;
                      }
                    }
                  }
                }
                if (sradj[1] - sradj[0] >= 3) {
                  // Check whether curvatures change sign
                  for (int dk = sradk[0]; dk <= sradk[1]; ++dk) {
                    for (int di = sradi[0]; di <= sradi[1]; ++di) {
                      const int dj = sradj[0];
                      const CCTK_REAL c0 = pcrse(di, dj + 0, dk) -
                                           2 * pcrse(di, dj + 1, dk) +
                                           pcrse(di, dj + 2, dk);
                      for (int dj = sradj[0] + 1; dj <= sradj[1] - 2; ++dj) {
                        const CCTK_REAL c = pcrse(di, dj + 0, dk) -
                                            2 * pcrse(di, dj + 1, dk) +
                                            pcrse(di, dj + 2, dk);
                        need_fallback |= c * c0 < 0;
                      }
                    }
                  }
                }
              }

              if constexpr (INTPK == CONS || INTPK == ENO) {
                if (false && sradk[1] - sradk[0] >= 2) {
                  // Check whether slopes change sign
                  for (int dj = sradj[0]; dj <= sradj[1]; ++dj) {
                    for (int di = sradi[0]; di <= sradi[1]; ++di) {
                      const int dk = sradk[0];
                      const CCTK_REAL s0 =
                          pcrse(di, dj, dk + 1) - pcrse(di, dj, dk + 0);
                      for (int dk = sradk[0] + 1; dk <= sradk[1] - 1; ++dk) {
                        const CCTK_REAL s =
                            pcrse(di, dj, dk + 1) - pcrse(di, dj, dk + 0);
                        need_fallback |= s * s0 < 0;
                      }
                    }
                  }
                }
                if (sradk[1] - sradk[0] >= 3) {
                  // Check whether curvatures change sign
                  for (int dj = sradj[0]; dj <= sradj[1]; ++dj) {
                    for (int di = sradi[0]; di <= sradi[1]; ++di) {
                      const int dk = sradk[0];
                      const CCTK_REAL c0 = pcrse(di, dj, dk + 0) -
                                           2 * pcrse(di, dj, dk + 1) +
                                           pcrse(di, dj, dk + 2);
                      for (int dk = sradk[0] + 1; dk <= sradk[1] - 2; ++dk) {
                        const CCTK_REAL c = pcrse(di, dj, dk + 0) -
                                            2 * pcrse(di, dj, dk + 1) +
                                            pcrse(di, dj, dk + 2);
                        need_fallback |= c * c0 < 0;
                      }
                    }
                  }
                }
              }
            }

            res = need_fallback ? val_lin : val;
          } // if FB != FB_NONE

          setfine(ifine[0], ifine[1], ifine[2], res);
        },
        imin, imax);

#ifdef CCTK_DEBUG
    // Check that the output values are finite
    amrex::ParallelFor(
        target_region,
        [=] CCTK_DEVICE(const int i, const int j, const int k)
            CCTK_ATTRIBUTE_ALWAYS_INLINE { assert(isfinite(fine(i, j, k))); });
#endif

  } // for comp

  // #ifdef __CUDACC__
  //   amrex::Gpu::synchronize();
  //   AMREX_GPU_ERROR_CHECK();
  // #endif
}

template <centering_t CENTI, centering_t CENTJ, centering_t CENTK,
          interpolation_t INTPI, interpolation_t INTPJ, interpolation_t INTPK,
          int ORDERI, int ORDERJ, int ORDERK, fallback_t FB>
void prolongate_3d_rf2<
    CENTI, CENTJ, CENTK, INTPI, INTPJ, INTPK, ORDERI, ORDERJ, ORDERK,
    FB>::interp_per_group(const amrex::FArrayBox &crse_box, const int crse_comp,
                          amrex::FArrayBox &fine_box, const int fine_comp,
                          const int ncomps, const amrex::Box &fine_region,
                          const amrex::IntVect &ratio,
                          const amrex::Geometry &crse_geom,
                          const amrex::Geometry &fine_geom,
                          amrex::Vector<amrex::BCRec> const &bcr,
                          const int actual_comp, const int actual_state,
                          const amrex::RunOn gpu_or_cpu) {
  DECLARE_CCTK_PARAMETERS;

  constexpr int maxncomps = 10;
  assert(ncomps <= maxncomps);

  static std::once_flag have_timers;
  static std::vector<Timer> timers;

  const int thread_num = omp_get_thread_num();

  call_once(have_timers, [&]() {
    const int num_threads = omp_get_num_threads();
    timers.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
      std::ostringstream buf;
      buf << "prolongate_3d_rf2<[" << CENTI << "," << CENTJ << "," << CENTK
          << "],[" << INTPI << "," << INTPJ << "," << INTPK << "],[" << ORDERI
          << "," << ORDERJ << "," << ORDERK << "," << FB << "]>[thread=" << i
          << "]";
      timers.emplace_back(buf.str());
    }
  });

  Timer &timer = timers.at(thread_num);
  Interval interval(timer);

  for (int d = 0; d < dim; ++d)
    assert(ratio.getVect()[d] == 2);
  // ??? assert(gpu_or_cpu == RunOn::Cpu);

  assert(actual_comp == 0);  // ???
  assert(actual_state == 0); // ???

  // Target box is intersection of fine_region and domain of fine
  const amrex::Box target_region = fine_region & fine_box.box();
  assert(target_region == fine_region);

  using STENCILI = interp1d<CENTI, INTPI, ORDERI>;
  using STENCILJ = interp1d<CENTJ, INTPJ, ORDERJ>;
  using STENCILK = interp1d<CENTK, INTPK, ORDERK>;

  constexpr vect<centering_t, dim> centering{CENTI, CENTJ, CENTK};
  constexpr vect<interpolation_t, dim> interpolation{INTPI, INTPJ, INTPK};
  constexpr vect<int, dim> order{ORDERI, ORDERJ, ORDERK};

  {
    static test_interp1d<CENTI, INTPI, ORDERI, CCTK_REAL> testi;
    static test_interp1d<CENTJ, INTPJ, ORDERJ, CCTK_REAL> testj;
    static test_interp1d<CENTK, INTPK, ORDERK, CCTK_REAL> testk;
  }

#ifdef CCTK_DEBUG
  // The points we will access, i.e. the coarsened fine region, with ghosts
  // added
  const amrex::Box source_region = CoarseBox(target_region, /*reffact*/ 2);
#endif

  const auto crsebox = crse_box.box();
  const auto finebox = fine_box.box();

  constexpr vect<int, dim> required_ghosts{
      STENCILI::required_ghosts,
      STENCILJ::required_ghosts,
      STENCILK::required_ghosts,
  };

  // Do we need shifted stencils?
  constexpr vect<bool, dim> use_shift{interpolation[0] == ENO && order[0] > 0,
                                      interpolation[1] == ENO && order[1] > 0,
                                      interpolation[2] == ENO && order[2] > 0};

  const CCTK_REAL *restrict const crseptr = crse_box.dataPtr(crse_comp);
  const std::ptrdiff_t crsenp = crse_box.dataPtr(1) - crse_box.dataPtr(0);
  CCTK_REAL *restrict fineptr = fine_box.dataPtr(fine_comp);
  const std::ptrdiff_t finenp = fine_box.dataPtr(1) - fine_box.dataPtr(0);

  const auto crse =
      [=] CCTK_DEVICE(const int i, const int j, const int k, const int comp)
          CCTK_ATTRIBUTE_ALWAYS_INLINE {
            const amrex::IntVect vcrse(i, j, k);
#ifdef CCTK_DEBUG
            assert(crsebox.contains(vcrse));
#endif
            return crseptr[crsebox.index(vcrse) + comp * crsenp];
          };
#ifdef CCTK_DEBUG
  const auto fine =
      [=] CCTK_DEVICE(const int i, const int j, const int k, const int comp)
          CCTK_ATTRIBUTE_ALWAYS_INLINE {
            const amrex::IntVect vfine(i, j, k);
#ifdef CCTK_DEBUG
            assert(finebox.contains(vfine));
#endif
            return fineptr[finebox.index(vfine) + comp * finenp];
          };
#endif
  const auto setfine =
      [=] CCTK_DEVICE(const int i, const int j, const int k, const int comp,
                      const CCTK_REAL val) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        const amrex::IntVect vfine(i, j, k);
#ifdef CCTK_DEBUG
        assert(finebox.contains(vfine));
#endif
        fineptr[finebox.index(vfine) + comp * finenp] = val;
      };

#ifdef CCTK_DEBUG
  // Check that the input values are finite
  amrex::ParallelFor(source_region,
                     [=] CCTK_DEVICE(const int i, const int j, const int k)
                         CCTK_ATTRIBUTE_ALWAYS_INLINE {
                           for (int comp = 0; comp < ncomps; ++comp)
                             assert(isfinite(crse(i, j, k, comp)));
                         });
#endif

  // Undivided differences
  // Maximum ENO shift
  constexpr vect<int, dim> maxshift{use_shift[0] ? required_ghosts[0] / 2 : 0,
                                    use_shift[1] ? required_ghosts[1] / 2 : 0,
                                    use_shift[2] ? required_ghosts[2] / 2 : 0};

  // We don't pre-calculate the undivided differences

  const vect<int, dim> imin{target_region.loVect()[0],
                            target_region.loVect()[1],
                            target_region.loVect()[2]};
  const vect<int, dim> imax{target_region.hiVect()[0] + 1,
                            target_region.hiVect()[1] + 1,
                            target_region.hiVect()[2] + 1};
  loop_region(
      [=] CCTK_DEVICE(
          const vect<int, dim> &ifine) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        // Redefine `constexpr` values since they are only captured as
        // `const` by nvcc
        constexpr vect<interpolation_t, dim> interpolation{INTPI, INTPJ, INTPK};
        constexpr vect<int, dim> order{ORDERI, ORDERJ, ORDERK};
        // Do we need shifted stencils?
        constexpr vect<bool, dim> use_shift{
            interpolation[0] == ENO && order[0] > 0,
            interpolation[1] == ENO && order[1] > 0,
            interpolation[2] == ENO && order[2] > 0};
        // Maximum ENO shift
        constexpr vect<int, dim> maxshift{
            use_shift[0] ? required_ghosts[0] / 2 : 0,
            use_shift[1] ? required_ghosts[1] / 2 : 0,
            use_shift[2] ? required_ghosts[2] / 2 : 0};

        const vect<int, dim> icrse = ifine >> 1;
        const vect<int, dim> off = ifine & 0x1;

        const auto pcrse = [&](const int di, const int dj, const int dk,
                               const int comp) {
          return crse(icrse[0] + di, icrse[1] + dj, icrse[2] + dk, comp);
        };

        // Choose stencil shift
        vect<int, dim> shift{0, 0, 0};
        constexpr bool any_use_shift = any(use_shift);
        if (any_use_shift) {
          CCTK_REAL min_dd = 1 / CCTK_REAL(0);
          // Loop over all possible shifts
          for (int sk = -maxshift[2]; sk <= +maxshift[2]; ++sk) {
            for (int sj = -maxshift[1]; sj <= +maxshift[1]; ++sj) {
              for (int si = -maxshift[0]; si <= +maxshift[0]; ++si) {

                // Stencil radius
                // This stencil radius already takes the shift into account,
                // but only for stencils that use a shift. Undo this; we add
                // the shift back later manually.
                vect<std::array<int, 2>, dim> stencil_radius{
                    STENCILI().stencil_radius(si, off[0]),
                    STENCILJ().stencil_radius(sj, off[1]),
                    STENCILK().stencil_radius(sk, off[2])};
                if (use_shift[0]) {
                  stencil_radius[0][0] -= si;
                  stencil_radius[0][1] -= si;
                }
                if (use_shift[1]) {
                  stencil_radius[1][0] -= sj;
                  stencil_radius[1][1] -= sj;
                }
                if (use_shift[2]) {
                  stencil_radius[2][0] -= sk;
                  stencil_radius[2][1] -= sk;
                }

                // Calculate all undivided differences in the x-direction,
                // looping over the y- and z-directions and finding the
                // maximum undivided difference there
                CCTK_REAL ddx = 0;
                if (use_shift[0]) {
                  for (int comp = 0; comp < ncomps; ++comp) {
                    for (int dk = stencil_radius[2][0];
                         dk <= stencil_radius[2][1]; ++dk) {
                      for (int dj = stencil_radius[1][0];
                           dj <= stencil_radius[1][1]; ++dj) {
                        const CCTK_REAL dd =
                            undivided_difference_1d<CENTI, INTPI, ORDERI>()(
                                [&](const int di) {
                                  return pcrse(si + di, sj + dj, sk + dk, comp);
                                });
                        ddx = fmax(ddx, fabs(dd));
                      }
                    }
                  }
                }

                // Same with y-undivided differences
                CCTK_REAL ddy = 0;
                if (use_shift[1]) {
                  for (int comp = 0; comp < ncomps; ++comp) {
                    for (int dk = stencil_radius[2][0];
                         dk <= stencil_radius[2][1]; ++dk) {
                      for (int di = stencil_radius[0][0];
                           di <= stencil_radius[0][1]; ++di) {
                        const CCTK_REAL dd =
                            undivided_difference_1d<CENTJ, INTPJ, ORDERJ>()(
                                [&](const int dj) {
                                  return pcrse(si + di, sj + dj, sk + dk, comp);
                                });
                        ddy = fmax(ddy, fabs(dd));
                      }
                    }
                  }
                }

                // Same with z-undivided differences
                CCTK_REAL ddz = 0;
                if (use_shift[2]) {
                  for (int comp = 0; comp < ncomps; ++comp) {
                    for (int dj = stencil_radius[1][0];
                         dj <= stencil_radius[1][1]; ++dj) {
                      for (int di = stencil_radius[0][0];
                           di <= stencil_radius[0][1]; ++di) {
                        const CCTK_REAL dd =
                            undivided_difference_1d<CENTK, INTPK, ORDERK>()(
                                [&](const int dk) {
                                  return pcrse(si + di, sj + dj, sk + dk, comp);
                                });
                        ddz = fmax(ddz, fabs(dd));
                      }
                    }
                  }
                }

                // Prefer centred stencils
                const CCTK_REAL penalty =
                    1 + sqrt(std::numeric_limits<CCTK_REAL>::epsilon()) *
                            (abs(si) + abs(sj) + abs(sk));
                const CCTK_REAL dd = penalty * fmax(fmax(ddx, ddy), ddz);
                if (dd < min_dd) {
                  min_dd = dd;
                  shift = {si, sj, sk};
                }
              }
            }
          }
        } // if any_use_shift

        std::array<CCTK_REAL, maxncomps> vals;
        for (int comp = 0; comp < ncomps; ++comp)
          vals[comp] = call_stencil_3d(
              [&](const int di, const int dj, const int dk) {
                return pcrse(di, dj, dk, comp);
              },
              [&](const auto &crse) {
                return interp1d<CENTI, INTPI, ORDERI>()(crse, shift[0], off[0]);
              },
              [&](const auto &crse) {
                return interp1d<CENTJ, INTPJ, ORDERJ>()(crse, shift[1], off[1]);
              },
              [&](const auto &crse) {
                return interp1d<CENTK, INTPK, ORDERK>()(crse, shift[2], off[2]);
              });

        std::array<CCTK_REAL, maxncomps> ress;

        if constexpr (FB == FB_NONE ||
                      (((INTPI != CONS && INTPI != ENO) || ORDERI <= 1) &&
                       ((INTPJ != CONS && INTPJ != ENO) || ORDERJ <= 1) &&
                       ((INTPK != CONS && INTPK != ENO) || ORDERK <= 1))) {
          // We are not falling back to linear interpolation

          for (int comp = 0; comp < ncomps; ++comp)
            ress[comp] = vals[comp];

        } else {
          // We might want to fall back to linear interpolation

          // Check whether we need to fall back
          bool need_fallback = false;
          {
            const std::array<int, 2> sradi =
                interp1d<CENTI, INTPI, ORDERI>().stencil_radius(shift[0],
                                                                off[0]);
            const std::array<int, 2> sradj =
                interp1d<CENTJ, INTPJ, ORDERJ>().stencil_radius(shift[1],
                                                                off[1]);
            const std::array<int, 2> sradk =
                interp1d<CENTK, INTPK, ORDERK>().stencil_radius(shift[2],
                                                                off[2]);
            // Fallback condition 1: The interpolated value introduces a new
            // extremum
            for (int comp = 0; comp < ncomps; ++comp) {
              CCTK_REAL minval = +1 / CCTK_REAL(0), maxval = -1 / CCTK_REAL(0);
              for (int dk = sradk[0]; dk <= sradk[1]; ++dk) {
                for (int dj = sradj[0]; dj <= sradj[1]; ++dj) {
                  for (int di = sradi[0]; di <= sradi[1]; ++di) {
                    using std::fmax, std::fmin;
                    minval = fmin(minval, pcrse(di, dj, dk, comp));
                    maxval = fmax(maxval, pcrse(di, dj, dk, comp));
                  }
                }
              }
              need_fallback |= vals[comp] < minval || vals[comp] > maxval;
            }

            // Fallback condition 2 (checked for every direction in
            // which the operator is conservative): The slope or
            // curvature of the values at the stencil points changes
            // sign anywhere in the stencil

            if constexpr (INTPI == CONS || INTPI == ENO) {
              if (false && sradi[1] - sradi[0] >= 2) {
                // Check whether slopes change sign
                for (int comp = 0; comp < ncomps; ++comp) {
                  for (int dk = sradk[0]; dk <= sradk[1]; ++dk) {
                    for (int dj = sradj[0]; dj <= sradj[1]; ++dj) {
                      const int di = sradi[0];
                      const CCTK_REAL s0 = pcrse(di + 1, dj, dk, comp) -
                                           pcrse(di + 0, dj, dk, comp);
                      for (int di = sradi[0] + 1; di <= sradi[1] - 1; ++di) {
                        const CCTK_REAL s = pcrse(di + 1, dj, dk, comp) -
                                            pcrse(di + 0, dj, dk, comp);
                        need_fallback |= s * s0 < 0;
                      }
                    }
                  }
                }
              }
              if (sradi[1] - sradi[0] >= 3) {
                // Check whether curvatures change sign
                for (int comp = 0; comp < ncomps; ++comp) {
                  for (int dk = sradk[0]; dk <= sradk[1]; ++dk) {
                    for (int dj = sradj[0]; dj <= sradj[1]; ++dj) {
                      const int di = sradi[0];
                      const CCTK_REAL c0 = pcrse(di + 0, dj, dk, comp) -
                                           2 * pcrse(di + 1, dj, dk, comp) +
                                           pcrse(di + 2, dj, dk, comp);
                      for (int di = sradi[0] + 1; di <= sradi[1] - 2; ++di) {
                        const CCTK_REAL c = pcrse(di + 0, dj, dk, comp) -
                                            2 * pcrse(di + 1, dj, dk, comp) +
                                            pcrse(di + 2, dj, dk, comp);
                        need_fallback |= c * c0 < 0;
                      }
                    }
                  }
                }
              }
            }

            if constexpr (INTPJ == CONS || INTPJ == ENO) {
              if (false && sradj[1] - sradj[0] >= 2) {
                // Check whether slopes change sign
                for (int comp = 0; comp < ncomps; ++comp) {
                  for (int dk = sradk[0]; dk <= sradk[1]; ++dk) {
                    for (int di = sradi[0]; di <= sradi[1]; ++di) {
                      const int dj = sradj[0];
                      const CCTK_REAL s0 = pcrse(di, dj + 1, dk, comp) -
                                           pcrse(di, dj + 0, dk, comp);
                      for (int dj = sradj[0] + 1; dj <= sradj[1] - 1; ++dj) {
                        const CCTK_REAL s = pcrse(di, dj + 1, dk, comp) -
                                            pcrse(di, dj + 0, dk, comp);
                        need_fallback |= s * s0 < 0;
                      }
                    }
                  }
                }
              }
              if (sradj[1] - sradj[0] >= 3) {
                // Check whether curvatures change sign
                for (int comp = 0; comp < ncomps; ++comp) {
                  for (int dk = sradk[0]; dk <= sradk[1]; ++dk) {
                    for (int di = sradi[0]; di <= sradi[1]; ++di) {
                      const int dj = sradj[0];
                      const CCTK_REAL c0 = pcrse(di, dj + 0, dk, comp) -
                                           2 * pcrse(di, dj + 1, dk, comp) +
                                           pcrse(di, dj + 2, dk, comp);
                      for (int dj = sradj[0] + 1; dj <= sradj[1] - 2; ++dj) {
                        const CCTK_REAL c = pcrse(di, dj + 0, dk, comp) -
                                            2 * pcrse(di, dj + 1, dk, comp) +
                                            pcrse(di, dj + 2, dk, comp);
                        need_fallback |= c * c0 < 0;
                      }
                    }
                  }
                }
              }
            }

            if constexpr (INTPK == CONS || INTPK == ENO) {
              if (false && sradk[1] - sradk[0] >= 2) {
                // Check whether slopes change sign
                for (int comp = 0; comp < ncomps; ++comp) {
                  for (int dj = sradj[0]; dj <= sradj[1]; ++dj) {
                    for (int di = sradi[0]; di <= sradi[1]; ++di) {
                      const int dk = sradk[0];
                      const CCTK_REAL s0 = pcrse(di, dj, dk + 1, comp) -
                                           pcrse(di, dj, dk + 0, comp);
                      for (int dk = sradk[0] + 1; dk <= sradk[1] - 1; ++dk) {
                        const CCTK_REAL s = pcrse(di, dj, dk + 1, comp) -
                                            pcrse(di, dj, dk + 0, comp);
                        need_fallback |= s * s0 < 0;
                      }
                    }
                  }
                }
              }
              if (sradk[1] - sradk[0] >= 3) {
                // Check whether curvatures change sign
                for (int comp = 0; comp < ncomps; ++comp) {
                  for (int dj = sradj[0]; dj <= sradj[1]; ++dj) {
                    for (int di = sradi[0]; di <= sradi[1]; ++di) {
                      const int dk = sradk[0];
                      const CCTK_REAL c0 = pcrse(di, dj, dk + 0, comp) -
                                           2 * pcrse(di, dj, dk + 1, comp) +
                                           pcrse(di, dj, dk + 2, comp);
                      for (int dk = sradk[0] + 1; dk <= sradk[1] - 2; ++dk) {
                        const CCTK_REAL c = pcrse(di, dj, dk + 0, comp) -
                                            2 * pcrse(di, dj, dk + 1, comp) +
                                            pcrse(di, dj, dk + 2, comp);
                        need_fallback |= c * c0 < 0;
                      }
                    }
                  }
                }
              }
            }
          }

          constexpr int LINORDERI = INTPI == CONS || INTPI == ENO ? 1 : ORDERI;
          constexpr int LINORDERJ = INTPJ == CONS || INTPJ == ENO ? 1 : ORDERJ;
          constexpr int LINORDERK = INTPK == CONS || INTPK == ENO ? 1 : ORDERK;
          std::array<CCTK_REAL, maxncomps> val_lins;
          for (int comp = 0; comp < ncomps; ++comp) {
            val_lins[comp] = call_stencil_3d(
                [&](const int di, const int dj, const int dk) {
                  return pcrse(di, dj, dk, comp);
                },
                [&](const auto &crse) {
                  return interp1d<CENTI, INTPI, LINORDERI>()(crse, 0, off[0]);
                },
                [&](const auto &crse) {
                  return interp1d<CENTJ, INTPJ, LINORDERJ>()(crse, 0, off[1]);
                },
                [&](const auto &crse) {
                  return interp1d<CENTK, INTPK, LINORDERK>()(crse, 0, off[2]);
                });
          }

          for (int comp = 0; comp < ncomps; ++comp)
            ress[comp] = need_fallback ? val_lins[comp] : vals[comp];

        } // if FB != FB_NONE

        for (int comp = 0; comp < ncomps; ++comp)
          setfine(ifine[0], ifine[1], ifine[2], comp, ress[comp]);
      },
      imin, imax);

#ifdef CCTK_DEBUG
  // Check that the output values are finite
  amrex::ParallelFor(target_region,
                     [=] CCTK_DEVICE(const int i, const int j, const int k)
                         CCTK_ATTRIBUTE_ALWAYS_INLINE {
                           for (int comp = 0; comp < ncomps; ++comp)
                             assert(isfinite(fine(i, j, k, comp)));
                         });
#endif

  // #ifdef __CUDACC__
  //   amrex::Gpu::synchronize();
  //   AMREX_GPU_ERROR_CHECK();
  // #endif
}

template <centering_t CENTI, centering_t CENTJ, centering_t CENTK,
          interpolation_t INTPI, interpolation_t INTPJ, interpolation_t INTPK,
          int ORDERI, int ORDERJ, int ORDERK, fallback_t FB>
void prolongate_3d_rf2<
    CENTI, CENTJ, CENTK, INTPI, INTPJ, INTPK, ORDERI, ORDERJ, ORDERK,
    FB>::interp(const amrex::FArrayBox &crse_box, const int crse_comp,
                amrex::FArrayBox &fine_box, const int fine_comp,
                const int ncomps, const amrex::Box &fine_region,
                const amrex::IntVect &ratio, const amrex::Geometry &crse_geom,
                const amrex::Geometry &fine_geom,
                amrex::Vector<amrex::BCRec> const &bcr, const int actual_comp,
                const int actual_state, const amrex::RunOn gpu_or_cpu) {
  DECLARE_CCTK_PARAMETERS;
  if (!prolongate_per_group)
    interp_per_var(crse_box, crse_comp, fine_box, fine_comp, ncomps,
                   fine_region, ratio, crse_geom, fine_geom, bcr, actual_comp,
                   actual_state, gpu_or_cpu);
  else
    interp_per_group(crse_box, crse_comp, fine_box, fine_comp, ncomps,
                     fine_region, ratio, crse_geom, fine_geom, bcr, actual_comp,
                     actual_state, gpu_or_cpu);
}

template <centering_t CENTI, centering_t CENTJ, centering_t CENTK,
          interpolation_t INTPI, interpolation_t INTPJ, interpolation_t INTPK,
          int ORDERI, int ORDERJ, int ORDERK, fallback_t FB>
void prolongate_3d_rf2<
    CENTI, CENTJ, CENTK, INTPI, INTPJ, INTPK, ORDERI, ORDERJ, ORDERK,
    FB>::interp_face(const amrex::FArrayBox &crse, const int crse_comp,
                     amrex::FArrayBox &fine, const int fine_comp,
                     const int ncomp, const amrex::Box &fine_region,
                     const amrex::IntVect &ratio,
                     const amrex::IArrayBox &solve_mask,
                     const amrex::Geometry &crse_geom,
                     const amrex::Geometry &fine_geom,
                     amrex::Vector<amrex::BCRec> const &bcr, const int bccomp,
                     const amrex::RunOn gpu_or_cpu) {
  // solve_mask; ???
  assert(bccomp == 0); // ???
  interp(crse, crse_comp, fine, fine_comp, ncomp, fine_region, ratio, crse_geom,
         fine_geom, bcr, 0, 0, gpu_or_cpu);
}

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_PROLONGATE_3D_RF2_IMPL_HXX

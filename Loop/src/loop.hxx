#ifndef CARPETX_LOOP_LOOP_HXX
#define CARPETX_LOOP_LOOP_HXX

#include <AMReX_FArrayBox.H>

#include <simd.hxx>
#include <vect.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <climits>
#include <ostream>
#include <string>
#include <type_traits>

#define CCTK_DEVICE AMREX_GPU_DEVICE
#define CCTK_HOST AMREX_GPU_HOST

namespace Loop {

using Arith::vect;

template <typename F> CCTK_ATTRIBUTE_NOINLINE auto noinline(const F &f) {
  return f();
}

constexpr int dim = 3;

// Value for undefined cctkGH entries
// Note: Don't use a negative value, which tends to leave bugs undetected. Large
// positive values often lead to segfaults, exposing bugs.
constexpr int undefined = (INT_MAX / 2 + 1) + 666;

enum class where_t {
  everywhere,
  interior,
  boundary,
#if 0
  ghosts_inclusive,
#endif
  ghosts,
};
std::ostream &operator<<(std::ostream &os, const where_t where);

struct GridDescBase;

template <typename T, int D> struct units_t {
  constexpr CCTK_ATTRIBUTE_ALWAYS_INLINE CCTK_DEVICE CCTK_HOST Arith::vect<T, D>
  operator[](const int d) const {
    return Arith::vect<T, D>::unit(d);
  }
};

struct PointDesc {
  units_t<int, dim> DI; // direction unit vectors

  int level, patch, component;
  vect<int, dim> I; // grid point
  int iter;         // iteration
  // outward boundary normal (if in outer boundary), else zero
  vect<int, dim> NI;
  vect<int, dim> I0; // nearest interior point
  // outward boundary normal (if on outermost interior point), else zero
  vect<int, dim> BI;
  vect<int, dim> BI1;
  vect<int, dim> BI2;

  // outer boundary points for this grid function (might be outside the current
  // grid function component)
  vect<int, dim> bnd_min, bnd_max;
  vect<int, dim> loop_min, loop_max; // loop shape

  vect<CCTK_REAL, dim> X;  // grid point coordinates
  vect<CCTK_REAL, dim> DX; // grid spacing

  CCTK_BOOLVEC mask; // mask for grid loop

  // Deprecated
  int imin, imax;       // loop bounds, for SIMD vectorization
  int i, j, k;          // grid point
  CCTK_REAL x, y, z;    // grid point coordinates
  CCTK_REAL dx, dy, dz; // grid spacing

  PointDesc() = delete;
  PointDesc(const PointDesc &) = default;
  PointDesc(PointDesc &&) = default;
  PointDesc &operator=(const PointDesc &) = default;
  PointDesc &operator=(PointDesc &&) = default;

  CCTK_ATTRIBUTE_ALWAYS_INLINE CCTK_DEVICE CCTK_HOST
  PointDesc(const int level, const int patch, const int component,
            const vect<int, dim> &I, const int iter, const vect<int, dim> &NI,
            const vect<int, dim> &I0, const vect<int, dim> &BI,
            const vect<int, dim> &BI1, const vect<int, dim> &BI2,
            const vect<int, dim> &bnd_min, const vect<int, dim> &bnd_max,
            const vect<int, dim> &loop_min, const vect<int, dim> &loop_max,
            const vect<CCTK_REAL, dim> &X, const vect<CCTK_REAL, dim> &DX)
      : level(level), patch(patch), component(component), I(I), iter(iter),
        NI(NI), I0(I0), BI(BI), BI1(BI1), BI2(BI2), bnd_min(bnd_min), bnd_max(bnd_max),
        loop_min(loop_min), loop_max(loop_max), X(X), DX(DX),
        mask(Arith::mask_for_loop_tail<CCTK_BOOLVEC>(I[0], loop_max[0])),
        imin(loop_min[0]), imax(loop_max[0]), i(I[0]), j(I[1]), k(I[2]),
        x(X[0]), y(X[1]), z(X[2]), dx(DX[0]), dy(DX[1]), dz(DX[2]) {}

  friend std::ostream &operator<<(std::ostream &os, const PointDesc &p);
};

struct GridDescBase {
  int level, patch, component;
  vect<int, dim> gsh;
  vect<int, dim> lbnd, ubnd;
  vect<int, dim> lsh;
  vect<int, dim> ash;
  vect<vect<int, dim>, 2> bbox;
  vect<int, dim> nghostzones;
  vect<int, dim> tmin, tmax;

  // for current level
  // TODO: these are still cell centred, and so are cctk_origin_space and
  // cctk_delta_space; fix this!
  vect<CCTK_REAL, dim> x0; // origin_space
  vect<CCTK_REAL, dim> dx; // delta_space

  friend std::ostream &operator<<(std::ostream &os, const GridDescBase &grid);

protected:
  GridDescBase();

public:
  GridDescBase(const GridDescBase &) = default;
  GridDescBase(GridDescBase &&) = default;
  GridDescBase &operator=(const GridDescBase &) = default;
  GridDescBase &operator=(GridDescBase &&) = default;

  GridDescBase(const cGH *cctkGH);

  CCTK_ATTRIBUTE_ALWAYS_INLINE CCTK_DEVICE CCTK_HOST PointDesc
  point_desc(const vect<bool, dim> &CI, const vect<int, dim> &I, const int iter,
             const vect<int, dim> &NI, const vect<int, dim> &I0,
             const vect<int, dim> &BI, const vect<int, dim> &BI1, const vect<int, dim> &BI2, 
             const vect<int, dim> &bnd_min, const vect<int, dim> &bnd_max, 
             const vect<int, dim> &loop_min, const vect<int, dim> &loop_max) const {
    const vect<CCTK_REAL, dim> X =
        x0 + (lbnd + I - vect<CCTK_REAL, dim>(!CI) / 2) * dx;
    const vect<CCTK_REAL, dim> DX = dx;
    return PointDesc(level, patch, component, I, iter, NI, I0, BI, BI1, BI2, bnd_min,
                     bnd_max, loop_min, loop_max, X, DX);
  }

  // Loop over a given box
  template <int CI, int CJ, int CK, int VS = 1, int N = 1, typename F>
  void loop_box(const vect<int, dim> &restrict bnd_min,
                const vect<int, dim> &restrict bnd_max,
                const vect<int, dim> &restrict loop_min,
                const vect<int, dim> &restrict loop_max, const F &f) const {
    static_assert(CI == 0 || CI == 1);
    static_assert(CJ == 0 || CJ == 1);
    static_assert(CK == 0 || CK == 1);
    static_assert(N >= 0);
    static_assert(VS > 0);

    if (N == 0 || any(loop_max <= loop_min))
      return;

    for (int iter = 0; iter < N; ++iter) {
      for (int k = loop_min[2]; k < loop_max[2]; ++k) {
        for (int j = loop_min[1]; j < loop_max[1]; ++j) {
#pragma omp simd
          for (int i = loop_min[0]; i < loop_max[0]; i += VS) {
            // Grid point
            const vect<int, dim> I = {i, j, k};
            // Outward boundary normal (if in outer boundary), else 0
            const vect<int, dim> NI =
                vect<int, dim>(I > bnd_max - 1) - vect<int, dim>(I < bnd_min);
            // Nearest interior point
            const vect<int, dim> I0 =
                if_else(NI == 0, 0, if_else(NI < 0, bnd_min, bnd_max - 1));
            // Outward boundary normal (if on outermost interior point), else 0
            const vect<int, dim> BI =
                vect<int, dim>(I == bnd_max - 1) - vect<int, dim>(I == bnd_min);
            const vect<int, dim> BI1 = vect<int, dim>(I == bnd_max - 2) - vect<int, dim>(I == bnd_min + 1);
            const vect<int, dim> BI2 = vect<int, dim>(I == bnd_max - 3) - vect<int, dim>(I == bnd_min + 2);
 
            const PointDesc p =
                point_desc({CI, CJ, CK}, I, iter, NI, I0, BI, BI1, BI2, bnd_min, bnd_max,
                           loop_min, loop_max);
            f(p);
          }
        }
      }
    }
  }

  // Box for outer boundaries (might be outside the current grid function
  // component)
  template <int CI, int CJ, int CK>
  void boundary_box(const vect<int, dim> &group_nghostzones,
                    vect<int, dim> &restrict bnd_min,
                    vect<int, dim> &restrict bnd_max) const {
    constexpr vect<int, dim> offset{CI, CJ, CK};
    // Boundary points
    bnd_min = if_else(bbox[0], nghostzones - lbnd, -1000000);
    bnd_max =
        if_else(bbox[1], gsh - offset - nghostzones - lbnd, gsh + 1000000);
  }

  // Box for all points and for interior (non-ghost) points in the current grid
  // function component (not restricted to a single tile)
  template <int CI, int CJ, int CK>
  void domain_boxes(const vect<int, dim> &group_nghostzones,
                    vect<int, dim> &restrict all_min,
                    vect<int, dim> &restrict all_max,
                    vect<int, dim> &restrict int_min,
                    vect<int, dim> &restrict int_max) const {
    constexpr vect<int, dim> offset{CI, CJ, CK};
    const vect<int, dim> ghost_offset = nghostzones - group_nghostzones;
    // All points
    all_min = ghost_offset;
    all_max = lsh - offset - ghost_offset;
    // Interior points
    int_min = nghostzones;
    int_max = lsh - offset - nghostzones;
  }

  // Box including all points in the current tile
  template <int CI, int CJ, int CK>
  void box_all(const vect<int, dim> &group_nghostzones,
               vect<int, dim> &restrict imin,
               vect<int, dim> &restrict imax) const {
    vect<int, dim> all_min, all_max, int_min, int_max;
    domain_boxes<CI, CJ, CK>(group_nghostzones, all_min, all_max, int_min,
                             int_max);
    using std::max, std::min;
    imin = max(all_min, tmin);
    imax = min(all_max, tmax);
  }

  // Box including all interior points in the current tile
  template <int CI, int CJ, int CK>
  void box_int(const vect<int, dim> &group_nghostzones,
               vect<int, dim> &restrict imin,
               vect<int, dim> &restrict imax) const {
    vect<int, dim> all_min, all_max, int_min, int_max;
    domain_boxes<CI, CJ, CK>(group_nghostzones, all_min, all_max, int_min,
                             int_max);
    using std::max, std::min;
    imin = max(int_min, tmin);
    imax = min(int_max, tmax);
  }

  // Loop over all points
  template <int CI, int CJ, int CK, int VS = 1, int N = 1, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE void
  loop_all(const vect<int, dim> &group_nghostzones, const F &f) const {
    vect<int, dim> bnd_min, bnd_max;
    boundary_box<CI, CJ, CK>(group_nghostzones, bnd_min, bnd_max);
    vect<int, dim> imin, imax;
    box_all<CI, CJ, CK>(group_nghostzones, imin, imax);
    loop_box<CI, CJ, CK, VS, N>(bnd_min, bnd_max, imin, imax, f);
  }

  // Loop over all interior points
  template <int CI, int CJ, int CK, int VS = 1, int N = 1, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE void
  loop_int(const vect<int, dim> &group_nghostzones, const F &f) const {
    vect<int, dim> bnd_min, bnd_max;
    boundary_box<CI, CJ, CK>(group_nghostzones, bnd_min, bnd_max);
    vect<int, dim> imin, imax;
    box_int<CI, CJ, CK>(group_nghostzones, imin, imax);
    loop_box<CI, CJ, CK, VS, N>(bnd_min, bnd_max, imin, imax, f);
  }

  // Loop over a part of the domain. Loop over the interior first,
  // then faces, then edges, then corners.
  template <int CI, int CJ, int CK, int VS = 1, int N = 1, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE void
  loop_there(const vect<int, dim> &group_nghostzones,
             const vect<vect<vect<bool, dim>, dim>, dim> &there,
             const F &f) const {
    constexpr vect<int, dim> offset{CI, CJ, CK};

    vect<int, dim> bnd_min, bnd_max;
    boundary_box<CI, CJ, CK>(group_nghostzones, bnd_min, bnd_max);
    vect<int, dim> all_min, all_max, int_min, int_max;
    domain_boxes<CI, CJ, CK>(group_nghostzones, all_min, all_max, int_min,
                             int_max);

    for (int rank = dim; rank >= 0; --rank) {

      for (int nk = -1; nk <= +1; ++nk) {
        for (int nj = -1; nj <= +1; ++nj) {
          for (int ni = -1; ni <= +1; ++ni) {
            if ((ni == 0) + (nj == 0) + (nk == 0) == rank) {

              if (there[ni + 1][nj + 1][nk + 1]) {

                const vect<int, dim> inormal{ni, nj, nk};

                vect<int, dim> imin, imax;
                for (int d = 0; d < dim; ++d) {
                  switch (inormal[d]) {
                  case -1: // lower boundary
                    imin[d] = all_min[d];
                    imax[d] = int_min[d];
                    break;
                  case 0: // interior
                    imin[d] = int_min[d];
                    imax[d] = int_max[d];
                    break;
                  case +1: // upper boundary
                    imin[d] = int_max[d];
                    imax[d] = all_max[d];
                    break;
                  default:
                    assert(0);
                  }

                  using std::max, std::min;
                  imin[d] = max(tmin[d], imin[d]);
                  imax[d] = min(tmax[d], imax[d]);
                }

                loop_box<CI, CJ, CK, VS, N>(bnd_min, bnd_max, imin, imax, f);
              }
            } // if rank
          }
        }
      }

    } // for rank
  }

  // Loop over all outer boundary points. This excludes ghost faces, but
  // includes ghost edges/corners on non-ghost faces. Loop over faces first,
  // then edges, then corners.
  template <int CI, int CJ, int CK, int VS = 1, int N = 1, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE void
  loop_bnd(const vect<int, dim> &group_nghostzones, const F &f) const {
    vect<int, dim> bnd_min, bnd_max;
    boundary_box<CI, CJ, CK>(group_nghostzones, bnd_min, bnd_max);
    vect<int, dim> all_min, all_max, int_min, int_max;
    domain_boxes<CI, CJ, CK>(group_nghostzones, all_min, all_max, int_min,
                             int_max);

    for (int rank = dim - 1; rank >= 0; --rank) {

      for (int nk = -1; nk <= +1; ++nk) {
        for (int nj = -1; nj <= +1; ++nj) {
          for (int ni = -1; ni <= +1; ++ni) {
            if ((ni == 0) + (nj == 0) + (nk == 0) == rank) {

              if ((ni != 0 && bbox[ni < 0 ? 0 : 1][0]) ||
                  (nj != 0 && bbox[nj < 0 ? 0 : 1][1]) ||
                  (nk != 0 && bbox[nk < 0 ? 0 : 1][2])) {

                const vect<int, dim> inormal{ni, nj, nk};

                vect<int, dim> imin, imax;
                for (int d = 0; d < dim; ++d) {
                  switch (inormal[d]) {
                  case -1: // lower boundary
                    imin[d] = all_min[d];
                    imax[d] = int_min[d];
                    break;
                  case 0: // interior
                    imin[d] = int_min[d];
                    imax[d] = int_max[d];
                    break;
                  case +1: // upper boundary
                    imin[d] = int_max[d];
                    imax[d] = all_max[d];
                    break;
                  default:
                    assert(0);
                  }

                  using std::min, std::max;
                  imin[d] = max(tmin[d], imin[d]);
                  imax[d] = min(tmax[d], imax[d]);
                }

                loop_box<CI, CJ, CK, VS, N>(bnd_min, bnd_max, imin, imax, f);
              }
            } // if rank
          }
        }
      }

    } // for rank
  }

#if 0
  // Loop over all outer ghost points. This includes ghost edges/corners on
  // non-ghost faces. Loop over faces first, then edges, then corners.
  template <int CI, int CJ, int CK, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE void
  loop_ghosts_inclusive(const vect<int, dim> &group_nghostzones,
                        const F &f) const {
    constexpr vect<int, dim> offset{CI, CJ, CK};

    for (int rank = dim - 1; rank >= 0; --rank) {

      for (int nk = -1; nk <= +1; ++nk) {
        for (int nj = -1; nj <= +1; ++nj) {
          for (int ni = -1; ni <= +1; ++ni) {
            if ((ni == 0) + (nj == 0) + (nk == 0) == rank) {

              if ((ni != 0 && !bbox[ni < 0 ? 0 : 1][0]) ||
                  (nj != 0 && !bbox[nj < 0 ? 0 : 1][1]) ||
                  (nk != 0 && !bbox[nk < 0 ? 0 : 1][2])) {

                const vect<int, dim> inormal{ni, nj, nk};

                vect<int, dim> imin, imax;
                for (int d = 0; d < dim; ++d) {
                  const int ghost_offset =
                      nghostzones[d] - group_nghostzones[d];
                  const int begin_bnd = ghost_offset;
                  const int begin_int = nghostzones[d];
                  const int end_int = lsh[d] - offset[d] - nghostzones[d];
                  const int end_bnd = lsh[d] - offset[d] - ghost_offset;
                  switch (inormal[d]) {
                  case -1: // lower boundary
                    imin[d] = begin_bnd;
                    imax[d] = begin_int;
                    break;
                  case 0: // interior
                    imin[d] = begin_int;
                    imax[d] = end_int;
                    break;
                  case +1: // upper boundary
                    imin[d] = end_int;
                    imax[d] = end_bnd;
                    break;
                  default:
                    assert(0);
                  }

                  imin[d] = std::max(tmin[d], imin[d]);
                  imax[d] = std::min(tmax[d], imax[d]);
                }

#ifdef CCTK_DEBUG
                bool isempty = false;
                for (int d = 0; d < dim; ++d)
                  isempty |= imin[d] >= imax[d];
                if (!isempty) {
                  vect<int, dim> all_imin, all_imax;
                  box_all<CI, CJ, CK>(group_nghostzones, all_imin, all_imax);
                  vect<int, dim> int_imin, int_imax;
                  box_int<CI, CJ, CK>(group_nghostzones, int_imin, int_imax);
                  for (int d = 0; d < dim; ++d) {
                    assert(all_imin[d] <= imin[d]);
                    assert(imax[d] <= all_imax[d]);
                  }
                  bool overlaps = true;
                  for (int d = 0; d < dim; ++d)
                    overlaps &=
                        !(imax[d] <= int_imin[d] || imin[d] >= int_imax[d]);
                  assert(!overlaps);
                }
#endif

                loop_box_boundary<CI, CJ, CK>(imin, imax, inormal, f);
              }
            } // if rank
          }
        }
      }
    } // for rank
  }
#endif

  // Loop over all outer ghost points. This excludes ghost edges/corners on
  // non-ghost faces. Loop over faces first, then edges, then corners.
  template <int CI, int CJ, int CK, int VS = 1, int N = 1, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE void
  loop_ghosts(const vect<int, dim> &group_nghostzones, const F &f) const {
    vect<int, dim> bnd_min, bnd_max;
    boundary_box<CI, CJ, CK>(group_nghostzones, bnd_min, bnd_max);
    vect<int, dim> all_min, all_max, int_min, int_max;
    domain_boxes<CI, CJ, CK>(group_nghostzones, all_min, all_max, int_min,
                             int_max);

    for (int rank = dim - 1; rank >= 0; --rank) {

      for (int nk = -1; nk <= +1; ++nk) {
        for (int nj = -1; nj <= +1; ++nj) {
          for (int ni = -1; ni <= +1; ++ni) {
            if ((ni == 0) + (nj == 0) + (nk == 0) == rank) {

              if ((ni == 0 || !bbox[ni < 0 ? 0 : 1][0]) &&
                  (nj == 0 || !bbox[nj < 0 ? 0 : 1][1]) &&
                  (nk == 0 || !bbox[nk < 0 ? 0 : 1][2])) {

                const vect<int, dim> inormal{ni, nj, nk};

                vect<int, dim> imin, imax;
                for (int d = 0; d < dim; ++d) {
                  switch (inormal[d]) {
                  case -1: // lower boundary
                    imin[d] = all_min[d];
                    imax[d] = int_min[d];
                    break;
                  case 0: // interior
                    imin[d] = int_min[d];
                    imax[d] = int_max[d];
                    break;
                  case +1: // upper boundary
                    imin[d] = int_max[d];
                    imax[d] = all_max[d];
                    break;
                  default:
                    assert(0);
                  }

                  using std::min, std::max;
                  imin[d] = max(tmin[d], imin[d]);
                  imax[d] = min(tmax[d], imax[d]);
                }

                loop_box<CI, CJ, CK, VS, N>(bnd_min, bnd_max, imin, imax, f);
              }
            } // if rank
          }
        }
      }
    } // for rank
  }

  // Loop over the outermost "boundary" points in the interior. They correspond
  // to points that are shifted inwards by = cctk_nghostzones[3] from those that
  // CarpetX identifies as boundary points. From the perspective of CarpetX (or
  // AMReX), these do not belong in the outer boundary, but rather the interior.
  // This excludes ghost faces, but includes ghost edges/corners on non-ghost
  // faces. Loop over faces first, then edges, then corners. Modified from
  // loop_bnd.
  template <int CI, int CJ, int CK, int VS = 1, int N = 1, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE void
  loop_outermost_int(const vect<int, dim> &group_nghostzones,
                     const F &f) const {
    // boundary_box sets bnd_min and bnd_max
    vect<int, dim> bnd_min, bnd_max;
    // if on (Carpetx) boundary points, then
    //   bnd_min = nghostzones - lbnd
    //   bnd_max = gsh - offset - nghostzones - lbnd
    //   where lbnd : an array of cctk dim integers containing the lowest index
    //   (in each direction) of the local grid, as seen on the global grid
    // else,
    //   bnd_min = -100000000
    //   bnd_max = gsh + 1000000
    // where cctk_gsh : an array of cctk dim integers with the global grid size
    boundary_box<CI, CJ, CK>(group_nghostzones, bnd_min, bnd_max);

    // domain_boxes sets all_min, all_max, int_min and int_max
    vect<int, dim> all_min, all_max, int_min, int_max;
    // Indices for the entire grid
    //   all_min = ghost_offset;
    //   all_max = lsh - offset - ghost_offset;
    // Indices for the interior
    //   int_min = nghostzones;
    //   int_max = lsh - offset - nghostzones;
    domain_boxes<CI, CJ, CK>(group_nghostzones, all_min, all_max, int_min,
                             int_max);

    // Check that the actual number of ghost zones isn't overridden
    for (int d = 0; d < dim; ++d) {
      assert(group_nghostzones[d] == nghostzones[d]);
    }

    // Shift the indices towards the interior by nghostzones
    all_min += nghostzones;
    int_min += nghostzones;
    bnd_min += nghostzones;
    all_max -= nghostzones;
    int_max -= nghostzones;
    bnd_max -= nghostzones;

    // rank selects: faces, edges, corners
    // rank(interior) = 3
    // rank(face)     = 2
    // rank(edge)     = 1
    // rank(corner)   = 0
    for (int rank = dim - 1; rank >= 0; --rank) {

      // Nested loops that determine {ni,nj,nk} components of the normal vector
      for (int nk = -1; nk <= +1; ++nk) {
        for (int nj = -1; nj <= +1; ++nj) {
          for (int ni = -1; ni <= +1; ++ni) {
            // True when the normal vector is normal to a {face, edge, corner}
            if ((ni == 0) + (nj == 0) + (nk == 0) == rank) {
              // True when point is on left/right boundary,
              // and vector is not parallel to a {face,corner,edge}
              // In either of the 3 directions
              if ((ni != 0 && bbox[ni < 0 ? 0 : 1][0]) ||
                  (nj != 0 && bbox[nj < 0 ? 0 : 1][1]) ||
                  (nk != 0 && bbox[nk < 0 ? 0 : 1][2])) {

                const vect<int, dim> inormal{ni, nj, nk}; // normal vector

                vect<int, dim> imin, imax;
                for (int d = 0; d < dim; ++d) {
                  switch (inormal[d]) {
                  case -1: // lower boundary
                    imin[d] = all_min[d];
                    imax[d] = int_min[d];
                    break;
                  case 0: // interior
                    imin[d] = int_min[d];
                    imax[d] = int_max[d];
                    break;
                  case +1: // upper boundary
                    imin[d] = int_max[d];
                    imax[d] = all_max[d];
                    break;
                  default:
                    assert(0);
                  }

                  using std::min, std::max;
                  // Adjust loop_box index range according to tile min, max
                  imin[d] = max(tmin[d], imin[d]);
                  imax[d] = min(tmax[d], imax[d]);
                }

                loop_box<CI, CJ, CK, VS, N>(bnd_min, bnd_max, imin, imax, f);
              }
            } // if rank
          }
        }
      }

    } // for rank
  }

  template <int CI, int CJ, int CK, where_t where, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE
      std::enable_if_t<(where == where_t::everywhere), void>
      loop(const vect<int, dim> &group_nghostzones, const F &f) const {
    loop_all<CI, CJ, CK>(group_nghostzones, f);
  }
  template <int CI, int CJ, int CK, where_t where, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE
      std::enable_if_t<(where == where_t::interior), void>
      loop(const vect<int, dim> &group_nghostzones, const F &f) const {
    loop_int<CI, CJ, CK>(group_nghostzones, f);
  }
  template <int CI, int CJ, int CK, where_t where, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE
      std::enable_if_t<(where == where_t::boundary), void>
      loop(const vect<int, dim> &group_nghostzones, const F &f) const {
    loop_bnd<CI, CJ, CK>(group_nghostzones, f);
  }
#if 0
  template <int CI, int CJ, int CK, where_t where, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE
      std::enable_if_t<(where == where_t::ghosts_inclusive), void>
      loop(const vect<int, dim> &group_nghostzones, const F &f) const {
    loop_ghosts_inclusive<CI, CJ, CK>(group_nghostzones, f);
  }
#endif
  template <int CI, int CJ, int CK, where_t where, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE
      std::enable_if_t<(where == where_t::ghosts), void>
      loop(const vect<int, dim> &group_nghostzones, const F &f) const {
    loop_ghosts<CI, CJ, CK>(group_nghostzones, f);
  }

  template <int CI, int CJ, int CK, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE void
  loop(where_t where, const vect<int, dim> &group_nghostzones,
       const F &f) const {
    switch (where) {
    case where_t::everywhere:
      return noinline([&] {
        return loop<CI, CJ, CK, where_t::everywhere>(group_nghostzones, f);
      });
    case where_t::interior:
      return noinline([&] {
        return loop<CI, CJ, CK, where_t::interior>(group_nghostzones, f);
      });
    case where_t::boundary:
      return noinline([&] {
        return loop<CI, CJ, CK, where_t::boundary>(group_nghostzones, f);
      });
#if 0
    case where_t::ghosts_inclusive:
      return noinline([&] {
        return loop<CI, CJ, CK, where_t::ghosts_inclusive>(group_nghostzones,
                                                           f);
      });
#endif
    case where_t::ghosts:
      return noinline([&] {
        return loop<CI, CJ, CK, where_t::ghosts>(group_nghostzones, f);
      });
    default:
      assert(0);
    }
  }

  template <int CI, int CJ, int CK, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE void loop(where_t where,
                                                const F &f) const {
    loop<CI, CJ, CK>(where, nghostzones, f);
  }

  template <int CI, int CJ, int CK, where_t where, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE void loop(const F &f) const {
    loop<CI, CJ, CK, where>(nghostzones, f);
  }

  template <typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE void
  loop_idx(where_t where, const vect<int, dim> &indextype,
           const vect<int, dim> &group_nghostzones, const F &f) const {
    switch (indextype[0] + 2 * indextype[1] + 4 * indextype[2]) {
    case 0b000:
      return noinline(
          [&] { return loop<0, 0, 0>(where, group_nghostzones, f); });
    case 0b001:
      return noinline(
          [&] { return loop<1, 0, 0>(where, group_nghostzones, f); });
    case 0b010:
      return noinline(
          [&] { return loop<0, 1, 0>(where, group_nghostzones, f); });
    case 0b011:
      return noinline(
          [&] { return loop<1, 1, 0>(where, group_nghostzones, f); });
    case 0b100:
      return noinline(
          [&] { return loop<0, 0, 1>(where, group_nghostzones, f); });
    case 0b101:
      return noinline(
          [&] { return loop<1, 0, 1>(where, group_nghostzones, f); });
    case 0b110:
      return noinline(
          [&] { return loop<0, 1, 1>(where, group_nghostzones, f); });
    case 0b111:
      return noinline(
          [&] { return loop<1, 1, 1>(where, group_nghostzones, f); });
    default:
      assert(0);
    }
  }

  template <typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE void
  loop_idx(where_t where, const vect<int, dim> &indextype, const F &f) const {
    loop_idx(where, indextype, nghostzones, f);
  }
};

template <typename F>
inline CCTK_ATTRIBUTE_ALWAYS_INLINE void
loop_idx(const cGH *cctkGH, where_t where, const vect<int, dim> &indextype,
         const vect<int, dim> &nghostzones, const F &f) {
  GridDescBase(cctkGH).loop_idx(where, indextype, nghostzones, f);
}

template <typename F>
inline CCTK_ATTRIBUTE_ALWAYS_INLINE void
loop_idx(const cGH *cctkGH, where_t where, const vect<int, dim> &indextype,
         const F &f) {
  GridDescBase(cctkGH).loop_idx(where, indextype, f);
}

template <int CI, int CJ, int CK, where_t where, typename F>
inline CCTK_ATTRIBUTE_ALWAYS_INLINE void loop(const cGH *cctkGH, const F &f) {
  GridDescBase(cctkGH).loop<CI, CJ, CK, where>(f);
}

template <int CI, int CJ, int CK, typename F>
inline CCTK_ATTRIBUTE_ALWAYS_INLINE void loop(const cGH *cctkGH, where_t where,
                                              const F &f) {
  GridDescBase(cctkGH).loop<CI, CJ, CK>(where, f);
}

// Keep these for convenience
template <int CI, int CJ, int CK, typename F>
inline CCTK_ATTRIBUTE_ALWAYS_INLINE void loop_all(const cGH *cctkGH,
                                                  const F &f) {
  loop<CI, CJ, CK, where_t::everywhere>(cctkGH, f);
}

template <int CI, int CJ, int CK, typename F>
inline CCTK_ATTRIBUTE_ALWAYS_INLINE void loop_int(const cGH *cctkGH,
                                                  const F &f) {
  loop<CI, CJ, CK, where_t::interior>(cctkGH, f);
}

template <int CI, int CJ, int CK, typename F>
inline CCTK_ATTRIBUTE_ALWAYS_INLINE void loop_bnd(const cGH *cctkGH,
                                                  const F &f) {
  loop<CI, CJ, CK, where_t::boundary>(cctkGH, f);
}

#if 0
template <int CI, int CJ, int CK, typename F>
inline CCTK_ATTRIBUTE_ALWAYS_INLINE void
loop_ghosts_inclusive(const cGH *cctkGH, const F &f) {
  loop<CI, CJ, CK, where_t::ghosts_inclusive>(cctkGH, f);
}
#endif

template <int CI, int CJ, int CK, typename F>
inline CCTK_ATTRIBUTE_ALWAYS_INLINE void loop_ghosts(const cGH *cctkGH,
                                                     const F &f) {
  loop<CI, CJ, CK, where_t::ghosts>(cctkGH, f);
}

////////////////////////////////////////////////////////////////////////////////

template <typename T, int CI, int CJ, int CK> struct GF3D {
  static_assert(CI == 0 || CI == 1);
  static_assert(CJ == 0 || CJ == 1);
  static_assert(CK == 0 || CK == 1);
  typedef T value_type;
  static constexpr int di = 1;
  const int dj, dk, np;
  const int ni, nj, nk;
  T *restrict const ptr;
  static constexpr CCTK_ATTRIBUTE_ALWAYS_INLINE vect<int, dim> indextype() {
    return {CI, CJ, CK};
  }
  GF3D() = delete;
  GF3D(const GF3D &) = default;
  GF3D(GF3D &&) = default;
  GF3D &operator=(const GF3D &) = default;
  GF3D &operator=(GF3D &&) = default;
  GF3D(const cGH *restrict cctkGH, T *restrict ptr)
      : dj(di * (cctkGH->cctk_ash[0] - CI)),
        dk(dj * (cctkGH->cctk_ash[1] - CJ)),
        np(dk * (cctkGH->cctk_ash[2] - CK)), ni(cctkGH->cctk_lsh[0] - CI),
        nj(cctkGH->cctk_lsh[1] - CJ), nk(cctkGH->cctk_lsh[2] - CK), ptr(ptr) {}
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE int offset(int i, int j, int k) const {
    // These index checks prevent vectorization. We thus only enable
    // them in debug mode.
#ifdef CCTK_DEBUG
    assert(i >= 0 && i < ni);
    assert(j >= 0 && j < nj);
    assert(k >= 0 && k < nk);
#endif
    return i * di + j * dj + k * dk;
  }
  inline int offset(const vect<int, dim> &I) const {
    return offset(I[0], I[1], I[2]);
  }
  inline T &restrict operator()(int i, int j, int k) const {
    return ptr[offset(i, j, k)];
  }
  inline T &restrict operator()(const vect<int, dim> &I) const {
    return ptr[offset(I)];
  }
};

////////////////////////////////////////////////////////////////////////////////

template <typename T> struct GF3D1 {
  typedef T value_type;
  T *restrict ptr;
#ifdef CCTK_DEBUG
  vect<int, dim> imin, imax;
  vect<int, dim> ash;
#endif
  static constexpr int di = 1;
  int dj, dk, np;
  int off;
  GF3D1() = delete;
  GF3D1(const GF3D1 &) = default;
  GF3D1(GF3D1 &&) = default;
  GF3D1 &operator=(const GF3D1 &) = default;
  GF3D1 &operator=(GF3D1 &&) = default;
  GF3D1(T *restrict ptr, const vect<int, dim> &imin, const vect<int, dim> &imax,
        const vect<int, dim> &ash)
      : ptr(ptr),
#ifdef CCTK_DEBUG
        imin(imin), imax(imax), ash(ash),
#endif
        dj(di * ash[0]), dk(dj * ash[1]), np(dk * ash[2]),
        off(imin[0] * di + imin[1] * dj + imin[2] * dk) {
  }
  GF3D1(const cGH *restrict cctkGH, const vect<int, dim> &indextype,
        const vect<int, dim> &nghostzones, T *restrict ptr) {
    for (int d = 0; d < dim; ++d)
      assert(indextype[d] == 0 || indextype[d] == 1);
    for (int d = 0; d < dim; ++d) {
      assert(nghostzones[d] >= 0);
      assert(nghostzones[d] <= cctkGH->cctk_nghostzones[d]);
    }
    vect<int, dim> imin, imax;
    for (int d = 0; d < dim; ++d) {
      imin[d] = cctkGH->cctk_nghostzones[d] - nghostzones[d];
      imax[d] = cctkGH->cctk_lsh[d] - indextype[d] -
                (cctkGH->cctk_nghostzones[d] - nghostzones[d]);
    }
    vect<int, dim> ash;
    for (int d = 0; d < dim; ++d)
      ash[d] = cctkGH->cctk_ash[d] - indextype[d] -
               2 * (cctkGH->cctk_nghostzones[d] - nghostzones[d]);
    *this = GF3D1(ptr, imin, imax, ash);
  }
  int offset(int i, int j, int k) const {
    // These index checks prevent vectorization. We thus only enable
    // them in debug mode.
#ifdef CCTK_DEBUG
    assert(i >= imin[0] && i < imax[0]);
    assert(j >= imin[1] && j < imax[1]);
    assert(k >= imin[2] && k < imax[2]);
#endif
    return i * di + j * dj + k * dk - off;
  }
  int offset(const vect<int, dim> &I) const { return offset(I[0], I[1], I[2]); }
  T &restrict operator()(int i, int j, int k) const {
    return ptr[offset(i, j, k)];
  }
  T &restrict operator()(const vect<int, dim> &I) const {
    return ptr[offset(I)];
  }
  T &restrict operator()(const PointDesc &p) const { return (*this)(p.I); }
};

////////////////////////////////////////////////////////////////////////////////

struct GF3D2layout {
#ifdef CCTK_DEBUG
  vect<int, dim> imin, imax;
  vect<int, dim> ash;
#endif
  static constexpr int di = 1;
  int dj, dk, np;
  int off;
  GF3D2layout() = delete;
  GF3D2layout(const GF3D2layout &) = default;
  GF3D2layout(GF3D2layout &&) = default;
  GF3D2layout &operator=(const GF3D2layout &) = default;
  GF3D2layout &operator=(GF3D2layout &&) = default;
  CCTK_DEVICE CCTK_HOST GF3D2layout(const vect<int, dim> &imin,
                                    const vect<int, dim> &imax,
                                    const vect<int, dim> &ash)
      :
#ifdef CCTK_DEBUG
        imin(imin), imax(imax), ash(ash),
#endif
        dj(di * ash[0]), dk(dj * ash[1]), np(dk * ash[2]),
        off(imin[0] * di + imin[1] * dj + imin[2] * dk) {
  }
  CCTK_DEVICE CCTK_HOST GF3D2layout(const vect<int, dim> &imin,
                                    const vect<int, dim> &imax)
      : GF3D2layout(imin, imax, imax - imin) {
#ifdef CCTK_DEBUG
    if (np > 0) {
      assert(linear(imin[0], imin[1], imin[2]) == 0);
      assert(linear(imax[0] - 1, imax[1] - 1, imax[2] - 1) == np - 1);
    }
#endif
  }
  CCTK_DEVICE CCTK_HOST GF3D2layout(const cGH *restrict cctkGH,
                                    const vect<int, dim> &indextype,
                                    const vect<int, dim> &nghostzones) {
    for (int d = 0; d < dim; ++d)
      assert(indextype[d] == 0 || indextype[d] == 1);
    for (int d = 0; d < dim; ++d) {
      assert(nghostzones[d] >= 0);
      assert(nghostzones[d] <= cctkGH->cctk_nghostzones[d]);
    }
    vect<int, dim> imin, imax;
    for (int d = 0; d < dim; ++d) {
      imin[d] = cctkGH->cctk_nghostzones[d] - nghostzones[d];
      imax[d] = cctkGH->cctk_lsh[d] - indextype[d] -
                (cctkGH->cctk_nghostzones[d] - nghostzones[d]);
    }
    vect<int, dim> ash;
    for (int d = 0; d < dim; ++d)
      ash[d] = cctkGH->cctk_ash[d] - indextype[d] -
               2 * (cctkGH->cctk_nghostzones[d] - nghostzones[d]);
    *this = GF3D2layout(imin, imax, ash);
  }
  CCTK_DEVICE CCTK_HOST GF3D2layout(const cGH *restrict cctkGH,
                                    const vect<int, dim> &indextype)
      : GF3D2layout(cctkGH, indextype,
                    {cctkGH->cctk_nghostzones[0], cctkGH->cctk_nghostzones[1],
                     cctkGH->cctk_nghostzones[2]}) {}
  CCTK_DEVICE CCTK_HOST bool operator==(const GF3D2layout &other) const {
    bool iseq = true;
#ifdef CCTK_DEBUG
    iseq &= std::equal_to<vect<int, dim> >()(imin, other.imin) &&
            std::equal_to<vect<int, dim> >()(imax, other.imax);
    iseq &= std::equal_to<vect<int, dim> >()(ash, other.ash);
#endif
    iseq &= dj == other.dj && dk == other.dk && np == other.np;
    iseq &= off == other.off;
    return iseq;
  }
  CCTK_DEVICE CCTK_HOST bool operator!=(const GF3D2layout &other) const {
    return !(*this == other);
  }
  CCTK_DEVICE CCTK_HOST int linear(int i, int j, int k) const {
    // These index checks prevent vectorization. We thus only enable
    // them in debug mode.
#ifdef CCTK_DEBUG
    assert(i >= imin[0] && i < imax[0]);
    assert(j >= imin[1] && j < imax[1]);
    assert(k >= imin[2] && k < imax[2]);
#endif
    return i * di + j * dj + k * dk - off;
  }
  CCTK_DEVICE CCTK_HOST int linear(const vect<int, dim> &I) const {
    return linear(I[0], I[1], I[2]);
  }
  CCTK_DEVICE CCTK_HOST int delta(int i, int j, int k) const {
    return i * di + j * dj + k * dk;
  }
  CCTK_DEVICE CCTK_HOST int delta(const vect<int, dim> &I) const {
    return delta(I[0], I[1], I[2]);
  }
};

struct GF3D2index {
#ifdef CCTK_DEBUG
  GF3D2layout layout;
#endif
  int m_linear;
  GF3D2index() = delete;
  GF3D2index(const GF3D2index &) = default;
  GF3D2index(GF3D2index &&) = default;
  GF3D2index &operator=(const GF3D2index &) = default;
  GF3D2index &operator=(GF3D2index &&) = default;
  CCTK_DEVICE CCTK_HOST GF3D2index(const GF3D2layout &layout,
                                   const vect<int, dim> &I)
      :
#ifdef CCTK_DEBUG
        layout(layout),
#endif
        m_linear(layout.linear(I)) {
  }
  CCTK_DEVICE CCTK_HOST int linear() const { return m_linear; }
};

template <typename T> struct GF3D2 {
  typedef T value_type;
  // TODO: disallow inf, nan
  // Note: Keep `ptr` as first member, this improves performance a bit (?)
  T *restrict ptr;
  GF3D2layout layout;
  GF3D2() = delete;
  GF3D2(const GF3D2 &) = default;
  GF3D2(GF3D2 &&) = default;
  GF3D2 &operator=(const GF3D2 &) = default;
  GF3D2 &operator=(GF3D2 &&) = default;
  CCTK_DEVICE CCTK_HOST GF3D2(const GF3D2layout &layout, T *restrict ptr)
      : ptr(ptr), layout(layout) {}
  CCTK_DEVICE CCTK_HOST GF3D2index index(const vect<int, dim> &I) const {
    return GF3D2index(layout, I);
  }
  CCTK_DEVICE CCTK_HOST int linear(const vect<int, dim> &I) const {
    return index(I).linear();
  }
  CCTK_DEVICE CCTK_HOST int linear(int i, int j, int k) const {
    return index(vect<int, dim>{i, j, k}).linear();
  }
  CCTK_DEVICE CCTK_HOST int delta(const vect<int, dim> &I) const {
    return layout.delta(I);
  }
  CCTK_DEVICE CCTK_HOST int delta(int i, int j, int k) const {
    return layout.delta(i, j, k);
  }
  CCTK_DEVICE CCTK_HOST T &restrict operator()(const GF3D2index &index) const {
#ifdef CCTK_DEBUG
    assert(index.layout == this->layout);
#endif
    return ptr[index.linear()];
  }
  CCTK_DEVICE CCTK_HOST T &restrict operator()(const vect<int, dim> &I) const {
    return ptr[linear(I)];
  }
  CCTK_DEVICE CCTK_HOST T &restrict operator()(int i, int j, int k) const {
    return ptr[linear(i, j, k)];
  }
  CCTK_DEVICE CCTK_HOST void store(const GF3D2index &index,
                                   const T &value) const {
    ptr[index.linear()] = value;
  }
  CCTK_DEVICE CCTK_HOST void store(const vect<int, dim> &I,
                                   const T &value) const {
    ptr[linear(I)] = value;
  }
#if 0
  struct simd_reference {
    using element_type = std::remove_cv_t<T>;
    using value_type = Arith::simd<element_type>;
    T *const ptr;
    const Arith::simdl<element_type> mask;
    template <typename U>
    CCTK_DEVICE CCTK_HOST simd_reference(T *const ptr,
                                         const Arith::simdl<U> &mask)
        : ptr(ptr), mask(mask) {}
    simd_reference() = delete;
    simd_reference(const simd_reference &) = default;
    simd_reference(simd_reference &&) = default;
    simd_reference &operator=(const simd_reference &) = delete;
    simd_reference &operator=(simd_reference &&) = delete;
    CCTK_DEVICE CCTK_HOST operator value_type() const {
      return Arith::maskz_loadu(mask, ptr);
    }
    // CCTK_DEVICE CCTK_HOST value_type operator()() const {
    //   return Arith::maskz_loadu(mask, ptr);
    // }
    template <typename U>
    CCTK_DEVICE CCTK_HOST simd_reference
    operator=(const Arith::simd<U> &value) {
      mask_storeu(mask, ptr, value);
      return *this;
    }
    template <typename U>
    CCTK_DEVICE CCTK_HOST simd_reference operator=(const U &value) {
      return *this = value_type(value);
    }
    friend CCTK_DEVICE CCTK_HOST auto operator+(const simd_reference &x) {
      return +value_type(x);
    };
    friend CCTK_DEVICE CCTK_HOST auto operator-(const simd_reference &x) {
      return -value_type(x);
    };
    friend CCTK_DEVICE CCTK_HOST auto operator+(const simd_reference &x,
                                                const simd_reference &y) {
      return value_type(x) + value_type(y);
    };
    friend CCTK_DEVICE CCTK_HOST auto operator-(const simd_reference &x,
                                                const simd_reference &y) {
      return value_type(x) - value_type(y);
    };
    friend CCTK_DEVICE CCTK_HOST auto operator*(const simd_reference &x,
                                                const simd_reference &y) {
      return value_type(x) * value_type(y);
    };
    friend CCTK_DEVICE CCTK_HOST auto operator/(const simd_reference &x,
                                                const simd_reference &y) {
      return value_type(x) / value_type(y);
    };
    friend CCTK_DEVICE CCTK_HOST auto operator%(const simd_reference &x,
                                                const simd_reference &y) {
      return value_type(x) % value_type(y);
    };
    template <typename X>
    friend CCTK_DEVICE CCTK_HOST auto operator*(const X &x,
                                                const simd_reference &y) {
      return x * value_type(y);
    };
  };
  CCTK_DEVICE CCTK_HOST simd_reference
  operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
             const GF3D2index &index) const {
    return simd_reference(&(*this)(index), mask);
  }
  CCTK_DEVICE CCTK_HOST simd_reference
  operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
             const vect<int, dim> &I) const {
    return (*this)(mask, GF3D2index(layout, I));
  }
#endif
  CCTK_DEVICE CCTK_HOST Arith::simd<std::remove_cv_t<T> >
  operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
             const GF3D2index &index) const {
    return Arith::maskz_loadu(mask, &(*this)(index));
  }
  CCTK_DEVICE CCTK_HOST Arith::simd<std::remove_cv_t<T> >
  operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
             const vect<int, dim> &I) const {
    return (*this)(mask, GF3D2index(layout, I));
  }
  CCTK_DEVICE CCTK_HOST Arith::simd<std::remove_cv_t<T> >
  operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
             const GF3D2index &index,
             const Arith::simd<std::remove_cv_t<T> > &other) const {
    return Arith::masko_loadu(mask, &(*this)(index), other);
  }
  template <typename U,
            std::enable_if_t<std::is_convertible_v<T, U> > * = nullptr>
  CCTK_DEVICE CCTK_HOST Arith::simd<std::remove_cv_t<T> >
  operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
             const GF3D2index &index, const U &other) const {
    return Arith::masko_loadu(mask, &(*this)(index), other);
  }
  CCTK_DEVICE CCTK_HOST Arith::simd<std::remove_cv_t<T> >
  operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
             const vect<int, dim> &I,
             const Arith::simd<std::remove_cv_t<T> > &other) const {
    return Arith::masko_loadu(mask, &(*this)(I), other);
  }
  template <typename U,
            std::enable_if_t<std::is_convertible_v<T, U> > * = nullptr>
  CCTK_DEVICE CCTK_HOST Arith::simd<std::remove_cv_t<T> >
  operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
             const vect<int, dim> &I, const U &other) const {
    return Arith::masko_loadu(mask, &(*this)(I), other);
  }
  CCTK_DEVICE CCTK_HOST void store(const Arith::simdl<T> &mask,
                                   const vect<int, dim> &I,
                                   const Arith::simd<T> &value) const {
    mask_storeu(mask, &ptr[linear(I)], value);
  }
  CCTK_DEVICE CCTK_HOST void store(const Arith::simdl<T> &mask,
                                   const GF3D2index &index,
                                   const Arith::simd<T> &value) const {
    mask_storeu(mask, &ptr[index.linear()], value);
  }
};

////////////////////////////////////////////////////////////////////////////////

template <int NI, int NJ, int NK, int OFF = 0> struct GF3D3layout {
  static_assert(NI >= 0);
  static_assert(NJ >= 0);
  static_assert(NK >= 0);

  static constexpr int ni = NI;
  static constexpr int nj = NJ;
  static constexpr int nk = NK;

  static constexpr int off = OFF;

  static constexpr int di = 1;
  static constexpr int dj = NI * di;
  static constexpr int dk = NJ * dj;
  static constexpr int np = NK * dk;

  constexpr int linear(int i, int j, int k) const {
    return i * di + j * dj + k * dk - OFF;
  }
  constexpr int linear(const vect<int, dim> &I) const {
    return (*this)(I[0], I[1], I[2]);
  }
};

template <int imin0, int imin1, int imin2, int imax0, int imax1, int imax2>
struct makeGF3D3layout {
private:
  static constexpr int ni = imax0 - imin0;
  static constexpr int nj = imax1 - imin1;
  static constexpr int nk = imax2 - imin2;
  static constexpr int di = GF3D3layout<ni, nj, nk>::di;
  static constexpr int dj = GF3D3layout<ni, nj, nk>::dj;
  static constexpr int dk = GF3D3layout<ni, nj, nk>::dk;
  static constexpr int off = imin0 * di + imin1 * dj + imin2 * dk;

public:
  typedef GF3D3layout<ni, nj, nk, off> type;
};
template <int imin0, int imin1, int imin2, int imax0, int imax1, int imax2>
using makeGF3D3layout_t =
    typename makeGF3D3layout<imin0, imin1, imin2, imax0, imax1, imax2>::type;

template <typename T, int NI, int NJ, int NK, int OFF = 0>
struct GF3D3 : GF3D3layout<NI, NJ, NK, OFF> {
  using GF3D3layout<NI, NJ, NK, OFF>::np;
  using GF3D3layout<NI, NJ, NK, OFF>::linear;

  vect<T, np> arr;

  constexpr T &restrict operator()(int i, int j, int k) {
    return arr[linear(i, j, k)];
  }
  constexpr const T &restrict operator()(int i, int j, int k) const {
    return arr[linear(i, j, k)];
  }
  constexpr T &restrict operator()(const vect<int, dim> &I) {
    return (*this)(I[0], I[1], I[2]);
  }
  constexpr const T &restrict operator()(const vect<int, dim> &I) const {
    return (*this)(I[0], I[1], I[2]);
  }
};

template <typename T, int NI, int NJ, int NK, int OFF = 0>
struct GF3D3ptr : GF3D3layout<NI, NJ, NK, OFF> {
  using GF3D3layout<NI, NJ, NK, OFF>::np;
  using GF3D3layout<NI, NJ, NK, OFF>::linear;

  T *restrict ptr;

  GF3D3ptr() = delete;
  GF3D3ptr(T *restrict ptr) : ptr(ptr) {}

  constexpr T &restrict operator()(int i, int j, int k) const {
    return ptr[linear(i, j, k)];
  }
  constexpr T &restrict operator()(const vect<int, dim> &I) const {
    return (*this)(I[0], I[1], I[2]);
  }
};

////////////////////////////////////////////////////////////////////////////////

struct GF3D5layout {
#ifdef CCTK_DEBUG
  vect<int, dim> imin, imax;
  vect<int, dim> ash;
#endif
  static constexpr int di = 1;
  int dj, dk, np;
  int off;
  GF3D5layout() = delete;
  GF3D5layout(const GF3D5layout &) = default;
  GF3D5layout(GF3D5layout &&) = default;
  GF3D5layout &operator=(const GF3D5layout &) = default;
  GF3D5layout &operator=(GF3D5layout &&) = default;
  CCTK_DEVICE CCTK_HOST GF3D5layout(const vect<int, dim> &imin,
                                    const vect<int, dim> &imax,
                                    const vect<int, dim> &ash)
      :
#ifdef CCTK_DEBUG
        imin(imin), imax(imax), ash(ash),
#endif
        dj(di * ash[0]), dk(dj * ash[1]), np(dk * ash[2]),
        off(imin[0] * di + imin[1] * dj + imin[2] * dk) {
  }
  CCTK_DEVICE CCTK_HOST GF3D5layout(const vect<int, dim> &imin,
                                    const vect<int, dim> &imax)
      : GF3D5layout(imin, imax, imax - imin) {
#ifdef CCTK_DEBUG
    if (np > 0) {
      assert(linear(imin[0], imin[1], imin[2]) == 0);
      assert(linear(imax[0] - 1, imax[1] - 1, imax[2] - 1) == np - 1);
    }
#endif
  }
  CCTK_DEVICE CCTK_HOST GF3D5layout(const cGH *restrict cctkGH,
                                    const vect<int, dim> &indextype,
                                    const vect<int, dim> &nghostzones) {
    for (int d = 0; d < dim; ++d)
      assert(indextype[d] == 0 || indextype[d] == 1);
    for (int d = 0; d < dim; ++d) {
      assert(nghostzones[d] >= 0);
      assert(nghostzones[d] <= cctkGH->cctk_nghostzones[d]);
    }
    vect<int, dim> imin, imax;
    for (int d = 0; d < dim; ++d) {
      imin[d] = cctkGH->cctk_nghostzones[d] - nghostzones[d];
      imax[d] = cctkGH->cctk_lsh[d] - indextype[d] -
                (cctkGH->cctk_nghostzones[d] - nghostzones[d]);
    }
    vect<int, dim> ash;
    for (int d = 0; d < dim; ++d)
      ash[d] = cctkGH->cctk_ash[d] - indextype[d] -
               2 * (cctkGH->cctk_nghostzones[d] - nghostzones[d]);
    *this = GF3D5layout(imin, imax, ash);
  }
  CCTK_DEVICE CCTK_HOST GF3D5layout(const cGH *restrict cctkGH,
                                    const vect<int, dim> &indextype)
      : GF3D5layout(cctkGH, indextype,
                    {cctkGH->cctk_nghostzones[0], cctkGH->cctk_nghostzones[1],
                     cctkGH->cctk_nghostzones[2]}) {}
  CCTK_DEVICE CCTK_HOST bool operator==(const GF3D5layout &other) const {
    bool iseq = true;
#ifdef CCTK_DEBUG
    iseq &= std::equal_to<vect<int, dim> >()(imin, other.imin) &&
            std::equal_to<vect<int, dim> >()(imax, other.imax);
    iseq &= std::equal_to<vect<int, dim> >()(ash, other.ash);
#endif
    iseq &= dj == other.dj && dk == other.dk && np == other.np;
    iseq &= off == other.off;
    return iseq;
  }
  CCTK_DEVICE CCTK_HOST int linear(int i, int j, int k) const {
    // These index checks prevent vectorization. We thus only enable
    // them in debug mode.
#ifdef CCTK_DEBUG
    assert(i >= imin[0] && i < imax[0]);
    assert(j >= imin[1] && j < imax[1]);
    assert(k >= imin[2] && k < imax[2]);
#endif
    return i * di + j * dj + k * dk - off;
  }
  CCTK_DEVICE CCTK_HOST int linear(const vect<int, dim> &I) const {
    return linear(I[0], I[1], I[2]);
  }
  CCTK_DEVICE CCTK_HOST int delta(int i, int j, int k) const {
    return i * di + j * dj + k * dk;
  }
  CCTK_DEVICE CCTK_HOST int delta(const vect<int, dim> &I) const {
    return delta(I[0], I[1], I[2]);
  }
};

struct GF3D5index {
#ifdef CCTK_DEBUG
  GF3D5layout layout;
#endif
  int m_linear;
  GF3D5index() = delete;
  GF3D5index(const GF3D5index &) = default;
  GF3D5index(GF3D5index &&) = default;
  GF3D5index &operator=(const GF3D5index &) = default;
  GF3D5index &operator=(GF3D5index &&) = default;
  CCTK_DEVICE CCTK_HOST GF3D5index(const GF3D5layout &layout,
                                   const vect<int, dim> &I)
      :
#ifdef CCTK_DEBUG
        layout(layout),
#endif
        m_linear(layout.linear(I)) {
  }
  CCTK_DEVICE CCTK_HOST int linear() const { return m_linear; }
};

template <typename T> struct GF3D5 {
  typedef T value_type;
#ifdef CCTK_DEBUG
  GF3D5layout layout;
#endif
  T *restrict ptr;
  GF3D5() = delete;
  GF3D5(const GF3D5 &) = default;
  GF3D5(GF3D5 &&) = default;
  GF3D5 &operator=(const GF3D5 &) = default;
  GF3D5 &operator=(GF3D5 &&) = default;
  CCTK_DEVICE CCTK_HOST GF3D5(const GF3D5layout &layout, T *restrict ptr)
      :
#ifdef CCTK_DEBUG
        layout(layout),
#endif
        ptr(ptr) {
#ifdef CCTK_DEBUG
    assert(&(*this)(layout, layout.imin) == ptr);
#endif
  }
  CCTK_DEVICE CCTK_HOST constexpr T &restrict
  operator()(const GF3D5index &index) const {
#ifdef CCTK_DEBUG
    assert(index.layout == this->layout);
#endif
    return ptr[index.linear()];
  }
  CCTK_DEVICE CCTK_HOST constexpr T &restrict
  operator()(const GF3D5layout &layout, const vect<int, dim> &I) const {
    return (*this)(GF3D5index(layout, I));
  }
  CCTK_DEVICE CCTK_HOST constexpr T &restrict
  operator()(const GF3D5layout &layout, int i, int j, int k) const {
    return (*this)(GF3D5index(layout, vect<int, dim>{i, j, k}));
  }
  CCTK_DEVICE CCTK_HOST void store(const GF3D5index &index,
                                   const T &value) const {
    operator()(index) = value;
  }
  CCTK_DEVICE CCTK_HOST void store(const GF3D5layout &layout,
                                   const vect<int, dim> &I,
                                   const T &value) const {
    operator()(layout, I) = value;
  }
  CCTK_DEVICE CCTK_HOST Arith::simd<std::remove_cv_t<T> >
  operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
             const GF3D5index &index) const {
    return Arith::maskz_loadu(mask, &(*this)(index));
  }
  CCTK_DEVICE CCTK_HOST Arith::simd<std::remove_cv_t<T> >
  operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
             const GF3D5layout &layout, const vect<int, dim> &I) const {
    return (*this)(mask, GF3D5index(layout, I));
  }
#if 0
  struct simd_reference {
    using element_type = std::remove_cv_t<T>;
    using value_type = Arith::simd<element_type>;
    T *const ptr;
    const Arith::simdl<element_type> mask;
    template <typename U>
    CCTK_DEVICE CCTK_HOST simd_reference(T *const ptr,
                                         const Arith::simdl<U> &mask)
        : ptr(ptr), mask(mask) {}
    simd_reference() = delete;
    // simd_reference(const simd_reference &) = default;
    // simd_reference(simd_reference &&) = default;
    simd_reference(const simd_reference &) = delete;
    simd_reference(simd_reference &&) = delete;
    simd_reference &operator=(const simd_reference &) = delete;
    simd_reference &operator=(simd_reference &&) = delete;
    CCTK_DEVICE CCTK_HOST operator value_type() const {
      return Arith::maskz_loadu(mask, ptr);
    }
    // CCTK_DEVICE CCTK_HOST value_type operator()() const {
    //   return Arith::maskz_loadu(mask, ptr);
    // }
    template <typename U>
    CCTK_DEVICE CCTK_HOST simd_reference
    operator=(const Arith::simd<U> &value) {
      mask_storeu(mask, ptr, value);
      return *this;
    }
    template <typename U>
    CCTK_DEVICE CCTK_HOST simd_reference operator=(const U &value) {
      return *this = value_type(value);
    }
  };
  CCTK_DEVICE CCTK_HOST simd_reference
  operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
             const GF3D5index &index) const {
    return simd_reference(&(*this)(index), mask);
  }
  CCTK_DEVICE CCTK_HOST simd_reference
  operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
             const GF3D5layout &layout, const vect<int, dim> &I) const {
    return (*this)(mask, GF3D5index(layout, I));
  }
#endif
  CCTK_DEVICE CCTK_HOST Arith::simd<std::remove_cv_t<T> >
  operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
             const GF3D5index &index,
             const Arith::simd<std::remove_cv_t<T> > &other) const {
    return Arith::masko_loadu(mask, &(*this)(index), other);
  }
  template <typename U,
            std::enable_if_t<std::is_convertible_v<T, U> > * = nullptr>
  CCTK_DEVICE CCTK_HOST Arith::simd<std::remove_cv_t<T> >
  operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
             const GF3D5index &index, const U &other) const {
    return Arith::masko_loadu(mask, &(*this)(index), other);
  }
  CCTK_DEVICE CCTK_HOST Arith::simd<std::remove_cv_t<T> >
  operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
             const GF3D5layout &layout, const vect<int, dim> &I,
             const Arith::simd<std::remove_cv_t<T> > &other) const {
    return (*this)(mask, GF3D5index(layout, I), other);
  }
  template <typename U,
            std::enable_if_t<std::is_convertible_v<T, U> > * = nullptr>
  CCTK_DEVICE CCTK_HOST Arith::simd<std::remove_cv_t<T> >
  operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
             const GF3D5layout &layout, const vect<int, dim> &I,
             const U &other) const {
    return (*this)(mask, GF3D5index(layout, I), other);
  }
  CCTK_DEVICE CCTK_HOST void store(const Arith::simdl<T> &mask,
                                   const GF3D5index &index,
                                   const Arith::simd<T> &value) const {
    mask_storeu(mask, &(*this)(index), value);
  }
  CCTK_DEVICE CCTK_HOST void store(const Arith::simdl<T> &mask,
                                   const GF3D5layout &layout,
                                   const vect<int, dim> &I,
                                   const Arith::simd<T> &value) const {
    store(mask, GF3D5index(layout, I), value);
  }
};

template <typename T> struct is_GF3D5 : std::false_type {};
template <typename T> struct is_GF3D5<GF3D5<T> > : std::true_type {};
template <typename T> inline constexpr bool is_GF3D5_v = is_GF3D5<T>::value;

////////////////////////////////////////////////////////////////////////////////

template <typename T> struct GF3D5vector {
  static_assert((std::is_same_v<T, amrex::Real>));
  typedef T value_type;
  GF3D5layout layout;

  amrex::FArrayBox fab;

private:
  static amrex::FArrayBox make_fab(const GF3D5layout &layout, const int nvars) {
    const amrex::Box box(amrex::IntVect(0, 0, 0),
                         amrex::IntVect(layout.np - 1, 0, 0));
    return amrex::FArrayBox(box, nvars, amrex::The_Async_Arena());
  }

public:
  GF3D5vector(const GF3D5layout &layout, const int nvars)
      : layout(layout), fab(make_fab(layout, nvars)) {}
  size_t size() const { return fab.nComp(); }
  GF3D5<T> operator()(const int n) const {
    assert(n >= 0 && n < int(size()));
    return GF3D5<T>(layout, (T *)fab.dataPtr(n));
  }
};

} // namespace Loop

// Macros for declaring variables using DECLARE_CCTK_ARGUMENTSX_func_name
#define CCTK_CENTERING_GRID const Loop::GridDescBase cctk_grid(cctkGH)
#define grid cctk_grid

#ifdef CARPETX_GF3D5

#define CCTK_CENTERING_LAYOUT(L, V)                                            \
  constexpr Arith::vect<int, Loop::dim> L##_centered V;                        \
  const Loop::GF3D5layout cctk_layout_##L(cctkGH, L##_centered)
#define CCTK_CENTERING_GF(C, L, N)                                             \
  const Loop::GF3D5<C CCTK_REAL> N(cctk_layout_##L, cctk_ptr_##N)

#else

#define CCTK_CENTERING_LAYOUT(L, V)                                            \
  constexpr Arith::vect<int, Loop::dim> L##_centered V;                        \
  const Loop::GF3D2layout cctk_layout_##L(cctkGH, L##_centered)
#define CCTK_CENTERING_GF(C, L, N)                                             \
  const Loop::GF3D2<C CCTK_REAL> N(cctk_layout_##L, cctk_ptr_##N)

#endif

#endif // #ifndef CARPETX_LOOP_LOOP_HXX

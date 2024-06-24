#include "driver.hxx"
#include "interp.hxx"
#include "mpi_types.hxx"
#include "reduction.hxx"
#include "schedule.hxx"

#include <defs.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Functions.h>
#include <cctk_Parameters.h>
#include <util_ErrorCodes.h>
#include <util_Table.h>

#include <AMReX_AmrParticles.H>
#include <AMReX_Particles.H>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace CarpetX {
using Arith::pown;

namespace {

// Interpolate a grid function at one point, dimensionally recursive
template <typename T, int order, int centering> struct interpolator {
  // TODO: `centering` is not used; remove it
  // TODO: remove `dx` in favour of `x1`?

  static constexpr vect<bool, dim> indextype{(centering & 0b100) != 0,
                                             (centering & 0b010) != 0,
                                             (centering & 0b001) != 0};

  const GridDescBase &grid;
  const int vi;
  const amrex::Array4<const T> &vars;
  const vect<int, dim> &derivs;
  // Allow outer boundaries as interpolation sources
  const bool allow_boundaries;

  static constexpr T eps() {
    using std::pow;
    return pow(std::numeric_limits<T>::epsilon(), T(3) / 4);
  }

  // TODO: Check whether interpolated variables are valid

  // Base case: only access a grid point
  template <int dir>
  std::enable_if_t<(dir == -1), T>
  interpolate(const vect<int, dim> &i, const vect<CCTK_REAL, dim> &di) const {
    const amrex::IntVect j(i[0] + vars.begin.x, i[1] + vars.begin.y,
                           i[2] + vars.begin.z);
#ifdef CCTK_DEBUG
    assert(vars.contains(j[0], j[1], j[2]));
#endif
    const T val = vars(j, vi);
#ifdef CCTK_DEBUG
    using std::isfinite;
    if (!(isfinite(val))) {
      std::cerr << "!isfinite i=" << i << " di=" << di << " val=" << val
                << "\n";
      for (int c = -1; c <= +1; ++c)
        for (int b = -1; b <= +1; ++b)
          for (int a = -1; a <= +1; ++a)
            if (vars.contains(j[0] + a, j[1] + b, j[2] + c))
              // std::cerr << "  val[" << a << "," << b << "," << c
              //           << "]=" << vars(j[0] + a, j[1] + b, j[2] + c, vi) <<
              //           "\n";
              std::fprintf(stderr, "val[%d,%d,%d]=%.17g\n", j[0] + a, j[1] + b,
                           j[2] + c,
                           double(vars(j[0] + a, j[1] + b, j[2] + c, vi)));
    }
    assert(isfinite(val));
#endif
    return val;
  }

  // General case: interpolate in one direction, then recurse
  template <int dir>
  std::enable_if_t<(dir >= 0), T>
  interpolate(const vect<int, dim> &i, const vect<CCTK_REAL, dim> &di) const {
    static_assert(dir < dim);
    const auto DI = vect<int, dim>::unit(dir);

    // Ignore the centering for interpolation
    // switch ((centering >> (2 - dir)) & 1)

    const T x = di[dir] - order / T(2);
    // #ifdef CCTK_DEBUG
    //     using std::fabs;
    //     assert(fabs(x) <= T(0.5) + eps());
    // #endif

    switch (order) {
    case 0: {
      const T y0 = interpolate<dir - 1>(i, di);
      switch (derivs[dir]) {
      case 0:
        return y0;
      case 1:
        return 0;
      case 2:
        return 0;
      }
    }
    case 1: {
      const T y0 = interpolate<dir - 1>(i, di);
      const T y1 = interpolate<dir - 1>(i + DI, di);
      switch (derivs[dir]) {
      case 0:
        return (1 / T(2) - x) * y0 + (1 / T(2) + x) * y1;
      case 1:
        return (-y0 + y1) / grid.dx[dir];
      case 2:
        return 0;
      }
    }
    case 2: {
      const T y0 = interpolate<dir - 1>(i, di);
      const T y1 = interpolate<dir - 1>(i + DI, di);
      const T y2 = interpolate<dir - 1>(i + 2 * DI, di);
      switch (derivs[dir]) {
      case 0:
        return (-1 / T(2) * x + 1 / T(2) * pown(x, 2)) * y0 +
               (1 - pown(x, 2)) * y1 +
               (1 / T(2) * x + 1 / T(2) * pown(x, 2)) * y2;
      case 1:
        return ((-1 / T(2) + x) * y0 - 2 * x * y1 + (1 / T(2) + x) * y2) /
               grid.dx[dir];
      case 2:
        return (y0 - 2 * y1 + y2) / pown(grid.dx[dir], 2);
      }
    }
    case 3: {
      const T y0 = interpolate<dir - 1>(i, di);
      const T y1 = interpolate<dir - 1>(i + DI, di);
      const T y2 = interpolate<dir - 1>(i + 2 * DI, di);
      const T y3 = interpolate<dir - 1>(i + 3 * DI, di);
      switch (derivs[dir]) {
      case 0:
        return (-1 / T(16) + 1 / T(24) * x + 1 / T(4) * pown(x, 2) -
                1 / T(6) * pown(x, 3)) *
                   y0 +
               (9 / T(16) - 9 / T(8) * x - 1 / T(4) * pown(x, 2) +
                1 / T(2) * pown(x, 3)) *
                   y1 +
               (9 / T(16) + 9 / T(8) * x - 1 / T(4) * pown(x, 2) -
                1 / T(2) * pown(x, 3)) *
                   y2 +
               (-1 / T(16) - 1 / T(24) * x + 1 / T(4) * pown(x, 2) +
                1 / T(6) * pown(x, 3)) *
                   y3;
      case 1:
        return ((1 / T(24) + 1 / T(2) * x - 1 / T(2) * pown(x, 2)) * y0 +
                (-9 / T(8) - 1 / T(2) * x + 3 / T(2) * pown(x, 2)) * y1 +
                (9 / T(8) - 1 / T(2) * x - 3 / T(2) * pown(x, 2)) * y2 +
                (-1 / T(24) + 1 / T(2) * x + 1 / T(2) * pown(x, 2)) * y3) /
               grid.dx[dir];
      case 2:
        return ((1 / T(2) - x) * y0 + (-1 / T(2) + 3 * x) * y1 +
                (-1 / T(2) - 3 * x) * y2 + (1 / T(2) + x) * y3) /
               pown(grid.dx[dir], 2);
      }
    }
    case 4: {
      const T y0 = interpolate<dir - 1>(i, di);
      const T y1 = interpolate<dir - 1>(i + DI, di);
      const T y2 = interpolate<dir - 1>(i + 2 * DI, di);
      const T y3 = interpolate<dir - 1>(i + 3 * DI, di);
      const T y4 = interpolate<dir - 1>(i + 4 * DI, di);
      switch (derivs[dir]) {
      case 0:
        return (1 / T(12) * x - 1 / T(24) * pown(x, 2) -
                1 / T(12) * pown(x, 3) + 1 / T(24) * pown(x, 4)) *
                   y0 +
               (-2 / T(3) * x + 2 / T(3) * pown(x, 2) + 1 / T(6) * pown(x, 3) -
                1 / T(6) * pown(x, 4)) *
                   y1 +
               (1 - 5 / T(4) * pown(x, 2) + 1 / T(4) * pown(x, 4)) * y2 +
               (2 / T(3) * x + 2 / T(3) * pown(x, 2) - 1 / T(6) * pown(x, 3) -
                1 / T(6) * pown(x, 4)) *
                   y3 +
               (-1 / T(12) * x - 1 / T(24) * pown(x, 2) +
                1 / T(12) * pown(x, 3) + 1 / T(24) * pown(x, 4)) *
                   y4;
      case 1:
        return ((1 / T(12) - 1 / T(12) * x - 1 / T(4) * pown(x, 2) +
                 1 / T(6) * pown(x, 3)) *
                    y0 +
                (-2 / T(3) + 4 / T(3) * x + 1 / T(2) * pown(x, 2) -
                 2 / T(3) * pown(x, 3)) *
                    y1 +
                (-5 / T(2) * x + pown(x, 3)) * y2 +
                (2 / T(3) + 4 / T(3) * x - 1 / T(2) * pown(x, 2) -
                 2 / T(3) * pown(x, 3)) *
                    y3 +
                (-1 / T(12) - 1 / T(12) * x + 1 / T(4) * pown(x, 2) +
                 1 / T(6) * pown(x, 3)) *
                    y4) /
               grid.dx[dir];
      case 2:
        return ((-1 / T(12) - 1 / T(2) * x + 1 / T(2) * pown(x, 2)) * y0 +
                (4 / T(3) + x - 2 * pown(x, 2)) * y1 +
                (-5 / T(2) + 3 * pown(x, 2)) * y2 +
                (4 / T(3) - x - 2 * pown(x, 2)) * y3 +
                (-1 / T(12) + 1 / T(2) * x + 1 / T(2) * pown(x, 2)) * y4) /
               pown(grid.dx[dir], 2);
      }
    }
    default:
      assert(0);
    } // switch order
  }

  template <typename Particles>
  void interpolate3d(const Particles &particles,
                     std::vector<T> &varresult) const {
    const auto x0 = grid.x0 + (2 * grid.lbnd - !indextype) * grid.dx / 2;
    // const auto x1 = x0 + (grid.lsh - 1 - indextype) * grid.dx;
    const auto dx = grid.dx;

    assert(vars.end.x - vars.begin.x == grid.lsh[0]);
    assert(vars.end.y - vars.begin.y == grid.lsh[1]);
    assert(vars.end.z - vars.begin.z == grid.lsh[2]);

    // We assume that the input is synchronized, i.e. that all ghost
    // zones are valid, but all outer boundaries are invalid.
    // TODO: Take multipatch boundary directions into account; forbid
    // only interpatch boundaries but allow true outer boundaries.
    vect<vect<bool, dim>, 2> allowed_boundaries;
    for (int f = 0; f < 2; ++f)
      for (int d = 0; d < dim; ++d)
        allowed_boundaries[f][d] = allow_boundaries ? true : !grid.bbox[f][d];

    // The point must lie inside the domain. At outer boundaries the
    // point may be in the boundary region, but at ghost boundaries
    // the point cannot be in the ghost region. We define as "ghost"
    // region here the interpolation stencil size which is `order /
    // 2`.
    // const auto x0_allowed =
    //     x0 +
    //     (!allowed_boundaries[0] * grid.nghostzones + (order - 1) / T(2)) * dx
    //     - eps();
    // const auto x1_allowed =
    //     x1 -
    //     (!allowed_boundaries[1] * grid.nghostzones + (order - 1) / T(2)) * dx
    //     + eps();

    // The allowed index range is [i0, i1)
    const auto i0_allowed = !allowed_boundaries[0] * grid.nghostzones;
    const auto i1_allowed =
        grid.lsh - (!allowed_boundaries[1] * grid.nghostzones + order);

    const int np = int(varresult.size());

#pragma omp simd
    for (int n = 0; n < np; ++n) {
      const vect<T, dim> x{particles[n].rdata(0), particles[n].rdata(1),
                           particles[n].rdata(2)};
      // // Ensure the point is inside the domain
      // assert(all(x >= x0_allowed && x <= x1_allowed));

      // Find stencil anchor (i.e. the leftmost stencil point)
      const auto qi = (x - x0) / dx;
      const auto lrint1 = [](auto a) {
        using std::lrint;
        return int(lrint(a));
      };
      auto i = fmap(lrint1, qi - order / T(2));
      auto di = qi - i;
      // Consistency check
      assert(all(i >= 0 && i + order < grid.lsh));

      // Push point away from boundaries if they are just a little outside
      for (int d = 0; d < dim; ++d) {
        if (i[d] + order / 2 < i0_allowed[d] &&
            di[d] - order / T(2) >= +T(0.5) - eps()) {
          i[d] += 1;
          di[d] -= 1;
        }
        if (i[d] >= i1_allowed[d] && di[d] - order / T(2) <= -T(0.5) + eps()) {
          i[d] -= 1;
          di[d] += 1;
        }
      }

      // Avoid points on boundaries
      const bool is_allowed = all(i >= i0_allowed && i < i1_allowed);
      assert(is_allowed);

      const T res = !is_allowed ? -2 : interpolate<dim - 1>(i, di);

      varresult[n] = res;
    }
  }
};

} // namespace

int InterpLocalUniform(int /*N_dims*/, int /*param_table_handle*/,
                       /***** coordinate system *****/
                       const CCTK_REAL /*coord_origin*/[],
                       const CCTK_REAL /*coord_delta*/[],
                       /***** interpolation points *****/
                       int /*N_interp_points*/, int /*interp_coords_type_code*/,
                       const void *const /*interp_coords*/[],
                       /***** input arrays *****/
                       int /*N_input_arrays*/,
                       const CCTK_INT /*input_array_dims*/[],
                       const CCTK_INT /*input_array_type_codes*/[],
                       const void *const /*input_arrays*/[],
                       /***** output arrays *****/
                       int /*N_output_arrays*/,
                       const CCTK_INT /*output_array_type_codes*/[],
                       void *const /*output_arrays*/[]) {
  CCTK_ERROR("Dummy InterpLocalUniform function called");
}

extern "C" CCTK_INT CarpetX_InterpGridArrays(
    cGH const *const cctkGH, int const N_dims, int const local_interp_handle,
    int const param_table_handle, int const coord_system_handle,
    int const N_interp_points, int const interp_coords_type_code,
    void const *const coords[], int const N_input_arrays,
    CCTK_INT const input_array_variable_indices[], int const N_output_arrays,
    CCTK_INT const output_array_type_codes[], void *const output_arrays[]) {
  /* TODO: verify that the interface with SymmetryInterpolate can be simply
     copied from Carpet like below */
  //  if (CCTK_IsFunctionAliased("SymmetryInterpolate")) {
  //    return SymmetryInterpolate(
  //        cctkGH, N_dims, local_interp_handle, param_table_handle,
  //        coord_system_handle, N_interp_points, interp_coords_type_code,
  //        coords, N_input_arrays, input_array_variable_indices,
  //        N_output_arrays, output_array_type_codes, output_arrays);
  //  } else {
  return CarpetX_DriverInterpolate(
      cctkGH, N_dims, local_interp_handle, param_table_handle,
      coord_system_handle, N_interp_points, interp_coords_type_code, coords,
      N_input_arrays, input_array_variable_indices, N_output_arrays,
      output_array_type_codes, output_arrays);
  //  }
}

extern "C" CCTK_INT CarpetX_DriverInterpolate(
    CCTK_POINTER_TO_CONST const cctkGH, CCTK_INT const N_dims,
    CCTK_INT const local_interp_handle, CCTK_INT const param_table_handle,
    CCTK_INT const coord_system_handle, CCTK_INT const N_interp_points,
    CCTK_INT const interp_coords_type_code,
    CCTK_POINTER_TO_CONST const coords[], CCTK_INT const N_input_arrays,
    CCTK_INT const input_array_variable_indices[],
    CCTK_INT const N_output_arrays, CCTK_INT const output_array_type_codes[],
    CCTK_POINTER const output_arrays[]) {
  DECLARE_CCTK_PARAMETERS;

  // We do not support local interpolators yet
  const int carpetx_interp_handle = CCTK_InterpHandle("CarpetX");
  assert(carpetx_interp_handle >= 0);
  if (carpetx_interp_handle != local_interp_handle) {
    CCTK_VERROR("Incorrect local interpolator handle provided, only 'CarpetX' "
                "is allowed: %d != %d",
                local_interp_handle, carpetx_interp_handle);
  }

  // This verifies that the order in param_table_handle matches the order of the
  // runtime parameter from CarpetX
  CCTK_INT order;
  int n_elems = Util_TableGetInt(param_table_handle, &order, "order");
  assert(n_elems == 1);
  assert(order == interpolation_order);

  std::vector<CCTK_INT> varinds;
  varinds.resize(N_output_arrays);
  n_elems = Util_TableGetIntArray(param_table_handle, N_output_arrays,
                                  varinds.data(), "operand_indices");
  if (n_elems == UTIL_ERROR_TABLE_NO_SUCH_KEY) {
    assert(N_input_arrays == N_output_arrays);
    for (int i = 0; i < N_input_arrays; i++) {
      varinds.at(i) = input_array_variable_indices[i];
    }
  } else if (n_elems == N_output_arrays) {
    for (int i = 0; i < n_elems; i++) {
      varinds.at(i) = input_array_variable_indices[varinds.at(i)];
    }
  } else {
    CCTK_VERROR("TableGetIntArray failed with error code %d", n_elems);
  }

  std::vector<CCTK_INT> operations;
  operations.resize(N_output_arrays, 0);
  n_elems = Util_TableGetIntArray(param_table_handle, N_output_arrays,
                                  operations.data(), "operation_codes");
  if (n_elems == UTIL_ERROR_TABLE_NO_SUCH_KEY) {
    assert(N_input_arrays == N_output_arrays);
  } else if (n_elems != N_output_arrays) {
    CCTK_ERROR("TableGetIntArray failed.");
  }

  const CCTK_POINTER resultptrs = (CCTK_POINTER)output_arrays;
  const bool allow_boundaries = true;
  CarpetX_Interpolate(
      cctkGH, N_interp_points, static_cast<const CCTK_REAL *>(coords[0]),
      static_cast<const CCTK_REAL *>(coords[1]),
      static_cast<const CCTK_REAL *>(coords[2]), N_output_arrays,
      varinds.data(), operations.data(), allow_boundaries, resultptrs);

  return 0;
}

extern "C" void CarpetX_Interpolate(const CCTK_POINTER_TO_CONST cctkGH_,
                                    const CCTK_INT npoints,
                                    const CCTK_REAL *restrict const globalsx,
                                    const CCTK_REAL *restrict const globalsy,
                                    const CCTK_REAL *restrict const globalsz,
                                    const CCTK_INT nvars,
                                    const CCTK_INT *restrict const varinds,
                                    const CCTK_INT *restrict const operations,
                                    const CCTK_INT allow_boundaries,
                                    const CCTK_POINTER resultptrs_) {
  DECLARE_CCTK_PARAMETERS;
  const cGH *restrict const cctkGH = static_cast<const cGH *>(cctkGH_);
  assert(in_global_mode(cctkGH));

  static const bool have_MultiPatch_GlobalToLocal2 =
      CCTK_IsFunctionAliased("MultiPatch_GlobalToLocal2");

  // Convert global to patch-local coordinates
  // TODO: Call this only if there is a non-trivial patch system
  // TODO: Don't call this for multipatch interpolation, precalculate this
  // instead
  std::vector<CCTK_INT> patches(npoints);
  std::vector<CCTK_REAL> localsx(npoints);
  std::vector<CCTK_REAL> localsy(npoints);
  std::vector<CCTK_REAL> localsz(npoints);
  if (have_MultiPatch_GlobalToLocal2) {
    MultiPatch_GlobalToLocal2(npoints, globalsx, globalsy, globalsz,
                              patches.data(), localsx.data(), localsy.data(),
                              localsz.data());
  } else {
    // TODO: Don't copy
    for (int n = 0; n < npoints; ++n) {
      patches[n] = 0;
      localsx[n] = globalsx[n];
      localsy[n] = globalsy[n];
      localsz[n] = globalsz[n];
    }
  }

  // Apply symmetries to coordinates
  std::vector<bool> symmetry_reflected_z;
  assert(!reflection_x);
  assert(!reflection_y);
  assert(!reflection_upper_x);
  assert(!reflection_upper_y);
  assert(!reflection_upper_z);
  if (reflection_z) {
    symmetry_reflected_z.resize(npoints);
    assert(ghext->num_patches() == 1);
    constexpr int patch = 0;
    const amrex::Geometry &geom = ghext->patchdata.at(patch).amrcore->Geom(0);
    const CCTK_REAL *restrict const xmin = geom.ProbLo();
    for (int n = 0; n < npoints; ++n) {
      const bool refl = localsz[n] < xmin[2];
      symmetry_reflected_z[n] = refl;
      if (refl)
        localsz[n] = 2 * xmin[2] - localsz[n];
    }
  }

  // Project particles into the domain for AMReX's distribution
  // AMReX silently drops particles that are outside the domain. We
  // can't have this. We thus push them back into the domain. Of
  // course, these modified coordinates are not useful for
  // interpolating, so we have both the `local` (true) and the `pos`
  // (AMReX) coordinates.
  std::vector<CCTK_REAL> posx(npoints);
  std::vector<CCTK_REAL> posy(npoints);
  std::vector<CCTK_REAL> posz(npoints);
  for (int n = 0; n < npoints; ++n) {
    const int patch = patches[n];
    const amrex::Geometry &geom = ghext->patchdata.at(patch).amrcore->Geom(0);
    const CCTK_REAL *restrict const xmin = geom.ProbLo();
    const CCTK_REAL *restrict const xmax = geom.ProbHi();
    const CCTK_REAL *restrict const dx = geom.CellSize();
    using std::clamp;
    // Push the particle at least 1/2 grid spacing into the domain
    // TODO: push by less because 1/2 is too much if there are many AMR levels
    posx[n] = clamp(localsx[n], xmin[0] + dx[0] / 2, xmax[0] - dx[0] / 2);
    posy[n] = clamp(localsy[n], xmin[1] + dx[1] / 2, xmax[1] - dx[1] / 2);
    posz[n] = clamp(localsz[n], xmin[2] + dx[2] / 2, xmax[2] - dx[2] / 2);
  }

  // Create particle containers
  using Container = amrex::AmrParticleContainer<3, 2>;
  using ParticleTile = Container::ParticleTileType;
  std::vector<Container> containers(ghext->num_patches());
  for (int patch = 0; patch < ghext->num_patches(); ++patch) {
    const auto &restrict patchdata = ghext->patchdata.at(patch);
    containers.at(patch) = Container(patchdata.amrcore.get());
    const int level = 0;
    const auto &restrict leveldata = patchdata.leveldata.at(level);
    const amrex::MFIter mfi(*leveldata.fab);
    assert(mfi.isValid());
    ParticleTile *const particle_tile = &containers.at(patch).GetParticles(
        level)[make_pair(mfi.index(), mfi.LocalTileIndex())];

    // Set particle positions
    const int proc = amrex::ParallelDescriptor::MyProc();
    for (int n = 0; n < npoints; ++n) {
      // TODO: Loop over points only once
      if (patches.at(n) == patch) {
        amrex::Particle<3, 2> p;
        p.id() = Container::ParticleType::NextID();
        p.cpu() = proc;
        p.pos(0) = posx[n]; // AMReX distribution position
        p.pos(1) = posy[n];
        p.pos(2) = posz[n];
        p.rdata(0) = localsx[n]; // actual particle coordinate
        p.rdata(1) = localsy[n];
        p.rdata(2) = localsz[n];
        p.idata(0) = proc; // source process
        p.idata(1) = n;    // source index
        particle_tile->push_back(p);
      }
    }
  }

  // Send particles to interpolation points
  for (auto &container : containers) {
#ifdef CCTK_DEBUG
    const int patch = int(&container - containers.data());
    std::size_t old_nparticles = 0;
    std::set<int> oldids;
    {
      const auto &levels = container.GetParticles();
      for (const auto &level : levels) {
        const int lev = int(&level - levels.data());
        for (amrex::ParConstIter<3, 2> pti(container, lev); pti.isValid();
             ++pti) {
          const auto &particles = pti.GetArrayOfStructs();
          const int component = MFPointer(pti).index();
          CCTK_VINFO("patch %d level %d component %d old_nparticles: %zu",
                     patch, lev, component, particles.size());
          for (const auto &particle : particles) {
            oldids.insert(particle.id());
            // CCTK_VINFO("    id=%d proc=%d pos=[%g,%g,%g]  locals=[%g,%g,%g] "
            //            "proc=%d n=%d",
            //            int(particle.id()), int(particle.cpu()),
            //            particle.pos(0), particle.pos(1), particle.pos(2),
            //            particle.rdata(0), particle.rdata(1),
            //            particle.rdata(2), particle.idata(0),
            //            particle.idata(1));
          }
          old_nparticles += particles.size();
        }
      }
      const MPI_Comm comm = amrex::ParallelDescriptor::Communicator();
      MPI_Allreduce(MPI_IN_PLACE, &old_nparticles, 1,
                    mpi_datatype<std::size_t>::value, MPI_SUM, comm);
    }
#endif

    container.Redistribute();

#ifdef CCTK_DEBUG
    std::size_t new_nparticles = 0;
    std::set<int> newids;
    {
      const auto &levels = container.GetParticles();
      for (const auto &level : levels) {
        const int lev = int(&level - levels.data());
        for (amrex::ParConstIter<3, 2> pti(container, lev); pti.isValid();
             ++pti) {
          const int component = MFPointer(pti).index();
          const auto &particles = pti.GetArrayOfStructs();
          CCTK_VINFO("patch %d level %d component %d new_nparticles: %zu",
                     patch, lev, component, particles.size());
          for (const auto &particle : particles) {
            newids.insert(particle.id());
            // CCTK_VINFO("    id=%d proc=%d pos=[%g,%g,%g]  locals=[%g,%g,%g] "
            //            "proc=%d n=%d",
            //            int(particle.id()), int(particle.cpu()),
            //            particle.pos(0), particle.pos(1), particle.pos(2),
            //            particle.rdata(0), particle.rdata(1),
            //            particle.rdata(2), particle.idata(0),
            //            particle.idata(1));
          }
          new_nparticles += particles.size();
        }
      }
      const MPI_Comm comm = amrex::ParallelDescriptor::Communicator();
      MPI_Allreduce(MPI_IN_PLACE, &new_nparticles, 1,
                    mpi_datatype<std::size_t>::value, MPI_SUM, comm);
    }
    if (new_nparticles != old_nparticles) {
      for (const auto oldid : oldids)
        if (!newids.count(oldid))
          CCTK_VINFO("old id %d not present in new ids", oldid);
      for (const auto newid : newids)
        if (!oldids.count(newid))
          CCTK_VINFO("new id %d not present in old ids", newid);
      CCTK_VERROR(
          "We lost interpolation points on patch %d. Before redistributing: "
          "%zu particles, after redistributing: %zu particles",
          patch, old_nparticles, new_nparticles);
    }
#endif
  }

  // Define result variables
  const int nprocs = amrex::ParallelDescriptor::NProcs();
  std::vector<std::vector<CCTK_REAL> > results(nprocs); // [nprocs]

  // Interpolate
  constexpr int tl = 0;
  struct givi_t {
    int gi, vi;
  };
  std::vector<givi_t> givis(nvars);
  for (int v = 0; v < nvars; ++v) {
    int gi = CCTK_GroupIndexFromVarI(varinds[v]);
    assert(gi >= 0);
    assert(gi < CCTK_NumGroups());
    int vi = varinds[v] - CCTK_FirstVarIndexI(gi);
    assert(vi >= 0);
    assert(vi < CCTK_NumVarsInGroupI(gi));
    givis.at(v) = {gi, vi};
  }

  // CCTK_VINFO("interpolating");
  for (const auto &patchdata : ghext->patchdata) {
    const int patch = patchdata.patch;
    for (const auto &leveldata : patchdata.leveldata) {
      const int level = leveldata.level;
      // TODO: use OpenMP
      for (amrex::ParIter<3, 2> pti(containers.at(patch), level); pti.isValid();
           ++pti) {
        const MFPointer mfp(pti);
        const GridDesc grid(leveldata, mfp);
        // const int component = mfp.index();

        const int np = pti.numParticles();
        const auto &particles = pti.GetArrayOfStructs();
        // CCTK_VINFO("patch=%d level=%d component=%d npoints=%d", patch, level,
        //            component, np);

        std::vector<std::vector<CCTK_REAL> > varresults(nvars);

        // TODO: Don't re-calculate interpolation coefficients for each
        // variable
        for (int v = 0; v < nvars; ++v) {
          const int gi = givis.at(v).gi;
          const int vi = givis.at(v).vi;
          const auto &restrict groupdata = *leveldata.groupdata.at(gi);
          const int centering = groupdata.indextype[0] * 0b100 +
                                groupdata.indextype[1] * 0b010 +
                                groupdata.indextype[2] * 0b001;
          assert(all(groupdata.nghostzones == grid.nghostzones));
          const amrex::Array4<const CCTK_REAL> &vars =
              groupdata.mfab.at(tl)->array(pti);
          vect<int, dim> derivs;
          int op = operations[v];
          while (op > 0) {
            const int dir = op % 10 - 1;
            if (dir >= 0) {
              assert(dir >= 0 && dir < dim);
              ++derivs[dir];
            }
            op /= 10;
          }
          auto &varresult = varresults.at(v);
          varresult.resize(np);

          switch (centering) {
          case 0b000: {
            // Vertex centering

            switch (interpolation_order) {
            case 0: {
              const interpolator<CCTK_REAL, 0, 0b000> interp{
                  grid, vi, vars, derivs, bool(allow_boundaries)};
              interp.interpolate3d(particles, varresult);
              break;
            }
            case 1: {
              const interpolator<CCTK_REAL, 1, 0b000> interp{
                  grid, vi, vars, derivs, bool(allow_boundaries)};
              interp.interpolate3d(particles, varresult);
              break;
            }
            case 2: {
              const interpolator<CCTK_REAL, 2, 0b000> interp{
                  grid, vi, vars, derivs, bool(allow_boundaries)};
              interp.interpolate3d(particles, varresult);
              break;
            }
            case 3: {
              const interpolator<CCTK_REAL, 3, 0b000> interp{
                  grid, vi, vars, derivs, bool(allow_boundaries)};
              interp.interpolate3d(particles, varresult);
              break;
            }
            case 4: {
              const interpolator<CCTK_REAL, 4, 0b000> interp{
                  grid, vi, vars, derivs, bool(allow_boundaries)};
              interp.interpolate3d(particles, varresult);
              break;
            }
            default:
              CCTK_VERROR("Interpolation order %d for centering [%d,%d,%d] not "
                          "yet supported",
                          int(interpolation_order), groupdata.indextype[0],
                          groupdata.indextype[1], groupdata.indextype[2]);
            } // switch interpolation_order
            break;
          } // case 0b000

          case 0b111: {
            // Cell centering

            switch (interpolation_order) {
            case 0: {
              const interpolator<CCTK_REAL, 0, 0b111> interp{
                  grid, vi, vars, derivs, bool(allow_boundaries)};
              interp.interpolate3d(particles, varresult);
              break;
            }
            case 1: {
              const interpolator<CCTK_REAL, 1, 0b111> interp{
                  grid, vi, vars, derivs, bool(allow_boundaries)};
              interp.interpolate3d(particles, varresult);
              break;
            }
            case 2: {
              const interpolator<CCTK_REAL, 2, 0b111> interp{
                  grid, vi, vars, derivs, bool(allow_boundaries)};
              interp.interpolate3d(particles, varresult);
              break;
            }
            case 3: {
              const interpolator<CCTK_REAL, 3, 0b111> interp{
                  grid, vi, vars, derivs, bool(allow_boundaries)};
              interp.interpolate3d(particles, varresult);
              break;
            }
            case 4: {
              const interpolator<CCTK_REAL, 4, 0b111> interp{
                  grid, vi, vars, derivs, bool(allow_boundaries)};
              interp.interpolate3d(particles, varresult);
              break;
            }
            default:
              CCTK_VERROR("Interpolation order %d for centering [%d,%d,%d] not "
                          "yet supported",
                          int(interpolation_order), groupdata.indextype[0],
                          groupdata.indextype[1], groupdata.indextype[2]);
            } // switch interpolation_order
            break;
          } // case 0b111

          default:
            CCTK_VERROR("Centering [%d,%d,%d] not yet supported",
                        groupdata.indextype[0], groupdata.indextype[1],
                        groupdata.indextype[2]);
          } // switch centering

        } // for var

        for (int n = 0; n < np; ++n) {
          const int proc = particles[n].idata(0);
          const int id = particles[n].idata(1);
          auto &result = results.at(proc);
          result.push_back(id);
          for (int v = 0; v < nvars; ++v)
            result.push_back(varresults.at(v).at(n));
        }
      }
    }
  }

  // CCTK_VINFO("interpolation results");
  // for (const auto &proc_result : results) {
  //   const int p = proc_result.first;
  //   const auto &result = proc_result.second;
  //   CCTK_VINFO("[%d] count=%zu", p, result.size());
  // }

  // Collect particles back
  // CCTK_VINFO("collecting results");
  const MPI_Comm comm = amrex::ParallelDescriptor::Communicator();
  const MPI_Datatype datatype = mpi_datatype<CCTK_REAL>::value;

  // int total_npoints;
  // MPI_Allreduce(&npoints, &total_npoints, 1, MPI_INT, MPI_SUM, comm);

  std::vector<int> sendcounts(nprocs);
  std::vector<int> senddispls(nprocs);
  int total_sendcount = 0;
  for (int p = 0; p < nprocs; ++p) {
    const auto &result = results.at(p);
    sendcounts.at(p) = result.size();
    senddispls.at(p) = total_sendcount;
    total_sendcount += sendcounts.at(p);
  }
  std::vector<int> recvcounts(nprocs);
  MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT,
               comm);
  std::vector<int> recvdispls(nprocs);
  int total_recvcount = 0;
  for (int p = 0; p < nprocs; ++p) {
    recvdispls.at(p) = total_recvcount;
    total_recvcount += recvcounts.at(p);
  }

  std::vector<CCTK_REAL> sendbuf(total_sendcount);
  for (int p = 0; p < nprocs; ++p) {
    // TODO: Don't copy, store data here right away
    assert(p >= 0);
    assert(p < int(results.size()));
    const auto &result = results.at(p);
    assert(sendcounts.at(p) == int(result.size()));
    assert(p >= 0);
    assert(p < int(senddispls.size()));
    assert(senddispls.at(p) >= 0);
    assert(senddispls.at(p) + sendcounts.at(p) <= int(sendbuf.size()));
    std::copy(result.begin(), result.end(), sendbuf.data() + senddispls.at(p));
  }
  std::vector<CCTK_REAL> recvbuf(total_recvcount);
  MPI_Alltoallv(sendbuf.data(), sendcounts.data(), senddispls.data(), datatype,
                recvbuf.data(), recvcounts.data(), recvdispls.data(), datatype,
                comm);
#ifdef CCTK_DEBUG
  // Check consistency of received ids
  std::vector<bool> idxs(npoints, false);
  for (int n = 0; n < npoints; ++n) {
    const int offset = (nvars + 1) * n;
    const int idx = int(recvbuf.at(offset));
    assert(!idxs.at(idx));
    idxs.at(idx) = true;
  }
  for (int n = 0; n < npoints; ++n)
    assert(idxs.at(n));
#endif

  // Set result
  CCTK_REAL *const restrict *const restrict resultptrs =
      static_cast<CCTK_REAL *const *>(resultptrs_);
  for (int n = 0; n < npoints; ++n) {
    const int offset = (nvars + 1) * n;
    const int idx = int(recvbuf.at(offset));
    for (int v = 0; v < nvars; ++v)
      resultptrs[v][idx] = recvbuf.at(offset + 1 + v);
  }

  // Apply symmetries to interpolated values
  assert(!reflection_x);
  assert(!reflection_y);
  assert(!reflection_upper_x);
  assert(!reflection_upper_y);
  assert(!reflection_upper_z);
  if (reflection_z) {
    // The code below is only valid for Psi4
    assert(nvars == 2);
    assert(varinds[0] == CCTK_VarIndex("Weyl::Psi4re"));
    assert(varinds[1] == CCTK_VarIndex("Weyl::Psi4im"));
    // l^a = et^a + er^a
    // n^a = et^a - er^a
    // m^a = etheta^a + i ephi^a
    // Psi4 = C_abcd m-bar^b n^b m-bar^c n^d
    for (int n = 0; n < npoints; ++n) {
      if (symmetry_reflected_z[n]) {
        resultptrs[0][n] = -resultptrs[0][n];
        resultptrs[1][n] = +resultptrs[1][n];
      }
    }
  }
}
} // namespace CarpetX

#include "driver.hxx"
#include "fillpatch.hxx"
#include "io.hxx"
#include "loop.hxx"
#include "schedule.hxx"
#include "task_manager.hxx"
#include "timer.hxx"
#include "valid.hxx"

#include <cctk.h>
#include <cctk_Parameters.h>
#include <cctk_Schedule.h>
#include <cctki_GHExtensions.h>
#include <cctki_ScheduleBindings.h>
#include <cctki_WarnLevel.h>

#include <AMReX_MultiFabUtil.H>

#if defined _OPENMP
#include <omp.h>
#elif defined __HIPCC__
#define omp_get_max_threads() 1
#define omp_get_num_threads() 1
#define omp_get_thread_num() 0
#define omp_in_parallel() 0
#else
static inline int omp_get_max_threads() { return 1; }
static inline int omp_get_num_threads() { return 1; }
static inline int omp_get_thread_num() { return 0; }
static inline int omp_in_parallel() { return 0; }
#endif

#include <sys/time.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace CarpetX {
using namespace std;

#ifndef CCTK_HAVE_CGH_LEVEL
#error                                                                         \
    "The Cactus flesh does not support cctk_level in the cGH structure. Update the flesh."
#endif
#ifndef CCTK_HAVE_CGH_PATCH
#error                                                                         \
    "The Cactus flesh does not support cctk_patch in the cGH structure. Update the flesh."
#endif
#ifndef CCTK_HAVE_CGH_COMPONENT
#error                                                                         \
    "The Cactus flesh does not support cctk_component in the cGH structure. Update the flesh."
#endif
#ifndef CCTK_HAVE_CGH_TILE
#error                                                                         \
    "The Cactus flesh does not support cctk_tile_min etc. in the cGH structure. Update the flesh."
#endif

#if defined _OPENMP
#if !defined AMREX_USE_OMP
#error                                                                         \
    "Cactus is configured with OpenMP, but AMReX is configured without OpenMP. This does not work."
#endif
#endif

namespace {
double gettime() {
  timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + tv.tv_usec / 1.0e+6;
}
} // namespace

// Used to pass active levels from AMReX's regridding functions
optional<active_levels_t> active_levels;

void Reflux(const cGH *cctkGH, int level);
void Restrict(const cGH *cctkGH, int level, const vector<int> &groups);
void Restrict(const cGH *cctkGH, int level);
void SyncAfterRestrict(const cGH *cctkGH, int level);

namespace {
// Convert a (direction, face) pair to an AMReX Orientation
amrex::Orientation orient(int d, int f) {
  return amrex::Orientation(d, amrex::Orientation::Side(f));
}
int GroupStorageCrease(const cGH *cctkGH, int n_groups, const int *groups,
                       const int *requested_tls, int *status, const bool inc);
} // namespace

////////////////////////////////////////////////////////////////////////////////

GridDesc::GridDesc(const GHExt::PatchData::LevelData &leveldata,
                   const MFPointer &mfp) {
  DECLARE_CCTK_PARAMETERS;

  // The number of ghostzones in each direction
  // for (int d = 0; d < dim; ++d)
  //   nghostzones[d] = mfp.nGrowVect()[d];
  nghostzones = {ghost_size >= 0 ? ghost_size : ghost_size_x,
                 ghost_size >= 0 ? ghost_size : ghost_size_y,
                 ghost_size >= 0 ? ghost_size : ghost_size_z};

  const auto &patchdata = ghext->patchdata.at(leveldata.patch);
  const amrex::IntVect ng(nghostzones[0], nghostzones[1], nghostzones[2]);
  const amrex::Box &domain = patchdata.amrcore->Geom(leveldata.level).Domain();
  const amrex::Box &vbx = mfp.validbox(); // interior region (without ghosts)
  const amrex::Box &fbx = mfp.fabbox(ng); // allocated array
  const amrex::Box &gbx = mfp.growntilebox(ng); // current region (with ghosts)

  for (int d = 0; d < dim; ++d)
    assert(domain.type(d) == amrex::IndexType::CELL);

  // Level, patch, and component
  level = leveldata.level;
  patch = leveldata.patch;
  component = mfp.index();

  // Global shape
  for (int d = 0; d < dim; ++d)
    gsh[d] = domain[orient(d, 1)] + 1 - domain[orient(d, 0)] + 1 +
             2 * nghostzones[d];

  // Local shape
  for (int d = 0; d < dim; ++d)
    lsh[d] = fbx[orient(d, 1)] - fbx[orient(d, 0)] + 1 + 1;

  // Allocated shape
  for (int d = 0; d < dim; ++d)
    ash[d] = lsh[d];

  // Local extent
  for (int d = 0; d < dim; ++d) {
    lbnd[d] = fbx[orient(d, 0)] + nghostzones[d];
    ubnd[d] = fbx[orient(d, 1)] + 1 + nghostzones[d];
  }

  // Boundaries
  for (int d = 0; d < dim; ++d)
    for (int f = 0; f < 2; ++f)
      bbox[f][d] = vbx[orient(d, f)] == domain[orient(d, f)];

  // Thread tile box
  for (int d = 0; d < dim; ++d) {
    tmin[d] = gbx[orient(d, 0)] - fbx[orient(d, 0)];
    // For vertex centred grids, the allocated box is 1 vertex larger
    // than the number of cells, and AMReX assigns this extra vertex
    // to the final tile
    assert(gbx[orient(d, 1)] <= fbx[orient(d, 1)]);
    tmax[d] = gbx[orient(d, 1)] + 1 - fbx[orient(d, 0)] +
              (gbx[orient(d, 1)] == fbx[orient(d, 1)]);
  }

  const amrex::Geometry &geom = patchdata.amrcore->Geom(0);
  const CCTK_REAL *restrict const global_x0 = geom.ProbLo();
  const CCTK_REAL *restrict const global_dx = geom.CellSize();
  for (int d = 0; d < dim; ++d) {
    const int levfac = 1 << leveldata.level;
    // Offset between this level's and the coarsest level's origin as
    // multiple of the grid spacing
    const int levoff = (1 - levfac) * (1 - 2 * nghostzones[d]);
    const int levoffdenom = 2;
    // Cell-centred coordinates on coarse level
    const CCTK_REAL origin_space =
        global_x0[d] + (1 - 2 * nghostzones[d]) * global_dx[d] / 2;
    const CCTK_REAL delta_space = global_dx[d];
    // Cell-centred coordinates on current level
    dx[d] = delta_space / levfac;
    x0[d] = origin_space + dx[d] * levoff / levoffdenom;
  }

  // Check constraints
  for (int d = 0; d < dim; ++d) {
    // Domain size
    assert(gsh[d] >= 0);

    // Local size
    assert(lbnd[d] >= 0);
    assert(lsh[d] >= 0);
    assert(lbnd[d] + lsh[d] <= gsh[d]);
    assert(ubnd[d] == lbnd[d] + lsh[d] - 1);

    // Internal representation
    assert(ash[d] >= 0);
    assert(ash[d] >= lsh[d]);

    // Ghost zones
    assert(nghostzones[d] >= 0);
    assert(2 * nghostzones[d] <= lsh[d]);

    // Tiles
    assert(tmin[d] >= 0);
    assert(tmin[d] <= tmax[d]);
    assert(tmax[d] <= lsh[d]);
  }
}

GridDesc::GridDesc(const GHExt::PatchData::LevelData &leveldata,
                   const int component) {
  // `global_component` is the global component index.
  // There is no tiling.

  const auto &patchdata = ghext->patchdata.at(leveldata.patch);

  const amrex::FabArrayBase &fab = *leveldata.fab;

  const amrex::Box &fbx = fab.fabbox(component); // allocated array
  const amrex::Box &vbx =
      fab.box(component);      // interior region (without ghosts)
  const amrex::Box &gbx = fbx; // current region (with ghosts)
  const amrex::Box &domain = patchdata.amrcore->Geom(leveldata.level).Domain();

  for (int d = 0; d < dim; ++d)
    assert(domain.type(d) == amrex::IndexType::CELL);

  // Level, patch, and component
  level = leveldata.level;
  patch = leveldata.patch;
  this->component = component;

  // The number of ghostzones in each direction
  for (int d = 0; d < dim; ++d)
    nghostzones[d] = fab.nGrowVect()[d];

  // Global shape
  for (int d = 0; d < dim; ++d)
    gsh[d] = domain[orient(d, 1)] + 1 - domain[orient(d, 0)] + 1 +
             2 * nghostzones[d];

  // Local shape
  for (int d = 0; d < dim; ++d)
    lsh[d] = fbx[orient(d, 1)] - fbx[orient(d, 0)] + 1 + 1;

  // Allocated shape
  for (int d = 0; d < dim; ++d)
    ash[d] = lsh[d];

  // Local extent
  for (int d = 0; d < dim; ++d) {
    lbnd[d] = fbx[orient(d, 0)] + nghostzones[d];
    ubnd[d] = fbx[orient(d, 1)] + 1 + nghostzones[d];
  }

  // Boundaries
  const auto &symmetries = ghext->patchdata.at(leveldata.patch).symmetries;
  for (int d = 0; d < dim; ++d)
    for (int f = 0; f < 2; ++f)
      bbox[f][d] = vbx[orient(d, f)] == domain[orient(d, f)] &&
                   symmetries[f][d] != symmetry_t::none;

  // Thread tile box
  for (int d = 0; d < dim; ++d) {
    tmin[d] = gbx[orient(d, 0)] - fbx[orient(d, 0)];
    // For vertex centred grids, the allocated box is 1 vertex larger
    // than the number of cells, and AMReX assigns this extra vertex
    // to the final tile
    assert(gbx[orient(d, 1)] <= fbx[orient(d, 1)]);
    tmax[d] = gbx[orient(d, 1)] + 1 - fbx[orient(d, 0)] +
              (gbx[orient(d, 1)] == fbx[orient(d, 1)]);
  }

  const amrex::Geometry &geom = patchdata.amrcore->Geom(0);
  const CCTK_REAL *restrict const global_x0 = geom.ProbLo();
  const CCTK_REAL *restrict const global_dx = geom.CellSize();
  for (int d = 0; d < dim; ++d) {
    const int levfac = 1 << leveldata.level;
    // Offset between this level's and the coarsest level's origin as
    // multiple of the grid spacing
    const int levoff = (1 - levfac) * (1 - 2 * nghostzones[d]);
    const int levoffdenom = 2;
    // Cell-centred coordinates on coarse level
    const CCTK_REAL origin_space =
        global_x0[d] + (1 - 2 * nghostzones[d]) * global_dx[d] / 2;
    const CCTK_REAL delta_space = global_dx[d];
    // Cell-centred coordinates on current level
    dx[d] = delta_space / levfac;
    x0[d] = origin_space + dx[d] * levoff / levoffdenom;
  }

  // Check constraints
  for (int d = 0; d < dim; ++d) {
    // Domain size
    assert(gsh[d] >= 0);

    // Local size
    assert(lbnd[d] >= 0);
    assert(lsh[d] >= 0);
    assert(lbnd[d] + lsh[d] <= gsh[d]);
    assert(ubnd[d] == lbnd[d] + lsh[d] - 1);

    // Internal representation
    assert(ash[d] >= 0);
    assert(ash[d] >= lsh[d]);

    // Ghost zones
    assert(nghostzones[d] >= 0);
    assert(2 * nghostzones[d] <= lsh[d]);

    // Tiles
    assert(tmin[d] >= 0);
    assert(tmin[d] <= tmax[d]);
    assert(tmax[d] <= lsh[d]);
  }
}

GridPtrDesc::GridPtrDesc(const GHExt::PatchData::LevelData &leveldata,
                         const MFPointer &mfp)
    : GridDesc(leveldata, mfp) {
  const amrex::IntVect ng(nghostzones[0], nghostzones[1], nghostzones[2]);
  const amrex::Box &fbx = mfp.fabbox(ng); // allocated array
  cactus_offset = lbound(fbx);
}

GridPtrDesc1::GridPtrDesc1(
    const GHExt::PatchData::LevelData &leveldata,
    const GHExt::PatchData::LevelData::GroupData &groupdata,
    const MFPointer &mfp)
    : GridDesc(leveldata, mfp) {
  DECLARE_CCTK_PARAMETERS;
  const amrex::IntVect ng(nghostzones[0], nghostzones[1], nghostzones[2]);
  const amrex::Box &fbx = mfp.fabbox(ng); // allocated array
  cactus_offset = lbound(fbx);
  for (int d = 0; d < dim; ++d) {
    assert(groupdata.nghostzones.at(d) >= 0);
    assert(groupdata.nghostzones.at(d) <= nghostzones[d]);
  }
  for (int d = 0; d < dim; ++d)
    gimin[d] = nghostzones[d] - groupdata.nghostzones.at(d);
  for (int d = 0; d < dim; ++d)
    gimax[d] = lsh[d] - groupdata.indextype.at(d) -
               (nghostzones[d] - groupdata.nghostzones.at(d));
  for (int d = 0; d < dim; ++d)
    gash[d] = ash[d] - groupdata.indextype.at(d) -
              2 * (nghostzones[d] - groupdata.nghostzones.at(d));
}

////////////////////////////////////////////////////////////////////////////////

cGH *copy_cctkGH(const cGH *restrict const sourceGH) {
  cGH *restrict const cctkGH = new cGH;

  // Copy all fields by default
  *cctkGH = *sourceGH;

  // Allocate most pointers anew
  const auto copy_array = [](const auto *restrict const srcptr, const int sz) {
    using T = decay_t<decltype(*srcptr)>;
    T *restrict const ptr = new T[sz];
    copy(srcptr, srcptr + sz, ptr);
    return ptr;
  };
  cctkGH->cctk_gsh = copy_array(sourceGH->cctk_gsh, dim);
  cctkGH->cctk_lsh = copy_array(sourceGH->cctk_lsh, dim);
  cctkGH->cctk_lbnd = copy_array(sourceGH->cctk_lbnd, dim);
  cctkGH->cctk_ubnd = copy_array(sourceGH->cctk_ubnd, dim);
  cctkGH->cctk_tile_min = copy_array(sourceGH->cctk_tile_min, dim);
  cctkGH->cctk_tile_max = copy_array(sourceGH->cctk_tile_max, dim);
  cctkGH->cctk_ash = copy_array(sourceGH->cctk_ash, dim);
  cctkGH->cctk_to = copy_array(sourceGH->cctk_to, dim);
  cctkGH->cctk_from = copy_array(sourceGH->cctk_from, dim);
  cctkGH->cctk_delta_space = copy_array(sourceGH->cctk_delta_space, dim);
  cctkGH->cctk_origin_space = copy_array(sourceGH->cctk_origin_space, dim);
  cctkGH->cctk_bbox = copy_array(sourceGH->cctk_bbox, 2 * dim);
  cctkGH->cctk_levfac = copy_array(sourceGH->cctk_levfac, dim);
  cctkGH->cctk_levoff = copy_array(sourceGH->cctk_levoff, dim);
  cctkGH->cctk_levoffdenom = copy_array(sourceGH->cctk_levoffdenom, dim);
  cctkGH->cctk_nghostzones = copy_array(sourceGH->cctk_nghostzones, dim);

  const int numvars = CCTK_NumVars();
  cctkGH->data = new void **[numvars];
  for (int vi = 0; vi < numvars; ++vi)
    cctkGH->data[vi] =
        copy_array(sourceGH->data[vi], CCTK_DeclaredTimeLevelsVI(vi));

  return cctkGH;
}

void delete_cctkGH(cGH *cctkGH) {
  delete[] cctkGH->cctk_gsh;
  delete[] cctkGH->cctk_lsh;
  delete[] cctkGH->cctk_lbnd;
  delete[] cctkGH->cctk_ubnd;
  delete[] cctkGH->cctk_tile_min;
  delete[] cctkGH->cctk_tile_max;
  delete[] cctkGH->cctk_ash;
  delete[] cctkGH->cctk_to;
  delete[] cctkGH->cctk_from;
  delete[] cctkGH->cctk_delta_space;
  delete[] cctkGH->cctk_origin_space;
  delete[] cctkGH->cctk_bbox;
  delete[] cctkGH->cctk_levfac;
  delete[] cctkGH->cctk_levoff;
  delete[] cctkGH->cctk_levoffdenom;
  delete[] cctkGH->cctk_nghostzones;
  const int numvars = CCTK_NumVars();
  for (int vi = 0; vi < numvars; ++vi)
    delete[] cctkGH->data[vi];
  delete[] cctkGH->data;
#ifdef CCTK_DEBUG
  memset(cctkGH, 0, sizeof *cctkGH);
#endif
  delete cctkGH;
}

enum class mode_t { unknown, local, patch, level, global, meta };

mode_t current_mode(const cGH *restrict cctkGH) {
  const bool have_local = cctkGH->cctk_component != undefined;
  const bool have_patch = cctkGH->cctk_patch != undefined;
  const bool have_level = cctkGH->cctk_level != undefined;
  const bool have_global = cctkGH->cctk_nghostzones[0] != undefined;
  if (have_local && have_patch && have_level && have_global)
    return mode_t::local;
  else if (!have_local && have_patch && have_level && have_global)
    return mode_t::patch;
  else if (!have_local && !have_patch && have_level && have_global)
    return mode_t::level;
  else if (!have_local && !have_patch && !have_level && have_global)
    return mode_t::global;
  else if (!have_local && !have_patch && !have_level && !have_global)
    return mode_t::meta;
  else
    assert(0);
}

bool in_local_mode(const cGH *restrict cctkGH) {
  return current_mode(cctkGH) == mode_t::local;
}

bool in_patch_mode(const cGH *restrict cctkGH) {
  return current_mode(cctkGH) == mode_t::patch;
}

bool in_level_mode(const cGH *restrict cctkGH) {
  return current_mode(cctkGH) == mode_t::level;
}

bool in_global_mode(const cGH *restrict cctkGH) {
  return current_mode(cctkGH) == mode_t::global;
}

bool in_meta_mode(const cGH *restrict cctkGH) {
  return current_mode(cctkGH) == mode_t::meta;
}

// Initialize cctkGH entries
void setup_cctkGH(cGH *restrict cctkGH) {
  DECLARE_CCTK_PARAMETERS;

  // Dimensions
  cctkGH->cctk_dim = 3;

  // Grid function alignment
  // TODO: Check whether AMReX guarantees a particular alignment
  cctkGH->cctk_alignment = 1;
  cctkGH->cctk_alignment_offset = 0;

  // The refinement factor in time over the top level (coarsest) grid
  cctkGH->cctk_timefac = 1; // no subcycling

  // The total number of patches
  cctkGH->cctk_npatches = ghext->num_patches();

  // The convergence level (numbered from zero upwards)
  cctkGH->cctk_convlevel = 0; // no convergence tests

  // Initialize grid spacing
  for (int d = 0; d < dim; ++d) {
    cctkGH->cctk_origin_space[d] = NAN;
    cctkGH->cctk_delta_space[d] = NAN;
  }

  // Initialize time stepping
  cctkGH->cctk_time = 0;
  cctkGH->cctk_delta_time = NAN;

  // init into meta mode
  cctkGH->cctk_component = undefined;
  cctkGH->cctk_level = undefined;
  cctkGH->cctk_patch = undefined;
  cctkGH->cctk_nghostzones[0] = undefined;
  assert(in_meta_mode(cctkGH));
}

// Update fields that carry state and change over time
void update_cctkGH(cGH *const cctkGH, const cGH *const sourceGH) {
  if (cctkGH == sourceGH)
    return;
  cctkGH->cctk_iteration = sourceGH->cctk_iteration;
  cctkGH->cctk_time = sourceGH->cctk_time;
  cctkGH->cctk_delta_time = sourceGH->cctk_delta_time;
  // for (int d = 0; d < dim; ++d)
  //   cctkGH->cctk_origin_space[d] = sourceGH->cctk_origin_space[d];
  // for (int d = 0; d < dim; ++d)
  //   cctkGH->cctk_delta_space[d] = sourceGH->cctk_delta_space[d];
}

// Set cctkGH entries for global mode
void enter_global_mode(cGH *restrict cctkGH) {
  DECLARE_CCTK_PARAMETERS;
  assert(in_meta_mode(cctkGH));

  // The number of ghostzones in each direction
  // TODO: Get this from mfab (mfab.fb_ghosts)
  cctkGH->cctk_nghostzones[0] = ghost_size >= 0 ? ghost_size : ghost_size_x;
  cctkGH->cctk_nghostzones[1] = ghost_size >= 0 ? ghost_size : ghost_size_y;
  cctkGH->cctk_nghostzones[2] = ghost_size >= 0 ? ghost_size : ghost_size_z;

  // Grid scalar pointers
  {
    auto &restrict globaldata = ghext->globaldata;
    const int num_groups = CCTK_NumGroups();
    for (int gi = 0; gi < num_groups; ++gi) {
      cGroup group;
      int ierr = CCTK_GroupData(gi, &group);
      assert(!ierr);

      if (group.grouptype != CCTK_SCALAR && group.grouptype != CCTK_ARRAY) {
        continue;
      } else { // CCTK_ARRAY or CCTK_SCALAR
        auto &restrict arraygroupdata = *globaldata.arraygroupdata.at(gi);
        for (int tl = 0; tl < int(arraygroupdata.data.size()); ++tl) {
          const auto &restrict vars = arraygroupdata.data.at(tl);
          for (int vi = 0; vi < arraygroupdata.numvars; ++vi)
            cctkGH->data[arraygroupdata.firstvarindex + vi][tl] =
                const_cast<CCTK_REAL *>(
                    &vars.at(vi * arraygroupdata.array_size));
        }
      }
    }
  }

  assert(in_global_mode(cctkGH));
}
void leave_global_mode(cGH *restrict cctkGH) {
  assert(in_global_mode(cctkGH));

  // Grid scalar pointers
  {
    auto &restrict globaldata = ghext->globaldata;
    const int num_groups = CCTK_NumGroups();
    for (int gi = 0; gi < num_groups; ++gi) {
      cGroup group;
      int ierr = CCTK_GroupData(gi, &group);
      assert(!ierr);

      if (group.grouptype != CCTK_SCALAR && group.grouptype != CCTK_ARRAY) {
        continue;
      } else { // CCTK_ARRAY or CCTK_SCALAR
        auto &restrict arraygroupdata = *globaldata.arraygroupdata.at(gi);
        for (int tl = 0; tl < int(arraygroupdata.data.size()); ++tl)
          for (int vi = 0; vi < arraygroupdata.numvars; ++vi)
            cctkGH->data[arraygroupdata.firstvarindex + vi][tl] = nullptr;
      }
    }
  }

  for (int d = 0; d < dim; ++d)
    cctkGH->cctk_nghostzones[d] = undefined;

  assert(in_meta_mode(cctkGH));
}

// Set cctkGH entries for level mode
void enter_level_mode(cGH *restrict cctkGH, const int level) {
  DECLARE_CCTK_PARAMETERS;
  assert(in_global_mode(cctkGH));

  cctkGH->cctk_level = level;
  for (int d = 0; d < dim; ++d) {
    // The refinement factor over the top level (coarsest) grid
    const int levfac = 1 << level;
    cctkGH->cctk_levfac[d] = levfac;
    // Offset between this level's and the coarsest level's origin as multiple
    // of the grid spacing
    const int levoff = (1 - levfac) * (1 - 2 * cctkGH->cctk_nghostzones[d]);
    const int levoffdenom = 2;
    cctkGH->cctk_levoff[d] = levoff;
    cctkGH->cctk_levoffdenom[d] = levoffdenom;
  }

  assert(in_level_mode(cctkGH));
}
void leave_level_mode(cGH *restrict cctkGH, const int level) {
  assert(in_level_mode(cctkGH));
  cctkGH->cctk_level = undefined;
  for (int d = 0; d < dim; ++d)
    cctkGH->cctk_levfac[d] = undefined;
  for (int d = 0; d < dim; ++d) {
    cctkGH->cctk_levoff[d] = undefined;
    cctkGH->cctk_levoffdenom[d] = 0;
  }
  assert(in_global_mode(cctkGH));
}

// Set cctkGH entries for patch mode
void enter_patch_mode(cGH *restrict cctkGH,
                      const GHExt::PatchData::LevelData &restrict leveldata) {
  DECLARE_CCTK_PARAMETERS;
  assert(in_level_mode(cctkGH));

  const auto &patchdata = ghext->patchdata.at(leveldata.patch);

  cctkGH->cctk_patch = leveldata.patch;
  const amrex::Box &domain = patchdata.amrcore->Geom(leveldata.level).Domain();
  const amrex::Geometry &geom = patchdata.amrcore->Geom(0);
  const CCTK_REAL *restrict const global_x0 = geom.ProbLo();
  const CCTK_REAL *restrict const global_dx = geom.CellSize();
  for (int d = 0; d < dim; ++d) {
    // Global shape
    assert(cctkGH->cctk_nghostzones[d] != undefined);
    assert(domain.type(d) == amrex::IndexType::CELL);
    cctkGH->cctk_gsh[d] = domain[orient(d, 1)] + 1 - domain[orient(d, 0)] + 1 +
                          2 * cctkGH->cctk_nghostzones[d];
    // Cell-centred coarse level coordinates
    // TODOPATCH: Shouldn't these be vertex centred?
    const CCTK_REAL origin_space =
        global_x0[d] + (1 - 2 * cctkGH->cctk_nghostzones[d]) * global_dx[d] / 2;
    const CCTK_REAL delta_space = global_dx[d];
    cctkGH->cctk_delta_space[d] = delta_space;
    cctkGH->cctk_origin_space[d] = origin_space;
  }

  assert(in_patch_mode(cctkGH));
}
void leave_patch_mode(cGH *restrict cctkGH,
                      const GHExt::PatchData::LevelData &restrict leveldata) {
  assert(in_patch_mode(cctkGH));
  cctkGH->cctk_patch = undefined;
  for (int d = 0; d < dim; ++d)
    cctkGH->cctk_gsh[d] = undefined;
  for (int d = 0; d < dim; ++d) {
    cctkGH->cctk_origin_space[d] = NAN;
    cctkGH->cctk_delta_space[d] = NAN;
  }
  assert(in_level_mode(cctkGH));
}

// Set cctkGH entries for local mode
// TODO: Have separate cctkGH for each patch, level, and local box
void enter_local_mode(cGH *restrict cctkGH,
                      const GHExt::PatchData::LevelData &restrict leveldata,
                      const MFPointer &mfp) {
  assert(in_patch_mode(cctkGH));
  const GridPtrDesc grid(leveldata, mfp);

  cctkGH->cctk_component = mfp.index();
  for (int d = 0; d < dim; ++d)
    cctkGH->cctk_lsh[d] = grid.lsh[d];
  for (int d = 0; d < dim; ++d)
    cctkGH->cctk_ash[d] = grid.ash[d];
  for (int d = 0; d < dim; ++d) {
    cctkGH->cctk_lbnd[d] = grid.lbnd[d];
    cctkGH->cctk_ubnd[d] = grid.ubnd[d];
  }
  for (int d = 0; d < dim; ++d) {
    cctkGH->cctk_tile_min[d] = grid.tmin[d];
    cctkGH->cctk_tile_max[d] = grid.tmax[d];
  }
  for (int d = 0; d < dim; ++d)
    for (int f = 0; f < 2; ++f)
      cctkGH->cctk_bbox[2 * d + f] = grid.bbox[f][d];

  // Grid function pointers
  const int num_groups = CCTK_NumGroups();
  for (int gi = 0; gi < num_groups; ++gi) {
    cGroup group;
    int ierr = CCTK_GroupData(gi, &group);
    assert(!ierr);

    if (group.grouptype != CCTK_GF)
      continue;

    auto &restrict groupdata = *leveldata.groupdata.at(gi);
    const GridPtrDesc1 grid1(leveldata, groupdata, mfp);
    for (int tl = 0; tl < int(groupdata.mfab.size()); ++tl) {
      const amrex::Array4<CCTK_REAL> vars =
          groupdata.mfab.at(tl)->array(mfp.index());
      for (int vi = 0; vi < groupdata.numvars; ++vi)
        cctkGH->data[groupdata.firstvarindex + vi][tl] = grid1.ptr(vars, vi);
    }
  }

  // Check constraints
  for (int d = 0; d < dim; ++d) {
    // Domain size
    assert(cctkGH->cctk_gsh[d] >= 0);

    // Local size
    assert(cctkGH->cctk_lbnd[d] >= 0);
    assert(cctkGH->cctk_lsh[d] >= 0);
    assert(cctkGH->cctk_lbnd[d] + cctkGH->cctk_lsh[d] <= cctkGH->cctk_gsh[d]);
    assert(cctkGH->cctk_ubnd[d] ==
           cctkGH->cctk_lbnd[d] + cctkGH->cctk_lsh[d] - 1);

    // Tile box
    assert(cctkGH->cctk_tile_min[d] >= 0);
    assert(cctkGH->cctk_tile_min[d] <= cctkGH->cctk_tile_max[d]);
    assert(cctkGH->cctk_tile_max[d] <= cctkGH->cctk_lsh[d]);

    // Internal representation
    assert(cctkGH->cctk_ash[d] >= 0);
    assert(cctkGH->cctk_ash[d] >= cctkGH->cctk_lsh[d]);

    // Ghost zones
    assert(cctkGH->cctk_nghostzones[d] >= 0);
    assert(2 * cctkGH->cctk_nghostzones[d] <= cctkGH->cctk_lsh[d]);
  }

  assert(in_local_mode(cctkGH));
}
void leave_local_mode(cGH *restrict cctkGH,
                      const GHExt::PatchData::LevelData &restrict leveldata,
                      const MFPointer &mfp) {
  assert(in_local_mode(cctkGH));
  cctkGH->cctk_component = undefined;
  for (int d = 0; d < dim; ++d)
    cctkGH->cctk_lsh[d] = undefined;
  for (int d = 0; d < dim; ++d)
    cctkGH->cctk_ash[d] = undefined;
  for (int d = 0; d < dim; ++d) {
    cctkGH->cctk_lbnd[d] = undefined;
    cctkGH->cctk_ubnd[d] = undefined;
  }
  for (int d = 0; d < dim; ++d) {
    cctkGH->cctk_tile_min[d] = undefined;
    cctkGH->cctk_tile_max[d] = undefined;
  }
  for (int d = 0; d < dim; ++d)
    for (int f = 0; f < 2; ++f)
      cctkGH->cctk_bbox[2 * d + f] = undefined;
  const int num_groups = CCTK_NumGroups();
  for (int gi = 0; gi < num_groups; ++gi) {
    cGroup group;
    int ierr = CCTK_GroupData(gi, &group);
    assert(!ierr);

    if (group.grouptype != CCTK_GF)
      continue;

    auto &restrict groupdata = *leveldata.groupdata.at(gi);
    for (int tl = 0; tl < int(groupdata.mfab.size()); ++tl)
      for (int vi = 0; vi < groupdata.numvars; ++vi)
        cctkGH->data[groupdata.firstvarindex + vi][tl] = nullptr;
  }
  assert(in_patch_mode(cctkGH));
}

// Should this be passed in `cGH`?
int CallFunction_count = -1;
extern "C" CCTK_INT CarpetX_GetCallFunctionCount() {
#ifndef AMREX_USE_GPU
  // CPU: use hardware thread index
  return omp_get_thread_num();
#else
  // GPU: count tiles
  assert(CallFunction_count >= 0);
  return CallFunction_count;
#endif
}

active_levels_t::active_levels_t(const int min_level, const int max_level,
                                 const int min_patch, const int max_patch)
    : min_level(min_level), max_level(max_level), min_patch(min_patch),
      max_patch(max_patch) {
  assert(min_level >= 0);
  assert(max_level <= ghext->num_levels());
  assert(min_patch >= 0);
  assert(max_patch <= ghext->num_patches());
}
active_levels_t::active_levels_t(const int min_level, const int max_level)
    : active_levels_t(min_level, max_level, 0, ghext->num_patches()) {}
active_levels_t::active_levels_t() : active_levels_t(0, 0) {}

void active_levels_t::assert_consistent_iterations() const {
  rat64 good_iteration = -1;
  for (int level = min_level; level < max_level; ++level) {
    for (int patch = min_patch; patch < max_patch; ++patch) {
      const auto &patchdata = ghext->patchdata.at(patch);
      if (level < int(patchdata.leveldata.size())) {
        const auto &leveldata = patchdata.leveldata.at(level);
        const auto &iteration = leveldata.iteration;
        if (good_iteration == -1)
          good_iteration = iteration;
        assert(iteration == good_iteration);
      }
    }
  }
}

// Loop over all active patches of all active levels from coarsest to
// finest
void active_levels_t::loop_coarse_to_fine(
    const std::function<void(GHExt::PatchData::LevelData &level)> &kernel)
    const {
  assert(omp_get_num_threads() == 1);
  assert_consistent_iterations();
  for (int level = min_level; level < max_level; ++level) {
    for (int patch = min_patch; patch < max_patch; ++patch) {
      auto &patchdata = ghext->patchdata.at(patch);
      if (level < int(patchdata.leveldata.size()))
        kernel(patchdata.leveldata.at(level));
    }
  }
}

// Loop over all active patches of all active levels from finest to
// coarsest
void active_levels_t::loop_fine_to_coarse(
    const std::function<void(GHExt::PatchData::LevelData &level)> &kernel)
    const {
  assert(omp_get_num_threads() == 1);
  assert_consistent_iterations();
  for (int level = max_level - 1; level >= min_level; --level) {
    for (int patch = min_patch; patch < max_patch; ++patch) {
      auto &patchdata = ghext->patchdata.at(patch);
      if (level < int(patchdata.leveldata.size()))
        kernel(patchdata.leveldata.at(level));
    }
  }
}

// Loop over all components of all active patches of all active levels
// in parallel
void active_levels_t::loop_parallel(
    const std::function<void(int patch, int level, int index, int component,
                             const cGH *cctkGH)> &kernel) const {
  assert(omp_get_num_threads() == 1);
  task_manager tasks;

  loop_coarse_to_fine([&](const auto &restrict leveldata) {
    int component = 0;
    const auto mfitinfo = amrex::MFItInfo().DisableDeviceSync().EnableTiling();
    const auto &fab0 = *leveldata.fab;
    for (amrex::MFIter mfi(fab0, mfitinfo); mfi.isValid(); ++mfi, ++component) {
      const int index = mfi.index();
      tasks.submit_serially([&kernel, &leveldata, index, component]() {
        const int patch = leveldata.patch;
        const int level = leveldata.level;
        cGH *restrict const localGH = leveldata.get_local_cctkGH(component);
        kernel(patch, level, index, component, localGH);
      });
    }
  });

  // Run all tasks
  tasks.run_tasks();
  // There is an implicit OpenMP barrier here.
}

// Loop over all components of all active patches of all active levels
// serially
void active_levels_t::loop_serially(
    const std::function<void(int patch, int level, int index, int component,
                             const cGH *cctkGH)> &kernel) const {
  loop_coarse_to_fine([&](const auto &restrict leveldata) {
    int component = 0;
    const auto mfitinfo = amrex::MFItInfo().DisableDeviceSync().EnableTiling();
    const auto &fab0 = *leveldata.fab;
    for (amrex::MFIter mfi(fab0, mfitinfo); mfi.isValid(); ++mfi, ++component) {
      const int index = mfi.index();
      const int patch = leveldata.patch;
      const int level = leveldata.level;
      cGH *restrict const localGH = leveldata.get_local_cctkGH(component);
      kernel(patch, level, index, component, localGH);
    }
  });
}

void synchronize() {
#ifdef AMREX_USE_GPU
  // TODO: Synchronize only if GPU kernels were actually launched
  amrex::Gpu::streamSynchronizeAll();
  AMREX_GPU_ERROR_CHECK();
#endif
}

void update_cctkGHs(cGH *restrict const cctkGH) {
  update_cctkGH(ghext->global_cctkGH.get(), cctkGH);
  for (auto &restrict level_cctkGH : ghext->level_cctkGHs) {
    update_cctkGH(level_cctkGH.get(), cctkGH);
  }
  for (auto &patch : ghext->patchdata) {
    for (auto &restrict level : patch.leveldata) {
      update_cctkGH(level.patch_cctkGH.get(), cctkGH);
    }
  }
  for (auto &patch : ghext->patchdata) {
    for (auto &restrict level : patch.leveldata) {
      for (auto &restrict local_cctkGH : level.local_cctkGHs) {
        update_cctkGH(local_cctkGH.get(), cctkGH);
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

extern "C" void CarpetX_GetLoopBoxAll(const void *restrict const cctkGH_,
                                      const CCTK_INT size,
                                      CCTK_INT *restrict const loop_min,
                                      CCTK_INT *restrict const loop_max) {
  const cGH *restrict const cctkGH = static_cast<const cGH *>(cctkGH_);
  assert(size == dim);
  const GridDescBase grid(cctkGH);
  vect<int, dim> imin, imax;
  // Assumd cell centred box
  grid.box_all<1, 1, 1>(grid.nghostzones, imin, imax);
  for (int d = 0; d < dim; ++d)
    loop_min[d] = imin[d];
  for (int d = 0; d < dim; ++d)
    loop_max[d] = imax[d];
}

extern "C" void CarpetX_GetLoopBoxInt(const void *restrict const cctkGH_,
                                      const CCTK_INT size,
                                      CCTK_INT *restrict const loop_min,
                                      CCTK_INT *restrict const loop_max) {
  const cGH *restrict const cctkGH = static_cast<const cGH *>(cctkGH_);
  assert(size == dim);
  const GridDescBase grid(cctkGH);
  vect<int, dim> imin, imax;
  // Assumd cell centred box
  grid.box_int<1, 1, 1>(grid.nghostzones, imin, imax);
  for (int d = 0; d < dim; ++d)
    loop_min[d] = imin[d];
  for (int d = 0; d < dim; ++d)
    loop_max[d] = imax[d];
}

////////////////////////////////////////////////////////////////////////////////

mode_t decode_mode(const cFunctionData *restrict attribute) {
  bool local_mode = attribute->local;
  bool level_mode = attribute->level;
  bool global_mode = attribute->global;
  bool meta_mode = attribute->meta;
  assert(int(local_mode) + int(level_mode) + int(global_mode) +
             int(meta_mode) <=
         1);
  if (attribute->local)
    return mode_t::local;
  if (attribute->level)
    return mode_t::level;
  if (attribute->global)
    return mode_t::global;
  if (attribute->meta)
    return mode_t::meta;
  return mode_t::local; // default
}

enum class rdwr_t { read, write, invalid };
ostream &operator<<(ostream &os, const rdwr_t rdwr) {
  switch (rdwr) {
  case rdwr_t::read:
    return os << "read";
  case rdwr_t::write:
    return os << "write";
  case rdwr_t::invalid:
    return os << "invalid";
  default:
    assert(0);
  }
}

struct clause_t {
  int gi, vi, tl;
  valid_t valid;

  friend bool operator==(const clause_t &x, const clause_t &y) {
    return make_tuple(x.gi, x.vi, x.tl, x.valid) ==
           make_tuple(y.gi, y.vi, y.tl, y.valid);
  }
  friend bool operator<(const clause_t &x, const clause_t &y) {
    return make_tuple(x.gi, x.vi, x.tl, x.valid) <
           make_tuple(y.gi, y.vi, y.tl, y.valid);
  }

  friend ostream &operator<<(ostream &os, const clause_t &cl) {
    return os << "clause_t{gi:" << cl.gi << ",vi:" << cl.vi << ",tl:" << cl.tl
              << ",valid:" << cl.valid << "}";
  }
};

vector<clause_t> decode_clauses(const cFunctionData *restrict attribute,
                                const rdwr_t rdwr) {
  vector<clause_t> result;
  result.reserve(attribute->n_RDWR);
  for (int n = 0; n < attribute->n_RDWR; ++n) {
    const RDWR_entry &restrict RDWR = attribute->RDWR[n];
    int gi = CCTK_GroupIndexFromVarI(RDWR.varindex);
    assert(gi >= 0);
    int vi = RDWR.varindex - CCTK_FirstVarIndexI(gi);
    assert(vi >= 0 && vi < CCTK_NumVarsInGroupI(gi));
    int tl = RDWR.timelevel;
    int where;
    switch (rdwr) {
    case rdwr_t::read:
      where = RDWR.where_rd;
      break;
    case rdwr_t::write:
      where = RDWR.where_wr;
      break;
    case rdwr_t::invalid:
      where = RDWR.where_inv;
      break;
    default:
      assert(0);
    }
    // Ignore clauses that have no effect
    if (where == 0)
      continue;
    valid_t valid;
    valid.valid_int = where & CCTK_VALID_INTERIOR;
    valid.valid_outer = where & CCTK_VALID_BOUNDARY;
    valid.valid_ghosts = where & CCTK_VALID_GHOSTS;
    result.push_back({gi, vi, tl, valid});
  }
  return result;
}

// Schedule initialisation
int Initialise(tFleshConfig *config) {
  DECLARE_CCTK_PARAMETERS;

  static Timer timer("Initialise");
  Interval interval(timer);

  cGH *restrict const cctkGH = CCTK_SetupGH(config, 0);
  CCTKi_AddGH(config, 0, cctkGH);

  // Check presync mode
  if (!CCTK_EQUALS(presync_mode, "mixed-error") &&
      !CCTK_EQUALS(presync_mode, "presync-only"))
    CCTK_ERROR("CarpetX currently requires Cactus::presync_mode = "
               "\"mixed-error\" or \"presync-only\"");

  // Initialise iteration and time
  cctkGH->cctk_iteration = 0;
  cctkGH->cctk_time = *static_cast<const CCTK_REAL *>(
      CCTK_ParameterGet("cctk_initial_time", "Cactus", nullptr));

  // Initialise schedule
  CCTKi_ScheduleGHInit(cctkGH);

  // Initialise all grid extensions
  CCTKi_InitGHExtensions(cctkGH);

  // Set up cctkGH
  setup_cctkGH(cctkGH);
  enter_global_mode(cctkGH);
  ghext->global_cctkGH = GHExt::cctkGHptr(copy_cctkGH(cctkGH));

  for (const auto &patchdata : ghext->patchdata)
    assert(patchdata.leveldata.empty());
  assert(!active_levels);
  active_levels = make_optional<active_levels_t>(0, 0);

  CCTK_Traverse(cctkGH, "CCTK_WRAGH");
  CCTK_Traverse(cctkGH, "CCTK_PARAMCHECK");
  CCTKi_FinaliseParamWarn();

  active_levels = optional<active_levels_t>();

  if (config->recovered) {
    // Recover
#pragma omp critical
    CCTK_VINFO("Recovering from checkpoint...");

    RecoverGridStructure(cctkGH);

    assert(!active_levels);
    active_levels = make_optional<active_levels_t>();

    CCTK_Traverse(cctkGH, "CCTK_BASEGRID");

    const char *recovery_mode = *static_cast<const char *const *>(
        CCTK_ParameterGet("recovery_mode", "Cactus", nullptr));
    if (!CCTK_Equals(recovery_mode, "strict")) {
      // Set up initial conditions
      CCTK_Traverse(cctkGH, "CCTK_INITIAL");
      CCTK_Traverse(cctkGH, "CCTK_POSTINITIAL");
      CCTK_Traverse(cctkGH, "CCTK_POSTPOSTINITIAL");
    }

    // Recover
    RecoverGH(cctkGH);
    CCTK_Traverse(cctkGH, "CCTK_RECOVER_VARIABLES");
    CCTK_Traverse(cctkGH, "CCTK_POST_RECOVER_VARIABLES");

    active_levels = optional<active_levels_t>();

    // Enable regridding
    for (auto &patchdata : ghext->patchdata)
      patchdata.amrcore->cactus_is_initialized = true;

    // Determine time step size
    {
      CCTK_REAL mindx = 1.0 / 0.0;
      for (const auto &patchdata : ghext->patchdata) {
        const amrex::Geometry &geom = patchdata.amrcore->Geom(0);
        const CCTK_REAL *restrict const dx = geom.CellSize();
        CCTK_REAL mindx1 = 1.0 / 0.0;
        for (int d = 0; d < dim; ++d)
          mindx1 = fmin(mindx1, dx[d]);
        mindx1 = ldexp(mindx1, -(int(patchdata.leveldata.size()) - 1));
        mindx = fmin(mindx, mindx1);
      }
      cctkGH->cctk_delta_time = dtfac * mindx;
#pragma omp critical
      CCTK_VINFO("Iteration: %d   time: %g   delta_time: %g",
                 cctkGH->cctk_iteration, double(cctkGH->cctk_time),
                 double(cctkGH->cctk_delta_time));
    }

  } else {
    // Set up initial conditions
#pragma omp critical
    CCTK_VINFO("Setting up initial conditions...");

    // Create coarse grid
    {
      static Timer timer("InitialiseRegrid [coarse]");
      Interval interval(timer);

      const CCTK_REAL time = 0; // dummy time
      for (const auto &patchdata : ghext->patchdata)
        patchdata.amrcore->MakeNewGrids(time);

      // Determine time step size
      {
        CCTK_REAL mindx = 1.0 / 0.0;
        for (const auto &patchdata : ghext->patchdata) {
          const amrex::Geometry &geom = patchdata.amrcore->Geom(0);
          const CCTK_REAL *restrict const dx = geom.CellSize();
          CCTK_REAL mindx1 = 1.0 / 0.0;
          for (int d = 0; d < dim; ++d)
            mindx1 = fmin(mindx1, dx[d]);
          mindx1 = ldexp(mindx1, -(int(patchdata.leveldata.size()) - 1));
          mindx = fmin(mindx, mindx1);
        }
        cctkGH->cctk_delta_time = dtfac * mindx;
#pragma omp critical
        CCTK_VINFO("Iteration: %d   time: %g   delta_time: %g",
                   cctkGH->cctk_iteration, double(cctkGH->cctk_time),
                   double(cctkGH->cctk_delta_time));
      }

      assert(!active_levels);
      active_levels = make_optional<active_levels_t>(0, 1);
      CCTK_Traverse(cctkGH, "CCTK_BASEGRID");
      // CCTK_Traverse(cctkGH, "CCTK_POSTREGRID");
      active_levels = optional<active_levels_t>();
    }

    // Output domain information
    if (CCTK_MyProc(nullptr) == 0) {
      for (const auto &patchdata : ghext->patchdata) {
        const int level = 0;
        const cGH *const patchGH =
            ghext->get_patch_cctkGH(level, patchdata.patch);
        const int *restrict const gsh = patchGH->cctk_gsh;
        const int *restrict const nghostzones = patchGH->cctk_nghostzones;
        CCTK_REAL x0[dim], x1[dim], dx[dim];
        for (int d = 0; d < dim; ++d) {
          dx[d] = patchGH->cctk_delta_space[d];
          x0[d] = patchGH->cctk_origin_space[d] +
                  (2 * nghostzones[d] - 1) * dx[d] / 2;
          x1[d] = x0[d] + (gsh[d] - 2 * nghostzones[d] - 1) * dx[d];
        }
#pragma omp critical
        {
          CCTK_VINFO("Patch %d:", patchdata.patch);
          CCTK_VINFO("  Grid extent:");
          CCTK_VINFO("    gsh=[%d,%d,%d]", gsh[0], gsh[1], gsh[2]);
          const auto &bf = patchdata.amrcore->blockingFactor(0);
          CCTK_VINFO("    blocking_factor=[%d,%d,%d]", bf[0], bf[1], bf[2]);
          const auto &mgs = patchdata.amrcore->maxGridSize(0);
          CCTK_VINFO("    max_grid_size=[%d,%d,%d]", mgs[0], mgs[1], mgs[2]);
          const auto mfitinfo = amrex::MFItInfo().EnableTiling();
          const auto &ts = mfitinfo.tilesize;
          CCTK_VINFO("    max_tile_size=[%d,%d,%d]", ts[0], ts[1], ts[2]);
          CCTK_VINFO("  Domain extent:");
          CCTK_VINFO("    xmin=[%.17g,%.17g,%.17g]", double(x0[0]),
                     double(x0[1]), double(x0[2]));
          CCTK_VINFO("    xmax=[%.17g,%.17g,%.17g]", double(x1[0]),
                     double(x1[1]), double(x1[2]));
          CCTK_VINFO("    base dx=[%.17g,%.17g,%.17g]", double(dx[0]),
                     double(dx[1]), double(dx[2]));
          // CCTK_VINFO("  Time stepping:");
          // CCTK_VINFO("    t0=%.17g", double(patchGH->cctk_time));
          // CCTK_VINFO("    dt=%.17g", double(patchGH->cctk_delta_time));
        }
      }
    }

    // Enable regridding. We can only enable regridding after the
    // coarse level has been initialized, since otherwise the error
    // estimate is undefined.
    for (auto &patchdata : ghext->patchdata)
      patchdata.amrcore->cactus_is_initialized = true;

    for (;;) {
      const int level = ghext->num_levels() - 1;
#pragma omp critical
      CCTK_VINFO("Initializing level %d...", level);

      // Check whether a patch has too many levels
      for (const auto &patchdata : ghext->patchdata)
        assert(patchdata.amrcore->finestLevel() <= level);

      assert(!active_levels);
      active_levels = make_optional<active_levels_t>(0, level + 1);

      InputGH(cctkGH);
      CCTK_Traverse(cctkGH, "CCTK_INITIAL");
      CCTK_Traverse(cctkGH, "CCTK_POSTINITIAL");
      CCTK_Traverse(cctkGH, "CCTK_POSTPOSTINITIAL");

      active_levels = optional<active_levels_t>();

      // Regrid
      bool did_modify_any_level;
      {
#pragma omp critical
        CCTK_VINFO("Regridding...");
        static Timer timer("InitialiseRegrid [refined]");
        Interval interval(timer);

        for (const auto &patchdata : ghext->patchdata) {

          const int old_numlevels = patchdata.amrcore->finestLevel() + 1;
          patchdata.amrcore->level_modified.clear();
          patchdata.amrcore->level_modified.resize(old_numlevels, false);
          const CCTK_REAL time = 0; // dummy time
          patchdata.amrcore->regrid(0, time);

          const int new_numlevels = patchdata.amrcore->finestLevel() + 1;
          const int max_numlevels = patchdata.amrcore->maxLevel() + 1;
          assert(new_numlevels >= 0 && new_numlevels <= max_numlevels);
          assert(new_numlevels == old_numlevels ||
                 new_numlevels == old_numlevels + 1);

#pragma omp critical
          {
            const double pts0 =
                patchdata.leveldata.at(0).fab->boxArray().d_numPts();
            for (const auto &leveldata : patchdata.leveldata) {
              const int sz = leveldata.fab->size();
              const double pts = leveldata.fab->boxArray().d_numPts();
              if (leveldata.level == 0) {
                CCTK_VINFO(
                    "  level %d: %d boxes, %.0f cells (%.4g%%)",
                    leveldata.level, sz, pts,
                    100 * pts /
                        (ldexp(CCTK_REAL(1), dim * leveldata.level) * pts0));
              } else {
                const double ptsc = patchdata.leveldata.at(leveldata.level - 1)
                                        .fab->boxArray()
                                        .d_numPts();
                CCTK_VINFO(
                    "  level %d: %d boxes, %.0f cells (%.4g%%, %.0f%%)",
                    leveldata.level, sz, pts,
                    100 * pts /
                        (ldexp(CCTK_REAL(1), dim * leveldata.level) * pts0),
                    100 * pts / (ldexp(CCTK_REAL(1), dim) * ptsc));
              }
            }
          } // omp critical
        }   // for patchdata

        int first_modified_level = INT_MAX;
        int last_modified_level = -1;
        for (const auto &patchdata : ghext->patchdata) {
          for (int lev = 0; lev < int(patchdata.amrcore->level_modified.size());
               ++lev) {
            if (patchdata.amrcore->level_modified.at(lev)) {
              first_modified_level = min(first_modified_level, lev);
              last_modified_level = max(last_modified_level, lev);
            }
          }
        }
        did_modify_any_level = last_modified_level >= first_modified_level;

        if (did_modify_any_level) {
          // Determine time step size
          {
            CCTK_REAL mindx = 1.0 / 0.0;
            for (const auto &patchdata : ghext->patchdata) {
              const amrex::Geometry &geom = patchdata.amrcore->Geom(0);
              const CCTK_REAL *restrict const dx = geom.CellSize();
              CCTK_REAL mindx1 = 1.0 / 0.0;
              for (int d = 0; d < dim; ++d)
                mindx1 = fmin(mindx1, dx[d]);
              mindx1 = ldexp(mindx1, -(int(patchdata.leveldata.size()) - 1));
              mindx = fmin(mindx, mindx1);
            }
            cctkGH->cctk_delta_time = dtfac * mindx;
#pragma omp critical
            CCTK_VINFO("Iteration: %d   time: %g   delta_time: %g",
                       cctkGH->cctk_iteration, double(cctkGH->cctk_time),
                       double(cctkGH->cctk_delta_time));
          }

          assert(!active_levels);
          active_levels = make_optional<active_levels_t>(
              first_modified_level, last_modified_level + 1);
          CCTK_Traverse(cctkGH, "CCTK_BASEGRID");
          CCTK_Traverse(cctkGH, "CCTK_POSTREGRID");
          active_levels = optional<active_levels_t>();
        }
      } // Regrid

      if (!did_modify_any_level)
        break;
    } // for level
  }
#pragma omp critical
  CCTK_VINFO("Initialized %d levels", ghext->num_levels());

  assert(!active_levels);
  active_levels = make_optional<active_levels_t>(0, ghext->num_levels());

  if (!restrict_during_sync) {
    // Restrict
    assert(active_levels);
    active_levels->loop_fine_to_coarse([&](const auto &leveldata) {
      if (leveldata.level != ghext->num_levels() - 1)
        Restrict(cctkGH, leveldata.level);
    });
    // Prolongation
    active_levels->loop_coarse_to_fine([&](const auto &leveldata) {
      if (leveldata.level != 0)
        SyncAfterRestrict(cctkGH, leveldata.level);
    });
    CCTK_Traverse(cctkGH, "CCTK_POSTRESTRICT");
  }

  // Checkpoint, analysis, output
  CCTK_Traverse(cctkGH, "CCTK_POSTSTEP");
  CCTK_Traverse(cctkGH, "CCTK_CPINITIAL");
  CCTK_Traverse(cctkGH, "CCTK_ANALYSIS");
  CCTK_OutputGH(cctkGH);

  active_levels = optional<active_levels_t>();

  return 0;
} // namespace CarpetX

bool EvolutionIsDone(cGH *restrict const cctkGH) {
  DECLARE_CCTK_PARAMETERS;

  if (terminate_next || CCTK_TerminationReached(cctkGH))
    return true;

  if (CCTK_Equals(terminate, "never"))
    return false;

  const bool max_iteration_reached = cctkGH->cctk_iteration >= cctk_itlast;

  const bool max_simulation_time_reached =
      cctk_initial_time < cctk_final_time
          ? cctkGH->cctk_time >= cctk_final_time
          : cctkGH->cctk_time <= cctk_final_time;

  int runtime = CCTK_RunTime();
  MPI_Bcast(&runtime, 1, MPI_INT, 0, MPI_COMM_WORLD);
  const bool max_runtime_reached = runtime >= 60 * max_runtime;

  if (CCTK_Equals(terminate, "iteration"))
    return max_iteration_reached;
  if (CCTK_Equals(terminate, "time"))
    return max_simulation_time_reached;
  if (CCTK_Equals(terminate, "runtime"))
    return max_runtime_reached;
  if (CCTK_Equals(terminate, "any"))
    return max_iteration_reached || max_simulation_time_reached ||
           max_runtime_reached;
  if (CCTK_Equals(terminate, "all"))
    return max_iteration_reached && max_simulation_time_reached &&
           max_runtime_reached;
  if (CCTK_Equals(terminate, "either"))
    return max_iteration_reached || max_simulation_time_reached;
  if (CCTK_Equals(terminate, "both"))
    return max_iteration_reached && max_simulation_time_reached;

  assert(0);
}

void InvalidateTimelevels(cGH *restrict const cctkGH) {
  DECLARE_CCTK_PARAMETERS;

  static Timer timer("InvalidateTimelevels");
  Interval interval(timer);

  // TODO: Parallelize over groups
  const int num_groups = CCTK_NumGroups();
  for (int gi = 0; gi < num_groups; ++gi) {
    cGroup group;
    int ierr = CCTK_GroupData(gi, &group);
    assert(!ierr);

    if (group.grouptype == CCTK_GF) {
      const auto &patchdata0 = ghext->patchdata.at(0);
      const auto &leveldata0 = patchdata0.leveldata.at(0);
      const auto &groupdata0 = *leveldata0.groupdata.at(gi);
      if (!groupdata0.do_checkpoint) {
        const int ntls0 = groupdata0.mfab.size();
        assert(active_levels);
        active_levels->loop_serially([&](const auto &restrict leveldata) {
          auto &restrict groupdata = *leveldata.groupdata.at(gi);
          // Invalidate all time levels
          const int ntls = groupdata.mfab.size();
          assert(ntls == ntls0);
          for (int tl = 0; tl < ntls; ++tl)
            for (int vi = 0; vi < groupdata.numvars; ++vi)
              groupdata.valid.at(tl).at(vi).set_all(valid_t(), []() {
                return "InvalidateTimelevels (invalidate all non-checkpointed "
                       "variables)";
              });
        });
        // TODO: Parallelize over timelevels and variables
        for (int tl = 0; tl < ntls0; ++tl)
          for (int vi = 0; vi < groupdata0.numvars; ++vi)
            poison_invalid_gf(*active_levels, gi, vi, tl);
      }
    } else { // CCTK_ARRAY or CCTK_SCALAR

      auto &restrict globaldata = ghext->globaldata;
      auto &restrict arraygroupdata = *globaldata.arraygroupdata.at(gi);
      if (!arraygroupdata.do_checkpoint) {
        // Invalidate all time levels
        const int ntls = arraygroupdata.data.size();
        for (int tl = 0; tl < ntls; ++tl)
          for (int vi = 0; vi < arraygroupdata.numvars; ++vi)
            // TODO: handle this more nicely
            arraygroupdata.valid.at(tl).at(vi).set_int(false, []() {
              return "InvalidateTimelevels (invalidate all non-checkpointed "
                     "variables)";
            });
        // TODO: Parallelize over timelevels and variables
        for (int tl = 0; tl < ntls; ++tl)
          for (int vi = 0; vi < arraygroupdata.numvars; ++vi)
            poison_invalid_ga(gi, vi, tl);
      }
    }

  } // for gi
}

void CycleTimelevels(cGH *restrict const cctkGH) {
  DECLARE_CCTK_PARAMETERS;

  static Timer timer("CycleTimelevels");
  Interval interval(timer);

  update_cctkGHs(cctkGH);

  // TODO: Parallelize over groups
  const int num_groups = CCTK_NumGroups();
  for (int gi = 0; gi < num_groups; ++gi) {
    cGroup group;
    int ierr = CCTK_GroupData(gi, &group);
    assert(!ierr);

    if (group.grouptype == CCTK_GF) {
      const auto &patchdata0 = ghext->patchdata.at(0);
      const auto &leveldata0 = patchdata0.leveldata.at(0);
      const auto &groupdata0 = *leveldata0.groupdata.at(gi);
      const int ntls0 = groupdata0.mfab.size();
      const nan_handling_t nan_handling = groupdata0.do_checkpoint
                                              ? nan_handling_t::forbid_nans
                                              : nan_handling_t::allow_nans;

      assert(active_levels);
      active_levels->loop_serially([&](auto &restrict leveldata) {
        auto &restrict groupdata = *leveldata.groupdata.at(gi);
        const int ntls = groupdata.mfab.size();
        assert(ntls == ntls0);
        // Rotate time levels and invalidate current time level
        if (ntls > 1) {
          rotate(groupdata.mfab.begin(), groupdata.mfab.end() - 1,
                 groupdata.mfab.end());
          rotate(groupdata.valid.begin(), groupdata.valid.end() - 1,
                 groupdata.valid.end());
          for (int vi = 0; vi < groupdata.numvars; ++vi)
            groupdata.valid.at(0).at(vi).set_all(valid_t(), []() {
              return "CycletimeLevels (invalidate current time level)";
            });
        }
        // All time levels (except the current) must be valid everywhere for
        // checkpointed groups
        if (groupdata.do_checkpoint)
          for (int tl = (ntls == 1 ? 0 : 1); tl < ntls; ++tl)
            for (int vi = 0; vi < groupdata.numvars; ++vi)
              error_if_invalid(groupdata, vi, tl, make_valid_all(), []() {
                return "CycleTimelevels for the state vector";
              });
      });
      for (int vi = 0; vi < groupdata0.numvars; ++vi) {
        if (ntls0 > 1)
          poison_invalid_gf(*active_levels, gi, vi, 0);
        for (int tl = 0; tl < ntls0; ++tl)
          check_valid_gf(*active_levels, gi, vi, tl, nan_handling,
                         []() { return "CycleTimelevels"; });
      }
    } else { // CCTK_ARRAY or CCTK_SCALAR

      auto &restrict globaldata = ghext->globaldata;
      auto &restrict arraygroupdata = *globaldata.arraygroupdata.at(gi);
      const nan_handling_t nan_handling = arraygroupdata.do_checkpoint
                                              ? nan_handling_t::forbid_nans
                                              : nan_handling_t::allow_nans;
      const int ntls = arraygroupdata.data.size();
      // Rotate time levels and invalidate current time level
      if (ntls > 1) {
        rotate(arraygroupdata.data.begin(), arraygroupdata.data.end() - 1,
               arraygroupdata.data.end());
        rotate(arraygroupdata.valid.begin(), arraygroupdata.valid.end() - 1,
               arraygroupdata.valid.end());
        for (int vi = 0; vi < arraygroupdata.numvars; ++vi) {
          arraygroupdata.valid.at(0).at(vi).set_int(false, []() {
            return "CycletimeLevels (invalidate current time level)";
          });
          poison_invalid_ga(gi, vi, 0);
        }
      }
      for (int tl = 0; tl < ntls; ++tl)
        for (int vi = 0; vi < arraygroupdata.numvars; ++vi)
          check_valid_ga(gi, vi, tl, nan_handling,
                         []() { return "CycleTimelevels"; });
    }

  } // for gi
}

// Schedule evolution
int Evolve(tFleshConfig *config) {
  DECLARE_CCTK_PARAMETERS;

  static Timer timer("Evolve");
  Interval interval(timer);

  assert(config);
  cGH *restrict const cctkGH = config->GH[0];
  assert(cctkGH);

#pragma omp critical
  CCTK_VINFO("Starting evolution...");

  double total_evolution_time = 0;
  double total_evolution_output_time = 0;
  int total_iterations = 0;
  double total_cell_updates = 0;

  std::ofstream performance_file;
  if (out_performance && CCTK_MyProc(NULL) == 0) {
    const int every =
        out_performance_every == -1 ? out_every : out_performance_every;
    if (every > 0) {
      std::ostringstream buf;
      buf << out_dir << "/performance.yaml";
      const std::string filename = buf.str();
      performance_file.open(filename);
      performance_file << "performance:\n" << flush;
    }
  }

  while (!EvolutionIsDone(cctkGH)) {

    const double start_time = gettime();

    assert(!active_levels);

    // TODO: Move regridding into a function
    if (regrid_every > 0 && cctkGH->cctk_iteration % regrid_every == 0) {
#pragma omp critical
      CCTK_VINFO("Regridding...");
      static Timer timer("EvolveRegrid");
      Interval interval(timer);

      for (const auto &patchdata : ghext->patchdata) {
        const int old_numlevels = patchdata.amrcore->finestLevel() + 1;
        patchdata.amrcore->level_modified.clear();
        patchdata.amrcore->level_modified.resize(old_numlevels, false);
        const CCTK_REAL time = 0; // dummy time
        patchdata.amrcore->regrid(0, time);

        const int new_numlevels = patchdata.amrcore->finestLevel() + 1;
        const int max_numlevels = patchdata.amrcore->maxLevel() + 1;
        assert(new_numlevels >= 0 && new_numlevels <= max_numlevels);

#pragma omp critical
        {
          CCTK_VINFO("  old levels %d, new levels %d", old_numlevels,
                     new_numlevels);
          double pts0 = patchdata.leveldata.at(0).fab->boxArray().d_numPts();
          assert(!active_levels);
          for (const auto &leveldata : patchdata.leveldata) {
            const int sz = leveldata.fab->size();
            const double pts = leveldata.fab->boxArray().d_numPts();
            if (leveldata.level == 0) {
              CCTK_VINFO(
                  "  level %d: %d boxes, %.0f cells (%.4g%%)", leveldata.level,
                  sz, pts,
                  100 * pts /
                      (ldexp(CCTK_REAL(1), dim * leveldata.level) * pts0));
            } else {
              const double ptsc = patchdata.leveldata.at(leveldata.level - 1)
                                      .fab->boxArray()
                                      .d_numPts();
              CCTK_VINFO(
                  "  level %d: %d boxes, %.0f cells (%.4g%%, %.0f%%)",
                  leveldata.level, sz, pts,
                  100 * pts /
                      (ldexp(CCTK_REAL(1), dim * leveldata.level) * pts0),
                  100 * pts / (ldexp(CCTK_REAL(1), dim) * ptsc));
            }
          }
        } // omp critical
      }   // for patchdata

      int first_modified_level = INT_MAX;
      int last_modified_level = -1;
      for (const auto &patchdata : ghext->patchdata) {
        for (int lev = 0; lev < int(patchdata.amrcore->level_modified.size());
             ++lev) {
          if (patchdata.amrcore->level_modified.at(lev)) {
            first_modified_level = min(first_modified_level, lev);
            last_modified_level = max(last_modified_level, lev);
          }
        }
      }
      const bool did_modify_any_level =
          last_modified_level >= first_modified_level;

      if (did_modify_any_level) {
        // Determine time step size
        {
          CCTK_REAL mindx = 1.0 / 0.0;
          for (const auto &patchdata : ghext->patchdata) {
            const amrex::Geometry &geom = patchdata.amrcore->Geom(0);
            const CCTK_REAL *restrict const dx = geom.CellSize();
            CCTK_REAL mindx1 = 1.0 / 0.0;
            for (int d = 0; d < dim; ++d)
              mindx1 = fmin(mindx1, dx[d]);
            mindx1 = ldexp(mindx1, -(int(patchdata.leveldata.size()) - 1));
            mindx = fmin(mindx, mindx1);
          }
          cctkGH->cctk_delta_time = dtfac * mindx;
#pragma omp critical
          CCTK_VINFO("Iteration: %d   time: %g   delta_time: %g",
                     cctkGH->cctk_iteration, double(cctkGH->cctk_time),
                     double(cctkGH->cctk_delta_time));
        }

        assert(!active_levels);
        active_levels = make_optional<active_levels_t>(first_modified_level,
                                                       last_modified_level + 1);
        CCTK_Traverse(cctkGH, "CCTK_BASEGRID");
        CCTK_Traverse(cctkGH, "CCTK_POSTREGRID");
        active_levels = optional<active_levels_t>();
      }
    } // Regrid

    // Find smallest iteration number. Levels at this iteration will
    // be evolved.
    rat64 iteration = ghext->patchdata.at(0).leveldata.at(0).iteration;
    using std::min;
    for (const auto &patchdata : ghext->patchdata)
      for (const auto &leveldata : patchdata.leveldata)
        iteration = min(iteration, leveldata.iteration);

    cctkGH->cctk_iteration += 1;

    // Loop over all levels, in batches that combine levels that don't
    // subcycle. The level range is [min_level, max_level).
    for (int min_level = 0, max_level = min_level + 1;
         min_level < ghext->num_levels();
         min_level = max_level, max_level = min_level + 1) {
      // Find end of batch
      while (max_level < ghext->num_levels()) {
        bool level_is_subcycling_level = false;
        for (const auto &patchdata : ghext->patchdata)
          if (max_level < int(patchdata.leveldata.size()))
            level_is_subcycling_level |=
                patchdata.leveldata.at(max_level).is_subcycling_level;
        if (level_is_subcycling_level)
          break;
        ++max_level;
      }

      // Skip this batch of levels if it is not active at the current
      // iteration
      rat64 level_iteration = -1;
      rat64 level_delta_iteration = -1;
      for (const auto &patchdata : ghext->patchdata) {
        if (min_level < int(patchdata.leveldata.size())) {
          level_iteration = patchdata.leveldata.at(min_level).iteration;
          level_delta_iteration =
              patchdata.leveldata.at(min_level).delta_iteration;
          break;
        }
      }
      assert(level_iteration != -1);
      assert(level_delta_iteration != -1);
      // TODO: if a break is always ok (eg if subcycling factors are strange
      // or if a for() loop with a continue is required.
      if (level_iteration > iteration)
        continue;

      // must not terminate loop iteration due to active_levels being reset at
      // bootom of loop body
      active_levels = make_optional<active_levels_t>(min_level, max_level);

      // Advance iteration number on this batch of levels
      level_iteration += level_delta_iteration;
      active_levels->loop_serially([&](auto &restrict leveldata) {
        leveldata.iteration += leveldata.delta_iteration;
        assert(level_iteration == leveldata.iteration);
        assert(level_delta_iteration == leveldata.delta_iteration);
      });

      // We cannot invalidate all non-evolved variables. ODESolvers
      // calculates things in ODESolvers_Poststep, and we want to use
      // them in the next iteration.
      // InvalidateTimelevels(cctkGH);

      CycleTimelevels(cctkGH);

      cctkGH->cctk_time =
          (use_subcycling_wip)
              ? cctkGH->cctk_delta_time * double(level_iteration)
              : cctkGH->cctk_time + cctkGH->cctk_delta_time;

      CCTK_Traverse(cctkGH, "CCTK_PRESTEP");
      CCTK_Traverse(cctkGH, "CCTK_EVOL");

      // Reflux
      // TODO: These loop bounds are wrong for subcycling
      assert(active_levels);
      for (int level = ghext->num_levels() - 2; level >= 0; --level)
        Reflux(cctkGH, level);

      if (!restrict_during_sync) {
        // Restrict
        for (int level = min_level - 1; level >= 0; --level) {
          rat64 coarse_iteration =
              ghext->patchdata.at(0).leveldata.at(level).iteration;
          if (coarse_iteration == level_iteration)
            Restrict(cctkGH, level);
          else
            break;
        }
        // Prolongation
        for (int level = 1; level < max_level; ++level) {
          rat64 coarse_iteration =
              ghext->patchdata.at(0).leveldata.at(level - 1).iteration;
          if (coarse_iteration == level_iteration)
            SyncAfterRestrict(cctkGH, level);
          else
            continue;
        }
        CCTK_Traverse(cctkGH, "CCTK_POSTRESTRICT");
      }

      for (int level = min_level - 1; level >= 0; --level) {
        rat64 coarse_iteration =
            ghext->patchdata.at(0).leveldata.at(level).iteration;
        if (coarse_iteration == level_iteration)
          min_level = level;
        else
          break;
      }
      active_levels = make_optional<active_levels_t>(min_level, max_level);
      CCTK_Traverse(cctkGH, "CCTK_POSTSTEP");
      CCTK_Traverse(cctkGH, "CCTK_CHECKPOINT");
      CCTK_Traverse(cctkGH, "CCTK_ANALYSIS");
      const double output_start_time = gettime();
      CCTK_OutputGH(cctkGH);
      const double output_finish_time = gettime();
      total_evolution_output_time += output_finish_time - output_start_time;

      active_levels = optional<active_levels_t>();
    } // while min_level

    const double finish_time = gettime();
    double num_cells = 0;
    for (const auto &patch : ghext->patchdata)
      for (const auto &level : patch.leveldata)
        num_cells += level.fab->boxArray().d_numPts();
    total_cell_updates += num_cells;
    ++total_iterations;
    const double iteration_time = finish_time - start_time;
    total_evolution_time += iteration_time;
    const double iterations_per_second = 1 / iteration_time;
    const double cell_updates_per_second = num_cells * iterations_per_second;
    const double total_evolution_compute_time =
        total_evolution_time - total_evolution_output_time;
    if (verbose) {
      CCTK_VINFO("Simulation time: %g   "
                 "Iterations per second: %g   "
                 "Simulation time per second: %g",
                 double(cctkGH->cctk_time), iterations_per_second,
                 double(cctkGH->cctk_delta_time * iterations_per_second)

      );
      // This is the same as H-AMR's "cell updates per second":
      CCTK_VINFO("Grid cells: %g   "
                 "Grid cell updates per second: %g",
                 num_cells, cell_updates_per_second);

      CCTK_VINFO("Performance:");
      CCTK_VINFO("  total evolution time:            %g sec",
                 total_evolution_time);
      CCTK_VINFO("  total evolution compute time:    %g sec",
                 total_evolution_compute_time);
      CCTK_VINFO("  total evolution output time:     %g sec",
                 total_evolution_output_time);
      CCTK_VINFO("  total iterations:                %d", total_iterations);
      CCTK_VINFO("  total cells updated:             %g", total_cell_updates);
      CCTK_VINFO("  average iterations per second: %g",
                 total_iterations / total_evolution_time);
      CCTK_VINFO("  average cell updates per second: %g",
                 total_cell_updates / total_evolution_time);
    }
    // TODO: Output this in a proper I/O method
    if (out_performance && CCTK_MyProc(NULL) == 0) {
      const int every =
          out_performance_every == -1 ? out_every : out_performance_every;
      if (every > 0 && cctkGH->cctk_iteration % every == 0)
        performance_file << "  " << total_iterations << ":\n"
                         << "    evolution-seconds: " << total_evolution_time
                         << "\n"
                         << "    evolution-compute-seconds: "
                         << total_evolution_compute_time << "\n"
                         << "    evolution-output-seconds: "
                         << total_evolution_output_time << "\n"
                         << "    evolution-cell-updates: " << total_cell_updates
                         << "\n"
                         << "    evolution-iterations: " << total_iterations
                         << "\n"
                         << flush;
    }

  } // main loop

  if (out_performance && CCTK_MyProc(NULL) == 0)
    performance_file.close();

  return 0;
} // namespace CarpetX

// Schedule shutdown
int Shutdown(tFleshConfig *config) {
  assert(config);
  cGH *restrict const cctkGH = config->GH[0];
  assert(cctkGH);

  static Timer timer("Shutdown");
  Interval interval(timer);

#pragma omp critical
  CCTK_VINFO("Shutting down...");

  assert(!active_levels);
  active_levels = make_optional<active_levels_t>();

  CCTK_Traverse(cctkGH, "CCTK_TERMINATE");

  active_levels = optional<active_levels_t>();
  active_levels = make_optional<active_levels_t>(0, 0);

  CCTK_Traverse(cctkGH, "CCTK_SHUTDOWN");

  active_levels = optional<active_levels_t>();
  assert(!ghext);

  return 0;
}

// Call a scheduled function
int CallFunction(void *function, cFunctionData *restrict attribute,
                 void *data) {
  DECLARE_CCTK_PARAMETERS;

  assert(function);
  assert(attribute);
  assert(data);

  cGH *restrict const cctkGH = static_cast<cGH *>(data);

  if (verbose)
#pragma omp critical
    CCTK_VINFO("CallFunction iteration %d %s: %s::%s", cctkGH->cctk_iteration,
               attribute->where, attribute->thorn, attribute->routine);

  static map<cFunctionData *restrict, Timer> timers;

  map<cFunctionData *restrict, Timer>::iterator timer_iter;
#pragma omp critical(CarpetX_CallFunction)
  {
    timer_iter = timers.find(attribute);
    if (timer_iter == timers.end()) {
      ostringstream buf;
      buf << "CallFunction " << attribute->where << ": " << attribute->thorn
          << "::" << attribute->routine;
      timer_iter = get<0>(timers.emplace(attribute, buf.str()));
    }
  }
  Timer &timer = timer_iter->second;
  Interval interval(timer);

  assert(active_levels);

  if (CCTK_EQUALS(presync_mode, "presync-only")) {
    const vector<clause_t> &reads = decode_clauses(attribute, rdwr_t::read);
    std::set<int> sync_set;
    for (const auto &rd : reads) {
      if (CCTK_GroupTypeI(rd.gi) == CCTK_GF) {

        active_levels->loop_serially([&](const auto &restrict leveldata) {
          const auto &restrict groupdata = *leveldata.groupdata.at(rd.gi);
          const valid_t &need = rd.valid;
          valid_t have = groupdata.valid.at(rd.tl).at(rd.vi).get();
          if (need.valid_ghosts && !have.valid_ghosts && have.valid_int)
            sync_set.insert(rd.gi);
        });
      }
    }
    if (!sync_set.empty()) {
      std::vector<int> sync_vec(sync_set.begin(), sync_set.end());
      SyncGroupsByDirI(cctkGH, sync_vec.size(), sync_vec.data(), nullptr);
    }
  }

  // Check whether input variables have valid data
  {
    const vector<clause_t> &reads = decode_clauses(attribute, rdwr_t::read);
    for (const auto &rd : reads) {
      if (CCTK_GroupTypeI(rd.gi) == CCTK_GF) {
        const auto &patchdata0 = ghext->patchdata.at(0);
        const auto &leveldata0 = patchdata0.leveldata.at(0);
        const auto &groupdata0 = *leveldata0.groupdata.at(rd.gi);
        const nan_handling_t nan_handling = groupdata0.do_checkpoint
                                                ? nan_handling_t::forbid_nans
                                                : nan_handling_t::allow_nans;

        active_levels->loop_serially([&](const auto &restrict leveldata) {
          const auto &restrict groupdata = *leveldata.groupdata.at(rd.gi);
          const valid_t &need = rd.valid;
          error_if_invalid(
              groupdata, rd.vi, rd.tl, need, [attribute, cctkGH]() {
                ostringstream buf;
                buf << "CallFunction iteration " << cctkGH->cctk_iteration
                    << " " << attribute->where << ": " << attribute->thorn
                    << "::" << attribute->routine << " checking input";
                return buf.str();
              });
        });
        check_valid_gf(*active_levels, rd.gi, rd.vi, rd.tl, nan_handling,
                       [attribute, cctkGH]() {
                         ostringstream buf;
                         buf << "CallFunction iteration "
                             << cctkGH->cctk_iteration << " "
                             << attribute->where << ": " << attribute->thorn
                             << "::" << attribute->routine << " checking input";
                         return buf.str();
                       });
      } else { // CCTK_ARRAY or CCTK_SCALAR

        const auto &restrict arraygroupdata =
            *ghext->globaldata.arraygroupdata.at(rd.gi);
        const nan_handling_t nan_handling = arraygroupdata.do_checkpoint
                                                ? nan_handling_t::forbid_nans
                                                : nan_handling_t::allow_nans;
        const valid_t &need = rd.valid;
        error_if_invalid(
            arraygroupdata, rd.vi, rd.tl, need, [attribute, cctkGH]() {
              ostringstream buf;
              buf << "CallFunction iteration " << cctkGH->cctk_iteration << " "
                  << attribute->where << ": " << attribute->thorn
                  << "::" << attribute->routine << " checking input";
              return buf.str();
            });
        check_valid_ga(
            rd.gi, rd.vi, rd.tl, nan_handling, [attribute, cctkGH]() {
              ostringstream buf;
              buf << "CallFunction iteration " << cctkGH->cctk_iteration << " "
                  << attribute->where << ": " << attribute->thorn
                  << "::" << attribute->routine << " checking input";
              return buf.str();
            });
      }
    }
  }

  // Poison those output variables that are not input variables
  if (poison_undefined_values) {
    map<clause_t, valid_t> isread;
    const vector<clause_t> &reads = decode_clauses(attribute, rdwr_t::read);
    for (const auto &rd : reads) {
      clause_t cl = rd;
      cl.valid = valid_t();
      assert(isread.count(cl) == 0);
      isread[cl] = rd.valid;
    }
    const vector<clause_t> &writes = decode_clauses(attribute, rdwr_t::write);
    for (const auto &wr : writes) {
      clause_t cl = wr;
      cl.valid = valid_t();
      valid_t need;
      if (isread.count(cl) > 0)
        need = isread[cl];

      if (CCTK_GroupTypeI(wr.gi) == CCTK_GF) {

        active_levels->loop_serially([&](auto &restrict leveldata) {
          auto &restrict groupdata = *leveldata.groupdata.at(wr.gi);
          const valid_t &provided = wr.valid;
          groupdata.valid.at(wr.tl).at(wr.vi).set_invalid(
              provided & ~need,
              [iteration = cctkGH->cctk_iteration, where = attribute->where,
               thorn = attribute->thorn, routine = attribute->routine] {
                ostringstream buf;
                buf << "CallFunction iteration " << iteration << " " << where
                    << ": " << thorn << "::" << routine
                    << ": Poison output variables that are not input variables";
                return buf.str();
              });
        });
        poison_invalid_gf(*active_levels, wr.gi, wr.vi, wr.tl);
      } else { // CCTK_ARRAY or CCTK_SCALAR
        auto &restrict arraygroupdata =
            *ghext->globaldata.arraygroupdata.at(wr.gi);
        const valid_t &provided = wr.valid;
        arraygroupdata.valid.at(wr.tl).at(wr.vi).set_invalid(
            provided & ~need,
            [iteration = cctkGH->cctk_iteration, where = attribute->where,
             thorn = attribute->thorn, routine = attribute->routine] {
              ostringstream buf;
              buf << "CallFunction iteration " << iteration << " " << where
                  << ": " << thorn << "::" << routine
                  << ": Poison output variables that are not input variables";
              return buf.str();
            });
        poison_invalid_ga(wr.gi, wr.vi, wr.tl);
      }
    }
  }

  // Calculate checksums over variables that are not written
  checksums_t checksums;
  if (poison_undefined_values) {
    const vector<clause_t> &writes = decode_clauses(attribute, rdwr_t::write);
    const int numgroups = CCTK_NumGroups();
    vector<vector<vector<valid_t> > > gfs(numgroups);
    for (int gi = 0; gi < numgroups; ++gi) {
      const int numvars = CCTK_NumVarsInGroupI(gi);
      gfs.at(gi).resize(numvars);
      for (int vi = 0; vi < numvars; ++vi) {
        const int numtimelevels = 1; // is expanded later if necessary
        gfs.at(gi).at(vi).resize(numtimelevels);
      }
    }
    for (const auto &wr : writes) {
      if (wr.tl >= int(gfs.at(wr.gi).at(wr.vi).size()))
        gfs.at(wr.gi).at(wr.vi).resize(wr.tl + 1);
      gfs.at(wr.gi).at(wr.vi).at(wr.tl) |= wr.valid;
    }

    checksums = calculate_checksums(gfs);
  }

  const mode_t mode = decode_mode(attribute);
  switch (mode) {
  case mode_t::local:
    // Call function once per tile
    active_levels->loop_parallel([&](int patch, int level, int index,
                                     int component, const cGH *local_cctkGH) {
      update_cctkGH(const_cast<cGH *>(local_cctkGH), cctkGH);
      CCTK_CallFunction(function, attribute, const_cast<cGH *>(local_cctkGH));
    });
    synchronize();
    break;

  case mode_t::meta:
  case mode_t::global:
  case mode_t::level:
    // Call function just once
    // Note: meta mode scheduling must continue to work even after we
    // shut down ourselves!
    CCTK_CallFunction(function, attribute, cctkGH);
    break;

  default:
    assert(0);
  }

  // Check checksums
  if (poison_undefined_values)
    check_checksums(checksums, [attribute, cctkGH]() {
      ostringstream buf;
      buf << "CallFunction iteration " << cctkGH->cctk_iteration << " "
          << attribute->where << ": " << attribute->thorn
          << "::" << attribute->routine << " checking output";
      return buf.str();
    });

  // Mark output variables as having valid data
  {
    const vector<clause_t> &writes = decode_clauses(attribute, rdwr_t::write);
    for (const auto &wr : writes) {
      if (CCTK_GroupTypeI(wr.gi) == CCTK_GF) {
        const auto &patchdata0 = ghext->patchdata.at(0);
        const auto &leveldata0 = patchdata0.leveldata.at(0);
        const auto &groupdata0 = *leveldata0.groupdata.at(wr.gi);
        const nan_handling_t nan_handling = groupdata0.do_checkpoint
                                                ? nan_handling_t::forbid_nans
                                                : nan_handling_t::allow_nans;

        active_levels->loop_serially([&](auto &restrict leveldata) {
          auto &restrict groupdata = *leveldata.groupdata.at(wr.gi);
          const valid_t &provided = wr.valid;
          groupdata.valid.at(wr.tl).at(wr.vi).set_valid(
              provided,
              [iteration = cctkGH->cctk_iteration, where = attribute->where,
               thorn = attribute->thorn, routine = attribute->routine] {
                ostringstream buf;
                buf << "CallFunction iteration " << iteration << " " << where
                    << ": " << thorn << "::" << routine
                    << ": Mark output variables as valid";
                return buf.str();
              });
        });
        check_valid_gf(*active_levels, wr.gi, wr.vi, wr.tl, nan_handling,
                       [attribute, cctkGH]() {
                         ostringstream buf;
                         buf << "CallFunction iteration "
                             << cctkGH->cctk_iteration << " "
                             << attribute->where << ": " << attribute->thorn
                             << "::" << attribute->routine
                             << " checking output";
                         return buf.str();
                       });
      } else { // CCTK_ARRAY or CCTK_SCALAR
        auto &restrict arraygroupdata =
            *ghext->globaldata.arraygroupdata.at(wr.gi);
        const nan_handling_t nan_handling = arraygroupdata.do_checkpoint
                                                ? nan_handling_t::forbid_nans
                                                : nan_handling_t::allow_nans;
        const valid_t &provided = wr.valid;
        arraygroupdata.valid.at(wr.tl).at(wr.vi).set_valid(
            provided,
            [iteration = cctkGH->cctk_iteration, where = attribute->where,
             thorn = attribute->thorn, routine = attribute->routine] {
              ostringstream buf;
              buf << "CallFunction iteration " << iteration << " " << where
                  << ": " << thorn << "::" << routine
                  << ": Mark output variables as valid";
              return buf.str();
            });
        check_valid_ga(
            wr.gi, wr.vi, wr.tl, nan_handling, [attribute, cctkGH]() {
              ostringstream buf;
              buf << "CallFunction iteration " << cctkGH->cctk_iteration << " "
                  << attribute->where << ": " << attribute->thorn
                  << "::" << attribute->routine << " checking output";
              return buf.str();
            });
      }
    }
  }

  // Mark invalid variables as having invalid data
  {
    const vector<clause_t> &invalids =
        decode_clauses(attribute, rdwr_t::invalid);
    for (const auto &inv : invalids) {
      if (CCTK_GroupTypeI(inv.gi) == CCTK_GF) {
        const auto &patchdata0 = ghext->patchdata.at(0);
        const auto &leveldata0 = patchdata0.leveldata.at(0);
        const auto &groupdata0 = *leveldata0.groupdata.at(inv.gi);
        const nan_handling_t nan_handling = groupdata0.do_checkpoint
                                                ? nan_handling_t::forbid_nans
                                                : nan_handling_t::allow_nans;

        active_levels->loop_serially([&](auto &restrict leveldata) {
          auto &restrict groupdata = *leveldata.groupdata.at(inv.gi);
          const valid_t &invalidated = inv.valid;
          groupdata.valid.at(inv.tl).at(inv.vi).set_invalid(
              invalidated,
              [iteration = cctkGH->cctk_iteration, where = attribute->where,
               thorn = attribute->thorn, routine = attribute->routine] {
                ostringstream buf;
                buf << "CallFunction iteration " << iteration << " " << where
                    << ": " << thorn << "::" << routine
                    << ": Mark invalid variables as invalid";
                return buf.str();
              });
        });
        check_valid_gf(*active_levels, inv.gi, inv.vi, inv.tl, nan_handling,
                       [attribute, cctkGH]() {
                         ostringstream buf;
                         buf << "CallFunction iteration "
                             << cctkGH->cctk_iteration << " "
                             << attribute->where << ": " << attribute->thorn
                             << "::" << attribute->routine
                             << " checking output";
                         return buf.str();
                       });
      } else { // CCTK_ARRAY or CCTK_SCALAR
        auto &restrict arraygroupdata =
            *ghext->globaldata.arraygroupdata.at(inv.gi);
        const nan_handling_t nan_handling = arraygroupdata.do_checkpoint
                                                ? nan_handling_t::forbid_nans
                                                : nan_handling_t::allow_nans;
        const valid_t &invalidated = inv.valid;
        arraygroupdata.valid.at(inv.tl).at(inv.vi).set_invalid(
            invalidated,
            [iteration = cctkGH->cctk_iteration, where = attribute->where,
             thorn = attribute->thorn, routine = attribute->routine] {
              ostringstream buf;
              buf << "CallFunction iteration " << iteration << " " << where
                  << ": " << thorn << "::" << routine
                  << ": Mark invalid variables as invalid";
              return buf.str();
            });
        check_valid_ga(
            inv.gi, inv.vi, inv.tl, nan_handling, [attribute, cctkGH]() {
              ostringstream buf;
              buf << "CallFunction iteration " << cctkGH->cctk_iteration << " "
                  << attribute->where << ": " << attribute->thorn
                  << "::" << attribute->routine << " checking output";
              return buf.str();
            });
      }
    }
  }

  constexpr int didsync = 0;
  return didsync;
}

bool sync_active = false; // Catch recursive calls

struct mark_sync_active {
  mark_sync_active() {
    if (sync_active)
      CCTK_ERROR(
          "Recursive call to SyncGroupsByDirI. Maybe you are syncing grid "
          "functions in the \"restrict\" bin while the parameter "
          "\"restrict_during_sync\" is true?");
    sync_active = true;
  }
  ~mark_sync_active() { sync_active = false; }
};

int SyncGroupsByDirI(const cGH *restrict cctkGH, int numgroups,
                     const int *groups0, const int *directions) {
  DECLARE_CCTK_PARAMETERS;

  assert(in_global_mode(cctkGH));

  mark_sync_active marked;

  static Timer timer("Sync");
  Interval interval(timer);

  assert(cctkGH);
  assert(numgroups >= 0);
  assert(groups0);

  if (verbose) {
    ostringstream buf;
    for (int n = 0; n < numgroups; ++n) {
      if (n != 0)
        buf << ", ";
      buf << CCTK_FullGroupName(groups0[n]);
    }
#pragma omp critical
    CCTK_VINFO("SyncGroups %s", buf.str().c_str());
  }

  const int gi_regrid_error = CCTK_GroupIndex("CarpetX::regrid_error");
  assert(gi_regrid_error >= 0);

  vector<int> groups;
  for (int n = 0; n < numgroups; ++n) {
    const int gi = groups0[n];
    if (CCTK_GroupTypeI(gi) != CCTK_GF)
      continue;
    // Don't restrict the regridding error
    if (gi == gi_regrid_error)
      continue;
    groups.push_back(gi);
  }

  // Skip groups that have valid ghosts and boundaries
  if (CCTK_EQUALS(presync_mode, "presync-only")) {
    active_levels->loop_serially([&](auto &restrict leveldata) {
      vector<int> new_groups;
      for (const int gi : groups) {
        auto &restrict groupdata = *leveldata.groupdata.at(gi);
        bool need_sync = false;
        for (int tl = 0; tl < int(groupdata.valid.size()); tl++) {
          if (need_sync)
            break;
          auto &timeleveldata = groupdata.valid.at(tl);
          for (int vi = 0; vi < int(timeleveldata.size()); vi++) {
            if (need_sync)
              break;
            valid_t have = groupdata.valid.at(tl).at(vi).get();
            if (!have.valid_ghosts || !have.valid_outer) {
              need_sync = true;
            }
          }
        }
        if (need_sync) {
          new_groups.push_back(gi);
        }
      }
      groups = new_groups;
    });
    if (groups.size() == 0) {
      return 0;
    }
  }

  if (restrict_during_sync) {
    active_levels->loop_fine_to_coarse([&](const auto &leveldata) {
      if (leveldata.level < ghext->num_levels() - 1)
        Restrict(cctkGH, leveldata.level, groups);
    });
    // FIXME: cannot call POSTRESTRICT since this could contain a SYNC leading
    // to an infinite loop. This means that outer boundaries will be left
    // invalid after an implicit restrict
    // CCTK_Traverse(cctkGH, "CCTK_POSTRESTRICT");
  }

  static const bool have_multipatch_boundaries =
      CCTK_IsFunctionAliased("MultiPatch_Interpolate");

  // Check preconditions
  for (const int gi : groups) {
    const auto &patchdata0 = ghext->patchdata.at(0);
    const auto &leveldata0 = patchdata0.leveldata.at(0);
    const auto &groupdata0 = *leveldata0.groupdata.at(gi);
    const nan_handling_t nan_handling = groupdata0.do_checkpoint
                                            ? nan_handling_t::forbid_nans
                                            : nan_handling_t::allow_nans;
    // We always sync all directions.
    // If there is more than one time level, then we don't sync the
    // oldest.
    // TODO: during evolution, sync only one time level
    const int ntls0 = groupdata0.mfab.size();
    const int sync_tl0 = ntls0 > 1 ? ntls0 - 1 : ntls0;

    active_levels->loop_serially([&](auto &restrict leveldata) {
      auto &restrict groupdata = *leveldata.groupdata.at(gi);

      if (leveldata.level > 0) {

        const int level = leveldata.level;
        const auto &restrict coarseleveldata =
            ghext->patchdata.at(leveldata.patch).leveldata.at(level - 1);
        auto &restrict coarsegroupdata = *coarseleveldata.groupdata.at(gi);
        assert(coarsegroupdata.numvars == groupdata.numvars);

        for (int tl = 0; tl < sync_tl0; ++tl) {
          for (int vi = 0; vi < groupdata.numvars; ++vi) {
            error_if_invalid(coarsegroupdata, vi, tl, make_valid_int(), []() {
              return "SyncGroupsByDirI on coarse level before prolongation";
            });
          }
        } // for tl

      } // if leveldata.level > 0

      for (int tl = 0; tl < sync_tl0; ++tl) {
        for (int vi = 0; vi < groupdata.numvars; ++vi) {
          // Synchronization only uses the interior
          error_if_invalid(groupdata, vi, tl, make_valid_int(),
                           []() { return "SyncGroupsByDirI before syncing"; });
          groupdata.valid.at(tl).at(vi).set_invalid(make_valid_ghosts(), []() {
            return "SyncGroupsByDirI before syncing: "
                   "Mark ghost zones as invalid";
          });
        }
      } // for tl
    });

    active_levels_t active_fine_levels = *active_levels;
    using std::max;
    active_fine_levels.min_level = max(active_fine_levels.min_level, 1);
    for (int tl = 0; tl < sync_tl0; ++tl) {
      for (int vi = 0; vi < groupdata0.numvars; ++vi) {
        check_valid_gf(active_fine_levels, gi, vi, tl, nan_handling, []() {
          return "SyncGroupsByDirI on coarse level before prolongation";
        });
        poison_invalid_gf(*active_levels, gi, vi, tl);
        check_valid_gf(*active_levels, gi, vi, tl, nan_handling,
                       []() { return "SyncGroupsByDirI before syncing"; });
      }
    } // for tl
  }   // for gi

  // We need to loop over groups, patches, and levels in a definite
  // order so that AMReX's communication pattern does not get
  // confused. Therefore all the loops here are serial. The only
  // parallelization happens within AMReX and within our boundary
  // conditions. This is not efficient.

  task_manager tasks1;
  task_manager tasks2;
  task_manager tasks3;

  for (const int gi : groups) {
    active_levels->loop_serially([&](auto &restrict leveldata) {
      auto &restrict groupdata = *leveldata.groupdata.at(gi);

      // We always sync all directions.
      // If there is more than one time level, then we don't sync the
      // oldest.
      // TODO: during evolution, sync only one time level
      const int ntls = groupdata.mfab.size();
      const int sync_tl = ntls > 1 ? ntls - 1 : ntls;

      if (leveldata.level == 0) {
        // Copy from adjacent boxes on same level

        for (int tl = 0; tl < sync_tl; ++tl) {
          tasks1.submit_serially([&tasks2, &tasks3, &leveldata, &groupdata,
                                  tl]() {
            FillPatch_Sync(tasks2, tasks3, groupdata, *groupdata.mfab.at(tl),
                           ghext->patchdata.at(leveldata.patch)
                               .amrcore->Geom(leveldata.level));
          });
        } // for tl

      } else { // if leveldata.level > 0
        // Copy from adjacent boxes on same level, and interpolate
        // from next coarser level

        const int level = leveldata.level;
        const auto &restrict coarseleveldata =
            ghext->patchdata.at(leveldata.patch).leveldata.at(level - 1);
        auto &restrict coarsegroupdata = *coarseleveldata.groupdata.at(gi);
        assert(coarsegroupdata.numvars == groupdata.numvars);

        amrex::Interpolater *const interpolator =
            get_interpolator(groupdata.indextype);

        for (int tl = 0; tl < sync_tl; ++tl) {

          tasks1.submit_serially([&tasks2, &tasks3, &leveldata, &groupdata,
                                  &coarsegroupdata, interpolator, tl]() {
            auto task2 = FillPatch_ProlongateGhosts(
                groupdata, *groupdata.mfab.at(tl), *coarsegroupdata.mfab.at(tl),
                ghext->patchdata.at(leveldata.patch)
                    .amrcore->Geom(leveldata.level - 1),
                ghext->patchdata.at(leveldata.patch)
                    .amrcore->Geom(leveldata.level),
                interpolator, groupdata.bcrecs);
            tasks2.submit([&tasks3, task2 = std::move(task2)]() {
              auto task3 = task2();
              tasks3.submit([task3 = std::move(task3)]() { task3(); });
            });
          });

        } // for tl

      } // if leveldata.level > 0
    });
  } // for gi

  tasks1.run_tasks_serially();
  synchronize();
  tasks2.run_tasks_serially();
  synchronize();
  tasks3.run_tasks_serially();
  synchronize();

  // Check postconditions
  for (const int gi : groups) {
    const auto &patchdata0 = ghext->patchdata.at(0);
    const auto &leveldata0 = patchdata0.leveldata.at(0);
    const auto &groupdata0 = *leveldata0.groupdata.at(gi);
    const nan_handling_t nan_handling = groupdata0.do_checkpoint
                                            ? nan_handling_t::forbid_nans
                                            : nan_handling_t::allow_nans;
    // We always sync all directions.
    // If there is more than one time level, then we don't sync the
    // oldest.
    // TODO: during evolution, sync only one time level
    const int ntls0 = groupdata0.mfab.size();
    const int sync_tl0 = ntls0 > 1 ? ntls0 - 1 : ntls0;

    active_levels->loop_serially([&](auto &restrict leveldata) {
      auto &restrict groupdata = *leveldata.groupdata.at(gi);

      for (int tl = 0; tl < sync_tl0; ++tl) {
        for (int vi = 0; vi < groupdata.numvars; ++vi) {
          groupdata.valid.at(tl).at(vi).set_ghosts(true, []() {
            return "SyncGroupsByDirI after syncing: "
                   "Mark ghost zones as valid";
          });
          if (groupdata.all_faces_have_symmetries_or_boundaries())
            groupdata.valid.at(tl).at(vi).set_outer(true, []() {
              return "SyncGroupsByDirI after syncing: "
                     "Mark outer boundaries as valid";
            });
        }
      } // for tl
    });

    for (int tl = 0; tl < sync_tl0; ++tl) {
      for (int vi = 0; vi < groupdata0.numvars; ++vi) {
        poison_invalid_gf(*active_levels, gi, vi, tl);
        // TODO: Check after applying multi-patch boundaries
        if (!have_multipatch_boundaries)
          check_valid_gf(*active_levels, gi, vi, tl, nan_handling,
                         []() { return "SyncGroupsByDirI after syncing"; });
      }
    } // for tl
  }   // for gi

  if (have_multipatch_boundaries) {
    std::vector<CCTK_INT> cactusvarinds;
    for (int group : groups) {
      const auto &groupdata =
          *ghext->patchdata.at(0).leveldata.at(0).groupdata.at(group);
      for (int var = 0; var < groupdata.numvars; ++var)
        cactusvarinds.push_back(groupdata.firstvarindex + var);
    }
    MultiPatch_Interpolate(cctkGH, cactusvarinds.size(), cactusvarinds.data());

    for (const int gi : groups) {
      const auto &patchdata0 = ghext->patchdata.at(0);
      const auto &leveldata0 = patchdata0.leveldata.at(0);
      const auto &groupdata0 = *leveldata0.groupdata.at(gi);
      const nan_handling_t nan_handling = groupdata0.do_checkpoint
                                              ? nan_handling_t::forbid_nans
                                              : nan_handling_t::allow_nans;
      // We always sync all directions.
      // If there is more than one time level, then we don't sync the
      // oldest.
      // TODO: during evolution, sync only one time level
      const int ntls0 = groupdata0.mfab.size();
      const int sync_tl0 = ntls0 > 1 ? ntls0 - 1 : ntls0;

      assert(active_levels->max_level == 1);

      for (int tl = 0; tl < sync_tl0; ++tl)
        for (int vi = 0; vi < groupdata0.numvars; ++vi)
          check_valid_gf(*active_levels, gi, vi, tl, nan_handling,
                         []() { return "SyncGroupsByDirI after syncing"; });

    } // for gi

  } else {
    assert(ghext->num_patches() == 1);
  }

  assert(sync_active);

  return numgroups; // number of groups synchronized
}

void Reflux(const cGH *cctkGH, int level) {
  DECLARE_CCTK_PARAMETERS;

  if (!do_reflux)
    return;

  static Timer timer("Reflux");
  Interval interval(timer);

  for (const auto &patchdata : ghext->patchdata) {
    if (level + 1 < int(patchdata.leveldata.size())) {
      auto &leveldata = patchdata.leveldata.at(level);
      const auto &fineleveldata = patchdata.leveldata.at(level + 1);
      for (int gi = 0; gi < int(leveldata.groupdata.size()); ++gi) {
        const int tl = 0;
        cGroup group;
        int ierr = CCTK_GroupData(gi, &group);
        assert(!ierr);

        if (group.grouptype != CCTK_GF)
          continue;

        auto &groupdata = *leveldata.groupdata.at(gi);
        const auto &finegroupdata = *fineleveldata.groupdata.at(gi);
        const nan_handling_t nan_handling = groupdata.do_checkpoint
                                                ? nan_handling_t::forbid_nans
                                                : nan_handling_t::allow_nans;

        // If the group has associated fluxes
        if (finegroupdata.freg) {

          // Check coarse and fine data and fluxes are valid
          for (int vi = 0; vi < finegroupdata.numvars; ++vi) {
            error_if_invalid(finegroupdata, vi, tl, make_valid_int(), []() {
              return "Reflux before refluxing: Fine level data";
            });
            error_if_invalid(groupdata, vi, tl, make_valid_int(), []() {
              return "Reflux before refluxing: Coarse level data";
            });
          }
          for (int d = 0; d < dim; ++d) {
            const int flux_gi = finegroupdata.fluxes.at(d);
            const auto &flux_finegroupdata =
                *fineleveldata.groupdata.at(flux_gi);
            const auto &flux_groupdata = *leveldata.groupdata.at(flux_gi);
            for (int vi = 0; vi < finegroupdata.numvars; ++vi) {
              error_if_invalid(
                  flux_finegroupdata, vi, tl, make_valid_int(), [&]() {
                    ostringstream buf;
                    buf << "Reflux: Fine level flux in direction " << d;
                    return buf.str();
                  });
              error_if_invalid(flux_groupdata, vi, tl, make_valid_int(), [&]() {
                ostringstream buf;
                buf << "Reflux: Coarse level flux in direction " << d;
                return buf.str();
              });
            }
          }

          for (int d = 0; d < dim; ++d) {
            const int flux_gi = finegroupdata.fluxes.at(d);
            const auto &flux_finegroupdata =
                *fineleveldata.groupdata.at(flux_gi);
            const auto &flux_groupdata = *leveldata.groupdata.at(flux_gi);
            finegroupdata.freg->CrseInit(*flux_groupdata.mfab.at(tl), d, 0, 0,
                                         flux_groupdata.numvars, -1);
            finegroupdata.freg->FineAdd(*flux_finegroupdata.mfab.at(tl), d, 0,
                                        0, flux_finegroupdata.numvars, 1);
          }
          const amrex::Geometry &geom = patchdata.amrcore->Geom(level);
          finegroupdata.freg->Reflux(*groupdata.mfab.at(tl), 1.0, 0, 0,
                                     groupdata.numvars, geom);

          const active_levels_t active_levels(level, level + 1);
          for (int vi = 0; vi < finegroupdata.numvars; ++vi)
            check_valid_gf(active_levels, gi, vi, tl, nan_handling, []() {
              return "Reflux after refluxing: Fine level data";
            });
        }
      } // for gi
    }   // if level exists
  }     // for patchdata
}

void Restrict(const cGH *cctkGH, int level, const vector<int> &groups) {
  DECLARE_CCTK_PARAMETERS;

#warning "TODO"
  assert(do_restrict);
  if (!do_restrict)
    return;

  static Timer timer("Restrict");
  Interval interval(timer);

  const int gi_regrid_error = CCTK_GroupIndex("CarpetX::regrid_error");
  assert(gi_regrid_error >= 0);

  for (const auto &patchdata : ghext->patchdata) {
    const int patch = patchdata.patch;
    if (level + 1 < int(patchdata.leveldata.size())) {
      auto &leveldata = patchdata.leveldata.at(level);
      const auto &fineleveldata = patchdata.leveldata.at(level + 1);
      const active_levels_t active_levels(level, level + 1, patch, patch + 1);
      const active_levels_t active_fine_levels(level + 1, level + 2, patch,
                                               patch + 1);

      for (const int gi : groups) {
        cGroup group;
        int ierr = CCTK_GroupData(gi, &group);
        assert(!ierr);

        assert(group.grouptype == CCTK_GF);

        auto &groupdata = *leveldata.groupdata.at(gi);
        const auto &finegroupdata = *fineleveldata.groupdata.at(gi);
        const amrex::IntVect reffact{2, 2, 2};
        const nan_handling_t nan_handling = groupdata.do_checkpoint
                                                ? nan_handling_t::forbid_nans
                                                : nan_handling_t::allow_nans;

        // Don't restrict the regridding error
        if (gi == gi_regrid_error)
          continue;
        // Don't restrict groups that have restriction disabled
        if (!groupdata.do_restrict)
          continue;

        // If there is more than one time level, then we don't restrict the
        // oldest.
        // TODO: during evolution, restrict only one time level
        int ntls = groupdata.mfab.size();
        int restrict_tl = ntls > 1 ? ntls - 1 : ntls;
        for (int tl = 0; tl < restrict_tl; ++tl) {

          for (int vi = 0; vi < groupdata.numvars; ++vi) {

            // Restriction only uses the interior
            error_if_invalid(finegroupdata, vi, tl, make_valid_int(), []() {
              return "Restrict on fine level before restricting";
            });
            poison_invalid_gf(active_fine_levels, gi, vi, tl);
            check_valid_gf(active_fine_levels, gi, vi, tl, nan_handling, []() {
              return "Restrict on fine level before restricting";
            });
            error_if_invalid(groupdata, vi, tl, make_valid_int(), []() {
              return "Restrict on coarse level before restricting";
            });
            poison_invalid_gf(active_levels, gi, vi, tl);
            check_valid_gf(active_levels, gi, vi, tl, nan_handling, []() {
              return "Restrict on coarse level before restricting";
            });
          }

#if 1
          {
            static Timer timer("Restrict::average_down");
            Interval interval(timer);
#warning                                                                       \
    "TODO: Allow different restriction operators, and ensure this is conservative"
            // rank: 0: vertex, 1: edge, 2: face, 3: volume
            int rank = 0;
            for (int d = 0; d < dim; ++d)
              rank += groupdata.indextype.at(d);
            switch (rank) {
            case 0:
              average_down_nodal(*finegroupdata.mfab.at(tl),
                                 *groupdata.mfab.at(tl), reffact);
              break;
            case 1:
              average_down_edges(*finegroupdata.mfab.at(tl),
                                 *groupdata.mfab.at(tl), reffact);
              break;
            case 2:
              average_down_faces(*finegroupdata.mfab.at(tl),
                                 *groupdata.mfab.at(tl), reffact);
              break;
            case 3:
              average_down(*finegroupdata.mfab.at(tl), *groupdata.mfab.at(tl),
                           0, groupdata.numvars, reffact);
              break;
            default:
              assert(0);
            }
          }
#endif

          // TODO: Also remember old why_valid for interior?
          for (int vi = 0; vi < groupdata.numvars; ++vi) {
            // Should we mark ghosts and maybe outer boundaries as
            // valid as well?
            groupdata.valid.at(tl).at(vi).set_invalid(
                make_valid_outer() | make_valid_ghosts(),
                []() { return "Restrict"; });
            poison_invalid_gf(active_levels, gi, vi, tl);
            check_valid_gf(active_levels, gi, vi, tl, nan_handling, []() {
              return "Restrict on coarse level after restricting";
            });
          }

        } // for tl
      }   // for gi
    }     // if level exists
  }       // for patchdata
}

void Restrict(const cGH *cctkGH, int level) {
  const int numgroups = CCTK_NumGroups();
  vector<int> groups;
  groups.reserve(numgroups);
  const auto &patchdata0 = ghext->patchdata.at(0);
  const auto &leveldata0 = patchdata0.leveldata.at(0);
  for (const auto &groupdataptr : leveldata0.groupdata) {
    // Restrict only grid functions
    if (groupdataptr) {
      auto &restrict groupdata = *groupdataptr;
      // Restrict only evolved grid functions
      if (groupdata.do_checkpoint)
        groups.push_back(groupdata.groupindex);
    }
  }
  Restrict(cctkGH, level, groups);
}

void SyncAfterRestrict(const cGH *cctkGH, int level) {
  const int numgroups = CCTK_NumGroups();
  vector<int> groups;
  groups.reserve(numgroups);
  const auto &patchdata0 = ghext->patchdata.at(0);
  const auto &leveldata0 = patchdata0.leveldata.at(0);
  for (const auto &groupdataptr : leveldata0.groupdata) {
    // Restrict only grid functions
    if (groupdataptr) {
      auto &restrict groupdata = *groupdataptr;
      // Restrict only evolved grid functions
      if (groupdata.do_checkpoint)
        groups.push_back(groupdata.groupindex);
    }
  }
  auto active_levels_bk = active_levels;
  active_levels = make_optional<active_levels_t>(level - 1, level + 1);
  SyncGroupsByDirI(cctkGH, groups.size(), groups.data(), nullptr);
  active_levels = active_levels_bk;
}

// storage handling
namespace {
int GroupStorageCrease(const cGH *cctkGH, int n_groups, const int *groups,
                       const int *requested_tls, int *status, const bool inc) {
  DECLARE_CCTK_PARAMETERS;

  assert(cctkGH);
  assert(n_groups >= 0);
  assert(groups);
  assert(requested_tls);
  for (int n = 0; n < n_groups; ++n) {
    if (groups[n] < 0 or groups[n] >= CCTK_NumGroups()) {
      CCTK_VWARN(CCTK_WARN_ALERT, "Group index %d is illegal", groups[n]);
      return -1;
    }
    assert(groups[n] >= 0 and groups[n] < CCTK_NumGroups());
    assert(requested_tls[n] >= 0 or requested_tls[n] == -1);
  }

  // sanitize list of requested timelevels
  std::vector<int> tls(n_groups);
  for (int n = 0; n < n_groups; ++n) {
    int ntls = requested_tls[n];
    int const declared_tls = CCTK_DeclaredTimeLevelsGI(groups[n]);
    if (inc and declared_tls < 2 and ntls > declared_tls) {
      CCTK_VWARN(CCTK_WARN_ALERT,
                 "Attempting to activate %d timelevels for group '%s' which "
                 "only has a single timelevel declared in interface.ccl. "
                 "Please declared at least 2 timelevels in interface.ccl to "
                 "allow more timelevels to be created at runtime.",
                 ntls, CCTK_FullGroupName(groups[n]));
      ntls = declared_tls;
    }
    if (ntls == -1) {
      ntls = declared_tls;
    }
    tls.at(n) = ntls;
  }

  // TODO: actually do something
  int min_num_timelevels = INT_MAX;
  for (int n = 0; n < n_groups; ++n) {
    int const gid = groups[n];

    cGroup group;
    int ierr = CCTK_GroupData(gid, &group);
    assert(not ierr);

    // Record previous number of allocated time levels
    if (status) {
      // Note: This remembers only the last level
      status[n] = group.numtimelevels;
    }

    // Record (minimum of) current number of time levels
    min_num_timelevels = min(min_num_timelevels, group.numtimelevels);
  } // for n
  if (min_num_timelevels == INT_MAX) {
    min_num_timelevels = 0;
  }

  return min_num_timelevels;
}
} // namespace

int GroupStorageIncrease(const cGH *cctkGH, int n_groups, const int *groups,
                         const int *tls, int *status) {
  DECLARE_CCTK_PARAMETERS;

  return GroupStorageCrease(cctkGH, n_groups, groups, tls, status, true);
}

int GroupStorageDecrease(const cGH *cctkGH, int n_groups, const int *groups,
                         const int *tls, int *status) {
  DECLARE_CCTK_PARAMETERS;

  return GroupStorageCrease(cctkGH, n_groups, groups, tls, status, false);
}

int EnableGroupStorage(const cGH *cctkGH, const char *groupname) {
  const int group = CCTK_GroupIndex(groupname);
  assert(group >= 0 and group < CCTK_NumGroups());
  // TODO: decide whether to use CCTK_MaxActiveTimeLevelsGI
  const int tls = CCTK_DeclaredTimeLevelsGI(group);
  int status;
  GroupStorageIncrease(cctkGH, 1, &group, &tls, &status);
  // Return whether storage was allocated previously
  return status;
}

int DisableGroupStorage(const cGH *cctkGH, const char *groupname) {
  const int group = CCTK_GroupIndex(groupname);
  assert(group >= 0 and group < CCTK_NumGroups());
  const int tls = 0;
  int status;
  GroupStorageDecrease(cctkGH, 1, &group, &tls, &status);
  // Return whether storage was allocated previously
  return status;
}

} // namespace CarpetX

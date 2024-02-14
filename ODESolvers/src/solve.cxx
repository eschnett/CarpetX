// TODO: Don't include files from other thorns; create a proper interface
#include "../../CarpetX/src/driver.hxx"
#include "../../CarpetX/src/schedule.hxx"
#include "../../CarpetX/src/timer.hxx"

#include <cctk.h>
#include <cctk_Parameters.h>
#include <cctk_Arguments.h>
#include <util_Table.h>

#include <div.hxx>

#include <AMReX_MultiFab.H>

#if defined _OPENMP || defined __HIPCC__
#include <omp.h>
#else
static inline int omp_get_max_threads() { return 1; }
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace ODESolvers {
using namespace std;

////////////////////////////////////////////////////////////////////////////////

// Taken from <https://en.cppreference.com/w/cpp/experimental/make_array>
namespace details {
template <class> struct is_ref_wrapper : std::false_type {};
template <class T>
struct is_ref_wrapper<std::reference_wrapper<T> > : std::true_type {};

template <class T>
using not_ref_wrapper = std::negation<is_ref_wrapper<std::decay_t<T> > >;

template <class D, class...> struct return_type_helper {
  using type = D;
};
template <class... Types>
struct return_type_helper<void, Types...> : std::common_type<Types...> {
  static_assert(std::conjunction_v<not_ref_wrapper<Types>...>,
                "Types cannot contain reference_wrappers when D is void");
};

template <class D, class... Types>
using return_type = std::array<typename return_type_helper<D, Types...>::type,
                               sizeof...(Types)>;
} // namespace details

template <class D = void, class... Types>
constexpr details::return_type<D, Types...> make_array(Types &&...t) {
  return {std::forward<Types>(t)...};
}

////////////////////////////////////////////////////////////////////////////////

// A state vector component, with mfabs for each level, group, and variable
struct statecomp_t {

  statecomp_t() = default;

  statecomp_t(statecomp_t &&) = default;
  statecomp_t &operator=(statecomp_t &&) = default;

  // Don't allow copies because we might own stuff
  statecomp_t(const statecomp_t &) = delete;
  statecomp_t &operator=(const statecomp_t &) = delete;

  vector<GHExt::PatchData::LevelData::GroupData *> groupdatas;
  vector<amrex::MultiFab *> mfabs;

  static void init_tmp_mfabs();
  static void free_tmp_mfabs();

  void set_valid(const valid_t valid) const;
  template <size_t N>
  static void combine_valids(const statecomp_t &dst, const CCTK_REAL scale,
                             const array<CCTK_REAL, N> &factors,
                             const array<const statecomp_t *, N> &srcs);
  void check_valid(const valid_t required, const function<string()> &why) const;
  void check_valid(const valid_t required, const string &why) const {
    check_valid(required, [=]() { return why; });
  }

  statecomp_t copy(const valid_t where) const;

  template <size_t N>
  static void lincomb(const statecomp_t &dst, CCTK_REAL scale,
                      const array<CCTK_REAL, N> &factors,
                      const array<const statecomp_t *, N> &srcs,
                      const valid_t where);
  template <size_t N>
  static void lincomb(const statecomp_t &dst, CCTK_REAL scale,
                      const array<CCTK_REAL, N> &factors,
                      const array<statecomp_t *, N> &srcs,
                      const valid_t where) {
    array<const statecomp_t *, N> srcs1;
    for (size_t n = 0; n < N; ++n)
      srcs1[n] = srcs[n];
    lincomb(dst, scale, factors, srcs1, where);
  }

  static void lincomb(const statecomp_t &dst, CCTK_REAL scale,
                      const vector<CCTK_REAL> &factors,
                      const vector<const statecomp_t *> &srcs,
                      const valid_t where);
};

template <std::size_t N> using reals = std::array<CCTK_REAL, N>;
template <std::size_t N> using states = std::array<const statecomp_t *, N>;

////////////////////////////////////////////////////////////////////////////////

// Initialize the temporary mfab mechanism
void statecomp_t::init_tmp_mfabs() {
  assert(CarpetX::active_levels);
  CarpetX::active_levels->loop_serially([&](const auto &leveldata) {
    for (const auto &groupdataptr : leveldata.groupdata) {
      if (groupdataptr == nullptr)
        continue;
      const auto &groupdata = *groupdataptr;
      groupdata.init_tmp_mfabs();
    }
  });
}

// Free all temporary mfabs that we might have allocated
void statecomp_t::free_tmp_mfabs() {
  assert(CarpetX::active_levels);
  CarpetX::active_levels->loop_serially([&](const auto &leveldata) {
    for (const auto &groupdataptr : leveldata.groupdata) {
      if (groupdataptr == nullptr)
        continue;
      const auto &groupdata = *groupdataptr;
      groupdata.free_tmp_mfabs();
    }
  });
}

// State that the state vector has valid data in the interior
void statecomp_t::set_valid(const valid_t valid) const {
  for (auto groupdata : groupdatas) {
    for (int vi = 0; vi < groupdata->numvars; ++vi) {
      const int tl = 0;
      groupdata->valid.at(tl).at(vi).set_int(valid.valid_int, [=]() {
        ostringstream buf;
        buf << "ODESolvers after lincomb: Mark interior as "
            << (valid.valid_int ? "valid" : "invalid");
        return buf.str();
      });
      groupdata->valid.at(tl).at(vi).set_outer(valid.valid_outer, [=]() {
        ostringstream buf;
        buf << "ODESolvers after lincomb: Mark outer boundary as "
            << (valid.valid_outer ? "valid" : "invalid");
        return buf.str();
      });
      groupdata->valid.at(tl).at(vi).set_ghosts(valid.valid_ghosts, [=]() {
        ostringstream buf;
        buf << "ODESolvers after lincomb: Mark ghosts as "
            << (valid.valid_int ? "valid" : "invalid");
        return buf.str();
      });
      // TODO: Parallelize over patches, levels, group, variables, and
      // timelevels
      const active_levels_t active_levels(
          groupdata->level, groupdata->level + 1, groupdata->patch,
          groupdata->patch + 1);
      CarpetX::poison_invalid_gf(active_levels, groupdata->groupindex, vi, tl);
    }
  }
}

// Combine validity information from several sources
template <size_t N>
void statecomp_t::combine_valids(const statecomp_t &dst, const CCTK_REAL scale,
                                 const array<CCTK_REAL, N> &factors,
                                 const array<const statecomp_t *, N> &srcs) {
  const int ngroups = dst.groupdatas.size();
  for (const auto &src : srcs)
    assert(int(src->groupdatas.size()) == ngroups);
  for (int group = 0; group < ngroups; ++group) {
    const auto &dstgroup = dst.groupdatas.at(group);
    const int nvars = dstgroup->numvars;
    for (const auto &src : srcs) {
      const auto &srcgroup = src->groupdatas.at(group);
      assert(srcgroup->numvars == nvars);
    }
  }

  for (int group = 0; group < ngroups; ++group) {
    const auto &dstgroup = dst.groupdatas.at(group);
    const int nvars = dstgroup->numvars;
    const int tl = 0;
    for (int vi = 0; vi < nvars; ++vi) {
      valid_t valid = valid_t(true);
      bool did_set_valid = false;
      if (scale != 0) {
        valid &= dstgroup->valid.at(tl).at(vi).get();
        did_set_valid = true;
      }
      for (size_t m = 0; m < srcs.size(); ++m) {
        if (factors.at(m) != 0) {
          const auto &src = srcs.at(m);
          const auto &srcgroup = src->groupdatas.at(group);
          valid &= srcgroup->valid.at(tl).at(vi).get();
          did_set_valid = true;
        }
      }
      if (!did_set_valid)
        valid = valid_t(false);
      dstgroup->valid.at(tl).at(vi) =
          why_valid_t(valid, []() { return "Set from RHS in ODESolvers"; });
    }
  }
}

// Ensure a state vector has valid data in the interior
void statecomp_t::check_valid(const valid_t required,
                              const function<string()> &why) const {
  for (const auto groupdata : groupdatas) {
    for (int vi = 0; vi < groupdata->numvars; ++vi) {
      const int tl = 0;
      CarpetX::error_if_invalid(*groupdata, vi, tl, required, why);
      // TODO: Parallelize over pathces, levels, group, variables, and
      // timelevels
      const active_levels_t active_levels(
          groupdata->level, groupdata->level + 1, groupdata->patch,
          groupdata->patch + 1);
      CarpetX::check_valid_gf(active_levels, groupdata->groupindex, vi, tl,
                              nan_handling_t::forbid_nans, why);
    }
  }
}

// Copy state vector into newly allocated memory
statecomp_t statecomp_t::copy(const valid_t where) const {
  const size_t size = mfabs.size();
  statecomp_t result;
  result.groupdatas.reserve(size);
  result.mfabs.reserve(size);
  for (size_t n = 0; n < size; ++n) {
    const auto groupdata = groupdatas.at(n);
#ifdef CCTK_DEBUG
    const auto &x = mfabs.at(n);
    if (x->contains_nan())
      CCTK_VERROR("statecomp_t::copy.x: Group %s contains nans",
                  groupdata->groupname.c_str());
#endif
    auto y = groupdata->alloc_tmp_mfab();
    result.groupdatas.push_back(groupdata);
    result.mfabs.push_back(y);
  }
  lincomb(result, 0, make_array(CCTK_REAL(1)), make_array(this), where);
#ifdef CCTK_DEBUG
  for (size_t n = 0; n < size; ++n) {
    const auto groupdata = result.groupdatas.at(n);
    const auto &y = result.mfabs.at(n);
    if (y->contains_nan())
      CCTK_VERROR("statecomp_t::copy.y: Group %s contains nans",
                  groupdata->groupname.c_str());
  }
#endif
  return result;
}

template <size_t N>
void statecomp_t::lincomb(const statecomp_t &dst, const CCTK_REAL scale,
                          const array<CCTK_REAL, N> &factors,
                          const array<const statecomp_t *, N> &srcs,
                          const valid_t where) {
  const size_t size = dst.mfabs.size();
  for (size_t n = 0; n < N; ++n)
    assert(srcs[n]->mfabs.size() == size);
  for (size_t m = 0; m < size; ++m) {
    const auto ncomp = dst.mfabs.at(m)->nComp();
    const auto ngrowvect = dst.mfabs.at(m)->nGrowVect();
    for (size_t n = 0; n < N; ++n) {
      assert(srcs[n]->mfabs.at(m)->nComp() == ncomp);
      assert(srcs[n]->mfabs.at(m)->nGrowVect() == ngrowvect);
    }
  }

  assert(isfinite(scale));
  const bool read_dst = scale != 0;
  for (size_t n = 0; n < N; ++n)
    assert(isfinite(factors[n]));

  statecomp_t::combine_valids(dst, scale, factors, srcs);

#ifndef AMREX_USE_GPU
  vector<function<void()> > tasks;
#endif

  for (size_t m = 0; m < size; ++m) {
    const ptrdiff_t ncomps = dst.mfabs.at(m)->nComp();
    const auto mfitinfo = amrex::MFItInfo().DisableDeviceSync();
    for (amrex::MFIter mfi(*dst.mfabs.at(m), mfitinfo); mfi.isValid(); ++mfi) {
      const amrex::Array4<CCTK_REAL> dstvar = dst.mfabs.at(m)->array(mfi);
      array<amrex::Array4<const CCTK_REAL>, N> srcvars;
      for (size_t n = 0; n < N; ++n)
        srcvars[n] = srcs[n]->mfabs.at(m)->const_array(mfi);
      for (size_t n = 0; n < N; ++n) {
        assert(srcvars[n].jstride == dstvar.jstride);
        assert(srcvars[n].kstride == dstvar.kstride);
        assert(srcvars[n].nstride == dstvar.nstride);
      }
      const ptrdiff_t nstride = dstvar.nstride;
      const ptrdiff_t npoints = nstride * ncomps;

      CCTK_REAL *restrict const dstptr = dstvar.dataPtr();
      array<const CCTK_REAL *restrict, N> srcptrs;
      for (size_t n = 0; n < N; ++n)
        srcptrs[n] = srcvars[n].dataPtr();

#ifndef AMREX_USE_GPU
      // CPU

      const ptrdiff_t ntiles = omp_get_max_threads();
      const ptrdiff_t tile_size =
          Arith::align_ceil(Arith::div_ceil(npoints, ntiles), ptrdiff_t(64));

      for (ptrdiff_t imin = 0; imin < npoints; imin += tile_size) {
        using std::min;
        const ptrdiff_t imax = min(npoints, imin + tile_size);

        if (!read_dst && N == 1 && factors[0] == 1) {
          // Copy

          auto task = [=]() {
            std::memcpy(&dstptr[imin], &srcptrs[0][imin],
                        (imax - imin) * sizeof *dstptr);
          };
          tasks.push_back(std::move(task));

        } else if (!read_dst && N >= 1 && factors[0] == 1) {
          // Write without scaling

          auto task = [=]() {
#pragma omp simd
            for (ptrdiff_t i = imin; i < imax; ++i) {
              CCTK_REAL accum = srcptrs[0][i];
              for (size_t n = 1; n < N; ++n)
                accum += factors[n] * srcptrs[n][i];
              dstptr[i] = accum;
            }
          };
          tasks.push_back(std::move(task));

        } else if (!read_dst) {
          // Write

          auto task = [=]() {
#pragma omp simd
            for (ptrdiff_t i = imin; i < imax; ++i) {
              CCTK_REAL accum = 0;
              for (size_t n = 0; n < N; ++n)
                accum += factors[n] * srcptrs[n][i];
              dstptr[i] = accum;
            }
          };
          tasks.push_back(std::move(task));

        } else if (scale == 1) {
          // Update without scaling

          auto task = [=]() {
#pragma omp simd
            for (ptrdiff_t i = imin; i < imax; ++i) {
              CCTK_REAL accum = dstptr[i];
              for (size_t n = 0; n < N; ++n)
                accum += factors[n] * srcptrs[n][i];
              dstptr[i] = accum;
            }
          };
          tasks.push_back(std::move(task));

        } else {
          // Update

          auto task = [=]() {
#pragma omp simd
            for (ptrdiff_t i = imin; i < imax; ++i) {
              CCTK_REAL accum = scale * dstptr[i];
              for (size_t n = 0; n < N; ++n)
                accum += factors[n] * srcptrs[n][i];
              dstptr[i] = accum;
            }
          };
          tasks.push_back(std::move(task));
        }
      } // for imin

#else
      // GPU

      const CCTK_REAL scale1 = scale;
      assert(npoints < INT_MAX);
      const amrex::Box box(
          amrex::IntVect(0, 0, 0), amrex::IntVect(npoints - 1, 0, 0),
          amrex::IntVect(amrex::IndexType::CELL, amrex::IndexType::CELL,
                         amrex::IndexType::CELL));

      if (!read_dst) {

        amrex::launch(box, [=] CCTK_DEVICE(const amrex::Box &box)
                               CCTK_ATTRIBUTE_ALWAYS_INLINE {
                                 const int i = box.smallEnd()[0];
                                 // const int j = box.smallEnd()[1];
                                 // const int k = box.smallEnd()[2];
                                 CCTK_REAL accum = 0;
                                 for (size_t n = 0; n < N; ++n)
                                   accum += factors[n] * srcptrs[n][i];
                                 dstptr[i] = accum;
                               });

      } else {

        amrex::launch(box, [=] CCTK_DEVICE(const amrex::Box &box)
                               CCTK_ATTRIBUTE_ALWAYS_INLINE {
                                 const int i = box.smallEnd()[0];
                                 // const int j = box.smallEnd()[1];
                                 // const int k = box.smallEnd()[2];
                                 CCTK_REAL accum = scale1 * dstptr[i];
                                 for (size_t n = 0; n < N; ++n)
                                   accum += factors[n] * srcptrs[n][i];
                                 dstptr[i] = accum;
                               });
      }

#endif
    }
  }

#ifndef AMREX_USE_GPU
  // run all tasks
#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < tasks.size(); ++i)
    tasks[i]();
#else
  // wait for all tasks
  amrex::Gpu::synchronize();
  AMREX_GPU_ERROR_CHECK();
#endif
}

namespace detail {
template <size_t N>
void call_lincomb(const statecomp_t &dst, const CCTK_REAL scale,
                  const vector<CCTK_REAL> &factors,
                  const vector<const statecomp_t *> &srcs,
                  const vector<size_t> &indices, const valid_t where) {
  assert(indices.size() == N);
  array<CCTK_REAL, N> factors1;
  array<const statecomp_t *, N> srcs1;
  for (size_t n = 0; n < N; ++n) {
    factors1[n] = factors.at(indices[n]);
    srcs1[n] = srcs.at(indices[n]);
  }
  statecomp_t::lincomb(dst, scale, factors1, srcs1, where);
}
} // namespace detail

void statecomp_t::lincomb(const statecomp_t &dst, const CCTK_REAL scale,
                          const vector<CCTK_REAL> &factors,
                          const vector<const statecomp_t *> &srcs,
                          const valid_t where) {
  const size_t N = factors.size();
  assert(srcs.size() == N);

  size_t NNZ = 0;
  for (size_t n = 0; n < N; ++n)
    NNZ += factors[n] != 0;
  vector<size_t> indices;
  indices.reserve(NNZ);
  for (size_t n = 0; n < N; ++n)
    if (factors[n] != 0)
      indices.push_back(n);
  assert(indices.size() == NNZ);

  switch (NNZ) {
  case 0:
    return detail::call_lincomb<0>(dst, scale, factors, srcs, indices, where);
  case 1:
    return detail::call_lincomb<1>(dst, scale, factors, srcs, indices, where);
  case 2:
    return detail::call_lincomb<2>(dst, scale, factors, srcs, indices, where);
  case 3:
    return detail::call_lincomb<3>(dst, scale, factors, srcs, indices, where);
  case 4:
    return detail::call_lincomb<4>(dst, scale, factors, srcs, indices, where);
  case 5:
    return detail::call_lincomb<5>(dst, scale, factors, srcs, indices, where);
  case 6:
    return detail::call_lincomb<6>(dst, scale, factors, srcs, indices, where);
  case 7:
    return detail::call_lincomb<7>(dst, scale, factors, srcs, indices, where);
  case 8:
    return detail::call_lincomb<8>(dst, scale, factors, srcs, indices, where);
  case 9:
    return detail::call_lincomb<9>(dst, scale, factors, srcs, indices, where);
  case 10:
    return detail::call_lincomb<10>(dst, scale, factors, srcs, indices, where);
  case 11:
    return detail::call_lincomb<11>(dst, scale, factors, srcs, indices, where);
  case 12:
    return detail::call_lincomb<12>(dst, scale, factors, srcs, indices, where);
  case 13:
    return detail::call_lincomb<13>(dst, scale, factors, srcs, indices, where);
  case 14:
    return detail::call_lincomb<14>(dst, scale, factors, srcs, indices, where);
  case 15:
    return detail::call_lincomb<15>(dst, scale, factors, srcs, indices, where);
  case 16:
    return detail::call_lincomb<16>(dst, scale, factors, srcs, indices, where);
  default:
    CCTK_VERROR("Unsupported vector length: %d", (int)NNZ);
  }
}

////////////////////////////////////////////////////////////////////////////////

int groupindex(const int other_gi, std::string gn) {
  // If the group name does not contain a colon, then prefix the current group's
  // implementation or thorn name
  if (gn.find(':') == std::string::npos) {
    const char *const thorn_or_impl = CCTK_GroupImplementationI(other_gi);
    assert(thorn_or_impl);
    const char *const impl = CCTK_ThornImplementation(thorn_or_impl);
    const char *const thorn = CCTK_ImplementationThorn(thorn_or_impl);
    assert(impl || thorn);
    const char *prefix;
    if (!impl) {
      prefix = thorn;
    } else if (!thorn) {
      prefix = impl;
    } else {
      assert(strcmp(impl, thorn) == 0);
      prefix = impl;
    }
    std::ostringstream buf;
    buf << prefix << "::" + gn;
    gn = buf.str();
  }
  const int gi = CCTK_GroupIndex(gn.c_str());
  return gi;
}

int get_group_rhs(const int gi) {
  assert(gi >= 0);
  const int tags = CCTK_GroupTagsTableI(gi);
  assert(tags >= 0);
  std::vector<char> rhs_buf(1000);
  const int iret =
      Util_TableGetString(tags, rhs_buf.size(), rhs_buf.data(), "rhs");
  if (iret == UTIL_ERROR_TABLE_NO_SUCH_KEY) {
    rhs_buf[0] = '\0'; // default: empty (no RHS)
  } else if (iret >= 0) {
    // do nothing
  } else {
    assert(0);
  }

  const std::string str(rhs_buf.data());
  if (str.empty())
    return -1; // No RHS specified

  const int rhs = groupindex(gi, str);
  if (rhs < 0)
    CCTK_VERROR("Variable group \"%s\" declares a RHS group \"%s\". "
                "That group does not exist.",
                CCTK_FullGroupName(gi), str.c_str());
  assert(rhs != gi);

  return rhs;
}

std::vector<int> get_group_dependents(const int gi) {
  assert(gi >= 0);
  const int tags = CCTK_GroupTagsTableI(gi);
  assert(tags >= 0);
  std::vector<char> dependents_buf(1000);
  const int iret = Util_TableGetString(tags, dependents_buf.size(),
                                       dependents_buf.data(), "dependents");
  if (iret == UTIL_ERROR_TABLE_NO_SUCH_KEY) {
    dependents_buf[0] = '\0'; // default: empty (no DEPENDENTS)
  } else if (iret >= 0) {
    // do nothing
  } else {
    assert(0);
  }

  std::vector<int> dependents;
  const std::string str(dependents_buf.data());
  std::size_t pos = 0;
  for (;;) {
    // Skip white space
    while (pos < str.size() && std::isspace(str[pos]))
      ++pos;
    if (pos == str.size())
      break;
    // Read group name
    const std::size_t group_begin = pos;
    while (pos < str.size() && !std::isspace(str[pos]))
      ++pos;
    const std::size_t group_end = pos;
    const std::string groupname =
        str.substr(group_begin, group_end - group_begin);
    const int dep_gi = groupindex(gi, groupname);
    if (dep_gi < 0)
      CCTK_VERROR("Variable group \"%s\" declares a dependent group \"%s\". "
                  "That group does not exist.",
                  CCTK_FullGroupName(gi), groupname.c_str());
    dependents.push_back(dep_gi);
  }

  return dependents;
}

// Mark groups as invalid
void mark_invalid(const std::vector<int> &groups) {
  CarpetX::active_levels->loop_serially([&](const auto &leveldata) {
    for (const int gi : groups) {
      auto &groupdata = *leveldata.groupdata.at(gi);
      // Invalidate all variables of the current time level
      const int tl = 0;
      for (auto &why_valid : groupdata.valid.at(tl))
        why_valid =
            why_valid_t([] { return "ODESolvers updated the state vector"; });
    }
  });
}

///////////////////////////////////////////////////////////////////////////////

extern "C" void ODESolvers_Solve(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_Solve;
  DECLARE_CCTK_PARAMETERS;

  static bool did_output = false;
  if (verbose || !did_output)
    CCTK_VINFO("Integrator is %s", method);
  did_output = true;

  static Timer timer("ODESolvers::Solve");
  Interval interval(timer);

  const CCTK_REAL dt = cctk_delta_time;
  const int tl = 0;

  static Timer timer_setup("ODESolvers::Solve::setup");
  std::optional<Interval> interval_setup(timer_setup);

  statecomp_t var, rhs;
  std::vector<int> var_groups, rhs_groups, dep_groups;
  int nvars = 0;
  bool do_accumulate_nvars = true;
  assert(CarpetX::active_levels);
  CarpetX::active_levels->loop_serially([&](const auto &leveldata) {
    for (const auto &groupdataptr : leveldata.groupdata) {
      // TODO: add support for evolving grid scalars
      if (groupdataptr == nullptr)
        continue;

      auto &groupdata = *groupdataptr;
      const int rhs_gi = get_group_rhs(groupdata.groupindex);
      if (rhs_gi >= 0) {
        assert(rhs_gi != groupdata.groupindex);
        auto &rhs_groupdata = *leveldata.groupdata.at(rhs_gi);
        assert(rhs_groupdata.numvars == groupdata.numvars);
        var.groupdatas.push_back(&groupdata);
        var.mfabs.push_back(groupdata.mfab.at(tl).get());
        rhs.groupdatas.push_back(&rhs_groupdata);
        rhs.mfabs.push_back(rhs_groupdata.mfab.at(tl).get());
        if (do_accumulate_nvars) {
          nvars += groupdata.numvars;
          var_groups.push_back(groupdata.groupindex);
          rhs_groups.push_back(rhs_gi);
          const auto &dependents = get_group_dependents(groupdata.groupindex);
          dep_groups.insert(dep_groups.end(), dependents.begin(),
                            dependents.end());
        }
      }
    }
    do_accumulate_nvars = false;
  });
  if (verbose)
    CCTK_VINFO("  Integrating %d variables", nvars);
  if (nvars == 0)
    CCTK_VWARN(CCTK_WARN_ALERT, "Integrating %d variables", nvars);

  {
    std::sort(var_groups.begin(), var_groups.end());
    const auto last = std::unique(var_groups.begin(), var_groups.end());
    assert(last == var_groups.end());
  }

  {
    std::sort(rhs_groups.begin(), rhs_groups.end());
    const auto last = std::unique(rhs_groups.begin(), rhs_groups.end());
    assert(last == rhs_groups.end());
  }

  // Add RHS variables to dependent variables
  dep_groups.insert(dep_groups.end(), rhs_groups.begin(), rhs_groups.end());

  {
    std::sort(dep_groups.begin(), dep_groups.end());
    const auto last = std::unique(dep_groups.begin(), dep_groups.end());
    dep_groups.erase(last, dep_groups.end());
  }

  for (const int gi : var_groups)
    assert(std::find(dep_groups.begin(), dep_groups.end(), gi) ==
           dep_groups.end());
  for (const int gi : rhs_groups)
    assert(std::find(var_groups.begin(), var_groups.end(), gi) ==
           var_groups.end());

  interval_setup.reset();

  {
    static Timer timer_alloc_temps("ODESolvers::Solve::alloc_temps");
    Interval interval_alloc_temps(timer_alloc_temps);
    statecomp_t::init_tmp_mfabs();
  }

  const CCTK_REAL saved_time = cctkGH->cctk_time;
  const CCTK_REAL old_time = cctkGH->cctk_time - dt;

  static Timer timer_lincomb("ODESolvers::Solve::lincomb");
  static Timer timer_rhs("ODESolvers::Solve::rhs");
  static Timer timer_poststep("ODESolvers::Solve::poststep");

  const auto copy_state = [](const auto &var) {
    return var.copy(make_valid_int());
  };
  const auto calcrhs = [&](const int n) {
    Interval interval_rhs(timer_rhs);
    if (verbose)
      CCTK_VINFO("Calculating RHS #%d at t=%g", n, double(cctkGH->cctk_time));
    CallScheduleGroup(cctkGH, "ODESolvers_RHS");
    rhs.check_valid(make_valid_int(),
                    "ODESolvers after calling ODESolvers_RHS");
  };
  // t = t_0 + c
  // var = a_0 * var + \Sum_i a_i * var_i
  const auto calcupdate = [&](const int n, const CCTK_REAL c,
                              const CCTK_REAL a0, const auto &as,
                              const auto &vars) {
    {
      Interval interval_lincomb(timer_lincomb);
      statecomp_t::lincomb(var, a0, as, vars, make_valid_int());
      var.check_valid(make_valid_int(),
                      "ODESolvers after defining new state vector");
      mark_invalid(dep_groups);
    }
    {
      Interval interval_poststep(timer_poststep);
      *const_cast<CCTK_REAL *>(&cctkGH->cctk_time) = old_time + c;
      CallScheduleGroup(cctkGH, "ODESolvers_PostStep");
      if (verbose)
        CCTK_VINFO("Calculated new state #%d at t=%g", n,
                   double(cctkGH->cctk_time));
    }
  };

  *const_cast<CCTK_REAL *>(&cctkGH->cctk_time) = old_time;

  if (CCTK_EQUALS(method, "constant")) {

    // y1 = y0

    // do nothing

  } else if (CCTK_EQUALS(method, "Euler")) {

    // k1 = f(y0)
    // y1 = y0 + h k1

    calcrhs(1);
    calcupdate(1, dt, 1.0, reals<1>{dt}, states<1>{&rhs});

  } else if (CCTK_EQUALS(method, "RK2")) {

    // k1 = f(y0)
    // k2 = f(y0 + h/2 k1)
    // y1 = y0 + h k2

    const auto old = copy_state(var);

    calcrhs(1);
    calcupdate(1, dt / 2, 1.0, reals<1>{dt / 2}, states<1>{&rhs});

    calcrhs(2);
    calcupdate(2, dt, 0.0, reals<2>{1.0, dt}, states<2>{&old, &rhs});

  } else if (CCTK_EQUALS(method, "RK3")) {

    // k1 = f(y0)
    // k2 = f(y0 + h/2 k1)
    // k3 = f(y0 - h k1 + 2 h k2)
    // y1 = y0 + h/6 k1 + 2/3 h k2 + h/6 k3

    const auto old = copy_state(var);

    calcrhs(1);
    const auto k1 = copy_state(rhs);
    calcupdate(1, dt / 2, 1.0, reals<1>{dt / 2}, states<1>{&k1});

    calcrhs(2);
    const auto k2 = copy_state(rhs);
    calcupdate(2, dt, 0.0, reals<3>{1.0, -dt, 2 * dt},
               states<3>{&old, &k1, &k2});

    calcrhs(3);
    calcupdate(3, dt, 0.0, reals<4>{1.0, dt / 6, 2 * dt / 3, dt / 6},
               states<4>{&old, &k1, &k2, &rhs});

  } else if (CCTK_EQUALS(method, "SSPRK3")) {

    // k1 = f(y0)
    // k2 = f(y0 + h k1)
    // k3 = f(y0 + h/4 k1 + h/4 k2)
    // y1 = y0 + h/6 k1 + h/6 k2 + 2/3 h k3

    const auto old = copy_state(var);

    calcrhs(1);
    const auto k1 = copy_state(rhs);
    calcupdate(1, dt, 1.0, reals<1>{dt}, states<1>{&k1});

    calcrhs(2);
    const auto k2 = copy_state(rhs);
    calcupdate(2, dt / 2, 0.0, reals<3>{1.0, dt / 4, dt / 4},
               states<3>{&old, &k1, &k2});

    calcrhs(3);
    calcupdate(3, dt, 0.0, reals<4>{1.0, dt / 6, dt / 6, 2 * dt / 3},
               states<4>{&old, &k1, &k2, &rhs});

  } else if (CCTK_EQUALS(method, "RK4")) {

    // k1 = f(y0)
    // k2 = f(y0 + h/2 k1)
    // k3 = f(y0 + h/2 k2)
    // k4 = f(y0 + h k3)
    // y1 = y0 + h/6 k1 + h/3 k2 + h/3 k3 + h/6 k4

    const auto old = copy_state(var);

    calcrhs(1);
    const auto kaccum = copy_state(rhs);
    calcupdate(1, dt / 2, 1.0, reals<1>{dt / 2}, states<1>{&kaccum});

    calcrhs(2);
    {
      Interval interval_lincomb(timer_lincomb);
      statecomp_t::lincomb(kaccum, 1.0, reals<1>{2.0}, states<1>{&rhs},
                           make_valid_int());
    }
    calcupdate(2, dt / 2, 0.0, reals<2>{1.0, dt / 2}, states<2>{&old, &rhs});

    calcrhs(3);
    {
      Interval interval_lincomb(timer_lincomb);
      statecomp_t::lincomb(kaccum, 1.0, reals<1>{2.0}, states<1>{&rhs},
                           make_valid_int());
    }
    calcupdate(3, dt, 0.0, reals<2>{1.0, dt}, states<2>{&old, &rhs});

    calcrhs(4);
    calcupdate(4, dt, 0.0, reals<3>{1.0, dt / 6, dt / 6},
               states<3>{&old, &kaccum, &rhs});

  } else if (CCTK_EQUALS(method, "RKF78")) {

    typedef CCTK_REAL T;
    const auto R = [](T x, T y) { return x / y; };
    const tuple<vector<tuple<T, vector<T> > >, vector<T> > tableau{
        {
            {/* 1 */ 0, {}},                                           //
            {/* 2 */ R(2, 27), {R(2, 27)}},                            //
            {/* 3 */ R(1, 9), {R(1, 36), R(3, 36)}},                   //
            {/* 4 */ R(1, 6), {R(1, 24), 0, R(3, 24)}},                //
            {/* 5 */ R(5, 12), {R(20, 48), 0, R(-75, 48), R(75, 48)}}, //
            {/* 6 */ R(1, 2), {R(1, 20), 0, 0, R(5, 20), R(4, 20)}},   //
            {/* 7 */ R(5, 6),
             {R(-25, 108), 0, 0, R(125, 108), R(-260, 108), R(250, 108)}}, //
            {/* 8 */ R(1, 6),
             {R(31, 300), 0, 0, 0, R(61, 225), R(-2, 9), R(13, 900)}}, //
            {/* 9 */ R(2, 3),
             {2, 0, 0, R(-53, 6), R(704, 45), R(-107, 9), R(67, 90), 3}}, //
            {/* 10 */ R(1, 3),
             {R(-91, 108), 0, 0, R(23, 108), R(-976, 135), R(311, 54),
              R(-19, 60), R(17, 6), R(-1, 12)}}, //
            {/* 11 */ 1,
             {R(2383, 4100), 0, 0, R(-341, 164), R(4496, 1025), R(-301, 82),
              R(2133, 4100), R(45, 82), R(45, 164), R(18, 41)}}, //
                                                                 // {/* 12 */ 0,
            //  {R(3, 205), 0, 0, 0, 0, R(-6, 41), R(-3, 205), R(-3, 41), R(3,
            //  41),
            //   R(6, 41)}}, //
            // {/* 13 */ 1,
            //  {R(-1777, 4100), 0, 0, R(-341, 164), R(4496, 1025), R(-289, 82),
            //   R(2193, 4100), R(51, 82), R(33, 164), R(12, 41), 0, 1}}, //
        },
        {
            R(41, 840), 0, 0, 0, 0, R(34, 105), R(9, 35), R(9, 35), R(9, 280),
            R(9, 280), R(41, 840),
            // 0,
            // 0,
        }};

    // Check Butcher tableau
    const size_t nsteps = get<0>(tableau).size();
    {
      for (size_t step = 0; step < nsteps; ++step) {
        // TODO: Could allow <=
        assert(get<1>(get<0>(tableau).at(step)).size() == step);
        const auto &c = get<0>(get<0>(tableau).at(step));
        const auto &as = get<1>(get<0>(tableau).at(step));
        T x = 0;
        for (const auto &a : as)
          x += a;
        assert(fabs(x - c) <= 10 * numeric_limits<T>::epsilon());
      }
      // TODO: Could allow <=
      assert(get<1>(tableau).size() == nsteps);
      const auto &bs = get<1>(tableau);
      T x = 0;
      for (const auto &b : bs)
        x += b;
      assert(fabs(x - 1) <= 10 * numeric_limits<T>::epsilon());
    }

    const auto old = copy_state(var);

    vector<statecomp_t> ks;
    ks.reserve(nsteps);
    for (size_t step = 0; step < nsteps; ++step) {
      // Skip the first state vector calculation, it is always trivial
      if (step > 0) {
	const auto &c = get<0>(get<0>(tableau).at(step));
	const auto &as = get<1>(get<0>(tableau).at(step));

        // Add scaled RHS to state vector
        vector<CCTK_REAL> factors;
        vector<const statecomp_t *> srcs;
        factors.reserve(as.size() + 1);
        srcs.reserve(as.size() + 1);
        factors.push_back(1.0);
        srcs.push_back(&old);
        for (size_t i = 0; i < as.size(); ++i) {
          if (as.at(i) != 0) {
            factors.push_back(as.at(i) * dt);
            srcs.push_back(&ks.at(i));
          }
        }
        calcupdate(step, c * dt, 0.0, factors, srcs);
        // TODO: Deallocate ks that are not needed any more
      }

      calcrhs(step + 1);
      ks.push_back(copy_state(rhs));
    }

    // Calculate new state vector
    const auto &bs = get<1>(tableau);
    vector<CCTK_REAL> factors;
    vector<const statecomp_t *> srcs;
    factors.reserve(bs.size() + 1);
    srcs.reserve(bs.size() + 1);
    factors.push_back(1);
    srcs.push_back(&old);
    for (size_t i = 0; i < bs.size(); ++i) {
      if (bs.at(i) != 0) {
        factors.push_back(bs.at(i) * dt);
        srcs.push_back(&ks.at(i));
      }
    }
    calcupdate(nsteps, dt, 0.0, factors, srcs);

  } else if (CCTK_EQUALS(method, "DP87")) {

    typedef CCTK_REAL T;
    const auto R = [](T x, T y) { return x / y; };
    // These coefficients are taken from the Einstein Toolkit, thorn
    // CactusNumerical/MoL, file RK87.c, written by Peter Diener,
    // following P. J. Prince and J. R. Dormand, Journal of
    // Computational and Applied Mathematics, volume 7, no 1, 1981
    const tuple<vector<vector<T> >, vector<T> > tableau{
        {
            {/*1*/},                                    //
            {/*2*/ R(1, 18)},                           //
            {/*3*/ R(1, 48), R(1, 16)},                 //
            {/*4*/ R(1, 32), 0, R(3, 32)},              //
            {/*5*/ R(5, 16), 0, -R(75, 64), R(75, 64)}, //
            {/*6*/ R(3, 80), 0, 0, R(3, 16), R(3, 20)}, //
            {/*7*/ R(29443841, 614563906), 0, 0, R(77736538, 692538347),
             -R(28693883, 1125000000), R(23124283, 1800000000)}, //
            {/*8*/ R(16016141, 946692911), 0, 0, R(61564180, 158732637),
             R(22789713, 633445777), R(545815736, 2771057229),
             -R(180193667, 1043307555)}, //
            {/*9*/ R(39632708, 573591083), 0, 0, -R(433636366, 683701615),
             -R(421739975, 2616292301), R(100302831, 723423059),
             R(790204164, 839813087), R(800635310, 3783071287)}, //
            {/*10*/ R(246121993, 1340847787), 0, 0,
             -R(37695042795, 15268766246), -R(309121744, 1061227803),
             -R(12992083, 490766935), R(6005943493, 2108947869),
             R(393006217, 1396673457), R(123872331, 1001029789)}, //
            {/*11*/ -R(1028468189, 846180014), 0, 0, R(8478235783, 508512852),
             R(1311729495, 1432422823), -R(10304129995, 1701304382),
             -R(48777925059, 3047939560), R(15336726248, 1032824649),
             -R(45442868181, 3398467696), R(3065993473, 597172653)}, //
            {/*12*/ R(185892177, 718116043), 0, 0, -R(3185094517, 667107341),
             -R(477755414, 1098053517), -R(703635378, 230739211),
             R(5731566787, 1027545527), R(5232866602, 850066563),
             -R(4093664535, 808688257), R(3962137247, 1805957418),
             R(65686358, 487910083)}, //
            {/*13*/ R(403863854, 491063109), 0, 0, -R(5068492393, 434740067),
             -R(411421997, 543043805), R(652783627, 914296604),
             R(11173962825, 925320556), -R(13158990841, 6184727034),
             R(3936647629, 1978049680), -R(160528059, 685178525),
             R(248638103, 1413531060), 0}, //
        },
        {R(14005451, 335480064), 0, 0, 0, 0, -R(59238493, 1068277825),
         R(181606767, 758867731), R(561292985, 797845732),
         -R(1041891430, 1371343529), R(760417239, 1151165299),
         R(118820643, 751138087), -R(528747749, 2220607170), R(1, 4)}};

    // Check Butcher tableau
    const size_t nsteps = get<0>(tableau).size();
    {
      for (size_t step = 0; step < nsteps; ++step)
        // TODO: Could allow <=
        assert(get<0>(tableau).at(step).size() == step);
      // TODO: Could allow <=
      assert(get<1>(tableau).size() == nsteps);
      const auto &bs = get<1>(tableau);
      T x = 0;
      for (const auto &b : bs)
        x += b;
      assert(fabs(x - 1) <= 10 * numeric_limits<T>::epsilon());
    }

    const auto old = copy_state(var);

    vector<statecomp_t> ks;
    ks.reserve(nsteps);
    for (size_t step = 0; step < nsteps; ++step) {
      // Skip the first state vector calculation, it is always trivial
      if (step > 0) {
	const auto &as = get<0>(tableau).at(step);
	T c = 0;
	for (const auto &a : as)
	  c += a;

        // Add scaled RHS to state vector
        vector<CCTK_REAL> factors;
        vector<const statecomp_t *> srcs;
        factors.reserve(as.size() + 1);
        srcs.reserve(as.size() + 1);
        factors.push_back(1.0);
        srcs.push_back(&old);
        for (size_t i = 0; i < as.size(); ++i) {
          if (as.at(i) != 0) {
            factors.push_back(as.at(i) * dt);
            srcs.push_back(&ks.at(i));
          }
        }
        calcupdate(step, c * dt, 0.0, factors, srcs);
        // TODO: Deallocate ks that are not needed any more
      }

      calcrhs(step + 1);
      ks.push_back(copy_state(rhs));
    }

    // Calculate new state vector
    const auto &bs = get<1>(tableau);
    vector<CCTK_REAL> factors;
    vector<const statecomp_t *> srcs;
    factors.reserve(bs.size() + 1);
    srcs.reserve(bs.size() + 1);
    factors.push_back(1);
    srcs.push_back(&old);
    for (size_t i = 0; i < bs.size(); ++i) {
      if (bs.at(i) != 0) {
        factors.push_back(bs.at(i) * dt);
        srcs.push_back(&ks.at(i));
      }
    }
    calcupdate(nsteps, dt, 0.0, factors, srcs);

  } else if (CCTK_EQUALS(method, "Implicit Euler")) {

    // Implicit definition:
    //   y1 = y0 + h/2 f(y0) + h/2 g(y1)
    //   y2 = y0 + h f(y1) + h g(y1)

    // Implicit RHS:
    //   u1 = G(u0, h)   where   u1 = u0 + h g(u1)

    // Explicit definition:
    //   k1 = f(y0)
    //   y1 = G(y0 + h/2 k1, h/2)
    //   k'2 = (y1 - y0 - h/2 k1) / (h/2)
    //   k2 = f(y1)
    //   y2 = y0 + h k2 + h k'2

    const auto y0 = var.copy(make_valid_int /*all*/ ());

    *const_cast<CCTK_REAL *>(&cctkGH->cctk_time) = old_time;
    if (verbose)
      CCTK_VINFO("Calculating RHS #1 at t=%g", double(cctkGH->cctk_time));
    CallScheduleGroup(cctkGH, "ODESolvers_RHS");
    const auto k1 = rhs.copy(make_valid_int());

    *const_cast<CCTK_REAL *>(&cctkGH->cctk_time) = old_time + dt / 2;
    statecomp_t::lincomb(var, 1, make_array(dt / 2), make_array(&rhs),
                         make_valid_int());
    var.check_valid(make_valid_int(),
                    "ODESolvers after defining new state vector");
    mark_invalid(dep_groups);
    CallScheduleGroup(cctkGH, "ODESolvers_PostStep");

    *const_cast<CCTK_REAL *>(&cctkGH->cctk_time) = old_time + dt / 2;
    *const_cast<CCTK_REAL *>(&cctkGH->cctk_delta_time) = dt / 2;
    if (verbose)
      CCTK_VINFO("Taking implicit step #1 at t=%g with dt=%g",
                 double(cctkGH->cctk_time), double(cctkGH->cctk_delta_time));
    CallScheduleGroup(cctkGH, "ODESolvers_ImplicitStep");
    *const_cast<CCTK_REAL *>(&cctkGH->cctk_delta_time) = dt;

    *const_cast<CCTK_REAL *>(&cctkGH->cctk_time) = old_time + dt;
    CallScheduleGroup(cctkGH, "ODESolvers_PostStep");
    const auto y1 = var.copy(make_valid_int /*all*/ ());

    statecomp_t kprime2;
    statecomp_t::lincomb(kprime2, 0,
                         make_array(-CCTK_REAL(1), +CCTK_REAL(1), -dt / 2),
                         make_array(&y0, &y1, &k1), make_valid_int());

    *const_cast<CCTK_REAL *>(&cctkGH->cctk_time) = old_time + dt;
    if (verbose)
      CCTK_VINFO("Calculating RHS #2 at t=%g", double(cctkGH->cctk_time));
    CallScheduleGroup(cctkGH, "ODESolvers_RHS");
    const auto k2 = rhs.copy(make_valid_int());

    statecomp_t::lincomb(var, 0, make_array(CCTK_REAL(1), dt, dt),
                         make_array(&y0, &k2, &kprime2), make_valid_int());
    var.check_valid(make_valid_int(),
                    "ODESolvers after defining new state vector");
    mark_invalid(dep_groups);

  } else {
    assert(0);
  }

  {
    static Timer timer_free_temps("ODESolvers::Solve::free_temps");
    Interval interval_free_temps(timer_free_temps);
    statecomp_t::free_tmp_mfabs();
  }

  // Reset current time
  *const_cast<CCTK_REAL *>(&cctkGH->cctk_time) = saved_time;

  // TODO: Update time here, and not during time level cycling in the driver
}

} // namespace ODESolvers

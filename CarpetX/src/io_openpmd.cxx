#include "io_openpmd.hxx"

#include "driver.hxx"
#include "timer.hxx"

#include <div.hxx>
#include <vect.hxx>

#include <CactusBase/IOUtil/src/ioutil_CheckpointRecovery.h>
#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <util_Network.h>

#ifdef HAVE_CAPABILITY_openPMD_api

#include <openPMD/openPMD.hpp>

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

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <ios>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#if !openPMD_HAVE_MPI
#error                                                                         \
    "CarpetX requires openPMD_api with MPI support. Please add -DopenPMD_USE_MPI=ON when building openPMD."
#endif

namespace CarpetX {

constexpr bool io_verbose = true;

openPMD::Format get_format() {
  DECLARE_CCTK_PARAMETERS;
  if (CCTK_EQUALS(openpmd_format, "HDF5"))
    return openPMD::Format::HDF5;
  if (CCTK_EQUALS(openpmd_format, "ADIOS1"))
    return openPMD::Format::ADIOS1;
  if (CCTK_EQUALS(openpmd_format, "ADIOS2_auto"))
#if OPENPMDAPI_VERSION_GE(0, 15, 0)
    return openPMD::Format::ADIOS2_BP5;
#else
    return openPMD::Format::ADIOS2;
#endif
#if OPENPMDAPI_VERSION_GE(0, 15, 0)
  if (CCTK_EQUALS(openpmd_format, "ADIOS2_BP"))
    return openPMD::Format::ADIOS2_BP;
  if (CCTK_EQUALS(openpmd_format, "ADIOS2_BP4"))
    return openPMD::Format::ADIOS2_BP4;
  if (CCTK_EQUALS(openpmd_format, "ADIOS2_BP5"))
    return openPMD::Format::ADIOS2_BP5;
#else
  if (CCTK_EQUALS(openpmd_format, "ADIOS2"))
    return openPMD::Format::ADIOS2;
#endif
  if (CCTK_EQUALS(openpmd_format, "ADIOS2_SST"))
    return openPMD::Format::ADIOS2_SST;
  if (CCTK_EQUALS(openpmd_format, "ADIOS2_SSC"))
    return openPMD::Format::ADIOS2_SSC;
  if (CCTK_EQUALS(openpmd_format, "JSON"))
    return openPMD::Format::JSON;
  CCTK_VERROR("The openPMD format \"%s\" is not supported in version %d.%d.%d "
              "of the openPMD_api library",
              openpmd_format, OPENPMDAPI_VERSION_MAJOR,
              OPENPMDAPI_VERSION_MINOR, OPENPMDAPI_VERSION_PATCH);
}

// - fileBased: One file per iteration. Needs templated file name to encode
//   iteration number.
// - groupBased: Multiple iterations per file
// - variableBased: Multiple iterations stored per variable. Needs special
//    support in the backend.
constexpr openPMD::IterationEncoding iterationEncoding =
    openPMD::IterationEncoding::fileBased;

//  constexpr const char options[]
// const std::string options = "{"
//                             "  \"adios2\": {"
//                             "    \"engine\": {"
//                             "      \"type\": \"BP4\","
//                             "      \"parameters\": {"
//                             "        \"BufferGrowthFactor\": \"2.0\""
//                             "      }"
//                             "    },"
//                             "    \"dataset\": {"
//                             "      \"operators\": {"
//                             "        \"type\": \"blosc\","
//                             "        \"parameters\": {"
//                             "          \"clevel\": \"9\","
//                             "          \"doshuffle\": \"BLOSC_BITSHUFFLE\""
//                             "        }"
//                             "      }"
//                             "    }"
//                             "  }"
//                             "}";
const std::string options = "{}";

constexpr bool input_ghosts = false;
constexpr bool output_ghosts = false;

////////////////////////////////////////////////////////////////////////////////

struct Const {
  // From: CODATA Internationally recommended 2018 values of the
  // Fundamental Physical Constants
  static constexpr CCTK_REAL c = 299792458;         // m s-1
  static constexpr CCTK_REAL G = 6.67430e-11;       // m3 kg-1 s-2
  static constexpr CCTK_REAL M_solar = 1.98847e+30; // kg
};

struct Unit {
  // We use c = G = 1, and M_solar as mass unit.
  static constexpr CCTK_REAL velocity = Const::c;                       // m s-1
  static constexpr CCTK_REAL mass = Const::M_solar;                     // kg
  static constexpr CCTK_REAL length = Const::G * mass / pow2(Const::c); // m
  static constexpr CCTK_REAL time = length / velocity;                  // s
};

////////////////////////////////////////////////////////////////////////////////

namespace {

template <typename T, std::size_t N>
constexpr std::array<T, N> reversed(const std::array<T, N> &arr) {
  std::array<T, N> res;
  for (std::size_t n = 0; n < N; ++n)
    res[n] = arr[N - 1 - n];
  return res;
}

template <typename T, int D>
constexpr Arith::vect<T, D> reversed(const Arith::vect<T, D> &arr) {
  Arith::vect<T, D> res;
  for (std::size_t d = 0; d < D; ++d)
    res[d] = arr[D - 1 - d];
  return res;
}

template <typename T> std::vector<T> reversed(const std::vector<T> &vec) {
  std::vector<T> res(vec.size());
  std::reverse_copy(vec.begin(), vec.end(), res.begin());
  return res;
}
template <typename T> std::vector<T> reversed(std::vector<T> &&vec) {
  std::vector<T> res(vec);
  std::reverse(res.begin(), res.end());
  return res;
}

template <typename T = std::uint64_t, typename C>
std::vector<T> to_vector(const C &source) {
  std::vector<T> res(source.size());
  for (std::size_t n = 0; n < res.size(); ++n)
    res[n] = source[n];
  return res;
}

template <typename T> std::string vector_string(const std::vector<T> &vec) {
  std::ostringstream buf;
  buf << "[";
  for (std::size_t n = 0; n < vec.size(); ++n) {
    if (n != 0)
      buf << ",";
    buf << vec[n];
  }
  buf << "]";
  return buf.str();
}

} // namespace

////////////////////////////////////////////////////////////////////////////////

std::string Geometry_string(const openPMD::Mesh::Geometry &geometry) {
  std::ostringstream buf;
  buf << geometry;
  return buf.str();
}

std::string UnitDimension_string(const std::array<double, 7> &unitDimension) {
  std::ostringstream buf;
  buf << "L:" << unitDimension[uint8_t(openPMD::UnitDimension::L)] << " "
      << "M:" << unitDimension[uint8_t(openPMD::UnitDimension::M)] << " "
      << "T:" << unitDimension[uint8_t(openPMD::UnitDimension::T)] << " "
      << "I:" << unitDimension[uint8_t(openPMD::UnitDimension::I)] << " "
      << "θ:" << unitDimension[uint8_t(openPMD::UnitDimension::theta)] << " "
      << "N:" << unitDimension[uint8_t(openPMD::UnitDimension::N)] << " "
      << "J:" << unitDimension[uint8_t(openPMD::UnitDimension::J)];
  return buf.str();
}

////////////////////////////////////////////////////////////////////////////////

struct carpetx_openpmd_t {
  carpetx_openpmd_t() = default;

  carpetx_openpmd_t(const carpetx_openpmd_t &) = delete;
  carpetx_openpmd_t(carpetx_openpmd_t &&) = default;
  carpetx_openpmd_t &operator=(const carpetx_openpmd_t &) = delete;
  carpetx_openpmd_t &operator=(carpetx_openpmd_t &&) = default;

  static std::optional<carpetx_openpmd_t> self;

  ////////////////////////////////////////////////////////////////////////////////

  template <typename T, std::size_t D> struct box_t {
    Arith::vect<T, D> lo, hi;
    constexpr friend bool operator==(const box_t &x, const box_t &y) {
      if (x.empty() && y.empty())
        return true;
      if (x.empty() || y.empty())
        return false;
      return all(x.lo == y.lo) && all(x.hi == y.hi);
    }
    constexpr friend bool operator!=(const box_t &x, const box_t &y) {
      return !(x == y);
    }
    constexpr bool empty() const { return any(hi < lo); }
    constexpr Arith::vect<T, D> shape() const {
      Arith::vect<T, D> sh;
      for (std::size_t d = 0; d < D; ++d)
        sh[d] = hi[d] < lo[d] ? T{0} : hi[d] - lo[d];
      return sh;
    }
    constexpr T size() const {
      const auto sh = shape();
      T sz{1};
      for (std::size_t d = 0; d < D; ++d)
        sz *= sh[d];
      return sz;
    }
    constexpr Arith::vect<T, D> stride() const {
      const Arith::vect<T, D> sh = shape();
      Arith::vect<T, D> str;
      T np{1};
      for (std::size_t d = 0; d < D; ++d)
        str[d] = (np *= sh[d]);
      return str;
    }
    constexpr T linear(const Arith::vect<T, D> &index) const {
      const Arith::vect<T, D> str = stride();
      T lin{0};
      if (D > 0) {
        lin = index[0];
        for (std::size_t d = 1; d < D; ++d)
          lin += index[d] * str[d - 1];
      }
      return lin;
    }
    friend std::ostream &operator<<(std::ostream &os, const box_t<T, D> &box) {
      return os << "box_t{lo:" << box.lo << ",hi:" << box.hi << "}";
    }
  };

  template <typename T, typename I, std::size_t D> struct level_t {
    box_t<T, D> rdomain;
    Arith::vect<bool, D> is_cell_centred;
    box_t<I, D> idomain;
    constexpr Arith::vect<T, D> rcoord(const Arith::vect<I, D> &icoord) const {
      Arith::vect<T, D> r;
      for (std::size_t d = 0; d < D; ++d) {
        const T rlo = rdomain.lo[d];
        const T rhi = rdomain.hi[d];
        const I ilo2 = 2 * idomain.lo[d] - is_cell_centred;
        const I ihi2 = 2 * idomain.hi[d] + is_cell_centred;
        const I i2 = 2 * icoord[d];
        // The expression below has been carefully constructed to
        // avoid round-off errors at i=ilo and i=ihi. Do not rearrange
        // the terms without preserving this property.
        r[d] = (T(ihi2 - i2) / T(ihi2 - ilo2)) * rlo[d] +
               (T(i2 - ilo2) / T(ihi2 - ilo2)) * rhi[d];
      }
      return r;
    }
    std::vector<box_t<I, D> > grids;
    Arith::vect<std::vector<I>, 2> offsets_sizes() const {
      std::vector<I> offsets(grids.size() + 1), sizes(grids.size());
      I offset{0};
      for (std::size_t n = 0; n < grids.size(); ++n) {
        const T size = grids[n].size();
        offsets[n] = offset;
        sizes[n] = size;
        offset += size;
      }
      offsets[grids.size()] = offset;
      return {offsets, sizes};
    }
  };

  template <typename T, typename I, std::size_t D> struct grid_structure_t {
    box_t<T, D> rdomain;
    std::vector<level_t<T, I, D> > levels;
  };

  ////////////////////////////////////////////////////////////////////////////////

  // Allowed characters are only [A-Za-z_]
  static std::string make_meshname(const int gi, const int patch,
                                   const int level) {
    std::string groupname = CCTK_FullGroupName(gi);
    groupname = std::regex_replace(groupname, std::regex("::"), "_");
    for (auto &ch : groupname)
      ch = std::tolower(ch);
    std::ostringstream buf;
    buf << groupname;
    if (patch != -1)
      buf << "_patch" << setw(2) << setfill('0') << patch;
    if (level != -1)
      // The suffix should be `_lvl<N>`. No `setfill`?
      buf << "_lev" << setw(2) << setfill('0') << level;
    return buf.str();
  }

#if 0
  static std_tuple<int, int> interpret_meshname(const std::string &meshname) {
    std::smatch match;
    const bool matched =
        std::regex_match(meshname, match, std::regex("(\\w+)_lev0*(\\d+)"));
    if (!matched)
      CCTK_VERROR("Cannot parse mesh name %s", meshname.c_str());
    const std::string groupname = match[1].str();
    const int gi = CCTK_GroupIndex(groupname.c_str());
    if (gi < 0)
      CCTK_VERROR("Unknown group name %s", groupname.c_str());
    const std::string levelstr = match[2].str();
    const int level = std::stoi(levelstr);
    return {gi, level};
  }
#endif

  // Allowed characters are only [A-Za-z_]
  static std::string make_componentname(const int gi, const int vi) {
    const int v0 = CCTK_FirstVarIndexI(gi);
    std::string varname = CCTK_FullVarName(v0 + vi);
    varname = std::regex_replace(varname, std::regex("::"), "_");
    for (auto &ch : varname)
      ch = std::tolower(ch);
    return varname;
  }

  ////////////////////////////////////////////////////////////////////////////////

  std::optional<std::string> filename;
  std::optional<openPMD::Series> series;
  // std::optional<openPMD::ReadIterations> read_iters;
  std::optional<openPMD::Iteration> read_iter;
  std::optional<openPMD::WriteIterations> write_iters;

  int InputOpenPMDParameters(const std::string &input_dir,
                             const std::string &input_file);
  void InputOpenPMDGridStructure(cGH *cctkGH, const std::string &input_dir,
                                 const std::string &input_file,
                                 int input_iteration);
  void InputOpenPMD(const cGH *const cctkGH,
                    const std::vector<bool> &input_group,
                    const std::string &input_dir,
                    const std::string &input_file);

  void OutputOpenPMD(const cGH *const cctkGH,
                     const std::vector<bool> &output_group,
                     const std::string &output_dir,
                     const std::string &output_file);
};

////////////////////////////////////////////////////////////////////////////////

std::optional<carpetx_openpmd_t> carpetx_openpmd_t::self;

int InputOpenPMDParameters(const std::string &input_dir,
                           const std::string &input_file) {
  if (!carpetx_openpmd_t::self)
    carpetx_openpmd_t::self = std::make_optional<carpetx_openpmd_t>();
  return carpetx_openpmd_t::self->InputOpenPMDParameters(input_dir, input_file);
}
void InputOpenPMDGridStructure(cGH *cctkGH, const std::string &input_dir,
                               const std::string &input_file,
                               int input_iteration) {
  if (!carpetx_openpmd_t::self)
    carpetx_openpmd_t::self = std::make_optional<carpetx_openpmd_t>();
  carpetx_openpmd_t::self->InputOpenPMDGridStructure(
      cctkGH, input_dir, input_file, input_iteration);
}
void InputOpenPMD(const cGH *cctkGH, const std::vector<bool> &input_group,
                  const std::string &input_dir, const std::string &input_file) {
  if (!carpetx_openpmd_t::self)
    carpetx_openpmd_t::self = std::make_optional<carpetx_openpmd_t>();
  carpetx_openpmd_t::self->InputOpenPMD(cctkGH, input_group, input_dir,
                                        input_file);
}

void OutputOpenPMD(const cGH *const cctkGH,
                   const std::vector<bool> &output_group,
                   const std::string &output_dir,
                   const std::string &output_file) {
  if (!carpetx_openpmd_t::self)
    carpetx_openpmd_t::self = std::make_optional<carpetx_openpmd_t>();
  carpetx_openpmd_t::self->OutputOpenPMD(cctkGH, output_group, output_dir,
                                         output_file);
}

void ShutdownOpenPMD() { carpetx_openpmd_t::self.reset(); }

////////////////////////////////////////////////////////////////////////////////

int carpetx_openpmd_t::InputOpenPMDParameters(const std::string &input_dir,
                                              const std::string &input_file) {
  DECLARE_CCTK_PARAMETERS;

  assert(!input_dir.empty());
  assert(!input_file.empty());

  // Set up timers
  static Timer timer("InputOpenPMDParameters");
  Interval interval(timer);

  if (io_verbose)
    CCTK_VINFO("InputOpenPMDParameters...");

  const openPMD::Format format = get_format();

  int input_iteration = -1;

  assert(!series);
  if (!series) {
    if (io_verbose)
      CCTK_VINFO("Creating openPMD object...");
    std::ostringstream buf;
    switch (iterationEncoding) {
    case openPMD::IterationEncoding::fileBased:
      buf << input_dir << "/" << input_file << ".it%08T"
          << openPMD::suffix(format);
      break;
    case openPMD::IterationEncoding::variableBased:
      buf << input_dir << "/" << input_file << openPMD::suffix(format);
      break;
    default:
      abort();
    }
    filename = std::make_optional<std::string>(buf.str());
    try {
      series = std::make_optional<openPMD::Series>(
          *filename, openPMD::Access::READ_ONLY, MPI_COMM_WORLD, options);
    } catch (const openPMD::no_such_file_error &) {
      // Did not find a checkpoint file
      if (io_verbose) {
        CCTK_VINFO("Not recovering parameters:");
        CCTK_VINFO("  Could not find an openPMD checkpoint file \"%s\"",
                   filename->c_str());
      }
      return -1; // no iteration found
    }

    // Find largest iteration
    for (auto iter = series->iterations.begin();
         iter != series->iterations.end(); ++iter)
      input_iteration = iter->first;

    if (input_iteration < 0) {
      // Did not find a checkpoint file
      if (io_verbose) {
        CCTK_VINFO("Not recovering parameters:");
        CCTK_VINFO("  Could not find an openPMD checkpoint file \"%s\"",
                   filename->c_str());
      }
      return -1; // no iteration found
    }

    read_iter = std::make_optional<openPMD::Iteration>(
        series->iterations[input_iteration]);
  }
  assert(filename);
  assert(series);
  // assert(read_iters);
  assert(read_iter);

  CCTK_VINFO("Recovering parameters from checkpoint file \"%s\" iteration %d",
             filename->c_str(), input_iteration);

  // Read metadata
  {
    const bool has_parameters = read_iter->containsAttribute("AllParameters");
    assert(has_parameters);
    const openPMD::Attribute parameters_attr =
        read_iter->getAttribute("AllParameters");
    assert(parameters_attr.dtype == openPMD::Datatype::STRING);
    const string parameters = parameters_attr.get<std::string>();
    IOUtil_SetAllParameters(parameters.data());
  }

  return input_iteration;
}

void carpetx_openpmd_t::InputOpenPMDGridStructure(cGH *cctkGH,
                                                  const std::string &input_dir,
                                                  const std::string &input_file,
                                                  const int input_iteration) {
  DECLARE_CCTK_PARAMETERS;

  assert(!input_dir.empty());
  assert(!input_file.empty());
  assert(input_iteration >= 0);

  // Set up timers
  static Timer timer("InputOpenPMDGridStructure");
  Interval interval(timer);

  if (io_verbose)
    CCTK_VINFO("InputOpenPMDGridStructure...");

  assert(filename);
  assert(series);
  // assert(read_iters);
  assert(read_iter);

  cctkGH->cctk_iteration = input_iteration;
  cctkGH->cctk_time = read_iter->time<double>();

  // TODO: Check whether attribute exists and has correct type
  const int ndims = read_iter->getAttribute("numDims").get<std::int64_t>();
  assert(ndims >= 0);

  const int npatches =
      read_iter->getAttribute("numPatches").get<std::int64_t>();
  assert(npatches == ghext->num_patches());
  std::vector<std::string> patch_suffixes;
  if (npatches == 1) {
    assert(read_iter->getAttribute("patchSuffixes").dtype ==
           openPMD::Datatype::STRING);
    const std::string patch_suffix =
        read_iter->getAttribute("patchSuffixes").get<std::string>();
    patch_suffixes.resize(1);
    patch_suffixes.at(0) = patch_suffix;
  } else {
    assert(read_iter->getAttribute("patchSuffixes").dtype ==
           openPMD::Datatype::VEC_STRING);
    patch_suffixes = read_iter->getAttribute("patchSuffixes")
                         .get<std::vector<std::string> >();
  }
  // TODOPATCH: Handle multiple patches
  assert(ghext->num_patches() == 1);
  const int patch = 0;

  const int nlevels =
      read_iter->getAttribute("numLevels" + patch_suffixes.at(patch))
          .get<std::int64_t>();
  assert(nlevels >= 0);
  std::vector<std::string> level_suffixes;
  if (nlevels == 1) {
    assert(read_iter->getAttribute("levelSuffixes" + patch_suffixes.at(patch))
               .dtype == openPMD::Datatype::STRING);
    const std::string level_suffix =
        read_iter->getAttribute("levelSuffixes" + patch_suffixes.at(patch))
            .get<std::string>();
    level_suffixes.resize(1);
    level_suffixes.at(0) = level_suffix;
  } else {
    assert(read_iter->getAttribute("levelSuffixes" + patch_suffixes.at(patch))
               .dtype == openPMD::Datatype::VEC_STRING);
    level_suffixes =
        read_iter->getAttribute("levelSuffixes" + patch_suffixes.at(patch))
            .get<std::vector<std::string> >();
  }
  assert(int(level_suffixes.size()) == nlevels);

  assert(ndims == 3);
  assert(nlevels > 0);
  auto &patchdata = ghext->patchdata.at(patch);
  patchdata.amrcore->SetFinestLevel(nlevels - 1);

  // Read FabArrayBase (component positions and shapes)
  for (int level = 0; level < nlevels; ++level) {
    const std::vector<std::int64_t> chunk_infos =
        read_iter->getAttribute("chunkInfo" + level_suffixes.at(level))
            .get<std::vector<std::int64_t> >();
    assert(chunk_infos.size() % (2 * ndims) == 0);
    const int nfabs = chunk_infos.size() / (2 * ndims);
    amrex::Vector<amrex::Box> levboxes(nfabs);
    for (int component = 0; component < nfabs; ++component) {
      const int offset = 2 * ndims * component;
      amrex::IntVect small, big;
      for (int d = 0; d < ndims; ++d) {
        small[d] = chunk_infos.at(offset + 0 * ndims + ndims - 1 - d);
        big[d] = chunk_infos.at(offset + 1 * ndims + ndims - 1 - d);
      }
      levboxes.at(component) = amrex::Box(small, big - 1);
    }

    // Don't set coarse level domain; this is already set by the driver
    if (level > 0) {
      amrex::Geometry geom = patchdata.amrcore->Geom(level - 1);
      geom.refine({2, 2, 2});
      patchdata.amrcore->SetGeometry(level, geom);
    }

    amrex::BoxList boxlist(std::move(levboxes));
    amrex::BoxArray boxarray(std::move(boxlist));
    patchdata.amrcore->SetBoxArray(level, boxarray);

    amrex::DistributionMapping dm(boxarray);
    patchdata.amrcore->SetDistributionMap(level, dm);

    patchdata.amrcore->SetupLevel(level, boxarray, dm,
                                  []() { return "Recovering"; });
  } // for level
}

void carpetx_openpmd_t::InputOpenPMD(const cGH *const cctkGH,
                                     const std::vector<bool> &input_group,
                                     const std::string &input_dir,
                                     const std::string &input_file) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  // Set up timers
  static Timer timer("InputOpenPMD");
  Interval interval(timer);

  if (std::count(input_group.begin(), input_group.end(), true) == 0)
    return;

  if (io_verbose)
    CCTK_VINFO("InputOpenPMD...");

  const bool is_root = CCTK_MyProc(nullptr) == 0;
  if (is_root) {
    CCTK_VINFO("  openPMD input for groups:");
    for (int gi = 0; gi < CCTK_NumGroups(); ++gi)
      if (input_group.at(gi))
        CCTK_VINFO("    %s", CCTK_FullGroupName(gi));
  }

  if (!series) {
    if (io_verbose)
      CCTK_VINFO("Creating openPMD object...");
    const openPMD::Format format = get_format();
    std::ostringstream buf;
    switch (iterationEncoding) {
    case openPMD::IterationEncoding::fileBased:
      buf << input_dir << "/" << input_file << ".it%08T"
          << openPMD::suffix(format);
      break;
    case openPMD::IterationEncoding::variableBased:
      buf << input_dir << "/" << input_file << openPMD::suffix(format);
      break;
    default:
      abort();
    }
    filename = std::make_optional<std::string>(buf.str());
    series = std::make_optional<openPMD::Series>(
        *filename, openPMD::Access::READ_ONLY, MPI_COMM_WORLD, options);
    assert(series->iterations.count(cctkGH->cctk_iteration));
    read_iter = std::make_optional<openPMD::Iteration>(
        series->iterations[cctkGH->cctk_iteration]);
  }
  assert(filename);
  assert(series);
  // assert(read_iters);
  assert(read_iter);

  CCTK_VINFO("  iteration: %d", cctkGH->cctk_iteration);

  const CCTK_REAL time = read_iter->time<CCTK_REAL>();
  const CCTK_REAL dt = read_iter->dt<CCTK_REAL>();
  const double timeUnitSI = read_iter->timeUnitSI();
  CCTK_VINFO("  time: %f", double(time));
  CCTK_VINFO("  dt: %f", double(dt));
  CCTK_VINFO("  time unit SI: %f", timeUnitSI);

  openPMD::Container<openPMD::Mesh> &meshes = read_iter->meshes;
  CCTK_VINFO("  found %d meshes", int(meshes.size()));

  if (io_verbose) {
    for (auto mesh_iter = meshes.begin(); mesh_iter != meshes.end();
         ++mesh_iter) {
      const std::string &mesh_name = mesh_iter->first;
      openPMD::Mesh &mesh = mesh_iter->second;
      CCTK_VINFO("    mesh: %s", mesh_name.c_str());

      const openPMD::Mesh::Geometry geometry = mesh.geometry();
      const std::vector<std::string> axisLabels = mesh.axisLabels();
      const std::vector<CCTK_REAL> gridSpacing = mesh.gridSpacing<CCTK_REAL>();
      const std::vector<double> gridGlobalOffset = mesh.gridGlobalOffset();
      const double gridUnitSI = mesh.gridUnitSI();
      const std::array<double, 7> unitDimension = mesh.unitDimension();
      const CCTK_REAL timeOffset = mesh.timeOffset<CCTK_REAL>();
      CCTK_VINFO("      geometry: %s", Geometry_string(geometry).c_str());
      CCTK_VINFO("      axis labels: %s",
                 vector_string(reversed(axisLabels)).c_str());
      CCTK_VINFO("      grid spacing: %s",
                 vector_string(reversed(gridSpacing)).c_str());
      CCTK_VINFO("      grid global offset: %s",
                 vector_string(reversed(gridGlobalOffset)).c_str());
      CCTK_VINFO("      grid unit SI: %f", gridUnitSI);
      CCTK_VINFO("      unit dimension: %s",
                 UnitDimension_string(unitDimension).c_str());
      CCTK_VINFO("      time offset: %f", double(timeOffset));

      CCTK_VINFO("      found %d components", int(mesh.size()));
      openPMD::Extent extent;
      for (auto record_component_iter = mesh.begin();
           record_component_iter != mesh.end(); ++record_component_iter) {
        const std::string &record_component_name = record_component_iter->first;
        openPMD::MeshRecordComponent &record_component =
            record_component_iter->second;
        CCTK_VINFO("        component: %s", record_component_name.c_str());

        const std::vector<CCTK_REAL> position =
            record_component.position<CCTK_REAL>();
        CCTK_VINFO("          position: %s",
                   vector_string(reversed(position)).c_str());

        const int ndims = record_component.getDimensionality();
        if (extent.empty())
          extent = record_component.getExtent();
        else
          assert(extent == record_component.getExtent());
        CCTK_VINFO("          ndims: %d", ndims);
        CCTK_VINFO("          extent: %s",
                   vector_string(reversed(extent)).c_str());

        const std::vector<openPMD::WrittenChunkInfo> chunks =
            record_component.availableChunks();
        CCTK_VINFO("          found %d chunks", int(chunks.size()));
        if (mesh_iter == meshes.begin() &&
            record_component_iter == mesh.begin()) {
          for (std::size_t n = 0; n < chunks.size(); ++n) {
            const openPMD::WrittenChunkInfo &chunk = chunks.at(n);
            CCTK_VINFO("            chunk: %d   start: %s   count: %s", int(n),
                       vector_string(reversed(chunk.offset)).c_str(),
                       vector_string(reversed(chunk.extent)).c_str());
          }
        }

      } // for record_component
    }   // for mesh
  }

  // First read grid functions in a loop over patches and levels

  // Loop over patches
  for (const auto &patchdata : ghext->patchdata) {
    // Loop over levels
    for (const auto &leveldata : patchdata.leveldata) {
      if (io_verbose)
        CCTK_VINFO("Reading patch %d level %d", patchdata.patch,
                   leveldata.level);

      // Determine grid structure

      const int *const nghosts = cctkGH->cctk_nghostzones;
      const amrex::Geometry &geom = patchdata.amrcore->Geom(leveldata.level);
      const amrex::Real *const xlo = geom.ProbLo();
      const amrex::Real *const xhi = geom.ProbHi();
      const amrex::Real *const dx = geom.CellSize();
      const box_t<CCTK_REAL, 3> rdomain{
          .lo = {xlo[0] - input_ghosts * nghosts[0] * dx[0],
                 xlo[1] - input_ghosts * nghosts[1] * dx[1],
                 xlo[2] - input_ghosts * nghosts[2] * dx[2]},
          .hi = {xhi[0] + input_ghosts * nghosts[0] * dx[0],
                 xhi[1] + input_ghosts * nghosts[1] * dx[1],
                 xhi[2] + input_ghosts * nghosts[2] * dx[2]}};
      const amrex::Box &dom = geom.Domain();
      const amrex::IntVect &ilo = dom.smallEnd();
      const amrex::IntVect &ihi = dom.bigEnd();
      // The domain is always vertex centred. The tensor components are
      // then staggered if necessary.
      const box_t<int, 3> idomain{
          .lo = {ilo[0] - input_ghosts * nghosts[0],
                 ilo[1] - input_ghosts * nghosts[1],
                 ilo[2] - input_ghosts * nghosts[2]},
          .hi = {ihi[0] + input_ghosts * nghosts[0] + 1 + 1,
                 ihi[1] + input_ghosts * nghosts[1] + 1 + 1,
                 ihi[2] + input_ghosts * nghosts[2] + 1 + 1}};
      if (io_verbose) {
        CCTK_VINFO("Level: %d", leveldata.level);
        CCTK_VINFO("  xmin: [%f,%f,%f]", double(rdomain.lo[0]),
                   double(rdomain.lo[1]), double(rdomain.lo[2]));
        CCTK_VINFO("  xmax: [%f,%f,%f]", double(rdomain.hi[0]),
                   double(rdomain.hi[1]), double(rdomain.hi[2]));
        CCTK_VINFO("  imin: [%d,%d,%d]", int(idomain.lo[0]), int(idomain.lo[1]),
                   int(idomain.lo[2]));
        CCTK_VINFO("  imax: [%d,%d,%d]", int(idomain.hi[0]), int(idomain.hi[1]),
                   int(idomain.hi[2]));
      }

      const int numgroups = CCTK_NumGroups();
      for (int gi = 0; gi < numgroups; ++gi) {
        if (input_group.at(gi)) {

          // Check group properties

          cGroup cgroup;
          const int ierr = CCTK_GroupData(gi, &cgroup);
          assert(!ierr);
          if (cgroup.grouptype != CCTK_GF)
            continue;
          assert(cgroup.vartype == CCTK_VARIABLE_REAL);
          assert(cgroup.dim == 3);
          // cGroupDynamicData cgroupdynamicdata;
          // ierr = CCTK_GroupDynamicData(cctkGH, gi, &cgroupdynamicdata);
          // assert(!ierr);
          // TODO: Check whether group has storage
          // TODO: Check whether data are valid

          if (io_verbose)
            CCTK_VINFO("Reading group %s...", CCTK_FullGroupName(gi));

          auto &groupdata = *leveldata.groupdata.at(gi);
          // const int firstvarindex = groupdata.firstvarindex;
          const int numvars = groupdata.numvars;
          const int tl = 0;
          amrex::MultiFab &mfab = *groupdata.mfab[tl];
          const amrex::IndexType &indextype = mfab.ixType();
          const Arith::vect<bool, 3> is_cell_centred{indextype.cellCentered(0),
                                                     indextype.cellCentered(1),
                                                     indextype.cellCentered(2)};

          const int num_local_components = mfab.local_size();

          // Read mesh

          const std::string meshname =
              make_meshname(gi, leveldata.patch, leveldata.level);
          if (io_verbose)
            CCTK_VINFO("Reading mesh %s...", meshname.c_str());
          assert(read_iter->meshes.count(meshname));
          const openPMD::Mesh &mesh = read_iter->meshes.at(meshname);
          // TODO: The openPMD standard says to add an attribute
          // `refinementRatio`, which is a vector of integers

          // Define tensor components

          // TODO: Set component names according to the tensor type
          std::vector<openPMD::MeshRecordComponent> record_components;
          record_components.reserve(numvars);
          openPMD::Extent extent;
          for (int vi = 0; vi < numvars; ++vi) {
            const std::string componentname = make_componentname(gi, vi);
            assert(mesh.count(componentname));
            record_components.push_back(mesh.at(componentname));
            const openPMD::MeshRecordComponent &record_component =
                record_components.back();
            if (vi == 0)
              extent = record_component.getExtent();
            else
              assert(extent == record_component.getExtent());
          }
          assert(int(record_components.size()) == numvars);

          // Read data

          if (io_verbose)
            CCTK_VINFO("Reading %d variables with %d components...", numvars,
                       num_local_components);

          const bool group_has_no_ghosts = groupdata.nghostzones[0] == 0 &&
                                           groupdata.nghostzones[1] == 0 &&
                                           groupdata.nghostzones[2] == 0;

          // Loop over components (AMReX boxes)
          for (int local_component = 0; local_component < num_local_components;
               ++local_component) {
            const int component = mfab.IndexArray().at(local_component);

            const amrex::Box &fabbox =
                mfab.fabbox(component); // exterior (with ghosts)
            const box_t<int, 3> extbox{
                .lo = {fabbox.smallEnd(0), fabbox.smallEnd(1),
                       fabbox.smallEnd(2)},
                .hi = {fabbox.bigEnd(0) + 1, fabbox.bigEnd(1) + 1,
                       fabbox.bigEnd(2) + 1}};
            const amrex::Box &validbox =
                mfab.box(component); // interior (without ghosts)
            const box_t<int, 3> intbox{
                .lo = {validbox.smallEnd(0), validbox.smallEnd(1),
                       validbox.smallEnd(2)},
                .hi = {validbox.bigEnd(0) + 1, validbox.bigEnd(1) + 1,
                       validbox.bigEnd(2) + 1}};
            const box_t<int, 3> &box = input_ghosts ? extbox : intbox;

            const openPMD::Offset start =
                to_vector(reversed(box.lo - idomain.lo));
            const openPMD::Extent count = to_vector(reversed(box.shape()));
            const int np = box.size();
            assert(int(count.at(0) * count.at(1) * count.at(2)) == np);
            for (int d = 0; d < 3; ++d)
              // assert(start.at(d) >= 0);
              assert(
                  start.at(d) <
                  std::numeric_limits<
                      std::remove_reference_t<decltype(start.at(d))> >::max() /
                      2);
            for (int d = 0; d < 3; ++d)
              assert(start.at(d) + count.at(d) <= extent.at(d));

            amrex::FArrayBox &fab = mfab[component];
            for (int vi = 0; vi < numvars; ++vi) {

              if (input_ghosts || intbox == extbox) {
                CCTK_REAL *const ptr = fab.dataPtr() + vi * np;
#if OPENPMDAPI_VERSION_GE(0, 15, 0)
                record_components.at(vi).loadChunkRaw(ptr, start, count);
#else
                record_components.at(vi).loadChunk(openPMD::shareRaw(ptr),
                                                   start, count);
#endif

              } else {
                const int amrex_size = extbox.size();
                CCTK_REAL *const amrex_var_ptr =
                    fab.dataPtr() + vi * amrex_size;
                const Arith::vect<int, 3> amrex_shape = extbox.shape();
                const Arith::vect<int, 3> amrex_offset = box.lo - extbox.lo;
                constexpr int amrex_di = 1;
                const int amrex_dj = amrex_di * amrex_shape[0];
                const int amrex_dk = amrex_dj * amrex_shape[1];
                // const int amrex_np = amrex_dk * amrex_shape[2];
                CCTK_REAL *const amrex_ptr =
                    amrex_var_ptr + amrex_di * amrex_offset[0] +
                    amrex_dj * amrex_offset[1] + amrex_dk * amrex_offset[2];
                const Arith::vect<int, 3> contig_shape = box.shape();
                constexpr int contig_di = 1;
                const int contig_dj = contig_di * contig_shape[0];
                const int contig_dk = contig_dj * contig_shape[1];
                const int contig_np = contig_dk * contig_shape[2];
                assert(contig_np == np);
                CCTK_REAL *const contig_ptr =
                    amrex_var_ptr + extbox.size() - box.size();
                // TODO: optimize memory layout
#if 0
              CCTK_REAL *const contig_ptr =
                  ptr + contig_di * (contig_shape[0] - 1) +
                  contig_dj * (contig_shape[1] - 1) +
                  contig_dk * (contig_shape[2] - 1) + 1 - contig_np;
              assert(&amrex_ptr[amrex_di * (amrex_shape[0] - 1) +
                                amrex_dj * (amrex_shape[1] - 1) +
                                amrex_dk * (amrex_shape[2] - 1)] ==
                     &contig_ptr[contig_di * (contig_shape[0] - 1) +
                                 contig_dj * (contig_shape[1] - 1) +
                                 contig_dk * (contig_shape[2] - 1)]);
#endif
                const auto expand_box = [=](CCTK_REAL *const contig_ptr) {
                  for (int k = 0; k < contig_shape[2]; ++k)
                    for (int j = 0; j < contig_shape[1]; ++j)
#pragma omp simd
                      for (int i = 0; i < contig_shape[0]; ++i)
                        amrex_ptr[amrex_di * i + amrex_dj * j + amrex_dk * k] =
                            contig_ptr[contig_di * i + contig_dj * j +
                                       contig_dk * k];
                  if (poison_undefined_values) {
                    // TODO: Get this from `valid.cxx`
#if defined CCTK_REAL_PRECISION_4
                    constexpr std::uint32_t ipoison = 0xffc00000UL + 0xdead;
#elif defined CCTK_REAL_PRECISION_8
                    constexpr std::uint64_t ipoison =
                        0xfff8000000000000ULL + 0xdeadbeef;
#endif
                    static_assert(sizeof ipoison == sizeof(CCTK_REAL));
                    CCTK_REAL poison;
                    std::memcpy(&poison, &ipoison, sizeof poison);
                    for (int k = extbox.lo[2]; k < extbox.hi[2]; ++k) {
                      for (int j = extbox.lo[1]; j < extbox.hi[1]; ++j) {
                        for (int i = extbox.lo[0]; i < extbox.hi[0]; ++i) {
                          const Arith::vect<int, dim> I{i, j, k};
                          if (any(I < box.lo || I >= box.hi))
                            amrex_var_ptr[amrex_di * (i - extbox.lo[0]) +
                                          amrex_dj * (j - extbox.lo[1]) +
                                          amrex_dk * (k - extbox.lo[2])] =
                                poison;
                        }
                      }
                    }
                  }
                };
                record_components.at(vi).loadChunk(
                    std::shared_ptr<CCTK_REAL>(contig_ptr, expand_box), start,
                    count);
              }

            } // for vi
          }   // for local_component

          // Mark read variables as valid
          for (int vi = 0; vi < numvars; ++vi)
            groupdata.valid.at(tl).at(vi).set_all(
                input_ghosts || group_has_no_ghosts ? make_valid_all()
                                                    : make_valid_int(),
                []() { return "read from openPMD file"; });
        }
      } // for gi

    } // for leveldata
  }   // for patchdata

  // Next read grid scalars and grid arrays

  {
    const int numgroups = CCTK_NumGroups();
    for (int gi = 0; gi < numgroups; ++gi) {
      if (input_group.at(gi)) {

        // Check group properties

        cGroup cgroup;
        const int ierr = CCTK_GroupData(gi, &cgroup);
        assert(!ierr);
        if (cgroup.grouptype == CCTK_GF)
          continue;
        assert(cgroup.vartype == CCTK_VARIABLE_REAL);
        assert(cgroup.disttype == CCTK_DISTRIB_CONSTANT);
        assert(cgroup.dim >= 0);
        assert(cgroup.dim <= 3);

        if (io_verbose)
          CCTK_VINFO("Reading group %d %s...", gi, CCTK_FullGroupName(gi));

        auto &groupdata = *ghext->globaldata.arraygroupdata.at(gi);
        // const int firstvarindex = groupdata.firstvarindex;
        const int numvars = groupdata.numvars;
        const int tl = 0;

        // Determine grid structure

        using ivect = Arith::vect<int, dim>;

        const box_t<int, 3> idomain{.lo = ivect{0, 0, 0},
                                    .hi = ivect(groupdata.gsh)};

        // Read mesh

        const std::string meshname = make_meshname(gi, -1, -1);
        assert(read_iter->meshes.count(meshname));
        const openPMD::Mesh &mesh = read_iter->meshes.at(meshname);
        // TODO: The openPMD standard says to add an attribute
        // `refinementRatio`, which is a vector of integers

        // Define tensor components

        // TODO: Set component names according to the tensor type
        std::vector<openPMD::MeshRecordComponent> record_components;
        record_components.reserve(numvars);
        openPMD::Extent extent;
        for (int vi = 0; vi < numvars; ++vi) {
          const std::string componentname = make_componentname(gi, vi);
          assert(mesh.count(componentname));
          record_components.push_back(mesh.at(componentname));
          const openPMD::MeshRecordComponent &record_component =
              record_components.back();
          if (vi == 0)
            extent = record_component.getExtent();
          else
            assert(extent == record_component.getExtent());
        }
        assert(int(record_components.size()) == numvars);

        // Read data

        if (io_verbose)
          CCTK_VINFO("Reading %d variables...", numvars);

        const bool group_has_no_ghosts = groupdata.nghostzones[0] == 0 &&
                                         groupdata.nghostzones[1] == 0 &&
                                         groupdata.nghostzones[2] == 0;

        // exterior (with ghosts)
        for (int d = 0; d < dim; ++d)
          assert(groupdata.lsh[d] == groupdata.gsh[d]);
        assert(all(Arith::vect<int, dim>(groupdata.lsh) ==
                   Arith::vect<int, dim>(groupdata.gsh)));
        const box_t<int, 3> extbox{.lo = ivect{0, 0, 0},
                                   .hi = ivect(groupdata.lsh)};
        // interior (without ghosts)
        const box_t<int, 3> intbox{.lo = ivect(groupdata.nghostzones),
                                   .hi = ivect(groupdata.lsh) -
                                         ivect(groupdata.nghostzones)};
        // It seems that openPMD assumes that chunks do not have ghost zones
        assert(!input_ghosts);
        const box_t<int, 3> &box = input_ghosts ? extbox : intbox;

        const openPMD::Offset start = to_vector(reversed(box.lo - idomain.lo));
        const openPMD::Extent count = to_vector(reversed(box.shape()));
        const int np = box.size();
        assert(int(count.at(0) * count.at(1) * count.at(2)) == np);
        for (int d = 0; d < 3; ++d)
          // assert(start.at(d) >= 0);
          assert(start.at(d) <
                 std::numeric_limits<
                     std::remove_reference_t<decltype(start.at(d))> >::max() /
                     2);
        for (int d = 0; d < 3; ++d)
          assert(start.at(d) + count.at(d) <= extent.at(d));

        const Arith::vect<int, 3> cactus_shape = extbox.shape();
        constexpr int cactus_di = 1;
        const int cactus_dj = cactus_di * cactus_shape[0];
        const int cactus_dk = cactus_dj * cactus_shape[1];
        const int cactus_np = cactus_dk * cactus_shape[2];
        assert(cactus_di > 0);
        assert(cactus_dj > 0);
        assert(cactus_dk > 0);
        assert(cactus_np > 0);
        for (int vi = 0; vi < numvars; ++vi) {
          CCTK_REAL *const cactus_var_ptr = groupdata.data.at(tl).dataPtr(vi);
          if (input_ghosts || intbox == extbox) {
#if OPENPMDAPI_VERSION_GE(0, 15, 0)
            record_components.at(vi).loadChunkRaw(cactus_var_ptr, start, count);
#else
            record_components.at(vi).loadChunk(
                openPMD::shareRaw(cactus_var_ptr), start, count);
#endif

          } else {
            const Arith::vect<int, 3> cactus_offset = box.lo - extbox.lo;
            CCTK_REAL *const cactus_ptr =
                cactus_var_ptr + cactus_di * cactus_offset[0] +
                cactus_dj * cactus_offset[1] + cactus_dk * cactus_offset[2];
            const Arith::vect<int, 3> contig_shape = box.shape();
            constexpr int contig_di = 1;
            const int contig_dj = contig_di * contig_shape[0];
            const int contig_dk = contig_dj * contig_shape[1];
            const int contig_np = contig_dk * contig_shape[2];
            assert(contig_np == np);
            CCTK_REAL *const contig_ptr =
                cactus_var_ptr + extbox.size() - box.size();
            // TODO: optimize memory layout
            const auto expand_box = [=](CCTK_REAL *const contig_ptr) {
              for (int k = 0; k < contig_shape[2]; ++k)
                for (int j = 0; j < contig_shape[1]; ++j)
#pragma omp simd
                  for (int i = 0; i < contig_shape[0]; ++i)
                    cactus_ptr[cactus_di * i + cactus_dj * j + cactus_dk * k] =
                        contig_ptr[contig_di * i + contig_dj * j +
                                   contig_dk * k];
              if (poison_undefined_values) {
                // TODO: Get this from `valid.cxx`
#if defined CCTK_REAL_PRECISION_4
                constexpr std::uint32_t ipoison = 0xffc00000UL + 0xdead;
#elif defined CCTK_REAL_PRECISION_8
                constexpr std::uint64_t ipoison =
                    0xfff8000000000000ULL + 0xdeadbeef;
#endif
                CCTK_REAL poison;
                std::memcpy(&poison, &ipoison, sizeof poison);
                for (int k = extbox.lo[2]; k < extbox.hi[2]; ++k) {
                  for (int j = extbox.lo[1]; j < extbox.hi[1]; ++j) {
                    for (int i = extbox.lo[0]; i < extbox.hi[0]; ++i) {
                      const Arith::vect<int, dim> I{i, j, k};
                      if (any(I < box.lo || I >= box.hi))
                        cactus_var_ptr[cactus_di * i + cactus_dj * j +
                                       cactus_dk * k] = poison;
                    }
                  }
                }
              }
            };
            record_components.at(vi).loadChunk(
                std::shared_ptr<CCTK_REAL>(contig_ptr, expand_box), start,
                count);
          }

          // Mark read variables as valid
          groupdata.valid.at(tl).at(vi).set_all(
              input_ghosts || group_has_no_ghosts ? make_valid_all()
                                                  : make_valid_int(),
              []() { return "read from openPMD file"; });

        } // for vi
      }
    } // for gi
  }

  if (io_verbose)
    CCTK_VINFO("InputOpenPMD: Performing all reads...");
  series->flush();

  if (io_verbose)
    CCTK_VINFO("InputOpenPMD: Closing iteration...");
  read_iter->close();

  if (io_verbose)
    CCTK_VINFO("InputOpenPMD: Deallocating objects...");
  read_iter.reset();
  series.reset();
  filename.reset();

  if (io_verbose)
    CCTK_VINFO("InputOpenPMD done.");

  if (io_verbose)
    timer.print();
}

////////////////////////////////////////////////////////////////////////////////

void carpetx_openpmd_t::OutputOpenPMD(const cGH *const cctkGH,
                                      const std::vector<bool> &output_group,
                                      const std::string &output_dir,
                                      const std::string &output_file) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  // Set up timers
  static Timer timer("OutputOpenPMD");
  Interval interval(timer);

  if (std::count(output_group.begin(), output_group.end(), true) == 0)
    return;

  if (io_verbose)
    CCTK_VINFO("OutputOpenPMD...");

  const openPMD::Format format = get_format();

  if (!series) {

    if (io_verbose)
      CCTK_VINFO("Creating openPMD object...");
    const int mode = 0755;
    static once_flag create_directory;
    call_once(create_directory, [&]() {
      const int ierr = CCTK_CreateDirectory(mode, output_dir.c_str());
      assert(ierr >= 0);
    });
    std::ostringstream buf;
    switch (iterationEncoding) {
    case openPMD::IterationEncoding::fileBased:
      buf << output_dir << "/" << output_file << ".it%08T"
          << openPMD::suffix(format);
      break;
    case openPMD::IterationEncoding::variableBased:
      buf << output_dir << "/" << output_file << openPMD::suffix(format);
      break;
    default:
      abort();
    }
    filename = std::make_optional<std::string>(buf.str());
    static bool is_first_output = true;
    const openPMD::Access access =
        iterationEncoding == openPMD::IterationEncoding::fileBased
            ? openPMD::Access::CREATE
        : iterationEncoding == openPMD::IterationEncoding::variableBased
            ? (is_first_output ? openPMD::Access::CREATE
                               : openPMD::Access::READ_WRITE)
            : openPMD::Access::READ_ONLY /*error*/;
    is_first_output = false;
    CCTK_VINFO("  options: %s", options.c_str());
    series = std::make_optional<openPMD::Series>(*filename, access,
                                                 MPI_COMM_WORLD, options);
    series->setIterationEncoding(iterationEncoding);

    {
      char const *const user = getenv("USER");
      if (user)
        series->setAuthor(user);
    }
    // Software is always "openPMD-api"
    // series->setSoftware("Einstein Toolkit <https://einsteintoolkit.org>");
    // series->setSoftwareVersion("...");
    // Date is set automatically
    // const std::time_t t = std::time(nullptr);
    // char date[100];
    // std::strftime(date, sizeof date, "%Y-%m-%dT%H:%M:%S",
    // std::localtime(&t)); series->setDate(date);
    // series->setSoftwareDependencies("...");
    {
      char hostname[1000];
      Util_GetHostName(hostname, sizeof hostname);
      series->setMachine(hostname);
    }

    write_iters =
        std::make_optional<openPMD::WriteIterations>(series->writeIterations());
  }
  assert(filename);
  assert(series);
  assert(write_iters);

  if (io_verbose)
    CCTK_VINFO("Creating iteration %d...", cctk_iteration);
  openPMD::Iteration iter = (*write_iters)[cctk_iteration];
  iter.setTime(cctk_time);
  iter.setDt(cctk_delta_time);
  iter.setTimeUnitSI(Unit::time);

  const int myproc = CCTK_MyProc(cctkGH);
  const int ioproc = 0;

  // Write parameters
  if (myproc == ioproc) {
    char *const data = IOUtil_GetAllParameters(cctkGH, 1 /*all*/);
    const std::string parameters(data);
    std::free(data);
    iter.setAttribute("AllParameters", parameters);
  }

  if (myproc == ioproc) {
    const int ndims = Loop::dim;
    iter.setAttribute<std::int64_t>("numDims", ndims);
    const int npatches = ghext->patchdata.size();
    iter.setAttribute<std::int64_t>("numPatches", npatches);
    std::vector<std::string> patch_suffixes(npatches);
    for (const auto &patchdata : ghext->patchdata) {
      const int patch = patchdata.patch;
      std::ostringstream buf;
      // String attributes cannot be empty; we thus always need to use a patch
      // suffix if (ghext->num_patches() > 1)
      buf << "_patch" << setw(2) << setfill('0') << patch;
      patch_suffixes.at(patch) = buf.str();
    }
    iter.setAttribute("patchSuffixes", patch_suffixes);
    for (const auto &patchdata : ghext->patchdata) {
      const int patch = patchdata.patch;
      const int nlevels = patchdata.leveldata.size();
      iter.setAttribute<std::int64_t>("numLevels" + patch_suffixes.at(patch),
                                      nlevels);
      std::vector<std::string> level_suffixes(nlevels);
      for (const auto &leveldata : patchdata.leveldata) {
        const int level = leveldata.level;
        std::ostringstream buf;
        buf << patch_suffixes.at(patch) << "_lev" << setw(2) << setfill('0')
            << level;
        level_suffixes.at(level) = buf.str();
      }
      iter.setAttribute("levelSuffixes" + patch_suffixes.at(patch),
                        level_suffixes);
      for (const auto &leveldata : patchdata.leveldata) {
        const int level = leveldata.level;
        const amrex::FabArrayBase &mfab = *leveldata.fab;
        const int nchunks = mfab.size();
        std::vector<std::int64_t> chunk_infos(2 * ndims * nchunks);
        for (int component = 0; component < nchunks; ++component) {
          const amrex::Box &box = mfab.box(component);
          for (int d = 0; d < ndims; ++d) {
            chunk_infos.at(2 * ndims * component + 0 * ndims + ndims - 1 - d) =
                box.smallEnd()[d];
            chunk_infos.at(2 * ndims * component + 1 * ndims + ndims - 1 - d) =
                box.bigEnd()[d] + 1;
          }
        }
        iter.setAttribute("chunkInfo" + level_suffixes.at(level), chunk_infos);
      }
    }
  } // if ioproc

  // First write grid functions in a loop over patches and levels

  // Loop over patches
  for (const auto &patchdata : ghext->patchdata) {
    // Loop over levels
    for (const auto &leveldata : patchdata.leveldata) {
      if (io_verbose)
        CCTK_VINFO("Writing patch %d level %d mesh...", patchdata.patch,
                   leveldata.level);

      // Determine grid structure

      const int *const nghosts = cctkGH->cctk_nghostzones;
      const amrex::Geometry &geom = patchdata.amrcore->Geom(leveldata.level);
      const amrex::Real *const xlo = geom.ProbLo();
      const amrex::Real *const xhi = geom.ProbHi();
      const amrex::Real *const dx = geom.CellSize();
      const box_t<CCTK_REAL, 3> rdomain{
          .lo = {xlo[0] - output_ghosts * nghosts[0] * dx[0],
                 xlo[1] - output_ghosts * nghosts[1] * dx[1],
                 xlo[2] - output_ghosts * nghosts[2] * dx[2]},
          .hi = {xhi[0] + output_ghosts * nghosts[0] * dx[0],
                 xhi[1] + output_ghosts * nghosts[1] * dx[1],
                 xhi[2] + output_ghosts * nghosts[2] * dx[2]}};
      const amrex::Box &dom = geom.Domain();
      const amrex::IntVect &ilo = dom.smallEnd();
      const amrex::IntVect &ihi = dom.bigEnd();
      // The domain is always vertex centred. The tensor components are
      // then staggered if necessary.
      const box_t<int, 3> idomain{
          .lo = {ilo[0] - output_ghosts * nghosts[0],
                 ilo[1] - output_ghosts * nghosts[1],
                 ilo[2] - output_ghosts * nghosts[2]},
          .hi = {ihi[0] + output_ghosts * nghosts[0] + 1 + 1,
                 ihi[1] + output_ghosts * nghosts[1] + 1 + 1,
                 ihi[2] + output_ghosts * nghosts[2] + 1 + 1}};
      if (io_verbose) {
        CCTK_VINFO("Patch: %d, Level: %d", patchdata.patch, leveldata.level);
        CCTK_VINFO("  xmin: [%f,%f,%f]", double(rdomain.lo[0]),
                   double(rdomain.lo[1]), double(rdomain.lo[2]));
        CCTK_VINFO("  xmax: [%f,%f,%f]", double(rdomain.hi[0]),
                   double(rdomain.hi[1]), double(rdomain.hi[2]));
        CCTK_VINFO("  imin: [%d,%d,%d]", int(idomain.lo[0]), int(idomain.lo[1]),
                   int(idomain.lo[2]));
        CCTK_VINFO("  imax: [%d,%d,%d]", int(idomain.hi[0]), int(idomain.hi[1]),
                   int(idomain.hi[2]));
      }

      // Create dataset

      const openPMD::Datatype datatype =
          openPMD::determineDatatype<CCTK_REAL>();
      const openPMD::Extent extent = to_vector(reversed(idomain.shape()));
      const openPMD::Dataset dataset(datatype, extent);

      const int numgroups = CCTK_NumGroups();
      for (int gi = 0; gi < numgroups; ++gi) {
        if (output_group.at(gi)) {

          // Check group properties

          cGroup cgroup;
          const int ierr = CCTK_GroupData(gi, &cgroup);
          assert(!ierr);
          if (cgroup.grouptype != CCTK_GF)
            continue;
          assert(cgroup.vartype == CCTK_VARIABLE_REAL);
          assert(cgroup.dim == 3);
          // cGroupDynamicData cgroupdynamicdata;
          // ierr = CCTK_GroupDynamicData(cctkGH, gi, &cgroupdynamicdata);
          // assert(!ierr);
          // TODO: Check whether group has storage
          // TODO: Check whether data are valid

          if (io_verbose)
            CCTK_VINFO("Writing group %d %s...", gi, CCTK_FullGroupName(gi));

          const auto &groupdata = *leveldata.groupdata.at(gi);
          // const int firstvarindex = groupdata.firstvarindex;
          const int numvars = groupdata.numvars;
          const int tl = 0;
          const amrex::MultiFab &mfab = *groupdata.mfab[tl];
          const amrex::IndexType &indextype = mfab.ixType();
          const Arith::vect<bool, 3> is_cell_centred{indextype.cellCentered(0),
                                                     indextype.cellCentered(1),
                                                     indextype.cellCentered(2)};

          const int num_local_components = mfab.local_size();

          // Create mesh

          const std::string meshname =
              make_meshname(gi, leveldata.patch, leveldata.level);
          if (io_verbose)
            CCTK_VINFO("Defining mesh %s...", meshname.c_str());
          assert(!iter.meshes.contains(meshname));
          openPMD::Mesh mesh = iter.meshes[meshname];

          mesh.setGeometry(openPMD::Mesh::Geometry::cartesian);
          mesh.setAxisLabels(reversed(std::vector<std::string>{"x", "y", "z"}));
          mesh.setGridSpacing(to_vector<CCTK_REAL>(
              reversed(fmap([](auto x, auto y) { return x / CCTK_REAL(y); },
                            rdomain.hi - rdomain.lo, idomain.shape() - 1))));
          mesh.setGridGlobalOffset(to_vector<double>(reversed(rdomain.lo)));
          mesh.setGridUnitSI(Unit::length);
          // const std::map<openPMD::UnitDimension, double> unitDimension{
          //     {openPMD::UnitDimension::L, 1}};
          // mesh.setUnitDimension(unitDimension);
          mesh.setTimeOffset(CCTK_REAL(0)); // TODO: check interface.ccl

          // Cell centred grids are offset by 1/2
          const Arith::vect<double, 3> position =
              fmap([](auto c) { return 0.5 * c; }, is_cell_centred);

          // Define tensor components

          // TODO: Set component names according to the tensor type
          std::vector<openPMD::MeshRecordComponent> record_components;
          record_components.reserve(numvars);
#if 0
        switch (numvars) {
        case 1:
          for (int vi = 0; vi < numvars; ++vi) {
           record_omponents.push_back(mesh[openPMD::MeshRecordComponent::SCALAR]);
          }
          break;
        case 3:
          for (int vi = 0; vi < numvars; ++vi) {
            const std::string cnames[] = {"x", "y", "z"};
            record_components.push_back(mesh[cnames[vi]]);
          }
          break;
        case 6:
          for (int vi = 0; vi < numvars; ++vi) {
            const std::string cnames[] = {"xx", "xy", "xz", "yy", "yz", "zz"};
            record_components.push_back(mesh[cnames[vi]]);
          }
          break;
        default:
          CCTK_VWARN(CCTK_WARN_ALERT, "unsupported tensor type gi=%d group=%s",
                     gi, CCTK_FullGroupName(gi));
          for (int vi = 0; vi < numvars; ++vi) {
          const std::string varname = CCTK_VarName(firstvarindex + vi);
            CCTK_VINFO("Creating component %d %s", vi, varname.c_str());
            record_components.push_back(mesh[varname]);
          }
          break;
        }
#endif
          for (int vi = 0; vi < numvars; ++vi) {
            const std::string componentname = make_componentname(gi, vi);
            record_components.push_back(mesh[componentname]);
            auto &record_component = record_components.back();
            record_component.setPosition(to_vector<double>(reversed(position)));
          }
          assert(int(record_components.size()) == numvars);

          // Write data

          if (io_verbose)
            CCTK_VINFO("Writing %d variables with %d components...", numvars,
                       num_local_components);

          for (int vi = 0; vi < numvars; ++vi)
            record_components.at(vi).resetDataset(dataset);

          // Loop over components (AMReX boxes)
          for (int local_component = 0; local_component < num_local_components;
               ++local_component) {
            const int component = mfab.IndexArray().at(local_component);

            const amrex::Box &fabbox =
                mfab.fabbox(component); // exterior (with ghosts)
            const box_t<int, 3> extbox{
                .lo = {fabbox.smallEnd(0), fabbox.smallEnd(1),
                       fabbox.smallEnd(2)},
                .hi = {fabbox.bigEnd(0) + 1, fabbox.bigEnd(1) + 1,
                       fabbox.bigEnd(2) + 1}};
            const amrex::Box &validbox =
                mfab.box(component); // interior (without ghosts)
            const box_t<int, 3> intbox{
                .lo = {validbox.smallEnd(0), validbox.smallEnd(1),
                       validbox.smallEnd(2)},
                .hi = {validbox.bigEnd(0) + 1, validbox.bigEnd(1) + 1,
                       validbox.bigEnd(2) + 1}};
            // It seems that openPMD assumes that chunks do not have
            // ghost zones
            assert(!output_ghosts);
            const box_t<int, 3> &box = output_ghosts ? extbox : intbox;

            const openPMD::Offset start =
                to_vector(reversed(box.lo - idomain.lo));
            const openPMD::Extent count = to_vector(reversed(box.shape()));
            const int np = box.size();
            assert(int(count.at(0) * count.at(1) * count.at(2)) == np);
            for (int d = 0; d < 3; ++d)
              // assert(start.at(d) >= 0);
              assert(
                  start.at(d) <
                  std::numeric_limits<
                      std::remove_reference_t<decltype(start.at(d))> >::max() /
                      2);
            for (int d = 0; d < 3; ++d)
              assert(start.at(d) + count.at(d) <= extent.at(d));

            const amrex::FArrayBox &fab = mfab[component];
            for (int vi = 0; vi < numvars; ++vi) {
              if (output_ghosts || intbox == extbox) {
                const CCTK_REAL *const ptr = fab.dataPtr() + vi * np;
#if OPENPMDAPI_VERSION_GE(0, 15, 0)
                record_components.at(vi).storeChunkRaw(ptr, start, count);
#else
                record_components.at(vi).storeChunk(openPMD::shareRaw(ptr),
                                                    start, count);
#endif
              } else {
                std::shared_ptr<CCTK_REAL> ptr(
                    new CCTK_REAL[np], std::default_delete<CCTK_REAL[]>());
                const Arith::vect<int, 3> amrex_shape = extbox.shape();
                const Arith::vect<int, 3> amrex_offset = box.lo - extbox.lo;
                constexpr int amrex_di = 1;
                const int amrex_dj = amrex_di * amrex_shape[0];
                const int amrex_dk = amrex_dj * amrex_shape[1];
                const int amrex_np = amrex_dk * amrex_shape[2];
                const CCTK_REAL *restrict const amrex_ptr =
                    fab.dataPtr() + vi * amrex_np + amrex_di * amrex_offset[0] +
                    amrex_dj * amrex_offset[1] + amrex_dk * amrex_offset[2];
                const Arith::vect<int, 3> contig_shape = box.shape();
                constexpr int contig_di = 1;
                const int contig_dj = contig_di * contig_shape[0];
                const int contig_dk = contig_dj * contig_shape[1];
                const int contig_np = contig_dk * contig_shape[2];
                assert(contig_np == np);
                CCTK_REAL *restrict const contig_ptr = ptr.get();
                for (int k = 0; k < contig_shape[2]; ++k)
                  for (int j = 0; j < contig_shape[1]; ++j)
#pragma omp simd
                    for (int i = 0; i < contig_shape[0]; ++i)
                      contig_ptr[contig_di * i + contig_dj * j +
                                 contig_dk * k] =
                          amrex_ptr[amrex_di * i + amrex_dj * j + amrex_dk * k];
                record_components.at(vi).storeChunk(std::move(ptr), start,
                                                    count);
              }
            } // for vi
          }   // for local_component
        }
      } // for gi

    } // for leveldata
  }   // for patchdata

  // Next write grid scalars and grid arrays

  if (myproc == ioproc) {
    const int numgroups = CCTK_NumGroups();
    for (int gi = 0; gi < numgroups; ++gi) {
      if (output_group.at(gi)) {

        // Check group properties

        cGroup cgroup;
        const int ierr = CCTK_GroupData(gi, &cgroup);
        assert(!ierr);
        if (cgroup.grouptype == CCTK_GF)
          continue;
        assert(cgroup.vartype == CCTK_VARIABLE_REAL);
        assert(cgroup.disttype == CCTK_DISTRIB_CONSTANT);
        assert(cgroup.dim >= 0);
        assert(cgroup.dim <= 3);

        if (io_verbose)
          CCTK_VINFO("Writing group %d %s...", gi, CCTK_FullGroupName(gi));

        const auto &groupdata = *ghext->globaldata.arraygroupdata.at(gi);
        // const int firstvarindex = groupdata.firstvarindex;
        const int numvars = groupdata.numvars;
        const int tl = 0;

        // Determine grid structure

        using ivect = Arith::vect<int, dim>;

        const box_t<int, 3> idomain{.lo = ivect{0, 0, 0},
                                    .hi = ivect(groupdata.gsh)};

        // Create dataset

        const openPMD::Datatype datatype =
            openPMD::determineDatatype<CCTK_REAL>();
        const openPMD::Extent extent = to_vector(reversed(idomain.shape()));
        const openPMD::Dataset dataset(datatype, extent);

        // Create mesh

        const std::string meshname = make_meshname(gi, -1, -1);
        openPMD::Mesh mesh = iter.meshes[meshname];

        // mesh.setGeometry(openPMD::Mesh::Geometry::cartesian);
        // mesh.setAxisLabels(reversed(std::vector<std::string>{"x", "y",
        // "z"}));
        // mesh.setGridSpacing(to_vector<CCTK_REAL>(
        //     reversed(fmap([](auto x, auto y) { return x / CCTK_REAL(y); },
        //                   rdomain.hi - rdomain.lo, idomain.shape() - 1))));
        // mesh.setGridGlobalOffset(to_vector<double>(reversed(rdomain.lo)));
        // mesh.setGridUnitSI(Unit::length);
        // // const std::map<openPMD::UnitDimension, double> unitDimension{
        // //     {openPMD::UnitDimension::L, 1}};
        // // mesh.setUnitDimension(unitDimension);
        mesh.setTimeOffset(CCTK_REAL(0)); // TODO: check interface.ccl

        // // Cell centred grids are offset by 1/2
        // const Arith::vect<double, 3> position{0, 0, 0};

        // Define tensor components

        // TODO: Set component names according to the tensor type
        std::vector<openPMD::MeshRecordComponent> record_components;
        record_components.reserve(numvars);
        for (int vi = 0; vi < numvars; ++vi) {
          const std::string componentname = make_componentname(gi, vi);
          record_components.push_back(mesh[componentname]);
          // auto &record_component = record_components.back();
          // record_component.setPosition(to_vector<double>(reversed(position)));
        }
        assert(int(record_components.size()) == numvars);

        // Write data

        if (io_verbose)
          CCTK_VINFO("Writing %d variables...", numvars);

        for (int vi = 0; vi < numvars; ++vi)
          record_components.at(vi).resetDataset(dataset);

        // exterior (with ghosts)
        for (int d = 0; d < dim; ++d)
          assert(groupdata.lsh[d] == groupdata.gsh[d]);
        assert(all(Arith::vect<int, dim>(groupdata.lsh) ==
                   Arith::vect<int, dim>(groupdata.gsh)));
        const box_t<int, 3> extbox{.lo = ivect{0, 0, 0},
                                   .hi = ivect(groupdata.lsh)};
        // interior (without ghosts)
        const box_t<int, 3> intbox{.lo = ivect(groupdata.nghostzones),
                                   .hi = ivect(groupdata.lsh) -
                                         ivect(groupdata.nghostzones)};
        // It seems that openPMD assumes that chunks do not have ghost zones
        assert(!output_ghosts);
        const box_t<int, 3> &box = output_ghosts ? extbox : intbox;

        const openPMD::Offset start = to_vector(reversed(box.lo - idomain.lo));
        const openPMD::Extent count = to_vector(reversed(box.shape()));
        const int np = box.size();
        assert(int(count.at(0) * count.at(1) * count.at(2)) == np);
        for (int d = 0; d < 3; ++d)
          // assert(start.at(d) >= 0);
          assert(start.at(d) <
                 std::numeric_limits<
                     std::remove_reference_t<decltype(start.at(d))> >::max() /
                     2);
        for (int d = 0; d < 3; ++d)
          assert(start.at(d) + count.at(d) <= extent.at(d));

        const Arith::vect<int, 3> cactus_shape = extbox.shape();
        constexpr int cactus_di = 1;
        const int cactus_dj = cactus_di * cactus_shape[0];
        const int cactus_dk = cactus_dj * cactus_shape[1];
        const int cactus_np = cactus_dk * cactus_shape[2];
        assert(cactus_di > 0);
        assert(cactus_dj > 0);
        assert(cactus_dk > 0);
        assert(cactus_np > 0);
        for (int vi = 0; vi < numvars; ++vi) {
          const CCTK_REAL *const var_ptr = groupdata.data.at(tl).dataPtr(vi);
          if (output_ghosts || intbox == extbox) {
            const CCTK_REAL *const ptr = var_ptr;
#if OPENPMDAPI_VERSION_GE(0, 15, 0)
            record_components.at(vi).storeChunkRaw(ptr, start, count);
#else
            record_components.at(vi).storeChunk(openPMD::shareRaw(ptr), start,
                                                count);
#endif
          } else {
            std::shared_ptr<CCTK_REAL> ptr(new CCTK_REAL[np],
                                           std::default_delete<CCTK_REAL[]>());
            const Arith::vect<int, 3> cactus_offset = box.lo - extbox.lo;
            const CCTK_REAL *restrict const cactus_ptr =
                var_ptr + cactus_di * cactus_offset[0] +
                cactus_dj * cactus_offset[1] + cactus_dk * cactus_offset[2];
            const Arith::vect<int, 3> contig_shape = box.shape();
            constexpr int contig_di = 1;
            const int contig_dj = contig_di * contig_shape[0];
            const int contig_dk = contig_dj * contig_shape[1];
            const int contig_np = contig_dk * contig_shape[2];
            assert(contig_np == np);
            CCTK_REAL *restrict const contig_ptr = ptr.get();
            for (int k = 0; k < contig_shape[2]; ++k)
              for (int j = 0; j < contig_shape[1]; ++j)
                for (int i = 0; i < contig_shape[0]; ++i)
                  contig_ptr[contig_di * i + contig_dj * j + contig_dk * k] =
                      cactus_ptr[cactus_di * i + cactus_dj * j + cactus_dk * k];
            record_components.at(vi).storeChunk(std::move(ptr), start, count);
          }
        } // for vi
      }
    }
  }

  if (io_verbose)
    CCTK_VINFO("Closing iteration...");
  iter.close();

  if (CCTK_MyProc(nullptr) == 0) {
    std::ostringstream buf;
    buf << output_dir << "/" << output_file << ".openpmd.visit";
    const std::string visitname = buf.str();
    std::ofstream visit(visitname, ios::app);
    assert(visit.good());
    switch (iterationEncoding) {
    case openPMD::IterationEncoding::fileBased:
      visit << output_file << ".it" << setw(8) << setfill('0') << cctk_iteration
            << openPMD::suffix(format) << "\n";
      break;
    case openPMD::IterationEncoding::variableBased:
      visit << output_file << openPMD::suffix(format) << "\n";
      break;
    default:
      abort();
    }
  }

  switch (iterationEncoding) {
  case openPMD::IterationEncoding::fileBased:
    write_iters.reset();
    series.reset();
    filename.reset();
    break;
  case openPMD::IterationEncoding::variableBased:
    // do nothing
    break;
  default:
    abort();
  }

  if (io_verbose)
    CCTK_VINFO("OutputOpenPMD done.");

  if (io_verbose)
    timer.print();
}

} // namespace CarpetX

#else

namespace CarpetX {
void ShutdownOpenPMD() {}
} // namespace CarpetX

#endif // #ifdef HAVE_CAPABILITY_OPENPMD

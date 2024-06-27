#include "io_tsv.hxx"

#include "driver.hxx"
#include "mpi_types.hxx"
#include "timer.hxx"

#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace CarpetX {
using namespace std;

void WriteTSVold(const cGH *restrict cctkGH, const string &filename, int gi,
                 const vector<string> &varnames) {
  ostringstream buf;
  buf << filename << ".tsv";
  ofstream file(buf.str());
  const string sep = "\t";

  // get more precision for floats, could also use
  // https://stackoverflow.com/a/30968371
  file << setprecision(numeric_limits<CCTK_REAL>::digits10 + 1) << scientific;

  // Output header
  file << "# 1:iteration" << sep << "2:time" << sep << "3:patch" << sep
       << "4:level" << sep << "5:component" << sep << "6:i" << sep << "7:j"
       << sep << "8:k" << sep << "9:x" << sep << "10:y" << sep << "11:z";
  int col = 12;
  for (const auto &varname : varnames)
    file << sep << col++ << ":" << varname;
  file << "\n";

  for (const auto &patchdata : ghext->patchdata) {
    for (const auto &leveldata : patchdata.leveldata) {
      const auto &groupdata = *leveldata.groupdata.at(gi);
      const int tl = 0;
      const auto &geom = patchdata.amrcore->Geom(leveldata.level);
      const auto &mfab = *groupdata.mfab.at(tl);
      for (amrex::MFIter mfi(mfab); mfi.isValid(); ++mfi) {
        const amrex::Array4<const CCTK_REAL> &vars = mfab.array(mfi);
        const auto &imin = vars.begin;
        const auto &imax = vars.end;
        for (int k = imin.z; k < imax.z; ++k) {
          for (int j = imin.y; j < imax.y; ++j) {
            for (int i = imin.x; i < imax.x; ++i) {
              const array<int, dim> I{i, j, k};
              array<CCTK_REAL, dim> x;
              for (int d = 0; d < dim; ++d)
                x[d] = geom.ProbLo(d) +
                       (I[d] + CCTK_REAL(0.5) * groupdata.indextype[d]) *
                           geom.CellSize(d);
              file << cctkGH->cctk_iteration << sep << cctkGH->cctk_time << sep
                   << patchdata.patch << sep << leveldata.level << sep
                   << mfi.index() << sep << I[0] << sep << I[1] << sep << I[2]
                   << sep << x[0] << sep << x[1] << sep << x[2];
              for (int n = 0; n < groupdata.numvars; ++n)
                file << sep << vars(i, j, k, n);
              file << "\n";
            }
          }
        }
      }
    }
  }

  file.close();
}

void OutputTSVold(const cGH *restrict cctkGH) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  if (!out_tsv)
    return;

  static Timer timer("OutputTSV");
  Interval interval(timer);

  const int numgroups = CCTK_NumGroups();
  for (int gi = 0; gi < numgroups; ++gi) {
    cGroup group;
    int ierr = CCTK_GroupData(gi, &group);
    assert(!ierr);

    if (group.grouptype != CCTK_GF)
      continue;

    auto &restrict groupdata0 =
        *ghext->patchdata.at(0).leveldata.at(0).groupdata.at(gi);
    if (groupdata0.mfab.size() > 0) {
      const int tl = 0;

      string groupname = CCTK_FullGroupName(gi);
      groupname = regex_replace(groupname, regex("::"), "-");
      for (auto &c : groupname)
        c = tolower(c);
      ostringstream buf;
      buf << out_dir << "/" << groupname;
      buf << ".it" << setw(6) << setfill('0') << cctk_iteration;
      buf << ".p" << setw(4) << setfill('0') << CCTK_MyProc(nullptr);
      const string filename = buf.str();

      amrex::Vector<string> varnames(groupdata0.numvars);
      for (int vi = 0; vi < groupdata0.numvars; ++vi) {
        ostringstream buf;
        buf << CCTK_VarName(groupdata0.firstvarindex + vi);
        for (int i = 0; i < tl; ++i)
          buf << "_p";
        varnames.at(vi) = buf.str();
      }

      WriteTSVold(cctkGH, filename, gi, varnames);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

void WriteTSVScalars(const cGH *restrict cctkGH, const string &filename,
                     const int gi) {
  // Output only on root process
  if (CCTK_MyProc(nullptr) > 0)
    return;

  const auto &arraygroupdata = *ghext->globaldata.arraygroupdata.at(gi);

  vector<string> varnames;
  for (int vi = 0; vi < arraygroupdata.numvars; ++vi)
    varnames.push_back(CCTK_VarName(arraygroupdata.firstvarindex + vi));

  const string sep = "\t";
  ofstream file(filename);
  // get more precision for floats, could also use
  // https://stackoverflow.com/a/30968371
  file << setprecision(numeric_limits<CCTK_REAL>::digits10 + 1) << scientific;

  // Output header
  file << "# 1:iteration" << sep << "2:time";
  int col = 3;
  for (const auto &varname : varnames)
    file << sep << col++ << ":" << varname;
  file << "\n";

  // Output data
  file << cctkGH->cctk_iteration << sep << cctkGH->cctk_time;
  const int tl = 0;
  for (int vi = 0; vi < arraygroupdata.numvars; ++vi)
    file << sep << *arraygroupdata.data.at(tl).dataPtr(vi);
  file << "\n";
}

void WriteTSVArrays(const cGH *restrict cctkGH, const string &filename,
                    const int gi, const int out_dir) {
  // Output only on root process
  if (CCTK_MyProc(nullptr) > 0)
    return;

  const auto &arraygroupdata = *ghext->globaldata.arraygroupdata.at(gi);

  if (out_dir >= arraygroupdata.dimension)
    return;

  vector<string> varnames;
  for (int vi = 0; vi < arraygroupdata.numvars; ++vi)
    varnames.push_back(CCTK_VarName(arraygroupdata.firstvarindex + vi));

  const string sep = "\t";
  ofstream file(filename);
  // get more precision for floats, could also use
  // https://stackoverflow.com/a/30968371
  file << setprecision(numeric_limits<CCTK_REAL>::digits10 + 1) << scientific;

  // Output header
  int col = 1;
  file << "# " << col++ << ":iteration";
  file << sep << col++ << ":time";
  for (int dir = 0; dir < arraygroupdata.dimension; ++dir)
    file << sep << col++ << ":"
         << "ijk"[dir];
  for (const auto &varname : varnames)
    file << sep << col++ << ":" << varname;
  file << "\n";

  constexpr int di = 1;
  const int dj = di * arraygroupdata.lsh[0];
  const int dk = dj * arraygroupdata.lsh[1];
  const int np = dk * arraygroupdata.lsh[2];
  const std::array<int, dim> DI{di, dj, dk};

  // Output data
  for (int i = 0; i < arraygroupdata.lsh[out_dir]; ++i) {
    file << cctkGH->cctk_iteration << sep << cctkGH->cctk_time;
    for (int dir = 0; dir < arraygroupdata.dimension; ++dir)
      file << sep << (dir == out_dir ? i : 0);
    const int tl = 0;
    for (int vi = 0; vi < arraygroupdata.numvars; ++vi)
      file << sep << arraygroupdata.data.at(tl).dataPtr(vi)[DI[out_dir] * i];
    file << "\n";
  }
}

void WriteTSVGFs(const cGH *restrict cctkGH, const string &filename,
                 const int gi, const vect<bool, dim> &outdirs,
                 const vect<CCTK_REAL, dim> &outcoords) {
  const auto &groupdata0 =
      *ghext->patchdata.at(0).leveldata.at(0).groupdata.at(gi);

  // Number of values transmitted per grid point
  const int nintvalues = 1                  // patch
                         + 1                // level
                         + dim              // grid point index
                         + 1;               // isghost
  const int nvalues = nintvalues            // integer values
                      + dim                 // coordinates
                      + groupdata0.numvars; // grid function values

  // Data transmitted from this process
  vector<CCTK_REAL> data;
  data.reserve(10000);
  for (const auto &patchdata : ghext->patchdata) {
    for (const auto &leveldata : patchdata.leveldata) {
      const auto &groupdata = *leveldata.groupdata.at(gi);

      const int tl = 0;

      // Convert a (direction, face) pair to an AMReX Orientation
      const auto orient = [&](int d, int f) {
        return amrex::Orientation(d, amrex::Orientation::Side(f));
      };

      const auto &geom = patchdata.amrcore->Geom(leveldata.level);
      const amrex::Box &domain =
          patchdata.amrcore->Geom(leveldata.level).Domain();
      vect<CCTK_REAL, dim> x0, dx;
      for (int d = 0; d < dim; ++d) {
        dx[d] = geom.CellSize(d);
        x0[d] = geom.ProbLo(d) + groupdata.indextype[d] * dx[d] / 2;
      }
      vect<int, dim> icoord;
      for (int d = 0; d < dim; ++d)
        if (outdirs[d])
          icoord[d] = INT_MIN;
        else
          icoord[d] = lrint((outcoords[d] - x0[d]) / dx[d]);

      const auto &symmetries = ghext->patchdata.at(leveldata.patch).symmetries;

      const auto &mfab = *groupdata.mfab.at(tl);

      const vect<int, dim> nghosts = {mfab.nGrow(0), mfab.nGrow(1),
                                      mfab.nGrow(2)};

      for (amrex::MFIter mfi(mfab); mfi.isValid(); ++mfi) {
        const amrex::Array4<const CCTK_REAL> &vars = mfab.array(mfi);

        const amrex::Box &vbx =
            mfi.validbox(); // interior region (without ghosts)
        vect<vect<bool, dim>, 2> bbox;
        for (int d = 0; d < dim; ++d)
          for (int f = 0; f < 2; ++f)
            bbox[f][d] = vbx[orient(d, f)] ==
                             domain[orient(d, f)] +
                                 (groupdata.indextype[d] == 0 && f == 1) &&
                         symmetries[f][d] != symmetry_t::none;

        const vect<int, dim> varmin = {vars.begin.x, vars.begin.y,
                                       vars.begin.z};
        const vect<int, dim> varmax = {vars.end.x, vars.end.y, vars.end.z};

        // Skip ghost points but keep boundary points
        const vect<int, dim> intmin = varmin + !bbox[0] * nghosts;
        const vect<int, dim> intmax = varmax - !bbox[1] * nghosts;

        bool output_something = true;
        vect<int, dim> imin, imax;
        for (int d = 0; d < dim; ++d) {
          if (outdirs[d]) {
            // output everything
            imin[d] = varmin[d];
            imax[d] = varmax[d];
          } else if (icoord[d] >= varmin[d] && icoord[d] < varmax[d]) {
            // output one point
            imin[d] = icoord[d];
            imax[d] = icoord[d] + 1;
          } else {
            // output nothing
            output_something = false;
          }
        }

        if (output_something) {
          for (int k = imin[2]; k < imax[2]; ++k) {
            for (int j = imin[1]; j < imax[1]; ++j) {
              for (int i = imin[0]; i < imax[0]; ++i) {
                const vect<int, dim> I{i, j, k};
                const auto old_size = data.size();
                data.push_back(patchdata.patch);
                data.push_back(leveldata.level);
                for (int d = 0; d < dim; ++d)
                  data.push_back(I[d]);
                const bool isghost = any(I < intmin || I >= intmax);
                data.push_back(isghost);
                for (int d = 0; d < dim; ++d)
                  data.push_back(x0[d] + I[d] * dx[d]);
                for (int vi = 0; vi < groupdata.numvars; ++vi)
                  data.push_back(vars(i, j, k, vi));
                assert(data.size() == old_size + nvalues);
              }
            }
          }
        } // if output_something
      }   // for mfi
    }     // for leveldata
  }       // for patchdata
  assert(data.size() % nvalues == 0);

  const MPI_Comm comm = amrex::ParallelDescriptor::Communicator();
  const int myproc = amrex::ParallelDescriptor::MyProc();
  const int nprocs = amrex::ParallelDescriptor::NProcs();
  const int ioproc = 0;

  assert(data.size() <= INT_MAX);
  const int npoints = data.size();

  vector<int> all_npoints;
  if (myproc == ioproc)
    all_npoints.resize(nprocs);
  MPI_Gather(&npoints, 1, MPI_INT, all_npoints.data(), 1, MPI_INT, ioproc,
             comm);

  int total_npoints = 0;
  vector<int> all_offsets;
  if (myproc == ioproc) {
    all_offsets.resize(nprocs);
    for (int p = 0; p < nprocs; ++p) {
      all_offsets.at(p) = total_npoints;
      assert(total_npoints <= INT_MAX - all_npoints.at(p));
      total_npoints += all_npoints.at(p);
    }
  }
  vector<CCTK_REAL> all_data;
  if (myproc == ioproc)
    all_data.resize(total_npoints);
  MPI_Gatherv(data.data(), npoints, mpi_datatype<CCTK_REAL>::value,
              all_data.data(), all_npoints.data(), all_offsets.data(),
              mpi_datatype<CCTK_REAL>::value, ioproc, comm);

  if (myproc == ioproc) {

    assert(total_npoints % nvalues == 0);
    vector<int> iptr(total_npoints / nvalues);
    iota(iptr.begin(), iptr.end(), 0);
    const auto compare_eq = [&](const int i, const int j) {
      array<int, nintvalues> pi, pj;
      for (int d = 0; d < nintvalues; ++d)
        pi[d] = int(all_data.at(i * nvalues + d));
      // Ignore `isghost` field
      pi[dim + 2] = 0;
      for (int d = 0; d < nintvalues; ++d)
        pj[d] = int(all_data.at(j * nvalues + d));
      // Ignore `isghost` field
      pj[dim + 2] = 0;
      return pi == pj;
    };
    const auto compare_lt = [&](const int i, const int j) {
      array<int, nintvalues> pi, pj;
      for (int d = 0; d < nintvalues; ++d)
        pi[d] = int(all_data.at(i * nvalues + d));
      for (int d = 0; d < nintvalues; ++d)
        pj[d] = int(all_data.at(j * nvalues + d));
      return pi < pj;
    };
    sort(iptr.begin(), iptr.end(), compare_lt);
    const auto last = unique(iptr.begin(), iptr.end(), compare_eq);
    iptr.erase(last, iptr.end());

    vector<string> varnames;
    for (int vi = 0; vi < groupdata0.numvars; ++vi)
      varnames.push_back(CCTK_VarName(groupdata0.firstvarindex + vi));

    const string sep = "\t";
    ofstream file(filename);
    // get more precision for floats, could also use
    // https://stackoverflow.com/a/30968371
    file << setprecision(numeric_limits<CCTK_REAL>::digits10 + 1) << scientific;

    // Output header
    int col = 0;
    file << "# " << ++col << ":iteration";
    file << sep << ++col << ":time";
    file << sep << ++col << ":patch";
    file << sep << ++col << ":level";
    for (int d = 0; d < dim; ++d)
      file << sep << ++col << ":"
           << "ijk"[d];
    for (int d = 0; d < dim; ++d)
      file << sep << ++col << ":"
           << "xyz"[d];
    for (const auto &varname : varnames)
      file << sep << ++col << ":" << varname;
    file << "\n";

    // Output data
    for (const auto i : iptr) {
      int pos = nvalues * i;
      file << cctkGH->cctk_iteration << sep << cctkGH->cctk_time;
      for (int v = 0; v < nintvalues; ++v)
        if (v != dim + 2) // skip `isghost` marker
          file << sep << int(all_data.at(pos++));
        else
          pos++;
      for (int v = nintvalues; v < nvalues; ++v)
        file << sep << all_data.at(pos++);
      file << "\n";
      assert(pos % nvalues == 0);
    }
  }
}

void OutputTSV(const cGH *restrict cctkGH) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  if (out_tsv_vars[0] == '\0')
    return;

  static Timer timer("OutputTSVUni");
  Interval interval(timer);

  // Find output groups
  const vector<bool> group_enabled = [&] {
    vector<bool> enabled(CCTK_NumGroups(), false);
    const auto callback{
        [](const int index, const char *const optstring, void *const arg) {
          vector<bool> &enabled = *static_cast<vector<bool> *>(arg);
          enabled.at(CCTK_GroupIndexFromVarI(index)) = true;
        }};
    CCTK_TraverseString(out_tsv_vars, callback, &enabled, CCTK_GROUP_OR_VAR);
    if (verbose) {
      CCTK_VINFO("TSV output for groups:");
      for (int gi = 0; gi < CCTK_NumGroups(); ++gi)
        if (group_enabled.at(gi))
          CCTK_VINFO("  %s", CCTK_FullGroupName(gi));
    }
    return enabled;
  }();
  const auto num_out_groups =
      count(group_enabled.begin(), group_enabled.end(), true);
  if (num_out_groups == 0)
    return;

  const int numgroups = CCTK_NumGroups();
  for (int gi = 0; gi < numgroups; ++gi) {
    if (group_enabled.at(gi)) {
      string groupname = CCTK_FullGroupName(gi);
      groupname = regex_replace(groupname, regex("::"), "-");
      for (auto &ch : groupname)
        ch = tolower(ch);
      ostringstream buf;
      buf << out_dir << "/" << groupname << ".it" << setw(6) << setfill('0')
          << cctk_iteration;
      const string basename = buf.str();
      switch (CCTK_GroupTypeI(gi)) {
      case CCTK_SCALAR:
        WriteTSVScalars(cctkGH, basename + ".tsv", gi);
        break;
      case CCTK_ARRAY:
        WriteTSVArrays(cctkGH, basename + ".x.tsv", gi, 0);
        WriteTSVArrays(cctkGH, basename + ".y.tsv", gi, 1);
        WriteTSVArrays(cctkGH, basename + ".z.tsv", gi, 2);
        break;
      case CCTK_GF:
        WriteTSVGFs(cctkGH, basename + ".x.tsv", gi, {true, false, false},
                    {0, out_xline_y, out_xline_z});
        WriteTSVGFs(cctkGH, basename + ".y.tsv", gi, {false, true, false},
                    {out_yline_x, 0, out_yline_z});
        WriteTSVGFs(cctkGH, basename + ".z.tsv", gi, {false, false, true},
                    {out_zline_x, out_zline_y, 0});
        break;
      default:
        assert(0);
      }
    }
  }
}

} // namespace CarpetX

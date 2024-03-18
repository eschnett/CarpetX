#include "cctk.h"
#include "cctk_Arguments.h"
#include "cctk_Parameters.h"
#include "loop.hxx"

extern "C"
void TestSubcycling_Init(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTSX_TestSubcycling_Init;

  CCTK_INFO("Initializing grid function");
  grid.loop_all<0,0,0>(grid.nghostzones, [=](const Loop::PointDesc &pt) {
    CCTK_REAL canary = 100 + 10 * cctk_iteration + 1 * cctk_level;
    iteration(pt.I) = canary;
  });

}

extern "C"
void TestSubcycling_Update(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTSX_TestSubcycling_Update;

  CCTK_VINFO("Updating grid function at iteration %d level %d time %g", cctk_iteration, cctk_level, cctk_time);
  grid.loop_int<0,0,0>(grid.nghostzones, [=] CCTK_HOST(const Loop::PointDesc &pt) {
    CCTK_REAL canary = 100 + 10 * cctk_iteration + 1 * cctk_level;
    iteration(pt.I) = canary;
  });

}

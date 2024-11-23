#ifndef _RAY_TRACER_H_
#define _RAY_TRACER_H_

#include <simobject.h>
#include <VX_types.h>
#include <cocogfx/include/fixed.hpp>
#include <cocogfx/include/math.hpp>
#include "types.h"
#include <rt_types.h>
#include "graphics.h"
#include "pipeline.h"

using namespace ray_tracing;

namespace vortex {

//class RayTracerUnit : public SimObject<RayTracerUnit> {
 //   public:
float IntersectBVH( Ray& ray, const uint nodeIdx, BVHNode bvhNode[], Tri tri[], uint32_t triIdx[] );
    
} // namespace vortex

#endif // _RAY_TRACER_H_
#include <simobject.h>
#include <VX_types.h>
#include <cocogfx/include/fixed.hpp>
#include <cocogfx/include/math.hpp>
#include "types.h"
#include <rt_types.h>
#include "graphics.h"
#include "pipeline.h"

namespace vortex {

//class RayTracerUnit : public SimObject<RayTracerUnit> {
 //   public:
    void IntersectTri( ray_tracing::Ray& ray, const ray_tracing::Tri& tri, float& distance );

    bool IntersectAABB( const ray_tracing::Ray& ray, const ray_tracing::float3 bmin, const ray_tracing::float3 bmax );

    float IntersectBVH( ray_tracing::Ray& ray, const uint nodeIdx, ray_tracing::BVHNode bvhNode[], ray_tracing::Tri tri[], uint32_t triIdx[] );
    
//}

}
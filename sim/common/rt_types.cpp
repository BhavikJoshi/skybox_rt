#include <cmath>
#include "rt_types.h"


namespace ray_tracing {

float3 cross( const float3& a, const float3& b ) { 
    float3 res(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); 
    return res;
}

} // namespace ray_tracing
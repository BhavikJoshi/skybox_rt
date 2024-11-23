#include "ray_tracer.h"
#include "graphics.h" // check if needed
#include <VX_config.h>
#include "mempool.h"  // check if needed
#include "mem.h"      // check if needed

using namespace ray_tracing;

namespace vortex {

static void IntersectTri( Ray& ray, const Tri& tri, float& distance)
{
    const float3 edge1 = tri.vertex1 - tri.vertex0;
    const float3 edge2 = tri.vertex2 - tri.vertex0;
    const float3 h = cross( ray.D, edge2 );
    const float a = dot( edge1, h );
    if (a > -0.0001f && a < 0.0001f) return; // ray parallel to triangle
    const float f = 1 / a;
    const float3 s = ray.O - tri.vertex0;
    const float u = f * dot( s, h );
    if (u < 0 || u > 1) return;
    const float3 q = cross( s, edge1 );
    const float v = f * dot( ray.D, q );
    if (v < 0 || u + v > 1) return;
    const float t = f * dot( edge2, q );
    if (t > 0.0001f && t < distance) distance = t;
}

static bool IntersectAABB( const Ray& ray, const float3& bmin, const float3& bmax )
{
    float tx1 = (bmin.x - ray.O.x) / ray.D.x, tx2 = (bmax.x - ray.O.x) / ray.D.x;
    float tmin = fminf( tx1, tx2 ), tmax = fmaxf( tx1, tx2 );
    float ty1 = (bmin.y - ray.O.y) / ray.D.y, ty2 = (bmax.y - ray.O.y) / ray.D.y;
    tmin = fmaxf( tmin, fminf( ty1, ty2 ) ), tmax = fminf( tmax, fmaxf( ty1, ty2 ) );
    float tz1 = (bmin.z - ray.O.z) / ray.D.z, tz2 = (bmax.z - ray.O.z) / ray.D.z;
    tmin = fmaxf( tmin, fminf( tz1, tz2 ) ), tmax = fminf( tmax, fmaxf( tz1, tz2 ) );
    return tmax >= tmin && tmin < ray.t && tmax > 0;
}

float IntersectBVH( Ray& ray, const uint nodeIdx, BVHNode bvhNode[], Tri tri[], uint32_t triIdx[])

{
    BVHNode& node = bvhNode[nodeIdx];
    float distance = 1e30;
    if (!IntersectAABB( ray, node.aabbMin, node.aabbMax )) return distance;
    if (node.isLeaf())
    {
        for (uint i = 0; i < node.triCount; i++ )
            IntersectTri( ray, tri[triIdx[node.leftFirst + i]], distance);
    }
    else
    {
        float leftResult = IntersectBVH( ray, node.leftFirst, bvhNode, tri, triIdx );
        float rightResult = IntersectBVH( ray, node.leftFirst + 1, bvhNode, tri, triIdx );
        distance = fminf( distance, fminf( leftResult, rightResult ) );
    }
    return distance;
}

} // namespace vortex
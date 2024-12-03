#ifndef _RT_TYPES_H_
#define _RT_TYPES_H_

#define NUM_TRIANGLES 64

namespace ray_tracing {

struct float3 {
    union { struct { float x, y, z; }; float cell[3]; };
    float& operator [] ( const int n ) { return cell[n]; }
    float3 operator-(const float3& a) const {
        return {x - a.x, y - a.y, z - a.z};
    }
    float3 operator+(const float3& a) const {
        return {x + a.x, y + a.y, z + a.z};
    }
    float3 operator*(const float& s) const {
        return {x * s, y * s, z * s};
    }
    float3& operator=(const float3& a) {
        x = a.x;
        y = a.y;
        z = a.z;
        return *this;
    }
    float3( const float a, const float b, const float c ) : x( a ), y( b ), z( c ) {}
    float3( const float a ) : x( a ), y( a ), z( a ) {}
    float3() : x( 0 ), y( 0 ), z( 0 ) {}
};

struct Tri { 
  float3 vertex0, vertex1, vertex2; 
  float3 centroid; 
  Tri() {}
};

struct Ray {
  float3 O, D;
  float t = 1e30f;
};

struct BVHNode {
    float3 aabbMin, aabbMax;
    int leftFirst, triCount;
    bool isLeaf() { return triCount > 0; }
};

inline float3 cross( const float3& a, const float3& b ) {
    float3 res(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
    return res;
}

inline float dot( const float3& a, const float3& b ) { return a.x * b.x + a.y * b.y + a.z * b.z; }

inline uint32_t RGB32FtoRGB8( float3 c )
{
	int r = (int)(fmin( c.x, 1.f ) * 255);
	int g = (int)(fmin( c.y, 1.f ) * 255);
	int b = (int)(fmin( c.z, 1.f ) * 255);
	return (r << 16) + (g << 8) + b;
}

void IntersectTri( Ray& ray, const Tri& tri, float& distance)
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

bool IntersectAABB( const Ray& ray, const float3& bmin, const float3& bmax )
{
    float tx1 = (bmin.x - ray.O.x) / ray.D.x, tx2 = (bmax.x - ray.O.x) / ray.D.x;
    float tmin = fminf( tx1, tx2 ), tmax = fmaxf( tx1, tx2 );
    float ty1 = (bmin.y - ray.O.y) / ray.D.y, ty2 = (bmax.y - ray.O.y) / ray.D.y;
    tmin = fmaxf( tmin, fminf( ty1, ty2 ) ), tmax = fminf( tmax, fmaxf( ty1, ty2 ) );
    float tz1 = (bmin.z - ray.O.z) / ray.D.z, tz2 = (bmax.z - ray.O.z) / ray.D.z;
    tmin = fmaxf( tmin, fminf( tz1, tz2 ) ), tmax = fminf( tmax, fmaxf( tz1, tz2 ) );
    return tmax >= tmin && tmin < ray.t && tmax > 0;
}

} // namespace ray_tracing

#endif // _RT_TYPES_H_
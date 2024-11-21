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

float3 cross( const float3& a, const float3& b );
float3 normalize(const float3& v1);
inline uint32_t RGB32FtoRGB8( float3 c );
inline float dot( const float3& a, const float3& b ) { return a.x * b.x + a.y * b.y + a.z * b.z; }


} // namespace ray_tracing

#endif // _RT_TYPES_H_
#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define KERNEL_ARG_DEV_MEM_ADDR 0x7ffff000

#define NUM_TRIANGLES 64

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

typedef struct {
  uint32_t grid_dim[2];
  uint32_t num_triangles;

  uint32_t dst_width;
  uint32_t dst_height;
 
  uint8_t  cbuf_stride;  
  uint32_t cbuf_pitch;  

  uint64_t bvh_addr;
  uint64_t tri_addr;
  uint64_t triIdx_addr;
  uint64_t cbuf_addr; 
} kernel_arg_t;

#endif
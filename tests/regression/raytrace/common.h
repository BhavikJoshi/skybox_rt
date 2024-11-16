#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define KERNEL_ARG_DEV_MEM_ADDR 0x7ffff000

#define NUM_TRIANGLES 64

struct float3 {
  float x, y, z;
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
};

struct Tri { 
  float3 vertex0, vertex1, vertex2; 
  float3 centroid; 
};

struct Ray {
  float3 O, D;
  float t = 1e30f;
};

struct BVHNode {
    float3 aabbMin, aabbMax;
    uint leftNode, firstTriIdx, triCount;
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
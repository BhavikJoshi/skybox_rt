#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define KERNEL_ARG_DEV_MEM_ADDR 0x7ffff000

typedef struct {
  uint32_t grid_dim[2];
  uint32_t num_triangles;

  uint32_t dst_width;
  uint32_t dst_height;
 
  uint8_t  cbuf_stride;  
  uint32_t cbuf_pitch;  

  uint64_t bvh_addr;
  uint64_t tri_addr;
  uint64_t cbuf_addr; 
} kernel_arg_t;

#endif
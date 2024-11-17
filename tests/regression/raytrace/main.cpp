#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <vortex.h>
#include <cmath>
#include <math.h>
#include <cocogfx/include/imageutil.hpp>
#include <graphics.h>
#include <gfxutil.h>
#include <VX_config.h>
#include <algorithm>
#include "common.h"
#include <cocogfx/include/imageutil.hpp>

#define FLOAT_ULP 6

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
	 cleanup();			                                              \
     exit(-1);                                                  \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

static void generate_triangles(Tri tri[], int N) {
    for (int i = 0; i < N; i++)
    {
        float3 r0(static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX);
        float3 r1(static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX);
        float3 r2(static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX);
        tri[i].vertex0 = r0 * 9 - float3( 5 );
        tri[i].vertex1 = tri[i].vertex0 + r1;
        tri[i].vertex2 = tri[i].vertex0 + r2;
    }
}

static void UpdateNodeBounds(BVHNode bvhNode[], Tri tri[], uint triIdx[], int N, uint nodeIdx, uint &nodesUsed)  {
    BVHNode& node = bvhNode[nodeIdx];
    node.aabbMin = float3(  1e30f );
    node.aabbMax = float3( -1e30f );
    for (uint first = node.leftFirst, i = 0; i < node.triCount; i++)
    {
        uint leafTriIdx = triIdx[first + i];
        Tri& leafTri = tri[leafTriIdx];
        node.aabbMin.x = fminf( node.aabbMin.x, leafTri.vertex0.x );
        node.aabbMin.x = fminf( node.aabbMin.x, leafTri.vertex1.x );
        node.aabbMin.x = fminf( node.aabbMin.x, leafTri.vertex2.x );
        node.aabbMin.y = fminf( node.aabbMin.y, leafTri.vertex0.y );
        node.aabbMin.y = fminf( node.aabbMin.y, leafTri.vertex1.y );
        node.aabbMin.y = fminf( node.aabbMin.y, leafTri.vertex2.y );
        node.aabbMin.z = fminf( node.aabbMin.z, leafTri.vertex2.z );
        node.aabbMin.z = fminf( node.aabbMin.z, leafTri.vertex0.z );
        node.aabbMin.z = fminf( node.aabbMin.z, leafTri.vertex1.z );
        node.aabbMax.x = fmaxf( node.aabbMax.x, leafTri.vertex0.x );
        node.aabbMax.x = fmaxf( node.aabbMax.x, leafTri.vertex1.x );
        node.aabbMax.x = fmaxf( node.aabbMax.x, leafTri.vertex2.x );
        node.aabbMax.y = fmaxf( node.aabbMax.y, leafTri.vertex0.y );
        node.aabbMax.y = fmaxf( node.aabbMax.y, leafTri.vertex1.y );
        node.aabbMax.y = fmaxf( node.aabbMax.y, leafTri.vertex2.y );
        node.aabbMax.z = fmaxf( node.aabbMax.z, leafTri.vertex0.z );
        node.aabbMax.z = fmaxf( node.aabbMax.z, leafTri.vertex1.z );
        node.aabbMax.z = fmaxf( node.aabbMax.z, leafTri.vertex2.z );
    }
}

static void Subdivide(BVHNode bvhNode[], Tri tri[], uint triIdx[], int N, uint nodeIdx, uint &nodePtr)
{
  // terminate recursion
  BVHNode& node = bvhNode[nodeIdx];
  if (node.triCount <= 2) return;
  // determine split axis and position
  float3 extent = node.aabbMax - node.aabbMin;
  int axis = 0;
  if (extent.y > extent.x) axis = 1;
  if (extent.z > extent[axis]) axis = 2;
  float splitPos = node.aabbMin[axis] + extent[axis] * 0.5f;
  // in-place partition
  int i = node.leftFirst;
  int j = i + node.triCount - 1;
  while (i <= j)
  {
      if (tri[triIdx[i]].centroid[axis] < splitPos)
          i++;
      else {
        auto temp = triIdx[i];
        triIdx[i] = triIdx[j];
        triIdx[j--] = temp;
      }
  }
  // abort split if one of the sides is empty
	int leftCount = i - node.leftFirst;
	if (leftCount == 0 || leftCount == node.triCount) return; // never happens for dragon mesh, nice
	// create child nodes
	int leftChildIdx = nodePtr++;
	int rightChildIdx = nodePtr++;
	bvhNode[leftChildIdx].leftFirst = node.leftFirst;
	bvhNode[leftChildIdx].triCount = leftCount;
	bvhNode[rightChildIdx].leftFirst = i;
	bvhNode[rightChildIdx].triCount = node.triCount - leftCount;
	node.leftFirst = leftChildIdx;
	node.triCount = 0;
	// recurse
   UpdateNodeBounds(bvhNode, tri, triIdx, N, leftChildIdx, nodePtr);
   UpdateNodeBounds(bvhNode, tri, triIdx, N, rightChildIdx, nodePtr);
   // recurse
   Subdivide(bvhNode, tri, triIdx, N, leftChildIdx, nodePtr);
   Subdivide(bvhNode, tri, triIdx, N, rightChildIdx, nodePtr);
 }

static void BuildBVH(BVHNode bvhNode[], Tri tri[], uint triIdx[], int N, uint rootNodeIdx, uint nodesUsed){
  for (int i = 0; i < N; i++) tri[i].centroid = 
            (tri[i].vertex0 + tri[i].vertex1 + tri[i].vertex2) * 0.3333f;
  // assign all triangles to root node
  for (int i = 0; i < NUM_TRIANGLES; i++) triIdx[i] = i;
  BVHNode& root = bvhNode[rootNodeIdx];
  root.leftFirst = 0;
  root.triCount = NUM_TRIANGLES;
  UpdateNodeBounds(bvhNode, tri, triIdx, N, rootNodeIdx, nodesUsed );
  // subdivide recursively
  Subdivide(bvhNode, tri, triIdx, N, rootNodeIdx, nodesUsed );
}

const char* kernel_file = "kernel.vxbin";
const char* output_file = "output.png";
uint32_t size = 128;

uint32_t clear_color = 0xff000000;

uint32_t dst_width  = size;
uint32_t dst_height = size;

uint32_t cbuf_stride;
uint32_t cbuf_pitch;
uint32_t cbuf_size;

vx_device_h device = nullptr;
vx_buffer_h BVH_buffer = nullptr;
vx_buffer_h tri_buffer = nullptr;
vx_buffer_h triIdx_buffer = nullptr;
vx_buffer_h color_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;

kernel_arg_t kernel_arg = {};

Tri tri[NUM_TRIANGLES];
uint triIdx[NUM_TRIANGLES];
BVHNode bvhNode[NUM_TRIANGLES * 2 - 1];

static void show_usage() {
   std::cout << "Vortex Ray Tracing Test." << std::endl;
   std::cout << "Usage: [-k: kernel] [-n size] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:o:k:h?")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
      dst_width  = size;
      dst_height = size;
      break;
    case 'k':
      kernel_file = optarg;
      break;
    case 'o':
      output_file = optarg;
      break;
    case 'h':
    case '?': {
      show_usage();
      exit(0);
    } break;
    default:
      show_usage();
      exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(BVH_buffer);
    vx_mem_free(tri_buffer);
    vx_mem_free(triIdx_buffer);
    vx_mem_free(color_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  // parse command arguments
  parse_args(argc, argv);

  std::srand(50);

  // open device connection
  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  std::cout << "output image size: " << dst_height << "x" << dst_width << std::endl;

  kernel_arg.grid_dim[0] = dst_height;
  kernel_arg.grid_dim[1] = dst_width;
  kernel_arg.num_triangles = NUM_TRIANGLES;

  cbuf_stride = 4;
  cbuf_pitch  = dst_width * cbuf_stride;
  cbuf_size   = dst_height * cbuf_pitch;

  int bvh_size = sizeof(BVHNode) * (2 * NUM_TRIANGLES - 1);
  int tri_size = sizeof(Tri) * NUM_TRIANGLES;
  int triIdx_size = sizeof(uint32_t) * NUM_TRIANGLES;

  generate_triangles(tri, NUM_TRIANGLES);
  BuildBVH(bvhNode, tri, triIdx, NUM_TRIANGLES, 0, 1);

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  // bvh buffer
  RT_CHECK(vx_mem_alloc(device, bvh_size, VX_MEM_READ, &BVH_buffer));
  RT_CHECK(vx_mem_address(BVH_buffer, &kernel_arg.bvh_addr));
  // tri buffer
  RT_CHECK(vx_mem_alloc(device, tri_size, VX_MEM_READ, &tri_buffer));
  RT_CHECK(vx_mem_address(tri_buffer, &kernel_arg.tri_addr));
  // triIdx buffer
  RT_CHECK(vx_mem_alloc(device, triIdx_size, VX_MEM_READ, &triIdx_buffer));
  RT_CHECK(vx_mem_address(triIdx_buffer, &kernel_arg.triIdx_addr));
  // color buffer
  RT_CHECK(vx_mem_alloc(device, cbuf_size, VX_MEM_WRITE, &color_buffer));
  RT_CHECK(vx_mem_address(color_buffer, &kernel_arg.cbuf_addr));

  std::cout << "bvh_addr=0x"    << std::hex << BVH_buffer    << std::endl;
  std::cout << "tri_addr=0x"    << std::hex << tri_buffer    << std::endl;
  std::cout << "triIdx_addr=0x" << std::hex << triIdx_buffer << std::endl;
  std::cout << "cbuf_addr=0x"   << std::hex << color_buffer  << std::endl;

  // upload bvh buffer
  {
    std::cout << "upload bvh buffer" << std::endl;
    RT_CHECK(vx_copy_to_dev(BVH_buffer, bvhNode, 0, bvh_size));
  }

//  // upload tri buffer
//   {
//     std::cout << "upload tri buffer" << std::endl;
//     RT_CHECK(vx_copy_to_dev(tri_addr, tri, 0, buf_size));
//   }

//   // upload triIdx buffer
//   {
//     std::cout << "upload triIdx buffer" << std::endl;
//     RT_CHECK(vx_copy_to_dev(triIdx_addr, triIdx, 0, buf_size));
//   }

  // upload program
  std::cout << "upload program" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
    {
      kernel_arg.dst_width   = dst_width;
      kernel_arg.dst_height  = dst_height;
      kernel_arg.cbuf_stride = cbuf_stride;
      kernel_arg.cbuf_pitch  = cbuf_pitch;
    }
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  auto time_start = std::chrono::high_resolution_clock::now();

  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  // save output image
  if (strcmp(output_file, "null") != 0) {
    std::cout << "save output image" << std::endl;
    std::vector<uint8_t> dst_pixels(cbuf_size);
    RT_CHECK(vx_copy_from_dev(dst_pixels.data(), color_buffer, 0, cbuf_size));
    //DumpImage(dst_pixels, dst_width, dst_height, 4);
    auto bits = dst_pixels.data() + (dst_height-1) * cbuf_pitch;
    RT_CHECK(SaveImage(output_file, cocogfx::FORMAT_A8R8G8B8, bits, dst_width, dst_height, -cbuf_pitch));
  }

  // verify result
  std::cout << "check image for result, no verification code yet" << std::endl;
  // TODO: add verification code

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  /*if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }*/

  std::cout << "PASSED!" << std::endl;

  return 0;
}
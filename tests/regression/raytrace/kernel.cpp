#include "common.h"
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include <cocogfx/include/color.hpp>
#include <cocogfx/include/math.hpp>
#include <vx_print.h>
#include <rt_types.h>
#include <graphics.h>

using namespace ray_tracing;

float3 output(0,0,0);
float3 camPos(0,0,15);
float3 p0( -1, 1, -15 ), p1( 1, 1, -15 ), p2( -1, -1, -15 );



void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	const cocogfx::ColorARGB default_color = 0xff000000;
	int col = blockIdx.x;
    int row = blockIdx.y;
	auto tri_addr = reinterpret_cast<Tri*>(arg->tri_addr);
	auto bvh_addr = reinterpret_cast<BVHNode*>(arg->bvh_addr);
	auto tri_idx_addr = reinterpret_cast<uint32_t*>(arg->triIdx_addr);
	auto dst = reinterpret_cast<uint32_t*>(arg->cbuf_addr + col * arg->cbuf_stride + row * arg->cbuf_pitch );
	float3 pixelPos = p0 + (p1 - p0) * (col / 100.0f) + (p2 - p0) * (row / 100.0f);

	Ray ray;
	ray.O = camPos;
	ray.D = normalize(pixelPos-camPos);

	csr_write(VX_CSR_RT_BVH_ADDR,bvh_addr);
	csr_write(VX_CSR_RT_TRI_ADDR,tri_addr);
	csr_write(VX_CSR_RT_TRI_IDX_ADDR,tri_idx_addr);
	TLASIntersect(&ray);
	ray.t = csr_read(VX_CSR_RT_HIT_DIST);

	//for texture/barycentric coordinates of intersection
	float u = vx_csr_read(VX_CSR_RT_U);
	float v = vx_csr_read(VX_CSR_RT_V);
	uint idx = vx_csr_read(VX_CSR_RT_HIT_IDX);


	if (ray.t != 1e30f) {
		float d = 4.0f / ray.t;
		output.x = d;
		output.y = d;
		output.z= d;
	}

	*dst = default_color.value | RGB32FtoRGB8(output);
}

int main() {
	auto __UNIFORM__ arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(2, &arg->grid_dim, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
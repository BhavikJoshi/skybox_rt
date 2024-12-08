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
float3 camPos(0,0,-18);
float3 p0( -1, 1, -15 ), p1( 1, 1, -15 ), p2( -1, -1, -15 );

float3 normalize(const float3& v1) {

    // Calculate the magnitude of the difference vector
    float magnitude = std::sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);

    // Check if the magnitude is not zero to avoid division by zero
    if (magnitude > 0.0f) {
        // Normalize the difference vector
        float3 normalized = {
            v1.x / magnitude,
            v1.y / magnitude,
            v1.z / magnitude
        };
        return normalized;
    } else {
        // Return a zero vector if the magnitude is zero
        return {0.0f, 0.0f, 0.0f};
    }
}

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	const cocogfx::ColorARGB black = 0xff000000;
	const cocogfx::ColorARGB white = 0xffffffff;
	int col = blockIdx.x;
    int row = blockIdx.y;
	auto tri_addr = reinterpret_cast<Tri*>(arg->tri_addr);
	auto bvh_addr = reinterpret_cast<BVHNode*>(arg->bvh_addr);
	auto tri_idx_addr = reinterpret_cast<uint32_t*>(arg->triIdx_addr);
	auto dst = reinterpret_cast<uint32_t*>(arg->cbuf_addr + col * arg->cbuf_stride + row * arg->cbuf_pitch );
	float dst_width = arg->dst_width;
	float dst_height = arg->dst_height;
	float3 pixelPos = p0 + (p1 - p0) * (col / dst_height) + (p2 - p0) * (row / dst_width);

	Ray ray;
	ray.O = camPos;
	ray.D = normalize(pixelPos-camPos);

	vx_printf("BVH_CSR_ADDRS: %x", VX_CSR_RT_BVH_ADDR);
	csr_write(VX_CSR_RT_BVH_ADDR,bvh_addr);
	vx_printf("Write succesful\n");
	csr_write(VX_CSR_RT_TRI_ADDR,tri_addr);
	csr_write(VX_CSR_RT_TRI_IDX_ADDR,tri_idx_addr);
	vx_bvh_ti((int)&ray);
	ray.t = csr_read(VX_CSR_RT_HIT_DIST);

	//for texture/barycentric coordinates of intersection
	//float u = csr_read(VX_CSR_RT_HIT_U);
	//float v = csr_read(VX_CSR_RT_HIT_V);
	//uint32_t idx = csr_read(VX_CSR_RT_HIT_IDX);

	*dst = ray.t == 1e30f ? black : white;
}

int main() {
	auto __UNIFORM__ arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(2, arg->grid_dim, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
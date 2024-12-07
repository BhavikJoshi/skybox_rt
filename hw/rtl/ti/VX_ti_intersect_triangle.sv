//!/bin/bash

// Copyright © 2024
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_raster_define.vh"
`include "float_dpi.vh"

module VX_ti_intersect_triangle import VX_ti_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter INSTANCE_IDX    = 0,
    parameter NUM_INSTANCES   = 1,
) 
(    
    // Inputs
    input wire enable;
    input wire [3*`TI_FLOAT_BITS-1:0] tri_vertex0;
    input wire [3*`TI_FLOAT_BITS-1:0] tri_vertex1;
    input wire [3*`TI_FLOAT_BITS-1:0] tri_vertex2;

    input wire [3*`TI_FLOAT_BITS-1:0] ray_origin;
    input wire [3*`TI_FLOAT_BITS-1:0] ray_dir;

    // Output
    output wire intersect;
    output wire [TI_FLOAT_BITS-1:0] hit_point_u;
    output wire [TI_FLOAT_BITS-1:0] hit_point_v;
    output wire [TI_FLOAT_BITS-1:0] dist;
    
);
    // Bit ranges for float3 X, Y, and Z
    localparam X_S = 0;
    local param X_E = 31;
    localparam Y_S = 32;
    localparam Y_E = 63;
    localparam Z_S = 64;
    localparam Z_E = 95;

    // TODO: make constants

    // Intermediate variables
    wire [3*`TI_FLOAT_BITS-1:0] edge1, edge2;
    wire [3*`TI_FLOAT_BITS-1:0] h;

    wire [`TI_FLOAT_BITS-1:0] c1_p1;
    wire [`TI_FLOAT_BITS-1:0] c1_p2;
    wire [`TI_FLOAT_BITS-1:0] c1_p3;
    wire [`TI_FLOAT_BITS-1:0] c1_p4;
    wire [`TI_FLOAT_BITS-1:0] c1_p5;
    wire [`TI_FLOAT_BITS-1:0] c1_p6;

    wire [`TI_FLOAT_BITS-1:0] a;

    wire [`TI_FLOAT_BITS-1:0] d1_p1;
    wire [`TI_FLOAT_BITS-1:0] d1_p2;
    wire [`TI_FLOAT_BITS-1:0] d1_p3;
    wire [`TI_FLOAT_BITS-1:0] d1_s1;

    wire a_gt_neg, a_lt_pos;

    wire [`TI_FLOAT_BITS-1:0] f;
    wire [3*`TI_FLOAT_BITS-1:0] s;
    wire [`TI_FLOAT_BITS-1:0] u;

    wire [`TI_FLOAT_BITS-1:0] d2_p1;
    wire [`TI_FLOAT_BITS-1:0] d2_p2;
    wire [`TI_FLOAT_BITS-1:0] d2_p3;
    wire [`TI_FLOAT_BITS-1:0] d2_s1;
    wire [`TI_FLOAT_BITS-1:0] d2_s2;

    wire u_lt_zero, u_gt_one;

    wire [3*`TI_FLOAT_BITS-1:0] q;

    wire [`TI_FLOAT_BITS-1:0] c2_p1;
    wire [`TI_FLOAT_BITS-1:0] c2_p2;
    wire [`TI_FLOAT_BITS-1:0] c2_p3;
    wire [`TI_FLOAT_BITS-1:0] c2_p4;
    wire [`TI_FLOAT_BITS-1:0] c2_p5;
    wire [`TI_FLOAT_BITS-1:0] c2_p6;

    wire [`TI_FLOAT_BITS-1:0] v;

    wire [`TI_FLOAT_BITS-1:0] d3_p1;
    wire [`TI_FLOAT_BITS-1:0] d3_p2;
    wire [`TI_FLOAT_BITS-1:0] d3_p3;
    wire [`TI_FLOAT_BITS-1:0] d3_s1;
    wire [`TI_FLOAT_BITS-1:0] d3_s2;

    wire [`TI_FLOAT_BITS-1:0] uvsum;
    wire v_lt_zero, uvsum_gt_one;

    wire [`TI_FLOAT_BITS-1:0] t;

    wire [`TI_FLOAT_BITS-1:0] d4_p1;
    wire [`TI_FLOAT_BITS-1:0] d4_p2;
    wire [`TI_FLOAT_BITS-1:0] d4_p3;
    wire [`TI_FLOAT_BITS-1:0] d4_s1;
    wire [`TI_FLOAT_BITS-1:0] d4_s2;

    // Result
    assign intersect = (!a_gt_neg && !a_lt_pos && !u_lt_zero && !u_gt_one && !v_lt_zero && !uvsum_gt_one && t_lt_dist && t_gt_zero);
    assign hit_point_u = u;
    assign hit_point_v = v;
    assign dist = t;

    // Intersection calculation
    always @ (*) begin
        // edge1 = tri_vertex1 - tri_vertex0
        dpi_sub (enable, `TI_FLOAT, tri_vertex1[X_E:X_S], tri_vertex0[X_E:X_S], 0, edge1[X_E:X_S], 0);
        dpi_sub (enable, `TI_FLOAT, tri_vertex1[Y_E:Y_S], tri_vertex0[Y_E:Y_S], 0, edge1[Y_E:Y_S], 0);
        dpi_sub (enable, `TI_FLOAT, tri_vertex1[Z_E:Z_S], tri_vertex0[Z_E:Z_S], 0, edge1[Z_E:Z_S], 0);
        // edge2 = tri_vertex2 - tri_vertex0
        dpi_sub (enable, `TI_FLOAT, tri_vertex2[X_E:X_S], tri_vertex0[X_E:X_S], 0, edge2[X_E:X_S], 0);
        dpi_sub (enable, `TI_FLOAT, tri_vertex2[Y_E:Y_S], tri_vertex0[Y_E:Y_S], 0, edge2[Y_E:Y_S], 0);
        dpi_sub (enable, `TI_FLOAT, tri_vertex2[Z_E:Z_S], tri_vertex0[Z_E:Z_S], 0, edge2[Z_E:Z_S], 0);
        // h = cross( ray_dir, edge2 )
        dpi_fmul (enable, `TI_FLOAT, ray_dir[Y_E:Y_S], edge2[Z_E:Z_S], 0, c1_p1, 0);
        dpi_fmul (enable, `TI_FLOAT, ray_dir[Z_E:Z_S], edge2[Y_E:Y_S], 0, c1_p2, 0);
        dpi_fsub (enable, `TI_FLOAT, c1_p1, c1_p2, 0, h[X_E:X_S], 0);
        dpi_fmul (enable, `TI_FLOAT, ray_dir[Z_E:Z_S], edge2[X_E:X_S], 0, c1_p3, 0);
        dpi_fmul (enable, `TI_FLOAT, ray_dir[X_E:X_S], edge2[Z_E:Z_S], 0, c1_p4, 0);
        dpi_fsub (enable, `TI_FLOAT, c1_p3, c1_p4, 0, h[Y_E:Y_S], 0);
        dpi_fmul (enable, `TI_FLOAT, ray_dir[X_E:X_S], edge2[Y_E:Y_S], 0, c1_p5, 0);
        dpi_fmul (enable, `TI_FLOAT, ray_dir[Y_E:Y_S], edge2[X_E:X_S], 0, c1_p6, 0);
        dpi_fsub (enable, `TI_FLOAT, c1_p5, c1_p6, 0, h[Z_E:Z_S], 0);
        // a = dot( edge1, h )
        dpi_fmul (enable, `TI_FLOAT, edge1[X_E:X_S], h[X_E:X_S], 0, d1_p1, 0);
        dpi_fmul (enable, `TI_FLOAT, edge1[Y_E:Y_S], h[Y_E:Y_S], 0, d1_p2, 0);
        dpi_fadd (enable, `TI_FLOAT, d1_p1, d1_p2, 0, d1_s1, 0);
        dpi_fmul (enable, `TI_FLOAT, edge1[Z_E:Z_S], h[Z_E:Z_S], 0, d1_p3, 0);
        dpi_fadd (enable, `TI_FLOAT, d1_s1, d1_p3, 0, a, 0);
        // a > -0.0001f && a < 0.0001f
        dpi_flt (enable, `TI_FLOAT, -0.0001f, a, a_gt_neg, 0);
        dpi_flt (enable, `TI_FLOAT, a, 0.0001f, a_lt_pos, 0);
        // f = 1 / a
        dpi_fdiv (enable, `TI_FLOAT, 1, a, 0, f, 0);
        // s = ray_origin - tri_vertex0
        dpi_fsub (enable, `TI_FLOAT, ray_origin[X_E:X_S], tri_vertex0[X_E:X_S], 0, s[X_E:X_S], 0);
        dpi_fsub (enable, `TI_FLOAT, ray_origin[Y_E:Y_S], tri_vertex0[Y_E:Y_S], 0, s[Y_E:Y_S], 0);
        dpi_fsub (enable, `TI_FLOAT, ray_origin[Z_E:Z_S], tri_vertex0[Z_E:Z_S], 0, s[Z_E:Z_S], 0);
        // u = f * dot( s, h )
        dpi_fmul (enable, `TI_FLOAT, s[X_E:X_S], h[X_E:X_S], 0, d2_p1, 0);
        dpi_fmul (enable, `TI_FLOAT, s[Y_E:Y_S], h[Y_E:Y_S], 0, d2_p2, 0);
        dpi_fadd (enable, `TI_FLOAT, d2_p1, d2_p2, 0, d2_s1, 0);
        dpi_fmul (enable, `TI_FLOAT, s[Z_E:Z_S], h[Z_E:Z_S], 0, d2_p3, 0);
        dpi_fadd (enable, `TI_FLOAT, d2_s1, d2_p3, 0, d2_s2, 0);
        dpi_fmul (enable, `TI_FLOAT, f, d2_s2, 0, u, 0);
        // u < 0 || u > 1
        dpi_flt (enable, `TI_FLOAT, 0, u, u_lt_zero, 0);
        dpi_flt (enable, `TI_FLOAT, u, 1, u_gt_one, 0);
        // q = cross( s, edge1 )
        dpi_fmul (enable, `TI_FLOAT, s[Y_E:Y_S], edge1[Z_E:Z_S], 0, c2_p1, 0);
        dpi_fmul (enable, `TI_FLOAT, s[Z_E:Z_S], edge1[Y_E:Y_S], 0, c2_p2, 0);
        dpi_fsub (enable, `TI_FLOAT, c2_p1, c2_p2, 0, q[X_E:X_S], 0);
        dpi_fmul (enable, `TI_FLOAT, s[Z_E:Z_S], edge1[X_E:X_S], 0, c2_p3, 0);
        dpi_fmul (enable, `TI_FLOAT, s[X_E:X_S], edge1[Z_E:Z_S], 0, c2_p4, 0);
        dpi_fsub (enable, `TI_FLOAT, c2_p3, c2_p4, 0, q[Y_E:Y_S], 0);
        dpi_fmul (enable, `TI_FLOAT, s[X_E:X_S], edge1[Y_E:Y_S], 0, c2_p5, 0);
        dpi_fmul (enable, `TI_FLOAT, s[Y_E:Y_S], edge1[X_E:X_S], 0, c2_p6, 0);
        dpi_fsub (enable, `TI_FLOAT, c2_p5, c2_p6, 0, q[Z_E:Z_S], 0);
        // v = f * dot( ray_dir, q )
        dpi_fmul (enable, `TI_FLOAT, ray_dir[X_E:X_S], q[X_E:X_S], 0, d3_p1, 0);
        dpi_fmul (enable, `TI_FLOAT, ray_dir[Y_E:Y_S], q[Y_E:Y_S], 0, d3_p2, 0);
        dpi_fadd (enable, `TI_FLOAT, d3_p1, d3_p2, 0, d3_s1, 0);
        dpi_fmul (enable, `TI_FLOAT, ray_dir[Z_E:Z_S], q[Z_E:Z_S], 0, d3_p3, 0);
        dpi_fadd (enable, `TI_FLOAT, d3_s1, d3_p3, 0, d3_s2, 0);
        dpi_fmul (enable, `TI_FLOAT, f, d3_s2, 0, v, 0);
        // v < 0 || u + v > 1
        dpi_flt (enable, `TI_FLOAT, 0, v, v_lt_zero, 0);
        dpi_fadd (enable, `TI_FLOAT, u, v, 0, uvsum, 0);
        dpi_flt (enable, `TI_FLOAT, uvsum, 1, uvsum_gt_one, 0);
        // t = f * dot( edge2, q )
        dpi_fmul (enable, `TI_FLOAT, edge2[X_E:X_S], q[X_E:X_S], 0, d4_p1, 0);
        dpi_fmul (enable, `TI_FLOAT, edge2[Y_E:Y_S], q[Y_E:Y_S], 0, d4_p2, 0);
        dpi_fadd (enable, `TI_FLOAT, d4_p1, d4_p2, 0, d4_s1, 0);
        dpi_fmul (enable, `TI_FLOAT, edge2[Z_E:Z_S], q[Z_E:Z_S], 0, d4_p3, 0);
        dpi_fadd (enable, `TI_FLOAT, d4_s1, d4_p3, 0, d4_s2, 0);
        dpi_fmul (enable, `TI_FLOAT, f, d4_s2, 0, t, 0);
        // t > 0.0001f && t < dist
        dpi_flt (enable, `TI_FLOAT, 0.0001f, t, t_gt_zero, 0);
        dpi_flt (enable, `TI_FLOAT, t, dist, t_lt_dist, 0);
    end

endmodule

///////////////////////////////////////////////////////////////////////////////
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

inline float3 cross( const float3& a, const float3& b ) {
    float3 res(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
    return res;
}

inline float dot( const float3& a, const float3& b ) { return a.x * b.x + a.y * b.y + a.z * b.z; }

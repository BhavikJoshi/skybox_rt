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

`include "VX_ti_define.vh"
`include "float_dpi.vh" 

module VX_ti_intersect_bounding_box import VX_ti_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter INSTANCE_IDX    = 0,
    parameter NUM_INSTANCES   = 1,
) 
(       
        // Inputs
        input wire enable;
        input wire [3*`TI_FLOAT_BITS-1:0] bmin;
        input wire [3*`TI_FLOAT_BITS-1:0] bmax;
        input wire [3*`TI_FLOAT_BITS-1:0] ray_origin;
        input wire [3*`TI_FLOAT_BITS-1:0] ray_dir;
        input wire [`TI_FLOAT_BITS-1:0] ray_t;
        
        // Output
        output wire intersect;
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
        wire [`TI_FLOAT_BITS-1:0] ta1, ta2, tx1, tx2;
        wire [`TI_FLOAT_BITS-1:0] tmin1, tmax1;
        wire [`TI_FLOAT_BITS-1:0] tb1, tb2, ty1, ty2;
        wire [`TI_FLOAT_BITS-1:0] tmin2, tmax2;
        wire [`TI_FLOAT_BITS-1:0] tmin3, tmax3;
        wire [`TI_FLOAT_BITS-1:0] tc1, tc2, tz1, tz2;
        wire [`TI_FLOAT_BITS-1:0] tmin4, tma4;
        wire [`TI_FLOAT_BITS-1:0] tmin, tmax;
        wire max_gt_min, min_lt_rayt, max_gt_zero;

        // Result
        assign intersect = max_gt_min && min_lt_rayt && max_gt_zero;

        // Intersection calculation
        always @ (*) begin
            // ta1 = bmin.x - ray_origin.x; tx1 = ta1 / ray_dir.x
            dpi_fsub (enable, `TI_FLOAT, bmin[X_E:X_S], ray_origin[X_E:X_S], 0, ta1, 0);
            dpi_fdiv (enable, `TI_FLOAT, ta1, ray_dir[X_E:X_S], 0, tx1, 0);
            // ta2 = bmax.x - ray_origin.x; tx2 = ta2 / ray_dir.x
            dpi_fsub (enable, `TI_FLOAT, bmax[X_E:X_S], ray_origin[X_E:X_S], 0, ta2, 0);
            dpi_fdiv (enable, `TI_FLOAT, ta2, ray_dir[X_E:X_S], 0, tx2, 0);
            // tmin1 = min(tx1, tx2); tmax1 = max(tx1, tx2)
            dpi_fmin (enable, `TI_FLOAT, tx1, tx2, tmin1, 0);
            dpi_fmax (enable, `TI_FLOAT, tx1, tx2, tmax1, 0);
            // tb1 = bmin.y - ray_origin.y; ty1 = tb1 / ray_dir.y
            dpi_fsub (enable, `TI_FLOAT, bmin[Y_E:Y_S], ray_origin[Y_E:Y_S], 0, tb1, 0);
            dpi_fdiv (enable, `TI_FLOAT, tb1, ray_dir[Y_E:Y_S], 0, ty1, 0);
            // tb2 = bmax.y - ray_origin.y; ty2 = tb2 / ray_dir.y
            dpi_fsub (enable, `TI_FLOAT, bmax[Y_E:Y_S], ray_origin[Y_E:Y_S], 0, tb2, 0);
            dpi_fdiv (enable, `TI_FLOAT, tb2, ray_dir[Y_E:Y_S], 0, ty2, 0);
            // tmin2 = min(ty1, ty2); tmax2 = max(ty1, ty2);
            //tmin3 = max(tmin1, tmin2); tmax3 = min(tmax1, tmax2)
            dpi_fmin (enable, `TI_FLOAT, ty1, ty2, tmin2, 0);
            dpi_fmax (enable, `TI_FLOAT, ty1, ty2, tmax2, 0);
            dpi_fmax (enable, `TI_FLOAT, tmin1, tmin2, tmin3, 0);
            dpi_fmin (enable, `TI_FLOAT, tmax1, tmax2, tmax3, 0);
            // tc1 = bmin.z - ray_origin.z; tz1 = tc1 / ray_dir.z
            dpi_fsub (enable, `TI_FLOAT, bmin[Z_E:Z_S], ray_origin[Z_E:Z_S], 0, tc1, 0);
            dpi_fdiv (enable, `TI_FLOAT, tc1, ray_dir[Z_E:Z_S], 0, tz1, 0);
            // tc2 = bmax.z - ray_origin.z; tz2 = tc2 / ray_dir.z
            dpi_fsub (enable, `TI_FLOAT, bmax[Z_E:Z_S], ray_origin[Z_E:Z_S], 0, tc2, 0);
            dpi_fdiv (enable, `TI_FLOAT, tc2, ray_dir[Z_E:Z_S], 0, tz2, 0);
            // tmin4 = min(tz1, tz2); tmax4 = max(tz1, tz2)
            dpi_fmin (enable, `TI_FLOAT, tz1, tz2, tmin4, 0);
            dpi_fmax (enable, `TI_FLOAT, tz1, tz2, tmax4, 0);
            // tmin = max(tmin3, tmin4); tmax = min(tmax3, tmax4)
            dpi_fmax (enable, `TI_FLOAT, tmin3, tmin4, tmin, 0);
            dpi_fmin (enable, `TI_FLOAT, tmax3, tmax4, tmax, 0);
            // max_gt_min = tmax > tmin
            dpi_fle  (enable, `TI_FLOAT, tmin, tmax, max_gt_min, 0);
            // min_lt_rayt = tmin < ray_t
            dpi_flt  (enable, `TI_FLOAT, tmin, ray_t, min_lt_rayt, 0);
            // max_gt_zero = tmax > 0
            dpi_flt  (enable, `TI_FLOAT, 0, tmax, max_gt_zero, 0);
        end

endmodule

///////////////////////////////////////////////////////////////////////////////

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

module VX_ti_intersect import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter INSTANCE_IDX    = 0,
    parameter NUM_INSTANCES   = 1,
) 
(    
        // Clock
        input wire clk,
        input wire reset,

        input wire ready,
        input wire isTriangle,
        input wire [3*32-1:0] aabb_min,
        input wire [3*32-1:0] aabb_max,
        input wire [3*32-1:0] tri_v0,
        input wire [3*32-1:0] tri_v1,
        input wire [3*32-1:0] tri_v2,
        input wire [3*32-1:0] ray_origin,
        input wire [3*32-1:0] ray_dir,
        
        output wire valid,
        output wire intersects,
        output wire [32-1:0] distance,
        output wire [32-1:0] hit_point_u,
        output wire [32-1:0] hit_point_v,
        output wire [32-1:0] hit_index

        // Instantiate FPU interface??
        


endmodule

///////////////////////////////////////////////////////////////////////////////

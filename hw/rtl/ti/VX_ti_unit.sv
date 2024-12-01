//!/bin/bash

// Copyright © 2019-2023
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

module VX_raster_unit import VX_gpu_pkg::*; import VX_raster_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter INSTANCE_IDX    = 0,
    parameter NUM_INSTANCES   = 1,
) (

    // Clock
    input wire clk,
    input wire reset,


    // Memory interface
    VX_mem_bus_if.master    cache_bus_if [RCACHE_NUM_REQS],


    // Outputs
    VX_ti_bus_if.master     ti_bus_if
);
    localparam BVH_NODE_BITS  = 32 << 3;
    localparam BVH_INDEX_BITS = 32;
    localparam TRI

    localparam IDLE = 0;
    localparam PUSH_NODE = 1;
    localparam POP_NODE = 2;
    localparam INTERSECT_AABB = 3;
    localparam INTERSECT_TRIANGLE = 4;
    localparam HIT_AABB = 5;
    localparam HIT_TRIANGLE = 6;
    localparam STACK_EMPTY = 7;

    reg [2:0] state, nextState;

    always @ (*) begin
        case (state)
            IDLE: begin
                nextState = IDLE;
                if (ti_bus_if.valid) begin
                  nextState = PUSH_NODE;
                end
            end
            PUSH_NODE: nextState = IDLE;
            POP_NODE: nextState = IDLE;
            INTERSECT_AABB: nextState = IDLE;
            INTERSECT_TRIANGLE: nextState = IDLE;
            HIT_AABB: nextState = IDLE;
            HIT_TRIANGLE: nextState = IDLE;
            STACK_EMPTY: nextState = IDLE;
        endcase
    end

    always @ (posedge clk) begin
        if (reset) begin
            state <= IDLE;
        end else begin
            state <= nextState;
        end
    end
    

    wire stackEmpty;
    reg [BVH_NODE_BITS-1:0] currNode;

    VX_ti_bvh_idx_stack  ti_bvh_stack (
        .clk            (clk),
        .reset          (reset),
        .is_empty       (stackEmpty),
        .cache_bus_if   (cache_bus_if),
        .ti_bus_if      (ti_bus_if)
    );

    VX_ti_mem raster_mem (
        .clk          (clk),
        .reset        (mem_reset),

        .start        (mem_unit_start),
        .mem_addr     (mem_unit_addr),
        .mem_size     (mem_unit_size),

        .mem_data     (mem_unit_data),
        .valid_out    (mem_unit_valid),
        .ready_out    (mem_unit_ready)
    );

   

endmodule

///////////////////////////////////////////////////////////////////////////////

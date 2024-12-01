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
    localparam TRI_INDEX_BITS = 32;
    localparam ADDR_BITS = 32;
    localparam TRI_NODE_BITS  = 48 << 3;
    localparam TRICOUNT_IDX_START = 224;
    localparam TRICOUNT_IDX_END = 256;

    // TODO: grab these from CSRs or as input into the module?
    // if grabbing from CSR, need an extra state after IDLE for grabbing addresses from CSR
    reg [ADDR_BITS-1:0] bvhBaseAddr;
    reg [AADR_BITS-1:0] triIdxBaseAddr;
    reg [ADDR_BITS-1:0] triBaseAddr;


    // T&I Unit States
    localparam IDLE = 0;
    localparam PUSH_STACK = 1;
    localparam POP_STACK = 2;
    localparam FETCH_BVH_NODE = 3;
    localparam FETCH_TRI_INDEX = 4;
    localparam FETCH_TRI_NODE = 5;
    localparam INTERSECT = 6;
    localparam MISS = 7;
    localparam HIT = 6;
    localparam STACK_EMPTY = 7;

    reg [2:0] state, nextState;
    wire stackEmpty;
    reg [BVH_INDEX_BITS-1:0] bvhIndexPush;
    reg push;
    reg pop;
    reg [BVH_INDEX_BITS-1:0] nextBvhIndex;
    reg [BVH_NODE_BITS-1:0] bvhBuffer;
    reg [TRI_INDEX_BITS-1:0] triIndexBuffer;
    reg [TRI_NODE_BITS-1:0] triBuffer;


    // Next state logic
    always @ (*) begin
        case (state)
            IDLE: begin
                if (ti_bus_if.valid) begin
                    nextState = PUSH_STACK;
                end
                else begin
                    nextState = IDLE;
                end
            end
            PUSH_STACK: begin
                nextState = POP_STACK;
            end
            POP_STACK: begin
                nextState = stackEmpty ? STACK_EMPTY : FETCH_BVH_NODE;
            end
            FETCH_BVH_NODE: begin
                // Once memory returns the BVH node, check if it's a leaf node
                if (bvhBufferValid) begin
                    // If not a leaf node, intersect the AABB; otherwise, fetch the triangle index
                    nextState = currNode[TRICOUNT_IDX_END-1:TRICOUNT_IDX_START] == 0 ? INTERSECT : FETCH_TRI_INDEX;
                end
                // Wait for memory to return the BVH node
                else begin
                    nextState = FETCH_BVH_NODE;
                end
            end
            FETCH_TRI_INDEX: begin
                if (triIdxValid) begin
                    nextState = FETCH_TRI_NODE;
                end
                // Wait for memory to return the triangle index
                else begin
                    nextState = FETCH_TRI_INDEX;
                end
            end
            FETCH_TRI_NODE: begin
                if (triNodeValid) begin
                    nextState = INTERSECT;
                end
                // Wait for memory to return the triangle node
                else begin
                    nextState = FETCH_TRI_NODE;
                end
            end
            INTERSECT: begin
                if (intersectValid) begin
                    if (intersects) begin
                        nextState = HIT;
                    end
                    else begin
                        nextState = MISS;
                    end
                end
                // Wait for intersection unit to return the result
                else begin
                    nextState = INTERSECT;
                end
            end
            MISS: begin
                nextState = POP_STACK;
            end
            HIT: begin
                nextState = isTriangleIntersect ? POP_STACK : PUSH_STACK;
            end
            STACK_EMPTY: begin
                nextState = IDLE;
            end
        endcase
    end

    // State register update
    always @ (posedge clk) begin
        if (reset) begin
            state <= IDLE;
        end else begin
            state <= nextState;
        end
    end

    // Get memory address of next fetch based on current state
    reg [ADDR_BITS-1:0] mem_unit_addr;
    always @ (*) begin
        if (state == FETCH_BVH_NODE) begin
            mem_unit_addr = bvhBaseAddr + nextBvhIndex << $CLOG2(BVH_NODE_BITS);
        end
        else if (state == FETCH_TRI_INDEX) begin
            mem_unit_addr = triIdxBaseAddr + (triIndexBuffer << 2);
        end
        else if (state == FETCH_TRI_NODE) begin
            mem_unit_addr = triBaseAddr + (triBuffer << 3);
        end
    end
    

    VX_ti_stack  ti_bvh_index_stack (
        .clk            (clk),
        .reset          (reset),
        .push           (push),
        .pop            (pop),
        .data_in        (bvhIndexPush),
        .data_out       (nextBvhIndex),
        .empty          (stackEmpty),
        .full           ()
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

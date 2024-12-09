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

`include "VX_ti_define.vh"

module VX_ti_unit import VX_gpu_pkg::*; import VX_ti_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter INSTANCE_IDX    = 0,
    parameter NUM_INSTANCES   = 1,
) (

    // Clock
    input wire clk,
    input wire reset,

    // Inputs
    VX_execute_if.slave     ti_execute_if,

    // CSR interface
    VX_sfu_csr_if.slave     ti_csr_if,
    
    // Memory interface (output request and receive response)
    VX_lsu_mem_if.master    lsu_mem_if,     

    // Outputs
    VX_commit_if.master     ti_commit_if
    // TODO: write to commit interface
);

    localparam RAY_BYTES = 28;
    localparam BVH_NODE_BYTES  = 32;
    localparam BVH_INDEX_BYTES = 4;
    localparam TRI_INDEX_BYTES = 4;
    localparam TRI_NODE_BYTES  = 48;
    localparam RAY_BITS = RAY_BYTES << 3;
    localparam BVH_NODE_BITS = BVH_NODE_BYTES << 3;
    localparam BVH_INDEX_BITS = BVH_INDEX_BYTES << 3;
    localparam TRI_INDEX_BITS = TRI_INDEX_BYTES << 3;
    localparam TRI_NODE_BITS  = TRI_NODE_BYTES << 3;

    localparam ADDR_BITS = 32;
    localparam TRI_NODE_BITS  = 48 << 3;

    wire [ADDR_BITS-1:0] bvhBaseAddr;
    wire [AADR_BITS-1:0] triIdxBaseAddr;
    wire [ADDR_BITS-1:0] triBaseAddr;

    ti_csrs_t reg_csrs;

    // Handle CSR writes
    always @(posedge clk) begin
        if (reset) begin
            reg_csrs <= '0;
        end else if (ti_csr_if.write_enable) begin
            case (ti_csr_if.write_addr)
                `VX_CSR_RT_BVH_ADDR: reg_csrs.bvh <= ti_csr_if.write_data;
                `VX_CSR_RT_TRI_ADDR: reg_csrs.tri <= ti_csr_if.write_data;
                `VX_CSR_RT_TRI_IDX_ADDR: reg_csrs.triIdx <= ti_csr_if.write_data;
                default:;
            endcase
        end
    end

    // Handle CSR reads
    assign bvhBaseAddr = reg_csrs.bvh;
    assign triIdxBaseAddr = reg_csrs.tri;
    assign triBaseAddr = reg_csrs.triIdx;

    // T&I Unit States
    localparam IDLE = 0;
    localparam SEND_RAY_REQ = 1;
    localparam WAIT_RAY_RSP = 2;
    localparam RECV_RAY_RSP = 3;
    localparam PUSH_STACK = 4;
    localparam POP_STACK = 5;
    localparam SEND_BVH_NODE_REQ = 6;
    localparam WAIT_BVH_NODE_RSP = 7;
    localparam RECV_BVH_NODE_RSP= 8;
    localparam INTERSECT_BOUNDING_BOX = 9;
    localparam HIT_BOUNDING_BOX = 10;
    localparam SEND_TRI_INDEX_REQ = 11;
    localparam WAIT_TRI_INDEX_RSP = 12;
    localparam RECV_TRI_INDEX_RSP = 13;
    localparam SEND_TRI_NODE_L_REQ = 14;
    localparam WAIT_TRI_NODE_L_RSP = 15;
    localparam RECV_TRI_NODE_L_RSP = 16;
    localparam SEND_TRI_NODE_H_REQ = 17;
    localparam WAIT_TRI_NODE_L_RSP = 18;
    localparam RECV_TRI_NODE_L_RSP = 19;
    localparam INTERSECT_TRIANGLE = 20;
    localparam HIT_TRIANGLE = 21;
    localparam MISS = 22;
    localparam STACK_EMPTY = 23;

    reg [4:0] state, nextState;
    wire stackEmpty;
    reg [BVH_INDEX_BITS-1:0] bvhIndexPush;
    reg push;
    reg pop;
    reg [BVH_INDEX_BITS-1:0] nextBvhIndex;
    reg [RAY_BITS-1:0] rayBuffer;
    reg [BVH_NODE_BITS-1:0] bvhBuffer;
    reg [TRI_INDEX_BITS-1:0] triIndexBuffer;
    reg [TRI_NODE_BITS-1:0] triBuffer;

    // State transition logic
    always @ (*) begin
        case (state)
            IDLE: begin
                // If there's a valid request, push the initial BVH index onto the stack
                if (ti_execute_if.valid) begin
                    nextState = SEND_RAY_REQ;
                end
                // Spin in IDLE state until a valid request is received
                else begin
                    nextState = IDLE;
                end
            end
            SEND_RAY_REQ: begin
            if (lsu_mem_if.req_data.req_ready) begin
                    nextState = WAIT_RAY_RSP;
            end
            // Wait for memory to return the triangle index
            else begin
                nextState = SEND_RAY_REQ;
            end
            WAIT_RAY_RSP: begin
                if (lsu_mem_if.req_data.rsp_valid) begin
                    nextState = RECV_RAY_RSP;
                end
                else begin
                    nextState = WAIT_RAY_RSP;
                end
            end
            RECV_RAY_RSP: begin
                nextState = PUSH_STACK;
            end
            PUSH_STACK: begin
                nextState = POP_STACK;
            end
            POP_STACK: begin
                // Check if stack is empty
                // Otherwise, fetch the next BVH node
                nextState = stackEmpty ? STACK_EMPTY : SEND_BVH_NODE_REQ;
            end
            SEND_BVH_NODE_REQ: begin
                // Once memory returns the BVH node, check if it's a leaf node
                if (lsu_mem_if.req_data.req_ready) begin
                    // If not a leaf node, intersect the AABB; otherwise, fetch the triangle index
                    nextState = WAIT_BVH_NODE_RSP;
                end
                // Wait for memory to return the BVH node
                else begin
                    nextState = SEND_BVH_NODE_REQ
                end
            end
            WAIT_BVH_NODE_RSP: begin
                if (lsu_mem_if.req_data.rsp_valid) begin
                    nextState = RECV_BVH_NODE_RSP;
                end
                else begin
                    nextState = WAIT_BVH_NODE_RSP;
                end
            end
            RECV_BVH_NODE_RSP: begin
                nextState = INTERSECT_BOUNDING_BOX;
            end
            INTERSECT_BOUNDING_BOX: begin
                wire isLeafNode;
                assign isLeafNode = currNode[TRICOUNT_IDX_END-1:TRICOUNT_IDX_START] != 0;
                // If we hit the bounding box
                //     If it is a leaf node, we need to intersect the triangles
                //     otherwise, add the sub-nodes to the stack
                // otherwise, we miss and go back to the stack
                nextState = (hitBoundingBox == 1'b1) ? (isLeafNode == 1'b1 ? SEND_TRI_INDEX_REQ : HIT_BOUNDING_BOX) : MISS;
            end
            HIT_BOUNDING_BOX: begin
                nextState = PUSH_STACK;
            end
            SEND_TRI_INDEX_REQ: begin
                if (lsu_mem_if.req_data.req_ready) begin
                    nextState = WAIT_TRI_INDEX_RSP;
                end
                // Wait for memory to return the triangle index
                else begin
                    nextState = SEND_TRI_INDEX_REQ;
                end
            end
            WAIT_TRI_INDEX_RSP: begin
                if (lsu_mem_if.req_data.rsp_valid) begin
                    nextState = RECV_TRI_INDEX_RSP;
                end
                // Wait for memory to return the triangle index
                else begin
                    nextState = WAIT_TRI_INDEX_RSP;
                end
            end
            RECV_TRI_INDEX_RSP: begin
                nextState = SEND_TRI_NODE_L_REQ;
            end
            SEND_TRI_NODE_L_REQ: begin
                if (lsu_mem_if.req_data.req_ready) begin
                    nextState = WAIT_TRI_NODE_L_RSP;
                end
                // Wait for memory to return the triangle node lower bits
                else begin
                    nextState = SEND_TRI_NODE_L_REQ;
                end
            end
            WAIT_TRI_NODE_L_RSP: begin
                if (lsu_mem_if.req_data.rsp_valid) begin
                    nextState = RECV_TRI_NODE_L_RSP;
                end
                // Wait for memory to return the triangle node higher bits
                else begin
                    nextState = WAIT_TRI_NODE_L_RSP;
                end
            end
            RECV_TRI_NODE_L_RSP: begin
                nextState = SEND_TRI_NODE_H_REQ;
            end
            SEND_TRI_NODE_H_REQ: begin
                if (lsu_mem_if.req_data.req_ready) begin
                    nextState = WAIT_TRI_NODE_H_RSP;
                end
                // Wait for memory to return the triangle node higher bits
                else begin
                    nextState = SEND_TRI_NODE_H_REQ;
                end
            end
            WAIT_TRI_NODE_H_RSP: begin
                if (lsu_mem_if.req_data.rsp_valid) begin
                    nextState = RECV_TRI_NODE_H_RSP;
                end
                // Wait for memory to return the triangle node higher bits
                else begin
                    nextState = WAIT_TRI_NODE_H_RSP;
                end
            end
            RECV_TRI_NODE_H_RSP: begin
                nextState = INTERSECT_TRIANGLE;
            end
            INTERSECT_TRIANGLE: begin
                nextState = (hitTriangle == 1'b1 ? HIT_TRIANGLE : MISS);
            end
            HIT_TRIANGLE: begin
                nextState = POP_STACK;
            end
            MISS: begin
                nextState = POP_STACK;
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

    // Memory storage updates
    always @ (posedge clk) begin
        if (reset) begin
            bvhBuffer <= '0;
            triIndexBuffer <= '0;
            triBuffer <= '0;
        end else begin
            case (state)
                RECV_RAY_RSP: begin
                   rayBuffer <= lsu_mem_if.rsp_data.data;
                end
                RECV_BVH_NODE_RSP: begin
                    bvhBuffer <= lsu_mem_if.rsp_data.data;
                end
                RECV_TRI_INDEX_RSP: begin
                    triIndexBuffer <= lsu_mem_if.rsp_data.data;
                end
                RECV_TRI_NODE_L_RSP: begin
                    triBuffer[((32<<3)-1):0] <= lsu_mem_if.rsp_data.data;
                end
                RECV_TRI_NODE_H_RSP: begin
                    triBuffer[TRI_NODE_BITS-1:((32<<3)-1)] <= lsu_mem_if.rsp_data.data;
                end
                default: begin
                    
                end
            endcase
        end
    end

    // LSU interface logic
    always @ (*) begin
        case (state)
            SEND_RAY_REQ: begin
                // Request data valid
                lsu_mem_if.req_valid = 1'b1;
                lsu_mem_if.rsp_ready = 1'b0;
                // Request data interface values
                lsu_mem_if.req_data.rw = 1'b0;
                lsu_mem_if.req_data.mask = 1'b1;
                lsu_mem_if.req_data.byteen = 1'b1;
                lsu_mem_if.req_data.addr = ti_execute_if.rs1_data[0];
                lsu_mem_if.req_data.atype = 2'b00;
                lsu_mem_if.req_data.tag = 2'b00;
            end
            WAIT_RAY_RSP: begin
                // Request data invalid, waiting for response
                lsu_mem_if.req_valid = 1'b0;
                // Request data interface values
                lsu_mem_if.req_data = '0;
            end
            RECV_RAY_RSP: begin
                // Acknowledge response
                lsu_mem_if.rsp_ready = 1'b1;
                // Request data interface values
                lsu_mem_if.req_data = '0;
            end
            SEND_BVH_NODE_REQ: begin
                // Request data valid
                lsu_mem_if.req_valid = 1'b1;
                lsu_mem_if.rsp_ready = 1'b0;
                // Request data interface values
                lsu_mem_if.req_data.rw = 1'b0;
                lsu_mem_if.req_data.mask = 1'b1;
                lsu_mem_if.req_data.byteen = 1'b1;
                lsu_mem_if.req_data.addr = bvhBaseAddr + nextBvhIndex << $CLOG2(BVH_NODE_BITS);
                lsu_mem_if.req_data.atype = 2'b00;
                lsu_mem_if.req_data.tag = 2'b00;
            end
            WAIT_BVH_NODE_RSP: begin
                // Request data invalid, waiting for response
                lsu_mem_if.req_valid = 1'b0;
                // Request data interface values
                lsu_mem_if.req_data = '0;
            end
            RECV_BVH_NODE_RSP: begin
                // Acknowledge response
                lsu_mem_if.rsp_ready = 1'b1;
                // Request data interface values
                lsu_mem_if.req_data = '0;
            end
            SEND_TRI_INDEX_REQ: begin
                // Request data valid
                lsu_mem_if.req_valid = 1'b1;
                lsu_mem_if.rsp_ready = 1'b0;
                 // Request data interface values
                lsu_mem_if.req_data.rw = 1'b0;
                lsu_mem_if.req_data.mask = 1'b1;
                lsu_mem_if.req_data.byteen = 1'b1;
                lsu_mem_if.req_data.addr = triIdxBaseAddr + (triIndexBuffer << 2);;
                lsu_mem_if.req_data.atype = 2'b00;
                lsu_mem_if.req_data.tag = 2'b00;
            end
            WAIT_TRI_INDEX_RSP: begin
                // Request data invalid, waiting for response
                lsu_mem_if.req_valid = 1'b0;
                // Request data interface values
                lsu_mem_if.req_data = '0;
            end
            RECV_TRI_INDEX_RSP: begin
                // Acknowledge response
                lsu_mem_if.rsp_ready = 1'b1;
                // Request data interface values
                lsu_mem_if.req_data = '0;
            end
            SEND_TRI_NODE_L_REQ: begin
                // Request data valid
                lsu_mem_if.req_valid = 1'b1;
                lsu_mem_if.rsp_ready = 1'b0;
                 // Request data interface values
                lsu_mem_if.req_data.rw = 1'b0;
                lsu_mem_if.req_data.mask = 1'b1;
                lsu_mem_if.req_data.byteen = 1'b1;
                lsu_mem_if.req_data.addr = triBaseAddr + (triBuffer << 3);
                lsu_mem_if.req_data.atype = 2'b00;
                lsu_mem_if.req_data.tag = 2'b00;
            end
            WAIT_TRI_NODE_L_RSP: begin
                // Request data invalid, waiting for response
                lsu_mem_if.req_valid = 1'b0;
                // Request data interface values
                lsu_mem_if.req_data = '0;
            end
            RECV_TRI_NODE_L_RSP: begin
                // Acknowledge response
                lsu_mem_if.rsp_ready = 1'b1;
                // Request data interface values
                lsu_mem_if.req_data = '0;
            end
            SEND_TRI_NODE_H_REQ: begin
                // Request data valid
                lsu_mem_if.req_valid = 1'b1;
                lsu_mem_if.rsp_ready = 1'b0;
                 // Request data interface values
                lsu_mem_if.req_data.rw = 1'b0;
                lsu_mem_if.req_data.mask = 1'b1;
                lsu_mem_if.req_data.byteen = 1'b1;
                lsu_mem_if.req_data.addr = triBaseAddr + (triBuffer << 3) + `DATA_SIZE;
                lsu_mem_if.req_data.atype = 2'b00;
                lsu_mem_if.req_data.tag = 2'b00;
            end
            WAIT_TRI_NODE_H_RSP: begin
                // Request data invalid, waiting for response
                lsu_mem_if.req_valid = 1'b0;
                // Request data interface values
                lsu_mem_if.req_data = '0;
            end
            RECV_TRI_NODE_H_RSP: begin
                // Acknowledge response
                lsu_mem_if.rsp_ready = 1'b1;
                // Request data interface values
                lsu_mem_if.req_data = '0;
            end
            default: begin
                // Request data valid
                lsu_mem_if.req_valid = 1'b0;
                lsu_mem_if.rsp_ready = 1'b0;
                // Request data interface values
                lsu_mem_if.req_data = '0;
            end
        endcase
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
    
    reg isTriangleIntersect;
    reg [3*32-1:0] aabbMin, aabbMax, triV0, triV1, triV2, rayOrigin, rayDir;
    wire intersects;


   

endmodule

///////////////////////////////////////////////////////////////////////////////
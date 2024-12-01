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

module VX_ti_stack import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter INSTANCE_IDX    = 0,
    parameter NUM_INSTANCES   = 1,
    // TODO: hardcoded stack size of 3
    // this can only handle a BVH of 1 billion nodes in the worst case
    // https://www.embree.org/papers/2019-HPG-ShortStack.pdf
    // changing our BVH algorithm can allow to use the short stack algorithm with only 5 entries for arbitrary BVH sizes
    parameter STACK_SIZE      = 30;
    parameter ENTRY_BITS      = 32;
) (

    // Clock
    input wire clk,
    input wire reset,
    input wire push,
    input wire pop,
    input wire [ENTRY_BITS-1:0] data_in,

    output wire [ENTRY_BITS-1:0] data_out,

    output wire empty,
    output wire full,

);
    reg [$CLOG2(STACK_SIZE)-1:0] sp;
    reg [ENTRY_BITS-1:0] stack [0:STACK_SIZE-1];
    assign empty = (sp == 0);
    assign full = (sp == STACK_SIZE);


endmodule

///////////////////////////////////////////////////////////////////////////////

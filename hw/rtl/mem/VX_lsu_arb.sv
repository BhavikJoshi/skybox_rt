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

`include "VX_define.vh"

`define IGNORE_WARNINGS_BEGIN
module VX_lsu_arb #(    
    parameter NUM_INPUTS     = 1, 
    parameter NUM_OUTPUTS    = 1,
    parameter DATA_SIZE      = 1,
    parameter MEM_ADDR_WIDTH = `MEM_ADDR_WIDTH,    
    parameter ADDR_WIDTH     = (MEM_ADDR_WIDTH-`CLOG2(DATA_SIZE)),
    parameter TAG_WIDTH      = 1,    
    parameter TAG_SEL_IDX    = 0,   
    parameter REQ_OUT_BUF    = 0,
    parameter RSP_OUT_BUF    = 0,
    parameter `STRING ARBITER = "R"
) (
    input wire              clk,
    input wire              reset,

    VX_lsu_bus_if.slave     bus_in_if [NUM_INPUTS],
    VX_lsu_bus_if.master    bus_out_if [NUM_OUTPUTS]
);       
    // TODO: currently copied from VX_mem_arb.sv, connected to lsu_bus_if and hollowed out. Need to fill with proper logic
endmodule
`define IGNORE_WARNINGS_END


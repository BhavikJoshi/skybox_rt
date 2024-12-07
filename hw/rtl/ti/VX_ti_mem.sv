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

module VX_ti_mem import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter INSTANCE_IDX    = 0,
    parameter NUM_INSTANCES   = 1,
) 
(    
        // Clock
        input wire clk,
        input wire reset,

        // Memory interface
        VX_mem_bus_if.master    cache_bus_if [RCACHE_NUM_REQS],

        // Outputs
        VX_ti_bus_if.master     ti_bus_if
);

endmodule

///////////////////////////////////////////////////////////////////////////////

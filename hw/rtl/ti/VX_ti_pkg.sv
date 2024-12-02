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

`ifndef VX_TI_PKG_VH
`define VX_TI_PKG_VH

`include "VX_ti_define.vh"

package VX_ti_pkg;

typedef struct packed {
    // logic [`TI_ADDR_BITS-1:0]   tbuf_addr;     // Tile buffer address
    // logic [`TI_TILE_BITS-1:0]   tile_count;    // Number of tiles in the tile buffer
    // logic [`TI_ADDR_BITS-1:0]   pbuf_addr;     // Primitive triangle data buffer start address
    // logic [`VX_TI_STRIDE_BITS-1:0] pbuf_stride; // Primitive data stride to fetch vertices
    // logic [`VX_TI_DIM_BITS-1:0] dst_xmin;      // Destination window xmin
    // logic [`VX_TI_DIM_BITS-1:0] dst_xmax;      // Destination window xmax
    // logic [`VX_TI_DIM_BITS-1:0] dst_ymin;      // Destination window ymin
    // logic [`VX_TI_DIM_BITS-1:0] dst_ymax;      // Destination window ymax
} ti_dcrs_t;

typedef struct packed {
    // logic [`VX_TI_DIM_BITS-2:0] pos_x;     // quad x position
    // logic [`VX_TI_DIM_BITS-2:0] pos_y;     // quad y position
    // logic [3:0]                     mask;      // quad mask
    // logic [2:0][3:0][31:0]          bcoords;   // barycentric coordinates
    // logic [`VX_TI_PID_BITS-1:0] pid;       // primitive index
} ti_stamp_t;

typedef struct packed {
    // capture all CSRs passed as arguments in instruction
    logic [31:0]           bvh; // pointer to bvh memory
    logic [31:0]           tri; // pointer to triangle array in memory
    logic [31:0]           triIdx; // pointer to triIdx array in memory
    // TODO: do we need to store output CSRs here as well? i.e. hit distance
} ti_csrs_t;

endpackage

`endif // VX_TI_PKG_VH

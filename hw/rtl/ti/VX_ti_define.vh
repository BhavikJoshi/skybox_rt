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

`ifndef VX_TI_DEFINE_VH
`define VX_TI_DEFINE_VH

`include "VX_define.vh"

`ifdef XLEN_64
`define TI_ADDR_BITS        32
`else
`define TI_ADDR_BITS        25
`endif

`define TRICOUNT_IDX_START  224
`define TRICOUNT_IDX_END    256

`define TI_FLOAT_BITS       32


`endif // VX_RASTER_DEFINE_VH
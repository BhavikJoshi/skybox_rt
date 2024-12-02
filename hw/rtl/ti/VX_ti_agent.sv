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

module VX_ti_agent #(
    parameter CORE_ID = 0,
    parameter NUM_LANES = 1
) (
    input wire clk,
    input wire reset,

    // Inputs
    VX_execute_if.slave    execute_if,
    VX_sfu_csr_if.slave    ti_csr_if,
    VX_ti_bus_if.slave ti_bus_if,

    // Outputs
    VX_commit_if.master    commit_if
);
    `UNUSED_PARAM (CORE_ID)
    localparam PID_BITS   = `CLOG2(`NUM_THREADS / NUM_LANES);
    localparam PID_WIDTH  = `UP(PID_BITS);

    wire ti_rsp_valid, ti_rsp_ready;

    // CSRs access

    wire csr_write_enable = ti_bus_if.req_valid && execute_if.valid && ti_rsp_ready;

    VX_ti_csr #(
        .CORE_ID   (CORE_ID),
        .NUM_LANES (NUM_LANES)
    ) ti_csr (
        .clk            (clk),
        .reset          (reset),
        // inputs
        .write_enable   (csr_write_enable),
        .write_uuid     (execute_if.data.uuid),
        .write_wid      (execute_if.data.wid),
        .write_tmask    (execute_if.data.tmask),
        .write_pid      (execute_if.data.pid),
        .write_data     (ti_bus_if.req_data.stamps),
        // outputs
        .ti_csr_if  (ti_csr_if)
    );

    // it is possible to have ready = f(valid) when using arbiters,
    // because of that we need to decouple execute_if and commit_if handshake with a pipe register

    assign execute_if.ready = ti_bus_if.req_valid && ti_rsp_ready;

    assign ti_bus_if.req_ready = execute_if.valid && ti_rsp_ready;

    assign ti_rsp_valid = execute_if.valid && ti_bus_if.req_valid;

    wire [NUM_LANES-1:0][31:0] response_data, commit_data;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign response_data[i] = {31'(ti_bus_if.req_data.stamps[i].pid), ~ti_bus_if.req_data.done};
    end

    VX_elastic_buffer #(
        .DATAW (`UUID_WIDTH + `NW_WIDTH + NUM_LANES + `PC_BITS + `NR_BITS + (NUM_LANES * 32) + PID_WIDTH + 1 + 1),
        .SIZE  (2)
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (ti_rsp_valid),
        .ready_in  (ti_rsp_ready),
        .data_in   ({execute_if.data.uuid, execute_if.data.wid, execute_if.data.tmask, execute_if.data.PC, execute_if.data.rd, response_data, execute_if.data.pid, execute_if.data.sop, execute_if.data.eop}),
        .data_out  ({commit_if.data.uuid, commit_if.data.wid, commit_if.data.tmask, commit_if.data.PC, commit_if.data.rd, commit_data, commit_if.data.pid, commit_if.data.sop, commit_if.data.eop}),
        .valid_out (commit_if.valid),
        .ready_out (commit_if.ready)
    );

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign commit_if.data.data[i] = `XLEN'(commit_data[i]);
    end

    assign commit_if.data.wb  = 1'b1;

// `ifdef DBG_TRACE_TI
//     always @(posedge clk) begin
//         if (execute_if.valid && execute_if.ready) begin
//             for (integer i = 0; i < NUM_LANES; ++i) begin
//                 `TRACE(1, ("%d: core%0d-ti-stamp[%0d]: valid=%b, wid=%0d, PC=0x%0h, done=%b, x=%0d, y=%0d, mask=%0d, pid=%0d, bcoords={{0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}} (#%0d)\n",
//                     $time, CORE_ID, i, execute_if.data.tmask, execute_if.data.wid, execute_if.data.PC,
//                     ti_bus_if.req_data.done,
//                     ti_bus_if.req_data.stamps[i].pos_x, ti_bus_if.req_data.stamps[i].pos_y, ti_bus_if.req_data.stamps[i].mask, ti_bus_if.req_data.stamps[i].pid,
//                     ti_bus_if.req_data.stamps[i].bcoords[0][0], ti_bus_if.req_data.stamps[i].bcoords[1][0], ti_bus_if.req_data.stamps[i].bcoords[2][0],
//                     ti_bus_if.req_data.stamps[i].bcoords[0][1], ti_bus_if.req_data.stamps[i].bcoords[1][1], ti_bus_if.req_data.stamps[i].bcoords[2][1],
//                     ti_bus_if.req_data.stamps[i].bcoords[0][2], ti_bus_if.req_data.stamps[i].bcoords[1][2], ti_bus_if.req_data.stamps[i].bcoords[2][2],
//                     ti_bus_if.req_data.stamps[i].bcoords[0][3], ti_bus_if.req_data.stamps[i].bcoords[1][3], ti_bus_if.req_data.stamps[i].bcoords[2][3], execute_if.data.uuid));
//             end
//         end
//     end
// `endif

endmodule

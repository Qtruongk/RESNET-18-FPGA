`timescale 1ns / 1ps

module pe_column #(
    parameter ROWS = 4
)(
    input  wire                   clk,
    input  wire                   rst,
    input  wire [(ROWS*8)-1:0]    a_in_bus,
    input  wire signed [7:0]      b_in_top,
    input  wire                   accumulate_done,
    input  wire                   shift_en,
    output wire signed [31:0]     psum_out_bottom
);

    // Mảng dây nối tầng (Cascade Wires) chạy dọc từ trên xuống dưới
    // Kích thước là ROWS + 1 để chứa cả biên trên (0) và biên dưới (ROWS)
    wire signed [7:0]  b_cascade    [0:ROWS];
    wire signed [31:0] psum_cascade [0:ROWS];

    // Xử lý điều kiện biên trên (Top Boundary) và biên dưới (Bottom Boundary)
    assign b_cascade[0]    = b_in_top;
    assign psum_cascade[0] = 32'sd0;              // Ghim psum_in của PE trên cùng về 0
    assign psum_out_bottom = psum_cascade[ROWS];  // Ngõ ra ở đáy cột

    // Khởi tạo tự động ROWS khối PE nối tiếp nhau theo chiều dọc
    genvar i;
    generate
        for (i = 0; i < ROWS; i = i + 1) begin : pe_col_gen
            pe pe_inst (
                .clk             (clk),
                .rst             (rst),
                .a_in            (a_in_bus[(i+1)*8-1 : i*8]),
                .b_in            (b_cascade[i]),
                .psum_in         (psum_cascade[i]),
                .accumulate_done (accumulate_done),
                .shift_en        (shift_en),
                .a_out           (),                  // Không sử dụng ở mô hình 1D
                .b_out           (b_cascade[i+1]),
                .psum_out        (psum_cascade[i+1])
            );
        end
    endgenerate

endmodule

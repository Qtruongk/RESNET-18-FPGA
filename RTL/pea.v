`timescale 1ns / 1ps

module pea #(
    parameter ROWS = 4,
    parameter COLS = 4
)(
    input  wire                   clk,
    input  wire                   rst,
    input  wire [(ROWS*8)-1:0]    a_in_bus,      // Dữ liệu A vào Biên Trái
    input  wire [(COLS*8)-1:0]    b_in_bus,      // Dữ liệu B vào Biên Trên
    input  wire                   accumulate_done,
    input  wire                   shift_en,
    output wire [(COLS*32)-1:0]   psum_out_bus   // Bus kết quả ở Đáy mảng
);

    // Mảng dây nối nội bộ 2D (Cascade Wires)
    wire signed [7:0]  a_casc    [0:ROWS-1][0:COLS];
    wire signed [7:0]  b_casc    [0:ROWS][0:COLS-1];
    wire signed [31:0] psum_casc [0:ROWS][0:COLS-1];

    genvar i, j;

    // 1. Xử lý điều kiện biên (Boundary Conditions)
    generate
        // Biên Trái: Bóc tách bus A cấp vào cột 0
        for (i = 0; i < ROWS; i = i + 1) begin : left_bound
            assign a_casc[i][0] = a_in_bus[(i+1)*8-1 : i*8];
        end

        // Biên Trên: Bóc tách bus B cấp vào hàng 0 và ghim psum_in = 0
        for (j = 0; j < COLS; j = j + 1) begin : top_bound
            assign b_casc[0][j]    = b_in_bus[(j+1)*8-1 : j*8];
            assign psum_casc[0][j] = 32'sd0;
        end

        // Biên Dưới: Gom các ngõ ra psum ở đáy mảng thành 1 bus duy nhất
        for (j = 0; j < COLS; j = j + 1) begin : bottom_bound
            assign psum_out_bus[(j+1)*32-1 : j*32] = psum_casc[ROWS][j];
        end
    endgenerate

    // 2. Khởi tạo lưới 2D (ROWS x COLS) khối PE
    generate
        for (i = 0; i < ROWS; i = i + 1) begin : row_gen
            for (j = 0; j < COLS; j = j + 1) begin : col_gen
                pe pe_inst (
                    .clk             (clk),
                    .rst             (rst),
                    .a_in            (a_casc[i][j]),
                    .a_out           (a_casc[i][j+1]),
                    .b_in            (b_casc[i][j]),
                    .b_out           (b_casc[i+1][j]),
                    .psum_in         (psum_casc[i][j]),
                    .psum_out        (psum_casc[i+1][j]),
                    .accumulate_done (accumulate_done),
                    .shift_en        (shift_en)
                );
            end
        end
    endgenerate

endmodule

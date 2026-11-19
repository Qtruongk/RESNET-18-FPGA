`timescale 1ns / 1ps

module skew_buffer #(
    parameter ROWS = 4
)(
    input  wire                  clk,
    input  wire                  rst,
    input  wire [(ROWS*8)-1:0]   a_flat_in,
    output wire [(ROWS*8)-1:0]   a_skewed_out
);

    genvar i;
    generate
        for (i = 0; i < ROWS; i = i + 1) begin : row_skew
            if (i == 0) begin : no_delay
                // Hàng 0: Đi thẳng, độ trễ = 0 chu kỳ
                assign a_skewed_out[7:0] = a_flat_in[7:0];
            end else begin : shift_delay
                // Hàng i: Tạo thanh ghi dịch gồm đúng i tầng Flip-Flop
                reg [7:0] delay_line [0:i-1];
                integer k;

                always @(posedge clk) begin
                    if (rst) begin
                        for (k = 0; k < i; k = k + 1) begin
                            delay_line[k] <= 8'd0;
                        end
                    end else begin
                        // Tầng đầu tiên nhận dữ liệu từ ngõ vào phẳng
                        delay_line[0] <= a_flat_in[(i+1)*8-1 : i*8];
                        // Các tầng sau dịch nối tiếp nhau
                        for (k = 1; k < i; k = k + 1) begin
                            delay_line[k] <= delay_line[k-1];
                        end
                    end
                end

                // Lấy ngõ ra từ tầng Flip-Flop cuối cùng của hàng i
                assign a_skewed_out[(i+1)*8-1 : i*8] = delay_line[i-1];
            end
        end
    endgenerate

endmodule

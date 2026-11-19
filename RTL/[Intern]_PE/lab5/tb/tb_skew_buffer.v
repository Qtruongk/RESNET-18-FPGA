`timescale 1ns / 1ps

module tb_skew_buffer;

    parameter ROWS = 4;

    reg                  clk;
    reg                  rst;
    reg  [(ROWS*8)-1:0]  a_flat_in;
    wire [(ROWS*8)-1:0]  a_skewed_out;

    // Tách bus ngõ ra thành 4 hàng để dễ quan sát sóng bậc thang
    wire signed [7:0] row0_out = a_skewed_out[7:0];
    wire signed [7:0] row1_out = a_skewed_out[15:8];
    wire signed [7:0] row2_out = a_skewed_out[23:16];
    wire signed [7:0] row3_out = a_skewed_out[31:24];

    skew_buffer #(.ROWS(ROWS)) dut (
        .clk          (clk),
        .rst          (rst),
        .a_flat_in    (a_flat_in),
        .a_skewed_out (a_skewed_out)
    );

    // Xung clock 100MHz (chu kỳ 10ns)
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        rst       = 1;
        a_flat_in = 0;

        // Giữ reset 1 chu kỳ
        @(posedge clk);
        rst <= 0;

        // Nhịp 1: Bơm cột dữ liệu phẳng 1: {Row3, Row2, Row1, Row0} = {4, 3, 2, 1}
        a_flat_in <= {8'sd4, 8'sd3, 8'sd2, 8'sd1};

        // Nhịp 2: Bơm cột dữ liệu phẳng 2: {40, 30, 20, 10}
        @(posedge clk);
        a_flat_in <= {8'sd40, 8'sd30, 8'sd20, 8'sd10};

        // Nhịp 3 trở đi: Ngừng bơm, chờ dữ liệu trôi hết qua các tầng trễ
        @(posedge clk);
        a_flat_in <= 0;

        repeat (5) @(posedge clk);
        $finish;
    end

    // In bảng log để thấy rõ đường chéo bậc thang
    initial begin
        $display("---------------------------------------------------------");
        $display(" Time | row0 (0 clk) | row1 (1 clk) | row2 (2 clk) | row3 (3 clk)");
        $display("---------------------------------------------------------");
        forever @(posedge clk) begin
            #1;
            $display("%5t | %12d | %12d | %12d | %12d",
                     $time, row0_out, row1_out, row2_out, row3_out);
        end
    end

endmodule

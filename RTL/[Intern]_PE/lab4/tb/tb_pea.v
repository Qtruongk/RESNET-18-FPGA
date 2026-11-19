`timescale 1ns / 1ps

module tb_pea;

    parameter ROWS = 2;
    parameter COLS = 2;

    reg                   clk;
    reg                   rst;
    reg  [(ROWS*8)-1:0]   a_in_bus;
    reg  [(COLS*8)-1:0]   b_in_bus;
    reg                   accumulate_done;
    reg                   shift_en;
    wire [(COLS*32)-1:0]  psum_out_bus;

    // Tách bus ngõ ra thành 2 cột riêng biệt để dễ đọc trên Waveform/Console
    wire signed [31:0] col0_out = psum_out_bus[31:0];
    wire signed [31:0] col1_out = psum_out_bus[63:32];

    pea #(
        .ROWS(ROWS),
        .COLS(COLS)
    ) dut (
        .clk             (clk),
        .rst             (rst),
        .a_in_bus        (a_in_bus),
        .b_in_bus        (b_in_bus),
        .accumulate_done (accumulate_done),
        .shift_en        (shift_en),
        .psum_out_bus    (psum_out_bus)
    );

    // Xung clock 100MHz (chu kỳ 10ns)
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        // Nhịp 0: Reset
        rst             = 1;
        a_in_bus        = 0;
        b_in_bus        = 0;
        accumulate_done = 0;
        shift_en        = 0;

        @(negedge clk);
        rst = 0;

        // Cấp dữ liệu dạng bậc thang (Skewed): {Row1, Row0} và {Col1, Col0}
        // Nhịp 1: A00=1, B00=5 (Hàng 1 và Cột 1 trễ 1 nhịp nên cấp 0)
        @(negedge clk);
        a_in_bus = {8'sd0, 8'sd1};
        b_in_bus = {8'sd0, 8'sd5};

        // Nhịp 2: A01=2, A10=3 | B10=7, B01=6
        @(negedge clk);
        a_in_bus = {8'sd3, 8'sd2};
        b_in_bus = {8'sd6, 8'sd7};

        // Nhịp 3: A11=4 | B11=8 (Hàng 0 và Cột 0 đã xong nên cấp 0)
        @(negedge clk);
        a_in_bus = {8'sd4, 8'sd0};
        b_in_bus = {8'sd8, 8'sd0};

        // Nhịp 4: Ngừng cấp dữ liệu, chờ PE(1,1) hoàn tất tầng nhân (Stage 1)
        @(negedge clk);
        a_in_bus = {8'sd0, 8'sd0};
        b_in_bus = {8'sd0, 8'sd0};

        // Nhịp 5: Bật accumulate_done chốt kết quả toàn mảng
        // Hàng dưới cùng xuất ngay ra đáy: col0_out = 43, col1_out = 50
        @(negedge clk);
        accumulate_done = 1;

        // Nhịp 6: Tắt done, bật shift_en để dịch hàng trên xuống đáy
        // Đáy mảng xuất tiếp: col0_out = 19, col1_out = 22
        @(negedge clk);
        accumulate_done = 0;
        shift_en        = 1;

        // Nhịp 7: Dịch thêm 1 nhịp để thấy số 0 ở đỉnh đẩy xuống
        @(negedge clk);
        shift_en = 0;

        #20;
        $finish;
    end

    // In log kiểm tra kết quả
    initial begin
        $display("---------------------------------------------------------");
        $display(" Time | done | shift | col0_out (C_i,0) | col1_out (C_i,1)");
        $display("---------------------------------------------------------");
        forever @(posedge clk) begin
            #1;
            $display("%5t | %4b | %5b | %16d | %16d",
                     $time, accumulate_done, shift_en, col0_out, col1_out);
        end
    end

endmodule

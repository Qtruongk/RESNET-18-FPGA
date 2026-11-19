`timescale 1ns / 1ps

module tb_column;

    parameter ROWS = 4;

    reg                   clk;
    reg                   rst;
    reg  [(ROWS*8)-1:0]   a_in_bus;
    reg  signed [7:0]     b_in_top;
    reg                   accumulate_done;
    reg                   shift_en;
    wire signed [31:0]    psum_out_bottom;

    pe_column #(.ROWS(ROWS)) dut (
        .clk             (clk),
        .rst             (rst),
        .a_in_bus        (a_in_bus),
        .b_in_top        (b_in_top),
        .accumulate_done (accumulate_done),
        .shift_en        (shift_en),
        .psum_out_bottom (psum_out_bottom)
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
        b_in_top        = 0;
        accumulate_done = 0;
        shift_en        = 0;

        @(negedge clk);
        rst = 0;

        // Cấp cố định A cho 4 hàng: PE0=1, PE1=2, PE2=3, PE3=4
        // Thứ tự ghép bus từ MSB xuống LSB: {A3, A2, A1, A0}
        a_in_bus = {8'sd4, 8'sd3, 8'sd2, 8'sd1};

        // Nhịp 1: Thả trọng số B = 10 vào đỉnh cột (PE0)
        @(negedge clk);
        b_in_top = 8'sd10;

        // Nhịp 2, 3, 4: Đưa B về 0, chờ số 10 rơi dần qua PE1 -> PE2 -> PE3
        @(negedge clk); b_in_top = 8'sd0;
        @(negedge clk);
        @(negedge clk);

        // Nhịp 5: Số 10 vừa nhân xong ở PE3 (tầng 1). Bật accumulate_done chốt kết quả cả 4 PE
        // Lúc này: PE0=10, PE1=20, PE2=30, PE3=40 (đáy cột xuất ngay 40)
        @(negedge clk);
        accumulate_done = 1;

        // Nhịp 6, 7, 8, 9: Tắt done, bật shift_en để xả lần lượt 30 -> 20 -> 10 -> 0 ra đáy cột
        @(negedge clk);
        accumulate_done = 0;
        shift_en        = 1;

        repeat (4) @(negedge clk);
        shift_en = 0;

        #20;
        $finish;
    end

    // In log theo từng chu kỳ để đối chiếu kết quả
    initial begin
        $display("-------------------------------------------------------------------------");
        $display(" Time | b_top | b_PE1 | b_PE2 | b_PE3 | done | shift | psum_out_bottom");
        $display("-------------------------------------------------------------------------");
        forever @(posedge clk) begin
            #1;
            $display("%5t | %5d | %5d | %5d | %5d | %4b | %5b | %15d",
                     $time,
                     dut.b_cascade[0],
                     dut.b_cascade[1],
                     dut.b_cascade[2],
                     dut.b_cascade[3],
                     accumulate_done,
                     shift_en,
                     psum_out_bottom);
        end
    end

endmodule

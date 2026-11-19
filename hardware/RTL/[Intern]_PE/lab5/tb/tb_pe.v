`timescale 1ns / 1ps

module tb_pe;

    // Khai báo tín hiệu đầu vào (reg)
    reg               clk;
    reg               rst;
    reg signed [7:0]  a_in;
    reg signed [7:0]  b_in;
    reg signed [31:0] psum_in;
    reg               accumulate_done;
    reg               shift_en;

    // Khai báo tín hiệu đầu ra (wire)
    wire signed [7:0]  a_out;
    wire signed [7:0]  b_out;
    wire signed [31:0] psum_out;

    // Khởi tạo khối DUT (Device Under Test)
    pe dut (
        .clk(clk),
        .rst(rst),
        .a_in(a_in),
        .b_in(b_in),
        .psum_in(psum_in),
        .accumulate_done(accumulate_done),
        .shift_en(shift_en),
        .a_out(a_out),
        .b_out(b_out),
        .psum_out(psum_out)
    );

    // 1. Tạo xung Clock 100MHz (Chu kỳ = 10ns)
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // posedge tại 5, 15, 25... negedge tại 10, 20, 30...
    end

    // 2. Kịch bản mô phỏng (Test Stimulus)
    // Lưu ý: Cấp tín hiệu vào sườn XUỐNG (negedge) để mạch lấy mẫu an toàn ở sườn LÊN (posedge)
    initial begin
        // --- CHU KỲ 0: Trạng thái ban đầu ---
        rst = 1;
        a_in = 0; b_in = 0; psum_in = 0;
        accumulate_done = 0; shift_en = 0;
        
        @(negedge clk); 
        rst = 0; // Tắt reset

        // --- TILE 1 (K=3): Mục tiêu tính ra 14 ---
        // Chu kỳ 1: Nạp Cặp 1 (A=1, B=1)
        @(negedge clk);
        a_in = 8'sd1; b_in = 8'sd1; accumulate_done = 0;

        // Chu kỳ 2: Nạp Cặp 2 (A=2, B=2)
        @(negedge clk);
        a_in = 8'sd2; b_in = 8'sd2; accumulate_done = 0;

        // Chu kỳ 3: Nạp Cặp 3 - Cuối Tile 1 (A=3, B=3)
        @(negedge clk);
        a_in = 8'sd3; b_in = 8'sd3; accumulate_done = 0;

        // --- TILE 2 (K=3): Mục tiêu tính ra 15 ---
        // Chu kỳ 4: Nạp Cặp 1 của Tile 2 (A=4, B=1)
        // QUAN TRỌNG: Bật cờ done để chốt kết quả Tile 1 ra Shadow Reg
        @(negedge clk);
        a_in = 8'sd4; b_in = 8'sd1; 
        accumulate_done = 1; // <--- KÍCH HOẠT CHỐT 1 NHỊP

        // Chu kỳ 5: Nạp Cặp 2 của Tile 2 (A=5, B=1)
        // Tắt cờ done. Bật thử shift_en để dịch psum_out đi, nhường chỗ cho dữ liệu từ trên dội xuống (psum_in)
        @(negedge clk);
        a_in = 8'sd5; b_in = 8'sd1; 
        accumulate_done = 0; // Tắt cờ chốt
        shift_en = 1;        // Mở MUX dịch dọc
        psum_in  = 32'sd999; // Giả sử có số 999 từ PE hàng trên chạy xuống

        // Chu kỳ 6: Nạp Cặp 3 - Cuối Tile 2 (A=6, B=1)
        @(negedge clk);
        a_in = 8'sd6; b_in = 8'sd1; 
        shift_en = 0;        // Tắt dịch dọc

        // --- TILE 3: Khởi động Tile mới để chốt sổ Tile 2 ---
        // Chu kỳ 7
        @(negedge clk);
        a_in = 8'sd0; b_in = 8'sd0;
        accumulate_done = 1; // <--- KÍCH HOẠT CHỐT KẾT QUẢ TILE 2

        // Chu kỳ 8: Xả nốt hệ thống
        @(negedge clk);
        accumulate_done = 0;

        #30;
        $finish; // Kết thúc mô phỏng
    end

    // 3. Monitor: In kết quả ra Console (Clock-by-Clock Trace)
    // Dùng đường dẫn nội bộ (dut.mul_reg, dut.acc_reg) để soi bên trong khối PE
    initial begin
        $display("-----------------------------------------------------------------------------------------");
        $display(" Time | a_in | b_in | done | mul_reg (T1) | acc_reg (T2) | psum_out (Shadow) | shift_en");
        $display("-----------------------------------------------------------------------------------------");
        // Theo dõi sự thay đổi ở cuối mỗi chu kỳ clock (ngay sau khi posedge thực thi xong)
        forever @(posedge clk) begin
            #1; // Đợi 1ns để tín hiệu ổn định sau sườn lên
            $display("%5t | %4d | %4d | %4b | %12d | %12d | %17d | %8b", 
                     $time, a_in, b_in, accumulate_done, dut.mul_reg, dut.acc_reg, psum_out, shift_en);
        end
    end

endmodule
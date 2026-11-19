`timescale 1ns/1ps
module tb_mac;
	reg               clk;
    reg               rst;
    reg signed [7:0]  a_in;
    reg signed [7:0]  b_in;
    wire signed [31:0] acc_out;
	
	mac_2stage dut (
        .clk(clk),
        .rst(rst),
        .a_in(a_in),
        .b_in(b_in),
        .acc_out(acc_out)
    );
	
	initial begin
        clk = 0;
        forever #5 clk = ~clk; // chu kỳ: 10ns
    end
	
	initial begin
        // Trạng thái ban đầu
        rst = 0;
        a_in = 8'sd0;
        b_in = 8'sd0;
        
        #15; // Giữ reset trong 1.5 chu kỳ
        rst = 0;
		// Mong đợi: mất 2 chu kỳ để mà ra kết quả
		@(posedge clk);
		rst = 1;
        a_in = 8'sd2; b_in = 8'sd3;
		#30;
		$finish;
	end
endmodule
module  mac_2stage (
    input wire                   clk,
    input wire                   rst,
    input wire signed [7:0]      a_in,
    input wire signed [7:0]      b_in,
    output reg signed [31:0]    acc_out
);
    reg signed [15:0] mul_reg;
	always@(posedge clk or negedge rst) begin
		if(!rst) begin
			acc_out <= 32'd0;
			mul_reg <= 16'd0;
		end else begin
			// stage 1:
			mul_reg <= a_in * b_in;
			// stage 2:
			acc_out <= acc_out + mul_reg;
		end
	end
endmodule
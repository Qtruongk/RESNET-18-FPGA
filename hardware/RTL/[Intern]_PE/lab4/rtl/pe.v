`timescale 1ns / 1ps

module pe (
    input  wire               clk,
    input  wire               rst,
    
    // Data & Weight Inputs
    input  wire signed [7:0]  a_in,
    input  wire signed [7:0]  b_in,
    
    // Systolic Shift / Drain Inputs
    input  wire signed [31:0] psum_in,
    input  wire               accumulate_done,
    input  wire               shift_en,
    
    // Forwarding Outputs (Registers)
    output reg  signed [7:0]  a_out,
    output reg  signed [7:0]  b_out,
    output reg  signed [31:0] psum_out
);

    reg signed [15:0] mul_reg;
    reg signed [31:0] acc_reg;

    always @(posedge clk or negedge rst) begin
        if(!rst) begin
            a_out       <= 8'd0;
            b_out       <= 8'd0;
            psum_out    <= 32'd0;
            mul_reg     <= 16'd0;
            acc_reg     <= 32'd0;
        end else begin
            a_out 		<= a_in;
            b_out 		<= b_in;
			// stage 1
			mul_reg     <= a_in * b_in;
			// stage 2 
			if(accumulate_done) begin
				psum_out <= acc_reg + mul_reg;
				acc_reg	 <= 32'd0;
			end else begin
				acc_reg <= acc_reg + mul_reg;
				
				if (shift_en) begin
					psum_out <= psum_in;
				end
			end
        end
    end

endmodule
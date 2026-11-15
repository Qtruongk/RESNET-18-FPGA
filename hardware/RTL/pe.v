/*
INT8 × INT8 → INT32, OS Dataflow

Một Processing Element (PE) chuẩn theo luồng dữ liệu Output Stationary (OS) 
trong kiến trúc Systolic Array sẽ giữ nguyên tổng riêng phần (Partial Sum - psum_out)
tại thanh ghi cục bộ để tích lũy qua mỗi chu kỳ xung nhịp, đồng thời truyền tiếp dữ liệu
Activation (a_in) sang phải và Weight (b_in) xuống dưới thông qua các thanh ghi trễ 1 
chu kỳ.Để mạch hoạt động chính xác với số nguyên có dấu (INT8 × INT8 → INT32) 
và truyền đồng bộ tín hiệu valid_out, thiết kế chuẩn cần bổ sung từ khóa signed cho các đường dữ liệu 
và thêm cổng điều khiển valid_in.
*/

module pe #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
) (
    input  wire                         clk,
    input  wire                         rst,        // Synchronous active-high reset
    input  wire                         acc_clear,  // Clear/Init accumulator for new tile
    input  wire                         valid_in,   // Enable MAC and forward valid

    // Activation: from left PE to right PE
    input  wire signed [DATA_WIDTH-1:0] a_in,
    output reg  signed [DATA_WIDTH-1:0] a_out,

    // Weight: from upper PE to lower PE
    input  wire signed [DATA_WIDTH-1:0] b_in,
    output reg  signed [DATA_WIDTH-1:0] b_out,

    // Valid forwarding
    output reg                          valid_out,

    // Local output stationary accumulator
    output reg  signed [ACC_WIDTH-1:0]  psum_out
);

    // Tích số INT8 x INT8 tạo ra kết quả 16-bit có dấu
    wire signed [2*DATA_WIDTH-1:0] product;
    assign product = a_in * b_in;

    always @(posedge clk) begin
        if (rst) begin
            a_out     <= {DATA_WIDTH{1'b0}};
            b_out     <= {DATA_WIDTH{1'b0}};
            valid_out <= 1'b0;
            psum_out  <= {ACC_WIDTH{1'b0}};
        end else begin
            a_out     <= a_in;
            b_out     <= b_in;
            valid_out <= valid_in;

            if (acc_clear) begin
                psum_out <= valid_in ? {{(ACC_WIDTH - 2*DATA_WIDTH){product[2*DATA_WIDTH-1]}}, product} 
                                     : {ACC_WIDTH{1'b0}};
            end else if (valid_in) begin
                psum_out <= psum_out + product;
            end
        end
    end

endmodule
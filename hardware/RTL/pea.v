/*
 * 2D Systolic Array - Output Stationary (OS) Dataflow
 * Kích thước mặc định: 4x4 (Có thể thay đổi qua parameter ROWS, COLS)
 */

module pe_array #(
    parameter ROWS       = 4,
    parameter COLS       = 4,
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
) (
    input  wire                                 clk,
    input  wire                                 rst,

    // Tín hiệu điều khiển cho từng hàng/cột tương ứng với sóng dữ liệu
    input  wire [ROWS-1:0]                      acc_clear_in,
    input  wire [ROWS-1:0]                      valid_in,

    // Dữ liệu Activation đi vào từ cạnh TRÁI (ROWS phần tử, mỗi phần tử 8-bit)
    input  wire [ROWS*DATA_WIDTH-1:0]           a_in_flat,

    // Dữ liệu Weight đi vào từ cạnh TRÊN (COLS phần tử, mỗi phần tử 8-bit)
    input  wire [COLS*DATA_WIDTH-1:0]           b_in_flat,

    // Cờ báo hiệu dữ liệu hợp lệ đi ra ở cạnh PHẢI của mỗi hàng
    output wire [ROWS-1:0]                      valid_out,

    // Toàn bộ kết quả tích lũy của mảng (ROWS x COLS phần tử 32-bit)
    output wire [ROWS*COLS*ACC_WIDTH-1:0]       psum_out_flat
);

    // =========================================================================
    // 1. Lưới dây nối nội bộ (Internal Interconnect Mesh)
    // =========================================================================
    // Kích thước (COLS+1) và (ROWS+1) giúp nối biên đầu vào và đầu ra đồng nhất
    wire signed [DATA_WIDTH-1:0] a_wire     [0:ROWS-1][0:COLS];
    wire signed [DATA_WIDTH-1:0] b_wire     [0:ROWS][0:COLS-1];
    wire                         valid_wire [0:ROWS-1][0:COLS];
    wire                         clear_wire [0:ROWS-1][0:COLS];
    wire signed [ACC_WIDTH-1:0]  psum_wire  [0:ROWS-1][0:COLS-1];

    genvar r, c;

    // =========================================================================
    // 2. Giải nén (Unpack) dữ liệu đầu vào vào biên Trái và biên Trên của lưới
    // =========================================================================
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : gen_row_inputs
            assign a_wire[r][0]     = a_in_flat[r*DATA_WIDTH +: DATA_WIDTH];
            assign valid_wire[r][0] = valid_in[r];
            assign clear_wire[r][0] = acc_clear_in[r];
            assign valid_out[r]     = valid_wire[r][COLS]; // Lấy valid ở cột cuối
        end

        for (c = 0; c < COLS; c = c + 1) begin : gen_col_inputs
            assign b_wire[0][c]     = b_in_flat[c*DATA_WIDTH +: DATA_WIDTH];
        end
    endgenerate

    // =========================================================================
    // 3. Khởi tạo mảng 2D các PE (2D Mesh Instantiation)
    // =========================================================================
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : gen_rows
            for (c = 0; c < COLS; c = c + 1) begin : gen_cols

                // Lan truyền tín hiệu acc_clear sang cột kế tiếp (trễ 1 nhịp giống valid)
                reg clear_reg;
                always @(posedge clk) begin
                    if (rst) clear_reg <= 1'b0;
                    else     clear_reg <= clear_wire[r][c];
                end
                assign clear_wire[r][c+1] = clear_reg;

                // Khởi tạo PE tại tọa độ (r, c)
                pe #(
                    .DATA_WIDTH (DATA_WIDTH),
                    .ACC_WIDTH  (ACC_WIDTH)
                ) u_pe (
                    .clk       (clk),
                    .rst       (rst),
                    .acc_clear (clear_wire[r][c]),
                    .valid_in  (valid_wire[r][c]),

                    // Ngang: Trái -> Phải
                    .a_in      (a_wire[r][c]),
                    .a_out     (a_wire[r][c+1]),

                    // Dọc: Trên -> Dưới
                    .b_in      (b_wire[r][c]),
                    .b_out     (b_wire[r+1][c]),

                    // Truyền tiếp cờ valid sang phải
                    .valid_out (valid_wire[r][c+1]),

                    // Kết quả tích lũy tại chỗ
                    .psum_out  (psum_wire[r][c])
                );

                // Đóng gói (Pack) kết quả 2D vào vector 1D đầu ra theo thứ tự Row-Major
                assign psum_out_flat[(r*COLS + c)*ACC_WIDTH +: ACC_WIDTH] = psum_wire[r][c];

            end
        end
    endgenerate

endmodule
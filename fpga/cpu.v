module cpu_module (
    input wire clk,
    input wire reset,
    output reg [31:0] data_out,
    output reg [1:0] error_flag  // [0] = counter overflow, [1] = reserved for future use
);

// Error flag encoding
localparam ERR_NONE     = 2'b00;
localparam ERR_OVERFLOW = 2'b01;  // bit 0: counter wrapped to max value

// Internal signals
reg [31:0] internal_data;
reg [31:0] counter;

always @(posedge clk or posedge reset) begin
    if (reset) begin
        data_out <= 32'b0;
        internal_data <= 32'b0;
        counter <= 32'b0;
        error_flag <= ERR_NONE;
    end else begin
        // Data output logic
        counter <= counter + 1;
        if (counter == 32'hFFFFFFFF) begin
            error_flag <= ERR_OVERFLOW; // Set overflow error flag when counter reaches max
        end else begin
            internal_data <= counter;
            data_out <= internal_data;
            error_flag <= ERR_NONE;
        end
    end
end

endmodule
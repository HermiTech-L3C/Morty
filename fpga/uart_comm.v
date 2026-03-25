module uart_comm (
    input wire clk,
    input wire reset,
    input wire [31:0] data_in,
    output reg [31:0] data_out,
    input wire uart_rx,
    output reg uart_tx,
    output reg [1:0] error_flags
);

// Internal signals
reg [7:0] rx_buffer;
reg [7:0] tx_buffer;
reg [3:0] bit_counter_rx;
reg [3:0] bit_counter_tx;
reg [1:0] state_rx;
reg [1:0] state_tx;
reg rx_ready;
reg tx_ready;

// Baud rate generator parameters
parameter integer BAUD_RATE_DIV = 10416; // 100 MHz / 9600 baud ≈ 10416 cycles per bit

// baud_rate_counter must be wide enough to hold BAUD_RATE_DIV (10416 requires 14 bits)
reg [13:0] baud_rate_counter;

always @(posedge clk or posedge reset) begin
    if (reset) begin
        rx_buffer <= 8'b0;
        tx_buffer <= 8'b0;
        bit_counter_rx <= 4'b0;
        bit_counter_tx <= 4'b0;
        baud_rate_counter <= 14'b0;
        state_rx <= 2'b0;
        state_tx <= 2'b0;
        rx_ready <= 1'b0;
        tx_ready <= 1'b0;
        error_flags <= 2'b0;
        uart_tx <= 1'b1;
    end else begin
        // Baud rate counter
        baud_rate_counter <= baud_rate_counter + 1;

        // UART receive logic
        if (baud_rate_counter == BAUD_RATE_DIV) begin
            baud_rate_counter <= 14'b0;
            case (state_rx)
                2'b00: begin // Start bit detection
                    if (!uart_rx) begin
                        state_rx <= 2'b01;
                    end
                end
                2'b01: begin // Receiving data bits
                    rx_buffer[bit_counter_rx] <= uart_rx;
                    bit_counter_rx <= bit_counter_rx + 1;
                    if (bit_counter_rx == 4'b1000) begin
                        state_rx <= 2'b10;
                        bit_counter_rx <= 4'b0;
                    end
                end
                2'b10: begin // Stop bit detection
                    if (uart_rx) begin
                        data_out <= {24'b0, rx_buffer}; // Assuming 8-bit data
                        rx_ready <= 1'b1;
                        state_rx <= 2'b00;
                    end else begin
                        error_flags[0] <= 1'b1; // Set error flag if stop bit is incorrect
                        state_rx <= 2'b00;
                    end
                end
            endcase
        end

        // UART transmit logic
        if (tx_ready) begin
            if (baud_rate_counter == BAUD_RATE_DIV) begin
                baud_rate_counter <= 14'b0;
                case (state_tx)
                    2'b00: begin // Start bit
                        uart_tx <= 1'b0;
                        state_tx <= 2'b01;
                    end
                    2'b01: begin // Transmitting data bits
                        uart_tx <= tx_buffer[bit_counter_tx];
                        bit_counter_tx <= bit_counter_tx + 1;
                        if (bit_counter_tx == 4'b1000) begin
                            state_tx <= 2'b10;
                            bit_counter_tx <= 4'b0;
                        end
                    end
                    2'b10: begin // Stop bit
                        uart_tx <= 1'b1;
                        tx_ready <= 1'b0;
                        state_tx <= 2'b00;
                    end
                endcase
            end
        end else if (!tx_ready && !rx_ready) begin
            tx_buffer <= data_in[7:0]; // Assuming 8-bit data
            tx_ready <= 1'b1;
        end
    end
end

endmodule
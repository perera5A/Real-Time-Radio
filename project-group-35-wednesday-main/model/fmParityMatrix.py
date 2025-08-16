import numpy as np
import math

parityMatrix = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1,    1, 1, 0, 1, 1, 0, 0, 1, 1, 1],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1,    1, 1, 1, 0, 1, 1, 0, 0, 1, 1],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0,    0, 0, 1, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,    1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,    0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1,    1, 1, 1, 0, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0,    0, 0, 1, 0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0,    1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1,    0, 1, 1, 0, 0, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1,    1, 0, 1, 1, 0, 0, 1, 1, 1, 1]
    ])

syndromes = {
    'A': [1, 1, 1, 1, 0, 1, 1, 0, 0, 0],
    'B': [1, 1, 1, 1, 0, 1, 0, 1, 0, 0],
    'C': [1, 0, 0, 1, 0, 1, 1, 1, 0, 0],
    'C\'': [1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
    'D': [1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
}

def compute_syndrome(window, parityMatrix):
    syndrome = []
    for column in parityMatrix:
        and_result = window * column
        xor_result = np.sum(and_result) % 2
        syndrome.append(xor_result)
    return syndrome

def get_syndrome_key(syndrome):
    for key, value in syndromes.items():
        if tuple(value) == tuple(syndrome):
            return key
    return None  

def ParityCheck(bit_stream, synchronized=False, prev_bits=None):
    i = 0
    if prev_bits is not None:
        bit_stream = np.concatenate((np.array(prev_bits), bit_stream))

    while not synchronized and i <= len(bit_stream) - 104:  # 4 blocks * 26 bits = 104
        sequence_keys = []
        for j in range(4):  # Try to get 4 syndromes 26 bits apart
            window = np.array(bit_stream[i + j*26 : i + (j+1)*26])

            
            syndrome = compute_syndrome(window, parityMatrix)
            key = get_syndrome_key(syndrome)
            if key:
                print(f"Processing block {j+1} at index {i + j*26}: {window}")
                print(f"Syndrome for block {j+1} at index {i + j*26}: {syndrome}, Key: {key}")
                sequence_keys.append(key)
            else:
                break

        if len(sequence_keys) == 4:
            # Check sequence A → B → C/C′ → D
            if sequence_keys[0] == 'A' and sequence_keys[1] == 'B' and \
               sequence_keys[2] in ['C', 'C\''] and sequence_keys[3] == 'D':
                synchronized = True
                print(f"Synchronized at index {i}, sequence: {sequence_keys}")
                i += 104  # Move past synchronized group
                break

        i += 1

    message = [0, [0, 0, 0, 0], 0, 0]
    radio_text = {}  # Dictionary to store radio text at different addresses
    program_info = ""
    count = 0
    address = 0
    pi_code = -1

    if synchronized:
        # Process blocks every 26 bits now
        while i <= len(bit_stream) - 26:
            window = np.array(bit_stream[i:i+26])
            
            syndrome = compute_syndrome(window, parityMatrix)
            key = get_syndrome_key(syndrome)

            window = window[0:16]
            
            if key == 'A':
                pi_code = getPIcode(window)
                message[0] = pi_code
            elif key == 'B':
                groupType = getGroupTypePlusPTY(window)
                message[1] = groupType
                if message[1][0] == 2:
                    address = int("".join(map(str, window[-5:])), 2)
            elif key == 'C':
                if message[1][0] == 2:
                    radio_text_data = getRadioText(window)
                    radio_text[address] = radio_text_data  # Append the radio text at the given address
            elif key == 'D':
                if message[1][0] == 2:
                    radio_text_data = getRadioText(window)
                    radio_text[address] += radio_text_data  # Append the radio text at the given address
                elif message[1][0] == 0:
                    program_info += getProgramInfo(window)
            
            count += 1
            if count % 4 == 0:
                # Combine radio text for all addresses
                # Join the radio texts properly by ensuring everything is a string
                all_radio_texts = ''.join([str(text) if isinstance(text, str) else ''.join(map(str, text)) for texts in radio_text.values() for text in texts])

                print(f"Station Group Type: {message[1][0]}{message[1][1]}, Station TP: {message[1][2]}, Program Type: {message[1][3]}")
                print(f"Program Info: {program_info}")
                print(all_radio_texts)

            i += 26  # Process next block
    prev_bits = bit_stream[i:] if i < len(bit_stream) else None
    print(synchronized)
    return message, synchronized, prev_bits

def getPIcode(message_bits):
    # Convert bit list to an integer (big-endian)
    pi_code_int = int("".join(map(str, message_bits)), 2)

    # Convert to hexadecimal (4-digit hex code)
    pi_code_hex = f"{pi_code_int:04X}"  # Uppercase hex

    return str(pi_code_hex)

def getGroupTypePlusPTY(message_bits):
    if message_bits is None or len(message_bits) < 11:
        return 0, 'A', 0, 0  # Default values (number=0, letter='A', TP=0, PTY=0)

    code = message_bits[0:4]  # Extract first 4 bits
    number = int("".join(map(str, code)), 2)  # Convert binary to decimal
    
    letter = 'A' if message_bits[4] == 0 else 'B'  # Determine letter

    TP = message_bits[5]

    PTY = int("".join(map(str, message_bits[6:11])), 2)
    
    return number, letter, TP, PTY

def getProgramInfo(message_bits):
    first_char = int("".join(map(str, message_bits[:8])), 2)
    second_char = int("".join(map(str, message_bits[8:])), 2)
    
    # Convert to characters (assuming ASCII/UTF-8)
    first_char = chr(first_char)
    second_char = chr(second_char)
    
    return first_char + second_char


def getRadioText(message_bits):
    first_char = int("".join(map(str, message_bits[:8])), 2)
    second_char = int("".join(map(str, message_bits[8:])), 2)
    
    # Convert to characters (assuming ASCII/UTF-8)
    first_char = chr(first_char)
    second_char = chr(second_char)
    
    return first_char + second_char


if __name__ == "__main__":
    checkword = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0]
    syndrome = compute_syndrome(checkword, parityMatrix)
    print(checkword[-10:])
    print(syndrome)
    key = get_syndrome_key(syndrome)
    print(key)
import socket
import random
from time import sleep, time
import numpy as np
import copy 

"""
This is the cascade algorithm.
It is a quantum error correction algorithm that uses a cascade of binary algorithms to correct errors in a key.
"""

def generate_key(N, seed = None):
    """
    Generate a key of length N using a seed.

    Parameters:
    N (int): The length of the key to be generated.
    seed (int): The seed to use for the random number generator.

    Returns:
    numpy.ndarray: The generated key.
    """
    # Seed the random number generator
    if seed is None:
        np.random.seed()
    else:
        np.random.seed(seed)
    # Generate the key
    key = np.random.randint(0, 2, N)

    return key

def add_noise_to_key(key, p):
    """
    Add noise to a key.

    Parameters:
    key (numpy.ndarray): The key to which to add noise.
    p (float): The probability of adding noise to each bit.

    Returns:
    numpy.ndarray: The key with added noise.
    """
    # Generate a random float for each element in the key
    random_floats = np.random.random(key.shape)

    # Where the random float is greater than p, keep the original bit, otherwise flip it
    noisy_key = np.where(random_floats > p, key, 1 - key)

    return noisy_key

def calculate_parity(block):
    """
    Calculate the parity of a block.

    Parameters:
    block (numpy.ndarray): The block for which to calculate the parity.

    Returns:
    int: The parity of the block (0 if even, 1 if odd).
    """
    return np.sum(block) % 2

def send_message(ip='127.0.0.1', port=1234, message=None, header=None):
    """
    Send a message to an IP socket.

    Parameters:
    ip (str): The IP address to send to.
    port (int): The port to send to.
    message (list): The message to send.
    header (str): The header to send. 2 for parity, 8 for done.

    Returns:
    None
    """
    global sent, i_am
    connected = False
    # Create a socket object
    while not connected:
        try:
            s = socket.socket()
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            #s.settimeout(1)
            # Connect to the server
            s.connect((ip, port))
            connected = True
        except:
            print("Connection failed, trying again")
            sleep(1)

    # Send the message
    if header:
        packet = [header]+message
        s.sendall(str(packet).encode('utf-8'))   
    else:
        s.sendall(str(message).encode('utf-8'))
    # Close the socket
    s.shutdown(socket.SHUT_RDWR)

    #if i_am == 'Bob':
        #with open('bob_sent.txt', 'a') as file:
        #    file.write(str(message[1:]) + '\n')
    #    sent += [message[1:]]
    s.close()
    
def receive_message(ip='127.0.0.1', port=1234, buffer_size=1024):
    """
    Receive a message from an IP socket.

    Parameters:
    ip (str): The IP address to listen on.
    port (int): The port to listen on.
    buffer_size (int): The maximum amount of data to be received at once.

    Returns:
    str: The received message.
    """
    # Create a socket object
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Bind to the IP and port
    s.bind((ip, port))

    # Listen for incoming connections
    s.listen(100)

    # Accept a connection
    c, addr = s.accept()

    # Receive the message
    buffer = b''
    while True:
        chunk = c.recv(buffer_size)
        if not chunk:
            break
        buffer += chunk
    
    data = buffer
    # Close the connection
    c.shutdown(socket.SHUT_RDWR)
    c.close()

    # Return the received message
    #if i_am == 'Alice':
    #    print("Received message: ", eval(data.decode('utf-8')))
    return eval(data.decode('utf-8'))

def binary_algorithm(block,i,location): #location is a string NN+i+L/R where NN is the top block number, i is the sub-block number, and L/R is the left or right sub-block
    global sub_blocks, alice_sub_block_parity, succesfull_access_dict, failed_access_dict, single_bit_parity_list
    
    sub_blocks[i][0], sub_blocks[i][1] = np.array_split(block, 2)

    if alice_sub_block_parity.get(str(location)):
        alice_parity = alice_sub_block_parity[str(location)]
        succesfull_access_dict += 1
    else:
        failed_access_dict += 1
        send_message(alice_ip, alice_port, sub_blocks[i][0].tolist(), 2)
        alice_parity = receive_message(bob_ip, bob_port, buffer_size)
        alice_sub_block_parity[str(location)] = alice_parity

    bob_parity = calculate_parity(key_new[sub_blocks[i][0]])

    if alice_parity != bob_parity:

        if len(sub_blocks[i][0]) > 1:
            #if (len(sub_blocks[i][0]) == 2):
                #don't split twos
                #return       
            binary_algorithm(sub_blocks[i][0],i,str(location)+',0')
        else:
            error_bits.append(sub_blocks[i][0][0])
            single_bit_parity_list.append(sub_blocks[i][0][0])
    else:
        if len(sub_blocks[i][1]) > 1:
           # if (len(sub_blocks[i][1]) == 2):
                #don't split twos
                #return
            binary_algorithm(sub_blocks[i][1],i,str(location)+',1')
        else:
            error_bits.append(sub_blocks[i][1][0])
    return

def cascade(i_am ='Alice', 
            alice_ip = '127.0.0.1', 
            alice_port = 1234, 
            bob_ip = '127.0.0.1',
            bob_port = 1235,
            buffer_size = 1024,
            key = None,
            key_length = 1000,
            key_noise_probability = 0.1,
            qber = None,
            N_top_blocks_scaler = 1.2):
    globals()['i_am'] = i_am
    globals()['alice_ip'] = alice_ip
    globals()['alice_port'] = alice_port
    globals()['bob_ip'] = bob_ip
    globals()['bob_port'] = bob_port
    globals()['buffer_size'] = buffer_size
    print("Starting cascade as: ", i_am)
    global key_new

    if key is None:
        is_simulation = True
    else:
        is_simulation = False

    # Check if simulation, if it is, generate key, add noise to key.
    if is_simulation:
        print("i_am: ", i_am)
        if i_am == 'Alice':
            print("Alice thinks this is a simulation, generating key and adding noise to key.")
            Alice_key = generate_key(key_length, 56)
            Bob_original_key = add_noise_to_key(Alice_key, key_noise_probability)
            print("Alice_key: ", Alice_key[:10])
            print("Bob_original_key: ", Bob_original_key[:10])
            send_message(bob_ip, bob_port, Bob_original_key.tolist(), 9)
        if i_am == 'Bob':
            print("Bob thinks this is a simulation, receiving key.")
            Bob_original_key = receive_message(bob_ip, bob_port, buffer_size)
            Bob_original_key = Bob_original_key[1:] #strip off the header
            if not qber:
                qber = key_noise_probability
            print("key received")
            key = copy.deepcopy(Bob_original_key)
            key_new = np.array(copy.deepcopy(key))
    else:
        if i_am == 'Alice':
            Alice_key = np.array(key)
        if i_am == 'Bob':
            key_new = np.array(copy.deepcopy(key))
            if not qber:
                print("No qber given, using key_noise_probability = 0.1")
                qber = key_noise_probability

    if i_am == 'Bob':
        global sent, sub_blocks, top_blocks_new, top_blocks_alice_parity, top_blocks_calculated_parity, N, alice_sub_block_parity, flipped_bits, N_odd_error_parity_zero_conts, single_bit_parity_list, succesfull_access_dict, failed_access_dict, error_bits

        succesfull_access_dict = 0
        failed_access_dict = 0
        error_bits = []
        start_time = time()
        assumed_number_of_errors = round(len(key) * qber)
        
        maps = []

        sent = []
        
        top_blocks_new =[]
        sub_blocks = []
        top_blocks_alice_parity = []
        top_blocks_calculated_parity = []
        N = 0
        done = False
        alice_sub_block_parity = {}
        flipped_bits = []
        N_odd_error_parity_zero_conts = 0
        single_bit_parity_list = []

        while not done:
            print("N: ", N)
            
            maps += [np.arange(len(key_new))]
            
            # Shuffle the map, but not for the first N as that represents the original key 
            if N != 0:
                np.random.shuffle(maps[N])
            
            # Scaling the number of top blocks with the number of errors, our goal is to have a little more top blocks than errors.
            # This ensures maximum error correction, but also ensures that we don't have too many top blocks, which would make the algorithm slow.
            # Also, we always want at least 2 top blocks, so we can look for odd error parities in the top blocks.
            if assumed_number_of_errors - len(flipped_bits) < 2:
                N_top_blocks = 2
            else:      
                N_top_blocks = round((assumed_number_of_errors - len(flipped_bits))*N_top_blocks_scaler)
        
            top_blocks_new += [np.array_split(maps[N], N_top_blocks)]
            top_blocks_alice_parity += [[None] * len(top_blocks_new[N])]
            top_blocks_calculated_parity += [[None] * len(top_blocks_new[N])]
            if N == 0:
                sub_blocks = [[0,1]] * len(top_blocks_new[N]) 
            N_odd_error_parity = 0       


            for NN in range(N,-1,-1):
                print("NN: ", NN)
                error_bits = []
                
                if NN != N: # Ran this NN before
                    for i in range(len(top_blocks_new[NN])):
                        top_blocks_calculated_parity[NN][i] = calculate_parity(key_new[top_blocks_new[NN][i]])   
                        if top_blocks_calculated_parity[NN][i] != top_blocks_alice_parity[NN][i]:
                            N_odd_error_parity += 1
                            binary_algorithm(top_blocks_new[NN][i],i,str(NN)+','+str(i))
                        
                else: # Have not run this NN before, so top block parity is known
                    for i in range(len(top_blocks_new[NN])):
                        top_blocks_calculated_parity[NN][i] = calculate_parity(key_new[top_blocks_new[NN][i]])
                                    
                        send_message(alice_ip, alice_port, top_blocks_new[NN][i].tolist(),2)
                        top_blocks_alice_parity[NN][i] = receive_message(bob_ip, bob_port, buffer_size)
                        if top_blocks_calculated_parity[NN][i] != top_blocks_alice_parity[NN][i]:
                            N_odd_error_parity += 1
                            binary_algorithm(top_blocks_new[NN][i],i,str(NN)+','+str(i))
            
                for bit_index in error_bits:
                    flipped_bits += [bit_index]
                    key_new[bit_index] = 1 - key_new[bit_index]
    
                print("Number of error bits found this iteration: ", len(error_bits))

            if N_odd_error_parity == 0: #No odd error parities in top blocks, meaning we could not correct any errors
                N_odd_error_parity_zero_conts +=1 #Count the number of times we have not been able to correct any errors

                if N_odd_error_parity_zero_conts > 1: #If we have not been able to correct any errors for two consecutive N's, we are done
                    done = True   
            elif N > 50: #If we have run the algorithm for 50 N's, we are done
                done=True
            else:
                N_odd_error_parity_zero_conts = 0 #Reset the counter
            
            N += 1
            
        #print(error_bits)
        #flipped_bits.sort()
        #print("flipped_bits", flipped_bits)
        if is_simulation:
            send_message(alice_ip, alice_port, [8])
            Alice_key = receive_message(bob_ip, bob_port, buffer_size)
            are_equal = np.array_equal(Alice_key,key_new)
            print("Did it work? ", are_equal)
            if not are_equal:
                print("Error, keys are not equal")
                difference = np.where(Alice_key != key_new)
                print("Difference: ", difference)
                print()
        else:
            print("Cascade algorithm done")
            send_message(alice_ip, alice_port, [8])
        #difference = np.where(np.array(Alice_key) != np.array(Bob_original_key))
        #print("correct Difference: ", difference)
        end_time = time()
        elapsed_time = end_time - start_time
        print("Elapsed time: ", elapsed_time, 's')
        #with open('bob_sent.txt', 'w') as file:
        #    file.write(str(sent))
        #print(alice_sub_block_parity)
        #print(len(alice_sub_block_parity))
        #print("succesfull_access_dict: ", succesfull_access_dict)
        #print("failed_access_dict: ", failed_access_dict)
        #print("Number of single bit parities checked: ", len(single_bit_parity_list))
        
        return key_new

    elif i_am == 'Alice':
        done = False
        while not done:
            message = receive_message(alice_ip, alice_port, buffer_size)
            if message[0] == 0:
                N = message[1]
                print("Sending parity of left sub-block: ", sub_blocks[N][0])
                send_message(bob_ip, bob_port, calculate_parity(sub_blocks[N][0]))
            elif message[0] == 1:
                N = message[1]
                send_message(bob_ip, bob_port, calculate_parity(sub_blocks[N][1]))
            elif message[0] == 2:

                indexes = message[1:]

                send_message(bob_ip, bob_port, calculate_parity(Alice_key[indexes]))
            elif message[0] == 8:
                done = True
                if is_simulation:
                    send_message(bob_ip, bob_port, Alice_key.tolist())
            elif message[0] == 9:
                #print("I am Alice, I received a message: ", message)
                #message = np.insert(Bob_original_key,0,9).tolist()
                #print("Sending message: ", message)
                send_message(bob_ip, bob_port, Bob_original_key.tolist(), 9)
        return Alice_key

#cascade(i_am = 'Bob') # Run the cascade algorithm


"""
Message headers
- 9: Key
- 8: Done
- 0: Left sub-block parity 
- 1: Right sub-block parity
- 2: Top-block parity
"""




import random
import numpy as np

#REPRODUCIBILITY
random.seed(42)


#IUPAC AMBIGUITY CODES
iupac_codes = {
    'M': ['A','C'],
    'R': ['A','G'],
    'Y': ['C','T'],
    'W': ['A','T'],
    'K': ['G','T'],
    'B': ['C','G','T'],
    'S': ['G','C']
}



def encode_6mer(sequence):
    base_to_int = {'A':0, 'T':1, 'G':2, 'C':3}
    int_arr = []

    #sort out ambiguities and converts the whole sequence to integers
    for base in sequence:
        if base in iupac_codes:
            base = random.choice(iupac_codes[base])
        
        int_arr.append(base_to_int[base])

    #make a multiple of 6
    while len(int_arr) % 6 != 0:
        int_arr.pop()
    
    #convert to numpy array --> faster
    int_arr = np.array(int_arr)

    pos1 = int_arr[0::6]
    pos2 = int_arr[1::6]
    pos3 = int_arr[2::6]
    pos4 = int_arr[3::6]
    pos5 = int_arr[4::6]
    pos6 = int_arr[5::6]

    tokens = pos1 * 4**5 + pos2 * 4**4 + pos3 * 4**3 + pos4 * 4**2 + pos5 * 4**1 + pos6

    return tokens

        
def decode_6mer(token):
    int_to_base = {0:'A', 1:'T', 2:'G', 3:'C'}
    
    six_mer = []

    dividend = token
    divisor = 4

    for i in range(6):
        remainder = dividend % divisor
        six_mer.append(int_to_base[remainder])

        quotient = dividend // divisor
        dividend = quotient
    
    six_mer.reverse()

    return ''.join(six_mer)






    


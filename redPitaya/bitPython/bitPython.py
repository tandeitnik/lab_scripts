import numpy as np

def bitString(decimalNumber,nBits, bitType = 'unsigned'):
    """

    Parameters
    ----------
    decimalNumber : integer
        integer number to be converted to a binary string.
    nBits : positive integer
        number of bits to be used.
    bitType : string, optional
        DESCRIPTION. The default is 'unsigned'. If 'signed', the signed representation is used

    Returns
    -------
    bit : string
        binary representation of decimalNumber.

    """
    bitStr = bin(abs(decimalNumber))[2:]
    
    assert len(bitStr) <= nBits, "the given number cannot be written with the number of bits asked"
    
    if bitType == "signed":
        
        bitStr = '0'*(nBits-len(bitStr))+bitStr
        
        if np.sign(decimalNumber) == -1: #evaluate the 2's complement
            
            bitStr = NOT(bitStr)
            tmp = bit2decimal(bitStr) + 1
            bitStr = bitString(tmp,nBits)
            
    if len(bitStr) < nBits:
        
        bitStr = '0'*(nBits-len(bitStr))+bitStr
        
    return bitStr

def bit2decimal(bitStr, bitType = 'unsigned'):
    """
    Parameters
    ----------
    bitStr : string
        bit string to be converted to decimal representation.
    bitType : string, optional
        DESCRIPTION. The default is 'unsigned'. How the number is represented. Can be 'unsigned' or 'signed'

    Returns
    -------
    ans : integer
        decimal representation of bitStr.

    """
    ans = 0
    
    if bitType == 'signed':
        
        if bitStr[0] == '1':
            
            tmp = bitSum(bitStr,bitString(1,len(bitStr)),subtract = True)
            tmp = NOT(tmp)
            
            for i in range(len(bitStr)):
                
                ans += int(tmp[-1-i])*2**i
                
            ans = ans*-1
            
        else:

            for i in range(len(bitStr)):
                
                ans += int(bitStr[-1-i])*2**i
                
    else:
        
        for i in range(len(bitStr)):
            
            ans += int(bitStr[-1-i])*2**i
        
    return ans

def fixedPoint(number,N_left,N_right):
    """
    Parameters
    ----------
    number : float
        number to be converted to fixed-point binary representation.
    N_left : integer
        number of binary digits to the left of the radix point.
    N_right : integer
        number of binary digits to the right of the radix point.

    Returns
    -------
    ans : string
        fixed-point binary representation of the input.

    """
    lowestNumber = -2**(N_left-1)
    highestNumber = 2**(N_left-1) - 2**(-N_right)
    
    assert (number >= lowestNumber) and (number <= highestNumber), "number is out of range"
    
    decimal = np.abs(number) - np.fix(np.abs(number))
    integer = int(np.fix(number))

    rightPoint = ""

    for i in range(1,N_right+1):
        
        rightPoint = rightPoint + str(int(decimal // 2**-i))
        decimal = decimal - (decimal // 2**-i)*(2**-i)
    
    if np.sign(number) == 1:
        if N_left == 1:
            leftPoint = '0'
        else:
            leftPoint = '0'+bitString(integer,N_left-1)
    else:
        leftPoint = bitString(integer,N_left)

    ans = leftPoint+rightPoint

    if np.sign(number) == -1:

        ans = NOT(ans)
        tmp = bit2decimal(ans) + 1
        ans = bitString(tmp,len(ans))

    return ans

def fixed2dec(number,N_left,N_right):
    
    return bit2decimal(fixedPoint(number,N_left,N_right))

def fixedBin2dec(bitStr,N_left,N_right):
    
    if bitStr[0] == '0':
    
        ans = 0
        
        for i in range(N_left):
            ans += int(bitStr[i])*2**(N_left-1-i)
        for i in range(N_right):
            ans += int(bitStr[i+N_left])*2**(-(i+1))
            
    else:
        
        ans = 2**(N_left-1)
        
        for i in range(1,N_left):
            ans -= int(bitStr[i])*2**(N_left-1-i)
        for i in range(N_right):
            ans -= int(bitStr[i+N_left])*2**(-(i+1))
        ans = ans*-1
    
    return ans

def SLL(bitString,n):
    
    ans = bitString[n:] + '0'*min(n,len(bitString))
    
    return ans

def SRL(bitString,n):
    
    ans = '0'*min(n,len(bitString))+bitString[:len(bitString)-n]
    
    return ans

def leftRoll(bitString,n):
    
    if n > len(bitString):
        
        n = n-len(bitString)
    
    ans = bitString[n:] + bitString[:n]
    
    return ans

def rightRoll(bitString,n):
    
    if n > len(bitString):
        
        n = n-len(bitString)
    
    ans = bitString[len(bitString)-n:]+bitString[:len(bitString)-n]

    return ans

def XOR(bitString0,bitString1):
    
    ans = ''
    
    for i in range(len(bitString0)):
        
        ans = ans + str(int(bitString0[i])^int(bitString1[i]))
        
    return ans

def AND(bitString0,bitString1):
    
    ans = ''
    
    for i in range(len(bitString0)):
        
        ans = ans + str(int(bitString0[i]) and int(bitString1[i]))
        
    return ans

def OR(bitString0,bitString1):
    
    ans = ''
    
    for i in range(len(bitString0)):
        
        ans = ans + str(int(bitString0[i]) or int(bitString1[i]))
        
    return ans

def NOT(bitString):
    """
    Parameters
    ----------
    bitString : string
        bit string.

    Returns
    -------
    ans : string
        The NOT of the input.

    """
    ans = ''
    
    for i in range(len(bitString)):
        
        ans = ans + str((int(bitString[i])+1)%2)
        
    return ans



def bitSum(bitString0,bitString1,subtract = False):
    """
    Parameters
    ----------
    bitString0 : string
        bit string to be added/subtracted.
    bitString1 : string
        bit string to be added/subtracted.
    subtract : boolean, optional
        DESCRIPTION. The default is False. If True, the operation is changed to subtraction.

    Returns
    -------
    ans : string
        Addition/subtraction of the inputs. If subtraction, the second input is subtracted from the first.

    """
    a = bit2decimal(bitString0)
    b = bit2decimal(bitString1)
    
    if subtract == True:
        ans = a-b
    else:
        ans = a+b
   
    ans = bitString(ans,max(np.floor(np.log2(ans)+1),len(bitString0),len(bitString1)))
    
    return ans


def tapList(n):
    """
    Parameters
    ----------
    n : integer
        number of bits for the LFSR.

    Returns
    -------
    list of integers
        position of the taps.

    """
    
    taps = [[2,1],
            [3,2],
            [4,3],
            [5,3],
            [6,5],
            [7,6],
            [8,6,5,4],
            [9,5],
            [10,7],
            [11,9],
            [12,6,4,1],
            [13,4,3,1],
            [14,5,4,1],
            [15,14],
            [16,15,13,4],
            [17,14],
            [18,11],
            [19,6,2,1],
            [20,17],
            [21,19],
            [22,21],
            [23,18],
            [24,23,22,17],
            [25,22],
            [26,6,2,1],
            [27,5,2,1],
            [28,25],
            [29,27],
            [30,6,4,1],
            [31,28],
            [32,22,2,1],
            [33,20],
            [34,27,2,1],
            [35,33],
            [36,25],
            [37,5,4,3,2,1],
            [38,6,5,1],
            [39,35],
            [40,38,21,19],
            [41,38],
            [42,41,20,19],
            [43,42,38,37],
            [44,43,18,17],
            [45,44,42,41],
            [46,45,26,25],
            [47,42],
            [48,47,21,20],
            [49,40],
            [50,49,24,23],
            [51,50,36,35],
            [52,49],
            [53,52,38,37],
            [54,53,18,17],
            [55,31],
            [56,55,35,34],
            [57,50],
            [58,39],
            [59,58,38,37],
            [60,59],
            [61,60,46,45],
            [62,61,6,5],
            [63,62],
            [64,63,61,60]]
    
    return taps[n-2]

def LFSR(n,N,leapSize,seed):
    """
    Parameters
    ----------
    n : output is n-bit
    N : N-bit words are used to generate LFSR sequence
    leapSize : size of the leap
    seed : first element of the sequence, must be a N-bit string

    Returns
    -------
    output : a n-bit string
    nextSequence : the next N-bit word of the sequence. Notice that this algorithm
    is leap forward, therefore the LFSR leaps in multiples of n.

    """
    assert len(seed) == N, 'length of seed must be equal to N'
    assert len(seed) <= N, 'length of output (n) must be equal or less to N'
    
    seedArray = np.zeros(N)
    for i in range(N):
        seedArray[i] = int(seed[i])
        
    taps = tapList(N)
    
    a = np.zeros([1,N])
    for i in range(len(taps)):
        a[0,taps[i]-1] = 1
        
    I = np.concatenate((np.eye(N-1),np.zeros([N-1,1])), axis = 1)

    M = np.concatenate((a,I), axis = 0)

    leapMatrix = np.linalg.matrix_power(M, leapSize)
    
    outputArray = (leapMatrix@seedArray) % 2

    output = ''
    nextSequence = ''

    for i in range(n):
        output =  str(int(outputArray[-1-i])) + output
        
    for i in range(N):
        nextSequence = nextSequence + str(int(outputArray[i]))
        
    return output, nextSequence

def seedGenerator(N,size):
    """
    Parameters
    ----------
    N : integer
        Number of bits of the seed.
    size : integer
        Number of seeds to be generated.

    Returns
    -------
    seeds : list
        A list containg the generated seeds.

    """
    seeds = []

    for j in range(size):

        seed = ''
        
        for i in range(N):
            
            seed = seed + str(int(np.round(np.random.uniform(0,1))))
            
        seeds.append(seed)
        
    return seeds

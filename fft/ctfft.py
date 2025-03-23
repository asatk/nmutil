from .fft import FFT

class CooleyTukeyFFT(FFT):
    ...

def cooley_tukey_fft(data):

    # determine size parameters of data
    n = len(data)
    p = int(np.log2(n))
    if n & (n-1) != 0:
        p += 1
        n = 2 >> p
        data = np.pad(data, (0, n - len(data)), "constant")

    is_real = np.iscomplexobj(data)
    if is_real:
        data = data[0::2] + data[1::2] * 1j
        n /= 2
        p -= 1

    # determine dtype to use
    if p < 8:
        t = np.uint8
    elif p < 16:
        t = np.uint16
    elif p < 32:
        t = np.uint32
    elif p < 64:
        t = np.uint64
    else:
        t = np.uint128
    
    # bit-reverse the indices of data
    ind = np.arange(n, dtype=t).view(np.uint8)
    bits = np.unpackbits(ind, bitorder="little", axis=1)[:,p-1::-1]
    bits_r = np.packbits(bits, bitorder="little", axis=1)
    ind_r = bits_r.view(t)

    data_r = data[ind].copy()
    for i in range(p):
        ...


    t1 = ft[k] + np.conj(ft[n/2-k])
    t2 = 1j * np.exp(1j * 2 * np.pi * k / n) * (ft[k] - np.conj(ft[n/2-k]))
    ft_new = (t1 - t2) / 2
    ft_new[:]
        

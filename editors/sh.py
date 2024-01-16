import math
import random

class Vec3d:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z
        
kCacheSize = 16

def Factorial(x):
    factorial_cache = [1, 1, 2, 6, 24, 120, 720, 5040,
                       40320, 362880, 3628800, 39916800,
                       479001600, 6227020800, 87178291200, 1307674368000]
    if x < kCacheSize:
        return factorial_cache[x]
    else:
        s = factorial_cache[kCacheSize - 1]
        for n in range(kCacheSize, x+1):
            s *= n
        return s
       
def DoubleFactorial(x):
    dbl_factorial_cache = [1, 1, 2, 3, 8, 15, 48, 105,
                           384, 945, 3840, 10395, 46080,
                           135135, 645120, 2027025]
    if x < kCacheSize:
        return dbl_factorial_cache[x]
    else:
        s = dbl_factorial_cache[kCacheSize - (2 if x % 2 == 0 else 1)]
        n = x
        while n >= kCacheSize:
            s *= n
            n -= 2.0
        return s
    
n_bands = 3
class SHSample:
    def __init__(self, sph, vec, coeff):
        self.sph = sph
        self.vec = vec
        self.coeff = coeff
 
# SHSampleList = []
# for i in range(sqrt_n_samples * sqrt_n_samples):
#     SHSampleList.append(SHSample(Vec3d(),Vec3d(),0))
#SHSampleList = [SHSample(Vec3d(),Vec3d(),0)]*(sqrt_n_samples * sqrt_n_samples)


MYPI = math.pi
sqrt2 = math.sqrt(2.0)

def P(l, m, x):
    # evaluate an Associated Legendre Polynomial P(l,m,x) at x
    pmm = 1.0
    if m > 0:
        somx2 = math.sqrt((1.0 - x)*(1.0 + x))
        fact = 1.0
        for i in range(1, m+1):
            pmm *= (-fact) * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2.0*m + 1.0) * pmm
    if l == m + 1:
        return pmmp1
    pll = 0.0
    for ll in range(m+2, l+1):
        pll = ((2.0*ll - 1.0)*x*pmmp1 - (ll + m - 1.0)*pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll

def K(l, m):
    # renormalisation constant for SH function
    temp = ((2.0*l + 1.0)*Factorial(l - m)) / (4.0*MYPI*Factorial(l + m))
    return math.sqrt(temp)

def SH(l, m, theta, phi):
    # return a point sample of a Spherical Harmonic basis function
    # l is the band, range [0..N]
    # m in the range [-l..l]
    # theta in the range [0..Pi]
    # phi in the range [0..2*Pi]
    if m == 0:
        return K(l, 0)*P(l, m, math.cos(theta))
    elif m > 0:
        return sqrt2*K(l, m)*math.cos(m*phi)*P(l, m, math.cos(theta))
    else:
        return sqrt2*K(l, -m)*math.sin(-m*phi)*P(l, -m, math.cos(theta))

def clamp(data, minimum, maximum):
    if data < minimum:
        data = minimum
    if data > maximum:
        data = maximum
    return data

def getLight(result, theta, phi):
    data = 0.0
    for l in range(n_bands):
        for m in range(-l, l+1):
            index = l*(l + 1) + m
            data += result[index]*SH(l, m, theta, phi)
    return clamp(data, 0.0, 1.0)


def SH_setup_spherical_samples(sqrt_n_samples = 100):
    # fill an N*N*2 array with uniformly distributed
    # samples across the sphere using jittered stratification
    samples = []
    for i in range(sqrt_n_samples * sqrt_n_samples):
        samples.append(SHSample(Vec3d(),Vec3d(),0))
    
    i = 0  # array index
    oneoverN = 1.0 / (sqrt_n_samples*1.0)
    for a in range(sqrt_n_samples):
        for b in range(sqrt_n_samples):
            # generate unbiased distribution of spherical coords
            x = (a + random.random()) * oneoverN  # do not reuse result
            y = (b + random.random()) * oneoverN  # each sample must be random
            theta = 2.0 * math.acos(math.sqrt(1.0 - x))
            phi = 2.0 * MYPI * y
            samples[i].sph = Vec3d(theta, phi, 1.0)
            # convert spherical coords to unit vector
            vec = Vec3d(math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta))
            samples[i].vec = vec
            samples[i].coeff = [0.0] * (n_bands * n_bands)
            # precompute all SH coefficients for this sample
            for l in range(n_bands):
                for m in range(-l, l+1):
                    index = l*(l + 1) + m
                    samples[i].coeff[index] = SH(l, m, theta, phi)
            i += 1
    return samples
            
            
def SH_project_polar_function(fn,r_thera,r_phi,sqrt_n_samples = 100):
    result = [0.0] * (n_bands*n_bands)
    samples = SH_setup_spherical_samples()
    n_coeff = n_bands * n_bands
    weight = 4.0 * math.pi
    n_samples = sqrt_n_samples*sqrt_n_samples*1.0
    
    # for each sample
    for i in range(int(n_samples)):
        theta = samples[i].sph.x
        phi = samples[i].sph.y
        theta = theta + r_thera
        phi = phi + r_phi
        #print(theta,phi)
        for n in range(n_coeff):
            #print(theta,phi)
            result[n] += fn(theta, phi) * samples[i].coeff[n]
    
    # divide the result by weight and number of samples
    factor = weight / n_samples
    #print(factor)
    for i in range(n_coeff):
        result[i] = result[i] * factor
    return result

# def getLightIntensity(theta, phi):
#     intensity = max(0.0, 5 * math.cos(theta) - 4) + max(0.0, -4 * math.sin(theta - math.pi) * math.cos(phi - 2.5) - 3)
#     return intensity
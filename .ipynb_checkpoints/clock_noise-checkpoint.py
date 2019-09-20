import numpy as np
import numpy.fft
import scipy as sp
import scipy.integrate

fSr = 429228004229873.1

##### LASERS

class generic_laser(object):
    
    def __init__(self):
        pass
        # self.allan = np.vectorize(self.__allan_one_point)
    
    def psd(self, f):
        '''Generic laser with 1/f noise'''
        return 1./f
    
    def allan(self, tau):
        # return np.vectorize(self.__allan_one_point)(tau)
        return np.vectorize(self.__allan_one_point)(tau)/fSr
        
    # def allan2(self, tau):
    #     return np.vectorize(self.__allan_one_point_slow)(tau)
            
    def __allan_one_point(self, tau):
        f = np.linspace(0, 6./tau, 1e4)[1:]
        kernel = 2 * np.sin(np.pi*tau*f)**4 / (np.pi*tau*f)**2
        integrand = self.psd(f)*kernel
        return np.sqrt(np.trapz(integrand, f))
    
    def __allan_one_point_slow(self, tau):
        result = sp.integrate.quad(lambda f: self.psd(f) \
                  * 2 * np.sin(np.pi*tau*f)**4 / (np.pi*tau*f)**2,
                  0,
                  np.inf)
        return np.sqrt(result[0])
        
    def generate_noise(self, ttotal=500, num_points=2**16):
        t = np.linspace(0, ttotal, num_points)
    
        # find the frequency spectrum that corresponds to the time values.
        # I'm going to assume an even number of time points.
        f = np.fft.fftfreq(len(t), t[1]-t[0])
        indPositiveFreq = f > 0
        tmp = np.sqrt(0.5*self.psd(f[indPositiveFreq])) * np.exp(1j*2*np.pi*np.random.rand(indPositiveFreq.sum()))
        yfft = np.concatenate(((0,), tmp, (0,), np.conj(np.flipud(tmp))))
    
        lasernoise = np.sqrt( len(t) / (t[1]-t[0]) ) * np.real(np.fft.ifft(yfft))
    
        return (t, lasernoise)


#Classic Mike Bishof model for MJM
class mjm(generic_laser):

    def __init__(self, hwhite=3.3e-3, hthermal = 1.5e-3, hresonance = [(21.87, 1.2, 0.03), (22.39, 0.6, 0.03),(29.45, 0.15, 0.1), (29.90, 0.08, 0.4), (60,0.012, 27)]):
        self.hwhite = hwhite
        self.hthermal = hthermal
        self.hresonance = hresonance
    
    def psd(self, f):

        return self.hwhite + self.hthermal/f + sum([x[1] / ( 1 + ( (f - x[0])/(x[2]/2) )**2 ) for x in self.hresonance])

    def ModSigma(self, tau):
        xi = np.logspace(np.log10(0.0001),np.log10(1000),10e3)
        yi = self.psd(xi)
        Syi = yi/(fSr**2)
        sigmaint = [np.sqrt(2*np.trapz(Syi*np.sin(np.pi*xi*a)**6/(np.pi*xi*a)**4,xi)) for a in tau]
        return sigmaint

#"Optimistic" MJM model.  Assumes lower white frequency noise and ignores resonances which dont agree with S1B20 counter measurements
class mjmOptimistic(generic_laser):

    def __init__(self, hwhite=1.5e-3, hthermal = 1.5e-3, hresonance = []):
        self.hwhite = hwhite
        self.hthermal = hthermal
        self.hresonance = hresonance
    
    def psd(self, f):

        return self.hwhite + self.hthermal/f + sum([x[1] / ( 1 + ( (f - x[0])/(x[2]/2) )**2 ) for x in self.hresonance])


    def ModSigma(self, tau):
        xi = np.logspace(np.log10(0.0001),np.log10(1000),10e3)
        yi = self.psd(xi)
        Syi = yi/(fSr**2)
        sigmaint = [np.sqrt(2*np.trapz(Syi*np.sin(np.pi*xi*a)**6/(np.pi*xi*a)**4,xi)) for a in tau]
        return sigmaint

    

    
#There are three Si classes below.  Basically, only the values in the _init_function are different but I include
#all three classes with explanations for clarity.

#Simple model for Silicon cavity.  Values for white frequency noise differ from the PTB PRL paper.  Also add in white phase noise term.
#This model only includes the two peaks around 50 Hz which I am 100% certain are coming from Si 3. 
#This model was tuned to give us good agreement with our three cornered hat measurements.
class si(generic_laser):
    def __init__(self,
                 hwhite=6e-34,
                 hthermal=1.7e-33,
                 hwalk=3e-36,
				 hwhitephase=3e-36,
                 hres = [(45.0, 1.0e-34, 4),(55.0, 4.0e-34, 1.2)],
                 f0=fSr):
        self.hwhite = hwhite
        self.hthermal = hthermal
        self.hwhitephase = hwhitephase
        self.hwalk = hwalk
        self.hres = hres
        self.f0 = f0


    def psd(self, f):
        fnp = np.array(f)
		
        fNoiseRes = sum([x[1] / ( 1 + ( (fnp - x[0])/(x[2]/2) )**2 )*fnp**2 for x in self.hres])

        fFractional = self.hthermal/fnp + self.hwalk/fnp**2 + self.hwhite*np.ones(len(fnp)) + fNoiseRes + self.hwhitephase*fnp**2

        return fFractional*self.f0**2


    def ModSigma(self, tau):
        xi = np.logspace(np.log10(0.0001),np.log10(1000),10e3)
        yi = self.psd(xi)
        Syi = yi/(fSr**2)
        sigmaint = [np.sqrt(2*np.trapz(Syi*np.sin(np.pi*xi*a)**6/(np.pi*xi*a)**4,xi)) for a in tau]
        return sigmaint

#Here's what the model looks like with no resonant peaks.  Probably not worth another class, but whatever.
class siNoPeaks(generic_laser):
    def __init__(self,
                 hwhite=6e-34,
                 hthermal=1.7e-33,
                 hwalk=3e-36,
				 hwhitephase=3e-36,
                 f0=fSr):
        self.hwhite = hwhite
        self.hthermal = hthermal
        self.hwhitephase = hwhitephase
        self.hwalk = hwalk
        #self.hres = [45.0, 1.0e-34, 4]
        self.f0 = f0

    def psd(self, f):
        fnp = np.array(f)
        fFractional = self.hthermal/fnp + self.hwalk/fnp**2 + self.hwhite*np.ones(len(fnp)) + self.hwhitephase*fnp**2
        return fFractional*self.f0**2

    def ModSigma(self, tau):
        xi = np.logspace(np.log10(0.0001),np.log10(1000),10e3)
        yi = self.psd(xi)
        Syi = yi/(fSr**2)
        sigmaint = [np.sqrt(2*np.trapz(Syi*np.sin(np.pi*xi*a)**6/(np.pi*xi*a)**4,xi)) for a in tau]
        return sigmaint

#Includes all of the peaks seen in Si3-MJM beat.  This "pessimistic" model assumes that all of these peaks
#are either from Si 3 or something else differential in the stability transfer setup (like the comb or fiber noise).
#If this pessimistic assumption is correct, then we would expect all of this noise to be written onto the clock laser when
#we lock the stability transfer servo.

class siPlusComb(generic_laser):
    def __init__(self,
                 hwhite=6e-34,
				 hwhitephase=3e-36,
                 hthermal=1.7e-33,
                 hwalk=3e-36,
                 hres = [(4.0, 8.0e-34, 0.6),(6.0, 5.0e-34, 0.6),(12.5,0.8e-34, 1.5),
                         (20.0,4.0e-34, 0.1),(45.0, 1.0e-34, 4),(55.0, 4.0e-34, 1.2)],
                 f0=fSr):
        self.hwhite = hwhite
        self.hthermal = hthermal
        self.hwhitephase = hwhitephase
        self.hwalk = hwalk
        self.hres = hres
        self.f0 = f0



    def psd(self, f):
        fnp = np.array(f)
        #fNoise_50Hz = self.hres[1]/(1+(fnp-self.hres[0])**2/(self.hres[2]/2)**2)*fnp**2
		
        fNoiseRes = sum([x[1] / ( 1 + ( (fnp - x[0])/(x[2]/2) )**2 )*fnp**2 for x in self.hres])

        fFractional = self.hthermal/fnp + self.hwalk/fnp**2 + self.hwhite*np.ones(len(fnp)) + fNoiseRes + self.hwhitephase*fnp**2

        return fFractional*self.f0**2

    def ModSigma(self, tau):
        xi = np.logspace(np.log10(0.0001),np.log10(1000),10e3)
        yi = self.psd(xi)
        Syi = yi/(fSr**2)
        sigmaint = [np.sqrt(2*np.trapz(Syi*np.sin(np.pi*xi*a)**6/(np.pi*xi*a)**4,xi)) for a in tau]
        return sigmaint

#This is the laser model used in Eric's paper
class siPlusComb2(generic_laser):
    def __init__(self,
                 hwhite=4.0e-34,
				 hwhitephase=3e-36,
                 hthermal=1.5e-33,
                 hwalk=3e-37,
                 hres = [(5.7, 7.0e-34, 1.0),(12.7,1.5e-34, 1.5),
                         (20.0,4.0e-34, 0.1),(30.0,5.0e-34, 0.1),(40.0,5.0e-34, 0.1),(45.0, 1.0e-34, 4),(55.0, 4.0e-34, 1.2)],
                 f0=fSr):
        self.hwhite = hwhite
        self.hthermal = hthermal
        self.hwhitephase = hwhitephase
        self.hwalk = hwalk
        self.hres = hres
        self.f0 = f0


    def psd(self, f):
        fnp = np.array(f)
        #fNoise_50Hz = self.hres[1]/(1+(fnp-self.hres[0])**2/(self.hres[2]/2)**2)*fnp**2
		
        fNoiseRes = sum([x[1] / ( 1 + ( (fnp - x[0])/(x[2]/2) )**2 )*fnp**2 for x in self.hres])

        fFractional = self.hthermal/fnp + self.hwalk/fnp**2 + self.hwhite*np.ones(len(fnp)) + fNoiseRes + self.hwhitephase*fnp**2

        return fFractional*self.f0**2

    def ModSigma(self, tau):
        xi = np.logspace(np.log10(0.0001),np.log10(1000),10e3)
        yi = self.psd(xi)
        Syi = yi/(fSr**2)
        sigmaint = [np.sqrt(2*np.trapz(Syi*np.sin(np.pi*xi*a)**6/(np.pi*xi*a)**4,xi)) for a in tau]
        return sigmaint

    

#"Optimistic" MJM model.  Assumes lower white frequency noise and ignores resonances which dont agree with S1B20 counter measurements
class portable(generic_laser):

    def __init__(self,
                 hwhite=5e-31,
				 hwhitephase=0,
                 hthermal=1.6e-31,
                 hwalk=0,
                 hres = [],
                 f0=fSr):
        self.hwhite = hwhite
        self.hthermal = hthermal
        self.hwhitephase = hwhitephase
        self.hwalk = hwalk
        self.hres = hres
        self.f0 = f0
    
    def psd(self, f):
        fnp = np.array(f)
        fNoiseRes = sum([x[1] / ( 1 + ( (fnp - x[0])/(x[2]/2) )**2 )*fnp**2 for x in self.hres])
        fFractional = self.hthermal/fnp + self.hwalk/fnp**2 + self.hwhite*np.ones(len(fnp)) + fNoiseRes + self.hwhitephase*fnp**2
        return fFractional*self.f0**2
    
    def ModSigma(self, tau):
        xi = np.logspace(np.log10(0.0001),np.log10(1000),10e3)
        yi = self.psd(xi)
        Syi = yi/(fSr**2)
        sigmaint = [np.sqrt(2*np.trapz(Syi*np.sin(np.pi*xi*a)**6/(np.pi*xi*a)**4,xi)) for a in tau]
        return sigmaint
    
    
    
#New syntheticLaser class:  Model for clock laser when stability transfer servo is locked.  Requires user to specify
#models for Si3 and MJM as well as the servo bandwidth and PI corner frequency for the stability transfer loop.
#Typically I've been using
#a double integrator at 1/10 of the bandwidth (Example: a 30 Hz servo with double PI corner at 3 Hz).
#This is conservative enough to keep the servo bump from getting too large.

class syntheticLaser(generic_laser):
    def __init__(self,
                 siModel = siPlusComb2(),
                 mjmModel = mjmOptimistic(),
                 bw = 30.0,
                 piCorner = 3.0,
                 f0=fSr):
        self.siModel = siModel
        self.mjmModel = mjmModel
        self.bw = bw
        self.piCorner = piCorner
        self.f0 = f0

    def loopTF(self,f):
        #Export loop transfer function for monitoring/sanity check purposes
        fnp = np.array(f)

        #Compute the loop transfer function (with double PI corner) for the chosen bandwidth
        Amplitude = (self.bw)**3/np.sqrt( (1.0- (self.bw/self.piCorner)**2 )**2 + 4.0*(self.bw/self.piCorner)**2 ) 
        #prescaler to get correct bw
        G = Amplitude*(1+1j*fnp/self.piCorner)**2/(1j*fnp)**3 #Complex valued loop transfer function

        mag = 20.0*np.log10(abs(G))  #Compute TF magnitude in dB
        phase = np.angle(G)*180.0/np.pi #Compute TF phase in degrees

        return mag, phase

    def psd(self, f):
        #Compute the synthetic laser frequency noise PSD from the Si3 and MJM models for the given servo parameters

        fnp = np.array(f)
		
        #Generate Si3 and MJM PSDs from chosen Si 3 and MJM models
        siPSD = self.siModel.psd(f)
        mjmPSD = self.mjmModel.psd(f)
		
        #Compute the loop transfer function (with double PI corner) for the chosen bandwidth
        Amplitude = (self.bw)**3/np.sqrt( (1.0- (self.bw/self.piCorner)**2 )**2 + 4.0*(self.bw/self.piCorner)**2 ) 
        #prescaler to get correct bw
        G = Amplitude*(1+1j*fnp/self.piCorner)**2/(1j*fnp)**3 #Complex valued loop transfer function

        #Compute resulting laser noise PSD (at 698 nm) by taking the quadrature sum of contributions from Si 3 and MJM
        return abs(1.0/(1.0+G))**2*mjmPSD + abs(G/(1.0+G))**2*siPSD


    def ModSigma(self, tau):
        xi = np.logspace(np.log10(0.0001),np.log10(1000),10e3)
        yi = self.psd(xi)
        Syi = yi/(fSr**2)
        sigmaint = [np.sqrt(2*np.trapz(Syi*np.sin(np.pi*xi*a)**6/(np.pi*xi*a)**4,xi)) for a in tau]
        return sigmaint


##### PULSE SEQUENCES

class pulse_sequence(object):
    
    def __init__(self, ttotal=1.):
        self.ttotal = ttotal
        self.average_zero = False
        pass
    
    def sensitivity(self, num_points=1e3):
        t = np.linspace(0, self.ttotal, num_points)
        y = self.sensitivity_function(t)
        return t, y
    
    def sensitivity_function(self, t):
        return np.zeros_like(t)
    
    def sensitivity_psd(self, num_points=1e3):
        # t = np.linspace(0, self.tpulse+self.tdead, num_points)
        t, y = self.sensitivity(num_points)
        f = np.fft.fftfreq(len(t), d=t[1]-t[0])
        fOneSided = f[f>=0]
        psd = np.abs(np.fft.fft(y)[f>=0])**2 / len(f)**2
        return (fOneSided, psd)     
        
    def sensitivity_psd_one_freq(self, f, num_points=1e4):
        '''Calculate the sensitivity function for one frequency.
        This is very inefficient.'''
        n = np.round(f * self.ttotal)
        f_alias = np.abs(f - (n/self.ttotal))
        
        t, y = self.sensitivity(num_points)
        return (f_alias,
                (np.abs(np.trapz(y
                      * np.exp(2*np.pi*1j*f*t), t))/(t[-1]-t[0]))**2 )

    def sensitivity_psd_continuous(self, fmin, num_points=1e4):
        t = np.linspace(0., 1., num_points) / fmin
        y = self.sensitivity_function(t)


        f, psd = sp.signal.periodogram(y, 1./(t[1]-t[0]), scaling='spectrum')

        psd = 0.5*psd / (fmin * self.ttotal)**2
        
        f_aliased = np.array([ np.abs(f0 \
                             - (np.round(f0 * self.ttotal)/self.ttotal)) \
                             for f0 in f ])
                             
        return (f, f_aliased, psd) 
        
class rabi(pulse_sequence):
    def __init__(self, tpulse, tdead):
        self.tpulse = tpulse
        self.tdead = tdead
        self.ttotal = tpulse + tdead
        self.omega_0 = np.pi/self.tpulse #T_pulse = pi/Omega_0  pi pulse time
        self.delta = 2.509/np.pi*self.omega_0 #Detuning with 0.5 excitation
        self.omega = np.sqrt(self.omega_0**2+self.delta**2)
        self.average_zero = False
        
    def sensitivity_function(self, t):
        return (np.pi/2)*(self.tpulse+self.tdead)/self.tpulse \
               * np.piecewise(t,
                         [t <= 0,
                          (t > 0) & (t <= self.tpulse),
                          t > self.tpulse],
                         [0,
                          lambda t: np.sin(np.pi*t/self.tpulse),
                          0])

    def sensitivity_psd_dm(self,f):
        #assumes Max frequency is a multiple of the FFT bin width fBin
        #f = np.linspace(fBin,fMax,fMax/fBin) 
        #Just Use closed form expression from MJM Thesis Eqn (6.63)
        mag = self.delta**2*self.omega_0**2/(self.omega**4*(np.pi*f*self.omega**2-4*np.pi**3*f**3)**2)*(self.omega*np.sin(np.pi**2*f/self.omega_0)*np.sin(np.pi*self.omega/self.omega_0)-4*np.pi*f*np.cos(np.pi**2*f/self.omega_0)*np.sin(np.pi*self.omega/2/self.omega_0)**2)**2
        f=0.000001
        mag0 = self.delta**2*self.omega_0**2/(self.omega**4*(np.pi*f*self.omega**2-4*np.pi**3*f**3)**2)*(self.omega*np.sin(np.pi**2*f/self.omega_0)*np.sin(np.pi*self.omega/self.omega_0)-4*np.pi*f*np.cos(np.pi**2*f/self.omega_0)*np.sin(np.pi*self.omega/2/self.omega_0)**2)**2
        return mag/mag0


def w(time,Tpi,Tdead,offset):
    out = []
    Delta = 0.4/Tpi
    theta = np.pi/2-np.arctan(2*Delta*Tpi)
    Omega_1 = lambda t:np.pi*np.sqrt(1+2.*Delta*Tpi)*t/Tpi
    Omega_2 = lambda t:np.pi*np.sqrt(1+2.*Delta*Tpi)*(Tpi-t)/Tpi
    for ti in time:
        t = ti%(Tpi+Tdead)
        if t<=(offset):
            out.append(0.0)
        elif (t>(offset))&(t<=(offset+Tpi)):            
            out.append(np.sin(np.pi*t/Tpi))
        else:
            out.append(0.0)
    return (np.pi/2)*(Tpi+Tdead)/Tpi*np.array(out)


#For computing Dick effect limit of the unsynchronized comparison
#in Eric's paper Tc_sr1 = 1.17s, Tc_sr2 = 1.5s
class rabiUnsync(pulse_sequence):
    def __init__(self):
        self.tpulse = 0.6
        self.tdead = 0.57
        self.ttotal1 = 0.6+0.57
        self.omega_0 = np.pi/self.tpulse #T_pulse = pi/Omega_0  pi pulse time
        self.delta = 2.509/np.pi*self.omega_0 #Detuning with 0.5 excitation
        self.omega = np.sqrt(self.omega_0**2+self.delta**2)

        self.tpulse2 = 0.75
        self.tdead2 = 0.75
        self.ttotal2 = 1.5
        self.omega2_0 = np.pi/self.tpulse2 #T_pulse = pi/Omega_0  pi pulse time
        self.delta2 = 2.509/np.pi*self.omega2_0 #Detuning with 0.5 excitation
        self.omega2 = np.sqrt(self.omega2_0**2+self.delta2**2)

        #Note: a full period takes 100*ttotal1 = 78*ttotal2
        self.ttotal = self.ttotal1*100.
        self.average_zero = False
        
    def sensitivity_function(self, t):
        #Note: a full period takes 100*ttotal = 78*ttotal2
        return (w(t,self.tpulse,self.tdead,0.0)-w(t,self.tpulse2,self.tdead2,0.0))/np.sqrt(2)

    def sensitivity_psd_dm(self,f):
        #assumes Max frequency is a multiple of the FFT bin width fBin
        #f = np.linspace(fBin,fMax,fMax/fBin) 
        #Just Use closed form expression from MJM Thesis Eqn (6.63)
        mag = self.delta**2*self.omega_0**2/(self.omega**4*(np.pi*f*self.omega**2-4*np.pi**3*f**3)**2)*(self.omega*np.sin(np.pi**2*f/self.omega_0)*np.sin(np.pi*self.omega/self.omega_0)-4*np.pi*f*np.cos(np.pi**2*f/self.omega_0)*np.sin(np.pi*self.omega/2/self.omega_0)**2)**2
        f=0.000001
        mag0 = self.delta**2*self.omega_0**2/(self.omega**4*(np.pi*f*self.omega**2-4*np.pi**3*f**3)**2)*(self.omega*np.sin(np.pi**2*f/self.omega_0)*np.sin(np.pi*self.omega/self.omega_0)-4*np.pi*f*np.cos(np.pi**2*f/self.omega_0)*np.sin(np.pi*self.omega/2/self.omega_0)**2)**2
        return mag/mag0


#Uses more exact sensitivity function from original Dick effect paper.  
#Dick effect estimate typically agrees with Ed's simplified equation to within 3%
class rabi2(pulse_sequence):
    def __init__(self, tpulse, tdead):
        self.tpulse = tpulse
        self.tdead = tdead
        self.ttotal = tpulse + tdead
        self.omega_0 = np.pi/self.tpulse #T_pulse = pi/Omega_0  pi pulse time
        self.delta = 2.509/np.pi*self.omega_0 #Detuning with 0.5 excitation
        self.Delta = 0.4/self.tpulse
        self.omega = np.sqrt(self.omega_0**2+self.delta**2)
        self.average_zero = False
        
    def sensitivity_function(self, t):
        Omega1 = lambda x : np.pi*np.sqrt(1+(2*self.tpulse*self.Delta)**2)*x/self.tpulse
        Omega2 = lambda x : np.pi*np.sqrt(1+(2*self.tpulse*self.Delta)**2)*(self.tpulse-x)/self.tpulse
        theta = np.pi/2-np.arctan(2*self.tpulse*self.Delta)
        return (np.pi/2)*(self.tpulse+self.tdead)/self.tpulse \
               *np.piecewise(t,
                         [t <= 0,
                          (t > 0) & (t <= self.tpulse),

                          t > self.tpulse],
                         [0,
                          lambda t: np.sin(theta)**2*np.cos(theta)*( (1-np.cos(Omega2(t)))*np.sin(Omega1(t)) + (1-np.cos(Omega1(t)))*np.sin(Omega2(t)) ),
                          0])

    def sensitivity_psd_dm(self,f):
        #assumes Max frequency is a multiple of the FFT bin width fBin
        #f = np.linspace(fBin,fMax,fMax/fBin) 
        #Just Use closed form expression from MJM Thesis Eqn (6.63)
        mag = self.delta**2*self.omega_0**2/(self.omega**4*(np.pi*f*self.omega**2-4*np.pi**3*f**3)**2)*(self.omega*np.sin(np.pi**2*f/self.omega_0)*np.sin(np.pi*self.omega/self.omega_0)-4*np.pi*f*np.cos(np.pi**2*f/self.omega_0)*np.sin(np.pi*self.omega/2/self.omega_0)**2)**2
        f=0.000001
        mag0 = self.delta**2*self.omega_0**2/(self.omega**4*(np.pi*f*self.omega**2-4*np.pi**3*f**3)**2)*(self.omega*np.sin(np.pi**2*f/self.omega_0)*np.sin(np.pi*self.omega/self.omega_0)-4*np.pi*f*np.cos(np.pi**2*f/self.omega_0)*np.sin(np.pi*self.omega/2/self.omega_0)**2)**2
        return mag/mag0

#Switch between condition 1 and condition 2 after two cycles.  This is what 
#Sr 1 ususally does for systematic evaluations and self comparisons
class rabiSelfComparison(pulse_sequence):
    def __init__(self, tpulse, tdead):
        self.tpulse = tpulse
        self.tdead = tdead
        self.ttotal = 4.0*(tpulse + tdead)
        self.tcycle = tpulse + tdead
        self.omega_0 = np.pi/self.tpulse #T_pulse = pi/Omega_0  pi pulse time
        self.delta = 2.509/np.pi*self.omega_0 #Detuning with 0.5 excitation
        self.omega = np.sqrt(self.omega_0**2+self.delta**2)
        self.average_zero = False
    
    #Use extended sensitivity function for four measurement cycles.  This assumes that
    #we switch between lock 1 and lock 2 after measuring both sides of one peak.  We make
    #two measurements quickly and then have an extended period of dead time while we make two 
    #measurements for the second lock.
    def sensitivity_function(self, t):
        return (np.pi)*(self.tpulse+self.tdead)/self.tpulse/np.sqrt(2.0) \
               * np.piecewise(t,
                         [t <= 0,
                          (t > 0) & (t <= self.tpulse),
                          (t > self.tpulse) & (t <= self.tcycle),
                          (t > self.tcycle) & (t <= self.tpulse + self.tcycle),
                          (t > self.tpulse + self.tcycle) & (t <= self.tcycle*2.0),
                          (t > self.tcycle*2.0) & (t <= self.tcycle*2.0+self.tpulse),
                          (t > self.tcycle*2.0+self.tpulse) & (t <= self.tcycle*3.0),
                          (t > self.tcycle*3.0) & (t <= self.tcycle*3.0+self.tpulse),
                          t>self.tcycle*3.0 + self.tpulse],
                         [0,
                          lambda t: np.sin(np.pi*t/self.tpulse),
                          0,
                          lambda t: np.sin(np.pi*(t-self.tcycle)/self.tpulse),
                          0,
                          lambda t: -np.sin(np.pi*(t-self.tcycle*2.0)/self.tpulse),
                          0,
                          lambda t: -np.sin(np.pi*(t-self.tcycle*3.0)/self.tpulse),
                          0])

#Switch between condition 1 and condition 2 after only 1 cycle
class rabiSelfComparison2(pulse_sequence):
    def __init__(self, tpulse, tdead):
        self.tpulse = tpulse
        self.tdead = tdead
        self.ttotal = 4.0*(tpulse + tdead)
        self.tcycle = tpulse + tdead
        self.omega_0 = np.pi/self.tpulse #T_pulse = pi/Omega_0  pi pulse time
        self.delta = 2.509/np.pi*self.omega_0 #Detuning with 0.5 excitation
        self.omega = np.sqrt(self.omega_0**2+self.delta**2)
        self.average_zero = False
    
    #Use extended sensitivity function for four measurement cycles.  This assumes that
    #we switch between lock 1 and lock 2 after each measurement.
    def sensitivity_function(self, t):
        return 0.75*(np.pi)*(self.tpulse+self.tdead)/self.tpulse/np.sqrt(2.0) \
               * np.piecewise(t,
                         [t <= 0,
                          (t > 0) & (t <= self.tpulse),
                          (t > self.tpulse) & (t <= self.tcycle),
                          (t > self.tcycle) & (t <= self.tpulse + self.tcycle),
                          (t > self.tpulse + self.tcycle) & (t <= self.tcycle*2.0),
                          (t > self.tcycle*2.0) & (t <= self.tcycle*2.0+self.tpulse),
                          (t > self.tcycle*2.0+self.tpulse) & (t <= self.tcycle*3.0),
                          (t > self.tcycle*3.0) & (t <= self.tcycle*3.0+self.tpulse),
                          t>self.tcycle*3.0 + self.tpulse],
                         [0,
                          lambda t: np.sin(np.pi*t/self.tpulse),
                          0,
                          lambda t: -np.sin(np.pi*(t-self.tcycle)/self.tpulse),
                          0,
                          lambda t: np.sin(np.pi*(t-self.tcycle*2.0)/self.tpulse),
                          0,
                          lambda t: -np.sin(np.pi*(t-self.tcycle*3.0)/self.tpulse),
                          0])

#Measure with condition 1 for four cycles before switching to condition 2
class rabiSelfComparison3(pulse_sequence):
    def __init__(self, tpulse, tdead):
        self.tpulse = tpulse
        self.tdead = tdead
        self.ttotal = 8.0*(tpulse + tdead)
        self.tcycle = tpulse + tdead
        self.omega_0 = np.pi/self.tpulse #T_pulse = pi/Omega_0  pi pulse time
        self.delta = 2.509/np.pi*self.omega_0 #Detuning with 0.5 excitation
        self.omega = np.sqrt(self.omega_0**2+self.delta**2)
        self.average_zero = False
    
    #Use extended sensitivity function for eight measurement cycles.  This assumes that
    #we switch between lock 1 and lock 2 after 4 measurements.
    def sensitivity_function(self, t):
        return 0.78*(np.pi)*(self.tpulse+self.tdead)/self.tpulse \
               * np.piecewise(t,
                         [t <= 0,
                          (t > 0) & (t <= self.tpulse),
                          (t > self.tpulse) & (t <= self.tcycle*2.0),
                          (t > self.tcycle*1.0) & (t <= self.tpulse + self.tcycle*1.0),
                          (t > self.tpulse + self.tcycle*1.0) & (t <= self.tcycle*2.0),
                          (t > self.tcycle*2.0) & (t <= self.tcycle*2.0 + self.tpulse),
                          (t > self.tcycle*2.0 + self.tpulse) & (t <= self.tcycle*3.0),
                          (t > self.tcycle*3.0) & (t <= self.tcycle*3.0 + self.tpulse),
                          (t > self.tcycle*3.0 + self.tpulse) & (t<= self.tcycle*4.0),
                          (t > self.tcycle*4.0) & (t <= self.tcycle*4.0 + self.tpulse),
                          (t > self.tcycle*4.0 + self.tpulse) & (t<= self.tcycle*5.0),
                          (t > self.tcycle*5.0) & (t <= self.tcycle*5.0 + self.tpulse),
                          (t > self.tcycle*5.0 + self.tpulse) & (t<= self.tcycle*6.0),
                          (t > self.tcycle*6.0) & (t <= self.tcycle*6.0 + self.tpulse),
                          (t > self.tcycle*6.0 + self.tpulse) & (t<= self.tcycle*7.0),
                          (t > self.tcycle*7.0) & (t <= self.tcycle*7.0 + self.tpulse),
                          t > self.tcycle*7.0 + self.tpulse],
                         [0,
                          lambda t: np.sin(np.pi*t/self.tpulse),
                          0,
                          lambda t: np.sin(np.pi*(t-self.tcycle*1.0)/self.tpulse),
                          0,
                          lambda t: np.sin(np.pi*(t-self.tcycle*2.0)/self.tpulse),
                          0,
                          lambda t: np.sin(np.pi*(t-self.tcycle*3.0)/self.tpulse),
                          0,
                          lambda t: -np.sin(np.pi*(t-self.tcycle*4.0)/self.tpulse),
                          0,
                          lambda t: -np.sin(np.pi*(t-self.tcycle*5.0)/self.tpulse),
                          0,
                          lambda t: -np.sin(np.pi*(t-self.tcycle*6.0)/self.tpulse),
                          0,
                          lambda t: -np.sin(np.pi*(t-self.tcycle*7.0)/self.tpulse),
                          0])

def wRamsey(time,Tdark,Tdead,Tpi,offset):
    out = []
    for ti in time:
        t = (ti-offset)%(Tpi+Tdead+Tdark)
        if t<=(0.0):
            out.append(0.0)
        elif (t>0.0)&(t<=(Tpi/2.0)):            
            out.append(np.sin(np.pi*t/Tpi))
        elif (t>Tpi/2.0)&(t<=Tpi/2.0+Tdark):
            out.append(1.0)
        elif (t>Tpi/2.0+Tdark)&(t<=Tpi+Tdark):
            out.append(np.sin(np.pi/Tpi*(t-Tdark)))
        else:
            out.append(0.0)
    return np.array(out)

#NOTE: I switched this to include a finite pi pulse duration
# This low passes the sensitivity function resulting in a more accurate
# depenence on high frequency laser noise
class ramsey(pulse_sequence):
    
    def __init__(self, tdark, tdead, tpi=0.01):
        self.tdark = tdark
        self.tdead = tdead
        self.tpi = tpi
        self.ttotal = tdark + tdead + tpi #Total time for Ed's calc
        self.T_meas = tdark + tpi #Interrigation time for Eric's calc
        self.omega_0 = np.pi/tpi #Nonzero for Eric's calc
        self.average_zero = False
        
    def sensitivity_function(self, t):
        return wRamsey(t,self.tdark,self.tdead,self.tpi,0.0)

    def sensitivity_psd(self, num_points=1e3):
        # t = np.linspace(0, self.tpulse+self.tdead, num_points)
        t, y = self.sensitivity(num_points)
        f = np.fft.fftfreq(len(t), d=t[1]-t[0])
        fOneSided = f[f>=0]
        psd = np.abs(np.fft.fft(y)[f>=0])**2 / len(f)**2
        #HACK: Ensure that normalization = 1 at low frequencies
        falias,psd_norm = self.sensitivity_psd_one_freq(f=1.0/(self.ttotal*100.), num_points=1e4)
        return (fOneSided, psd/psd_norm)     

    def sensitivity_psd_dm(self,f):
        #assumes Max frequency is a multiple of the FFT bin width fBin
        #f = np.linspace(fBin,fMax,fMax/fBin) 
        #Use closed form expression from MJM Thesis Eqn (6.57) with n=0 
        fnp = np.array(f)
        tau = self.T_meas
        mag = self.omega_0**2/(np.pi*fnp*self.omega_0**2-4*(np.pi*fnp)**3)**2*(2*np.pi*fnp*np.cos(np.pi*fnp*tau)+self.omega_0*np.sin(np.pi*fnp*(tau-np.pi/self.omega_0)))**2*np.sin(np.pi/2*(1+2*fnp*tau))**2/np.cos(np.pi*fnp*tau)**2
        fnp=0.000001
        mag0 = self.omega_0**2/(np.pi*fnp*self.omega_0**2-4*(np.pi*fnp)**3)**2*(2*np.pi*fnp*np.cos(np.pi*fnp*tau)+self.omega_0*np.sin(np.pi*fnp*(tau-np.pi/self.omega_0)))**2*np.sin(np.pi/2*(1+2*fnp*tau))**2/np.cos(np.pi*fnp*tau)**2
        return mag/mag0

class ramseySelfComparison(pulse_sequence):
    def __init__(self, tdark, tdead,tpi):
        self.tdark = tdark
        self.tdead = tdead
        self.tpi = tpi
        self.ttotal = 4.0*(tdark + tdead+tpi)
        self.tcycle = tdark + tdead + tpi
        self.average_zero = False
    
    #Use extended sensitivity function for four measurement cycles.  This assumes that
    #we switch between lock 1 and lock 2 after two measurements.
    def sensitivity_function(self, t):
        #Note: Normalization is currently a fudge factor to give sensitivity = 1 at 0 Hz
        return 0.35*(wRamsey(t,self.tdark,self.tdead+self.tcycle*3.0,self.tpi,0.0) +\
                wRamsey(t,self.tdark,self.tdead+self.tcycle*3.0,self.tpi,self.tcycle) -\
                wRamsey(t,self.tdark,self.tdead+self.tcycle*3.0,self.tpi,self.tcycle*2.0) -\
                wRamsey(t,self.tdark,self.tdead+self.tcycle*3.0,self.tpi,self.tcycle*3.0))

#Sensitivity function for a single tweezer cycle
#Set Tprep=0 when we aren't preparing a new sample
def wTweezerSingle(time,Tdark,Tdead,Tpi,offset,Tprep=0.0):
    out = []
    for ti in time:
        t = (ti-offset)
        if t<=(Tprep):
            out.append(0.0)
        elif (t>Tprep)&(t<=(Tprep+Tpi/2.0)):            
            out.append(np.sin(np.pi*(t-Tprep)/Tpi))
        elif (t>Tprep+Tpi/2.0)&(t<=Tprep+Tpi/2.0+Tdark):
            out.append(1.0)
        elif (t>Tpi/2.0+Tdark+Tprep)&(t<=Tpi+Tdark+Tprep):
            out.append(np.sin(np.pi/Tpi*(t-Tdark-Tprep)))
        else:
            out.append(0.0)
    return np.array(out)

#Sensitivity function for entire tweezer sequence
def wTweezer(time,Tdark,Tdead,Tpi,Tprep,n=1):
    tCycle = Tdark+Tdead+Tpi
    out = wTweezerSingle(time,Tdark,Tdead,Tpi,0.0,Tprep) #First cycle with prep time included
    for m in range(1,n): #Add in the short cycles
        out = out + wTweezerSingle(time,Tdark,Tdead,Tpi,offset=Tprep+m*tCycle,Tprep=0.0)
    return np.array(out)

class ramseyTweezer(pulse_sequence):
    
    def __init__(self, tdark, tdead, tprep, n=1, tpi=0.030):
        self.tdark = tdark
        self.tdead = tdead
        self.tprep = tprep
        self.tpi = tpi
        self.n = n #Number of tweezer cycles before reloading atoms (n>=1)
        self.ttotal = tprep + n*(tdark + tdead + tpi)
        self.average_zero = False
        
    def sensitivity_function(self, t):
        return wTweezer(t,self.tdark,self.tdead,self.tpi,self.tprep,self.n)

    def sensitivity_psd(self, num_points=1e3):
        # t = np.linspace(0, self.tpulse+self.tdead, num_points)
        t, y = self.sensitivity(num_points)
        f = np.fft.fftfreq(len(t), d=t[1]-t[0])
        fOneSided = f[f>=0]
        psd = np.abs(np.fft.fft(y)[f>=0])**2 / len(f)**2
        #HACK: Ensure that normalization = 1 at low frequencies
        falias,psd_norm = self.sensitivity_psd_one_freq(f=1.0/(self.ttotal*100.), num_points=1e4)
        return (fOneSided, psd/psd_norm)     
                          
class triangle(pulse_sequence):
    def __init__(self, tpulse, tdead):
        self.tpulse = tpulse
        self.tdead = tdead
        self.ttotal = tpulse + tdead
        self.average_zero = False
        
    def sensitivity_function(self, t):
        print([t <= 0,
                          (t > 0) & (t <= self.tpulse/2.),
                          (t > self.tpulse/2.) & (t <= self.tpulse),
                          t > self.tpulse]),
                          
        return ((self.tpulse+self.tdead)/self.tpulse) \
               * np.piecewise(t,
                         [t <= 0,
                          (t > 0) & (t <= self.tpulse/2.),
                          (t > self.tpulse/2.) & (t <= self.tpulse),
                          t > self.tpulse],
                         [0,
                          t/(self.tpulse/2.),
                          2.-t/(self.tpulse/2.),
                          0])

class spin_echo(pulse_sequence):
    def __init__(self, tdark, tdead, tpi=0.001, num_echos=1):
        self.tdark = tdark
        self.tdead = tdead
        self.num_echos = num_echos
        self.tpi = tpi
        self.ttotal = (1+num_echos)*tdark + tdead #Total time for Ed's calc
        self.T_meas = (1+num_echos)*tdark + num_echos*tpi #Interrogation time for Eric's calc
        self.omega_0 = np.pi/tpi #Nonzero for Eric's calc
        self.average_zero = True

    
#    def __init__(self, tpulse, tdead, num_echos=1):
#        
#        self.tpulse = tpulse
#        self.tdead = tdead
#        self.num_echos = num_echos
#        self.ttotal = (1+num_echos)*tpulse + tdead
#        self.average_zero = True
        
    def sensitivity_function(self, t):
        conditions = [ t <= 0 ] \
                     + [ (t > i*self.tdark) & (t <= (i+1)*self.tdark) \
                         for i in range(1+self.num_echos) ] \
                     + [ t > (1+self.num_echos)*self.tdark ]
        results = [ 0 ] \
                   + [ (-1)**i \
                         for i in range(1+self.num_echos) ] \
                   + [ 0 ]
                    
                     
        return ((self.tdark+self.tdead)/self.tdark)*np.piecewise(t, conditions, results)

    def sensitivity_psd_dm(self,f):
        #assumes Max frequency is a multiple of the FFT bin width fBin
        f = np.array(f) 
        #Use closed form expression from MJM Thesis Eqn (6.57)
        tau = self.T_meas/(1.0+self.num_echos)
        mag = self.omega_0**2/(np.pi*f*self.omega_0**2-4*(np.pi*f)**3)**2*(2*np.pi*f*np.cos(np.pi*f*tau)+self.omega_0*np.sin(np.pi*f*(tau-np.pi/self.omega_0)))**2*np.sin(np.pi/2*(1.0+self.num_echos)*(1+2*f*tau))**2/np.cos(np.pi*f*tau)**2 
        return mag#/mag[0]    


##### ATOMS
    
class atoms(object):
    
    def __init__(self,
                 pulse_sequence,
                 laser=mjm(),
                 f0=fSr):
        self.pulse_sequence = pulse_sequence
        self.laser = laser
        self.f0 = f0

    def allan_dick_effect(self, num_points=1e4):
        ''' Allan deviation at one second (not at the duty cycle) '''
        f, g2 = self.pulse_sequence.sensitivity_psd(num_points)
        return np.sqrt( np.sum( g2[1:] \
                        * self.laser.psd(f[1:]) ) ) / self.f0

    #Using Eqn 4.15 from Travis Nicholson's thesis
    def allan_dick_effect_async(self, num_points = 1e4, t_offset = 0.0):
        f, g2 = self.pulse_sequence.sensitivity_psd(num_points)
        k = np.array([n+1 for n in range(len(g2)-1)]) #indicies for summation
        return np.sqrt(4*np.sum(g2[1:]*np.sin(np.pi*k*t_offset/self.pulse_sequence.ttotal)**2 \
                               * self.laser.psd(f[1:]) ) ) / self.f0 / np.sqrt(2)


    def allan_qpn(self, N = 1000):
        ''' Allan deviation at one second (not at the duty cycle) '''
        if isinstance(self.pulse_sequence,rabi):
            adev_1s = 0.80/self.pulse_sequence.tpulse/(3.03*self.f0*np.sqrt(N))*np.sqrt(self.pulse_sequence.ttotal)
        elif isinstance(self.pulse_sequence,rabiSelfComparison):
            adev_1s = 0.80/self.pulse_sequence.tpulse/(3.03*self.f0*np.sqrt(N))*np.sqrt(self.pulse_sequence.tcycle)
        elif isinstance(self.pulse_sequence,rabiSelfComparison2):
            adev_1s = 0.80/self.pulse_sequence.tpulse/(3.03*self.f0*np.sqrt(N))*np.sqrt(self.pulse_sequence.tcycle)
        elif isinstance(self.pulse_sequence,rabiSelfComparison3):
            adev_1s = 0.80/self.pulse_sequence.tpulse/(3.03*self.f0*np.sqrt(N))*np.sqrt(self.pulse_sequence.tcycle)
        elif isinstance(self.pulse_sequence,ramsey):
            adev_1s = 1.0/(2.0*self.pulse_sequence.tdark)/(np.pi*self.f0*np.sqrt(N))*np.sqrt(self.pulse_sequence.ttotal)
        elif isinstance(self.pulse_sequence,ramseySelfComparison):
            adev_1s = 1.0/(2.0*self.pulse_sequence.tdark)/(np.pi*self.f0*np.sqrt(N))*np.sqrt(self.pulse_sequence.ttotal)
        else:
            adev_1s=0.0
            print('QPN calc is undefined for this pulse sequence.  Setting adev=0\n')
        return adev_1s
    
    # def allan_dick_effect_echo(self, num_points=1e4):
    #     f, g2 = self.pulse_sequence.sensitivity_psd(num_points)
    #     return np.sqrt( np.sum( ( g2[1:] ) \
    #                     * self.laser.psd(f[1:]) ) )
    
    def psd_dick_effect(self, f, num_points=1e4):
        '''Units are Hz^2/Hz'''
        adev = self.allan_dick_effect(num_points) * self.f0
        psd = 2*adev**2 + np.zeros_like(f)
        if self.pulse_sequence.average_zero == False:
            tcycle = self.pulse_sequence.ttotal
			#What is this??? Ask Ed later
#            psd += adev**2 / (2*np.log(2) * f * 2*tcycle)
        return psd

    def psd_qpn(self, f, N = 1000):
        ''' Units are Hz^2/Hz '''
        adev = self.allan_qpn(N) * self.f0
        psd = 2*adev**2 + np.zeros_like(f)
        return psd
    
    def frequency_record(self, ttotal=1000., num_points=2**16, signal=None):
        '''num_points is not the actual number of points, but a large number
        used to generate the noise spectrum from the laser before it's 
        calculates for the atoms'''
        t, df = self.laser.generate_noise(ttotal=ttotal, num_points=num_points) 

        if signal != None:
            df += signal(t)
        
        atom_df = []
        atom_t = []
        tcycle = self.pulse_sequence.ttotal
        dt = t[1]-t[0]
        
        for i in range(np.floor(max(t)/tcycle).astype(int)):
            ind = (t >= i*tcycle) \
                  & (t < (i+1)*tcycle)
            g = self.pulse_sequence.sensitivity_function(
                    t[ind]-i*tcycle)
                
            # print np.sum(g)
            # print dt
            # print len(g)
            # atom_df.append(np.sum(g*df[ind])/np.sum(g))
            atom_t.append(min(t[ind]))
            atom_df.append(np.sum(g*df[ind])/len(g))
        return np.array(atom_t), np.array(atom_df)

    #New feature.  Generate timeseries data for clock stability using laser noise model
    #Similar to Ed's frequency_record except we generate data for two identical clocks
    #and take their difference divided by sqrt2 before applying the atomic response. 
    #Essentially I'm computing a frequency record for an asynchronous comparison between two identical clocks.
    def self_stability_record(self, ttotal=1000., num_points=2**16, signal=None):
        '''num_points is not the actual number of points, but a large number
        used to generate the noise spectrum from the laser before it's 
        calculates for the atoms'''
        
        #Start by generating laser noise data for two identical cavity+clock systems
        t1, df1 = self.laser.generate_noise(ttotal=ttotal, num_points=num_points) 
        t2, df2 = self.laser.generate_noise(ttotal=ttotal, num_points=num_points) 

        atom_df = []
        atom_t = []
        tcycle = self.pulse_sequence.ttotal
        dt = t1[1]-t1[0]
        
        #Compute atomic response to laser noise.
        for i in range(np.floor(max(t1)/tcycle).astype(int)):
            ind = (t1 >= i*tcycle) \
                  & (t1 < (i+1)*tcycle)
            g = self.pulse_sequence.sensitivity_function(
                    t1[ind]-i*tcycle)
                
            # print np.sum(g)
            # print dt
            # print len(g)
            # atom_df.append(np.sum(g*df[ind])/np.sum(g))
            atom_t.append(min(t1[ind]))
            atom_df.append(np.sum(g* (df1[ind]-df2[ind])/np.sqrt(2.0) )/len(g))
        return np.array(atom_t), np.array(atom_df)



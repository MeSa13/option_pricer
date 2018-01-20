from math import *
#import numpy as np
#from numpy import linspace
#import matplotlib.pyplot as plt  
import random
from scipy import special
import copy
    
    
#class asset:
#    """asset with a given price and volatility"""
#    def __init__(self,price,vol):
#        self.price=price
#        self.vol=vol
#    def __str__(self):
#        return "current asset price=%s and volatility=%s"% (self.price,self.vol)

def gauss_seed(lchain):
    z=[]
    for i in range(lchain):
        dz=random.gauss(0.0,1.0)
        z.append(dz)
    return z

class option:
    """option of a given asset with price s, volatility vol, strick price K, maturity time T and interest rate r"""
    def __init__(self,s,vol,K,T,r):
        self.asset=s
        self.vol=vol
        self.strickp=K
        self.strickt=T
        self.interest=r
        
    def __str__(self):
        return "Option of an asset with price %s, vol %s, strick price %s, maturity time %s year(s) and risk-free interest rate %s per year"%(self.asset,self.vol,self.strickp,self.strickt,self.interest)
    
    def EU_analytic(self,ty):
        """price of European options using Black-Scholes-Merton analytic
        formula. ty (can be a character or a list of characters) determines the option type.
        'c' for call and 'p' for put.
        if ty is one character, it returns the option price. if ty is a string,
        it returns a list of option prices."""
        s=self.asset
        k=self.strickp
        sigma=self.vol
        t=self.strickt
        r=self.interest
        d1=(log(s/k)+(r+sigma**2/2)*t)/(sigma*sqrt(t))
        d2=d1-sigma*sqrt(t)
        out=[]
        for char in ty:
            if char=="c": #call option
                c= s/2*(1+special.erf(d1/sqrt(2)))-k*exp(-r*t)/2*(1+special.erf(d2/sqrt(2)))
                out.append(c)
            elif char=="p": #put option
                p= k*exp(-r*t)/2*(1+special.erf(-d2/sqrt(2)))-s/2*(1+special.erf(-d1/sqrt(2)))
                out.append(p)
            else:
                out.append(None)
        if len(ty)>1:
            return out
        else:
            return out[0]
        
    def simulate(self,z):
        s=[self.asset]
        dt=self.strickt/len(z)
        for i in range(len(z)):
            coeff=self.interest*dt+self.vol*sqrt(dt)*z[i]
            s.append(s[i]*(1+coeff))
        return s

    def MC_1sample(self,ty,z):
        s=self.simulate(z)
        if ty == "c":
            return (exp(-self.interest*self.strickt)*max(s[len(s)-1] - self.strickp , 0))
        elif ty == "p":
            return (exp(-self.interest*self.strickt)*max(self.strickp - s[len(s)-1]  , 0))
    
    def MC(self,ty,sample=1000,lchain=100):
        """European option pricing using Monte Carlo method. ty determines the
        option type. 'c' for call and 'p' for put.
        sample determines the number of simulations. lchain determines
        the length of each simulated chain.
        if ty is a character, it returns (option price,Delta).
        if ty is a string, it returns a list of (option price,Delta)"""
        r=self.interest
        t=self.strickt
        dt=t/lchain
        out=[]
        for char in ty:
            if char in "cp":
                price_list=[]
                for j in range(sample):
                    z=gauss_seed(lchain)
                    s=self.simulate(z)
                    price=self.MC_1sample(char,z)
                    price_list.append(price)
                out.append(sum(price_list)/sample)
            else:
                out.append(None)
        if len(ty)>1:
            return out
        else:
            return out[0]
        
    def binomial(self,ty,lchain=10):
        """European call option pricing using binomial method. lchain determines
        the length of the binomial tree.
        ty determines the option type. if ty is a character, it returns the option price.
        if ty is a string, it returns a list of option prices."""
        if len(ty)>1: #to break down ty into its characters
            out=[]
            for char in ty:
                out.append(self.binomial(char,lchain))
            return out
        if not (ty in "cp"):
            return None
        if lchain==0 and ty=="c": #value at the final node
            return max(self.asset - self.strickp,0)
        if lchain==0 and ty=="p": #value at the final node
            return max(self.strickp - self.asset,0)
        dt=self.strickt/lchain
        a=exp(self.interest*dt) #Q: these numbers do not change.
        #Is there a way to carry them out and not calculate them every time?
        u=exp(self.vol*sqrt(dt))
        d=exp(-self.vol*sqrt(dt))
        p=(a-d)/(u-d)
        op_up=copy.deepcopy(self) #representing the option in up node
        op_down=copy.deepcopy(self) #representing the option in down node
        op_up.strickt -= dt
        op_down.strickt -= dt
        op_up.asset=u*self.asset
        op_down.asset=d*self.asset
        return (p*op_up.binomial(ty,lchain-1)+(1-p)*op_down.binomial(ty,lchain-1))/a

    def Delta_EU_analytic(self,ty,eps=10.0**-3):
        """Calculates the Delta of European option using the EU_analytic pricer.
        ty determines the option type."""
        temp_op=copy.deepcopy(self)
        temp_op.asset += eps
        out=[]
        for char in ty:
            if char in "cp":
                out.append((temp_op.EU_analytic(char) - self.EU_analytic(char))/eps)
            else:
                out.append(None)
        if len(ty)>1:
            return out
        else:
            return out[0]

    def Delta_MC(self,ty,sample=1000,lchain=100,eps=10.0**-3):
        r=self.interest
        t=self.strickt
        dt=t/lchain
        out=[]
        for char in ty:
            if char in "cp":
                Delta_list=[]
                for j in range(sample):
                    z=gauss_seed(lchain)
                    temp_op=copy.deepcopy(self)
                    temp_op.asset += eps
                    price1=self.MC_1sample(char,z)
                    price2=temp_op.MC_1sample(char,z)
                    Delta_list.append((price2 - price1)/eps)
                out.append(sum(Delta_list)/sample)
            else:
                out.append(None)
        if len(ty)>1:
            return out
        else:
            return out[0]
       

    
op=option(100.0,0.3,100.0,1.0,0.1)

print op.EU_analytic("cpd")
print op.MC("cpd")
print op.binomial("cpd")
print op.Delta_EU_analytic("cpd")
print op.Delta_MC("cpf")

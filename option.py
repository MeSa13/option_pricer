from math import *
#import numpy as np
#from numpy import linspace
#import matplotlib.pyplot as plt  
import random
from scipy import special
import copy


    
    
class asset:
    """asset with a given price and volatility"""
    def __init__(self,price,vol):
        self.price=price
        self.vol=vol
    def __str__(self):
        return "current asset price=%s and volatility=%s"% (self.price,self.vol)


class option:
    """option of a given stock with strick price K, maturity time T and interest rate r"""
    def __init__(self,stock,K,T,r):
        self.stock=stock
        self.strickp=K
        self.strickt=T
        self.interest=r
    def __str__(self):
        return "Option of an asset [%s] with strick price %s, maturity time %s year(s) and risk-free interest rate %s per year"%(self.stock,self.strickp,self.strickt,self.interest)
    def EU_analytic(self,ty):
        """price of European options using Black-Scholes-Merton analytic
        formula. ty (can be a character or a list of characters) determines the option type.
        'c' for call and 'p' for put.
        if ty is one character, it returns the option price. if ty is a string,
        it returns a list of option prices."""
        s=self.stock.price
        k=self.strickp
        sigma=self.stock.vol
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
        
    def EU_MC(self,ty,sample=1000,lchain=100,eps=10.0**-3):
        """European option pricing using Monte Carlo method. ty determines the
        option type. 'c' for call and 'p' for put.
        sample determines the number of simulations. lchain determines
        the length of each simulated chain.
        if ty is a character, it returns (option price,Delta).
        if ty is a string, it returns a list of (option price,Delta)"""
        c=[]
        p=[]
        c_eps=[]
        p_eps=[]
        s=self.stock.price
        s_eps=s+eps
        k=self.strickp
        sigma=self.stock.vol
        t=self.strickt
        r=self.interest
        dt=t/lchain
        for j in range(sample):
            ss=s
            ss_eps=s_eps
            for i in range(lchain):
                dz=random.gauss(0.0,1.0)
                coeff=r*dt+sigma*sqrt(dt)*dz
                ss=ss*(1+coeff)
                ss_eps=ss_eps*(1+coeff)
            c.append(exp(-r*t)*max(ss-k,0))
            p.append(exp(-r*t)*max(k-ss,0))
            c_eps.append(exp(-r*t)*max(ss_eps-k,0))
            p_eps.append(exp(-r*t)*max(k-ss_eps,0))
        c_price=sum(c)/sample
        p_price=sum(p)/sample
        c_eps_price=sum(c_eps)/sample
        p_eps_price=sum(p_eps)/sample
        out=[]
        for char in ty:
            if char=="c":
                out.append((c_price,(c_eps_price - c_price)/eps))
            elif char=="p":
                out.append((p_price,(p_eps_price - p_price)/eps))               
            else:
                out.append((None,None))
        if len(ty)>1:
            return out
        else:
            return out[0]
    def binomial(self,ty,lchain=10):
        """European call option pricing using binomial method. lchain determines
        the length of the binomial tree.
        ty determines the option type. if ty is a character, it returns the option price.
        if ty is a string, it returns a list of option prices."""
        if len(ty)>1:
            out=[]
            for char in ty:
                out.append(self.binomial(char,lchain))
            return out
        if not (ty in "cp"):
            return None
        if lchain==0 and ty=="c":            
            return max(self.stock.price-self.strickp,0)
        if lchain==0 and ty=="p":
            return max(self.strickp-self.stock.price,0)
        dt=self.strickt/lchain
        a=exp(self.interest*dt)
        u=exp(self.stock.vol*sqrt(dt))
        d=exp(-self.stock.vol*sqrt(dt))
        p=(a-d)/(u-d)
        tempopu=copy.deepcopy(self)
        tempopd=copy.deepcopy(self)
        tempopu.strickt-=dt
        tempopd.strickt-=dt
        tempopu.stock.price=u*self.stock.price
        tempopd.stock.price=d*self.stock.price
        return (p*tempopu.binomial(ty,lchain-1)+(1-p)*tempopd.binomial(ty,lchain-1))/a



    def Delta_EUput_analytic(self,eps=10.0**-3):
        temp_op=copy.deepcopy(self)
        temp_op.stock.price+=eps
        return (temp_op.EUput_analytic() - self.EUput_analytic())/eps

    def Delta_EUcall_analytic(self,eps=10.0**-3):
        temp_op=copy.deepcopy(self)
        temp_op.stock.price+=eps
        return (temp_op.EUcall_analytic() - self.EUcall_analytic())/eps    
    
google=asset(100.0,0.3)
op=option(google,100.0,1.0,0.1)
MC=op.EU_MC("p")

print op.EU_analytic("c")
print MC
print op.binomial("pcd")


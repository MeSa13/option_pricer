from math import *
#import numpy as np
#from numpy import linspace
#import matplotlib.pyplot as plt  
import random
from scipy import special
import copy


class Delta:
    def __init__(self,op,eps=10.0**-3):
        self.option=op
        self.eps=eps
    def EUcall_analytic(self):
        op_eps=copy.deepcopy(self.option)
        op_eps.stock.price+=Delta.eps
        return (self.option.EUcall_analytic)
    

    
    
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
        return "Option of a stock [%s] with strick price %s, maturity time %s year(s) and risk-free interest rate %s per year"%(self.stock,self.strickp,self.strickt,self.interest)
    def EUcall_analytic(self):
        """price of a European call option using Black-Scholes-Merton analytic formula and its Delta"""
        s=self.stock.price
        k=self.strickp
        sigma=self.stock.vol
        t=self.strickt
        r=self.interest
        d1=(log(s/k)+(r+sigma**2/2)*t)/(sigma*sqrt(t))
        d2=d1-sigma*sqrt(t)
        c= s/2*(1+special.erf(d1/sqrt(2)))-k*exp(-r*t)/2*(1+special.erf(d2/sqrt(2)))
        return c
    def EUput_analytic(self):
        """price of a European put option using Black-Scholes-Merton analytic formula"""
        s=self.stock.price
        k=self.strickp
        sigma=self.stock.vol
        t=self.strickt
        r=self.interest
        d1=(log(s/k)+(r+sigma**2/2)*t)/(sigma*sqrt(t))
        d2=d1-sigma*sqrt(t)
        p= k*exp(-r*t)/2*(1+special.erf(-d2/sqrt(2)))-s/2*(1+special.erf(-d1/sqrt(2)))
        return p
    def EU_MC(self,sample=1000,lchain=100,eps=10.0**-3):
        """European call and put option pricing using Monte Carlo method.
        sample determines the number of simulations. lchain determines
        the length of each simulated chain. The return value is (call,put) price."""
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
        return sum(c)/sample,sum(p)/sample,(sum(c_eps)-sum(c))/(eps*sample),(sum(p_eps)-sum(p))/(eps*sample)
    def EUcall_binom(self,lchain=10):
        """European call option pricing using binomial method. lchain determines
        the length of the binomial tree."""
        if lchain==0:
            return max(self.stock.price-self.strickp,0)
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
        return (p*tempopu.EUcall_binom(lchain-1)+(1-p)*tempopd.EUcall_binom(lchain-1))/a
    def EUput_binom(self,lchain=10):
        """European put option pricing using binomial method. lchain determines
        the length of the binomial tree."""
        if lchain==0:
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
        return (p*tempopu.EUput_binom(lchain-1)+(1-p)*tempopd.EUput_binom(lchain-1))/a
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
MC=op.EU_MC(sample=10000,lchain=1000)

print op.EUcall_analytic(),op.EUput_analytic(), op.Delta_EUcall_analytic(),op.Delta_EUput_analytic()
print MC
print op.EUcall_binom(),op.EUput_binom()

print op.EUcall_analytic()+op.strickp*exp(-op.interest*op.strickt)-google.price-op.EUput_analytic()
print op.EUcall_binom()+op.strickp*exp(-op.interest*op.strickt)-google.price-op.EUput_binom()
print MC[0]+op.strickp*exp(-op.interest*op.strickt)-google.price-MC[1]



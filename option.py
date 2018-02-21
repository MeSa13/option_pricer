# from numpy import linspace
# import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import time
from abc import ABC, abstractmethod


class Asset:
    """asset with a given price and volatility"""

    def __init__(self, price, vol):
        self.price = price
        self.vol = vol

    def __str__(self):
        return "current asset price=%s and volatility=%s" % (self.price, self.vol)

    def simulate(self, duration, growth_rate, number_simulations, steps):
        wiener_process = np.random.normal(0.0, 1.0, (number_simulations, steps))
        dt = duration / steps
        movement_coefficient = 1 + growth_rate * dt + self.vol * np.sqrt(dt) * wiener_process
        simulated_price = np.ones((number_simulations, 1)) * self.price
        for i in range(steps):
            new_price = simulated_price[:, i] * movement_coefficient[:, i]
            new_price.shape = (number_simulations, 1)
            simulated_price = np.append(simulated_price, new_price, axis=1)
        return simulated_price


class Option(ABC):
    """option of a given asset, strike price, maturity time and interest rate"""

    def __init__(self, option_class, option_type, asset, strike, term, interest_rate, pricing_method):
        self.option_class = option_class
        assert self.option_class in ["Eu", "American"]
        self.option_type = option_type
        assert self.option_type in ["Call", "Put"]
        self.asset = asset
        self.strike = strike
        self.term = term
        self.interest_rate = interest_rate
        self.pricing_method = pricing_method
        assert self.pricing_method in ["Binomial", "MonteCarlo", "Analytic"]

    def __str__(self):
        return "Option of an asset with price %s, vol %s, strike price %s, maturity time %s year(s) and risk-free " \
               "interest rate %s per year" % (
                   self.asset.price, self.asset.vol, self.strike, self.term, self.interest_rate)

    def payoff(self, s):
        if self.option_type is "Put":
            return np.maximum(self.strike - s, 0)
        elif self.option_type is "Call":
            return np.maximum(s - self.strike, 0)

    @abstractmethod
    def binomial(self, step):
        pass

    @abstractmethod
    def monte_carlo(self, number_simulations, steps):
        pass

    @abstractmethod
    def analytic(self):
        pass

    def price(self, *args):
        if self.pricing_method is "Binomial":
            return self.binomial(*args)
        elif self.pricing_method is "MonteCarlo":
            return self.monte_carlo(*args)
        elif self.pricing_method is "Analytic":
            return self.analytic()

    def delta(self, *args, eps=0.01): #QQ: first *args or eps?
        st = np.random.get_state()
        price1 = self.price(*args)
        self.asset.price += eps
        np.random.set_state(st)
        price2 = self.price(*args)
        self.asset.price -= eps
        return (price2 - price1) / eps

    def gamma(self, *args, eps=1.0):
        st = np.random.get_state()
        price1 = self.price(*args)
        np.random.set_state(st)
        self.asset.price += eps
        price2 = self.price(*args)
        np.random.set_state(st)
        self.asset.price -= 2 * eps
        price3 = self.price(*args)
        self.asset.price += eps
        return (price2 + price3 - 2*price1) / eps ** 2

    def vega(self, *args, eps=0.01):
        st = np.random.get_state()
        price1 = self.price(*args)
        self.asset.vol += eps
        np.random.set_state(st)
        price2 = self.price(*args)
        self.asset.vol -= eps
        return 0.01 * (price2 - price1) / eps

    def rho(self, *args, eps=0.01):
        st = np.random.get_state()
        price1 = self.price(*args)
        self.interest_rate += eps
        np.random.set_state(st)
        price2 = self.price(*args)
        self.interest_rate -= eps
        return 0.01 * (price2 - price1) / eps

    def theta(self, *args, eps=0.001):
        st = np.random.get_state()
        price1 = self.price(*args)
        np.random.set_state(st)
        self.term += eps
        price2 = self.price(*args)
        self.term -= eps
        return (price2 - price1) / (365*eps)


class EuOption(Option):
    """something here!"""

    def __init__(self, option_type, asset, strike, term, interest_rate, pricing_method):
        Option.__init__(self, "Eu", option_type, asset, strike, term, interest_rate, pricing_method)

    def analytic(self):
        """price of European options using Black-Scholes-Merton analytic formula."""

        d1 = (np.log(self.asset.price / self.strike)
              + (self.interest_rate + self.asset.vol ** 2 / 2) * self.term) \
             / (self.asset.vol * np.sqrt(self.term))
        d2 = d1 - self.asset.vol * np.sqrt(self.term)
        if self.option_type is "Put":
            price = self.strike * np.exp(-self.interest_rate * self.term) * scipy.stats.norm.cdf(-d2) \
                    - self.asset.price * scipy.stats.norm.cdf(-d1)
        elif self.option_type is "Call":
            price = - self.strike * np.exp(-self.interest_rate * self.term) * scipy.stats.norm.cdf(d2) \
                    + self.asset.price * scipy.stats.norm.cdf(d1)
        return price

    def binomial(self, step=1000):
        dt = 1.0 * self.term / step
        a = np.exp(self.interest_rate * dt)
        u = np.exp(self.asset.vol * np.sqrt(dt))
        d = np.exp(-self.asset.vol * np.sqrt(dt))
        p = (a - d) / (u - d)
        option_price = np.zeros((step+1, step+1))
        for up_mov in range(step + 1):
            option_price[up_mov, step - up_mov] = self.payoff(u ** up_mov * d ** (step - up_mov) * self.asset.price)
        for layer in range(step - 1, -1, -1):
            for up_mov in range(layer + 1):
                down_mov = layer - up_mov
                option_price[up_mov, down_mov] = (p * option_price[up_mov + 1, down_mov] \
                                                  + (1 - p) * option_price[up_mov, down_mov + 1]) / a
        return option_price[0, 0]

    def monte_carlo(self, number_simulation=10000, steps=100):
        """option pricing using Monte Carlo method.
        sample: number of simulations
        steps: length of each simulated chain"""
        simulations = self.asset.simulate(self.term, self.interest_rate, number_simulation, steps)
        asset_price_list = simulations[:, -1]
        payoff_list = self.payoff(asset_price_list)
        price = np.exp(-self.interest_rate * self.term) * np.mean(payoff_list)
        return price


class AmericanOption(Option):

    def __init__(self, option_type, asset, strike, term, interest_rate):
        Option.__init__(self, "American", option_type, asset, strike, term, interest_rate, "Binomial")

    def analytic(self):
        raise AttributeError("'AmericanOption' has no attribute 'analytic'")

    def monte_carlo(self, number_simulations, steps):
        raise AttributeError("'AmericanOption' has no attribute 'monte_carlo'")

    def binomial(self, step=1000):
        dt = 1.0 * self.term / step
        a = np.exp(self.interest_rate * dt)
        u = np.exp(self.asset.vol * np.sqrt(dt))
        d = np.exp(-self.asset.vol * np.sqrt(dt))
        p = (a - d) / (u - d)
        option_price = np.zeros((step + 1, step + 1))
        for up_mov in range(step + 1):
            option_price[up_mov, step - up_mov] = self.payoff(u ** up_mov * d ** (step - up_mov) * self.asset.price)
        for layer in range(step - 1, -1, -1):
            for up_mov in range(layer + 1):
                down_mov = layer - up_mov
                option_price[up_mov, down_mov] = (p * option_price[up_mov + 1, down_mov] \
                                                  + (1 - p) * option_price[up_mov, down_mov + 1]) / a
                current_payoff = self.payoff(u ** up_mov * d ** down_mov * self.asset.price)
                option_price[up_mov, down_mov] = max(current_payoff, option_price[up_mov, down_mov])
        return option_price[0, 0]


example_asset_price = 100.0
example_strike = 100.0
example_vol = 0.3
example_term = 1.0
example_interest = 0.1
stock = Asset(example_asset_price, example_vol)
eu_call_analytic = EuOption("Call", stock, example_strike, example_term, example_interest, "Analytic")
eu_call_binomial = EuOption("Call", stock, example_strike, example_term, example_interest, "Binomial")
eu_call_MC = EuOption("Call", stock, example_strike, example_term, example_interest, "MonteCarlo")
eu_put_analytic = EuOption("Put", stock, example_strike, example_term, example_interest, "Analytic")
eu_put_binomial = EuOption("Put", stock, example_strike, example_term, example_interest, "Binomial")
eu_put_MC = EuOption("Put", stock, example_strike, example_term, example_interest, "MonteCarlo")
american_call = AmericanOption("Call", stock, example_strike, example_term, example_interest)
american_put = AmericanOption("Put", stock, example_strike, example_term, example_interest)

print("option price")
print(eu_call_analytic.price(), eu_put_analytic.price())
print(eu_call_binomial.price(), eu_put_binomial.price())
print(eu_call_MC.price(), eu_put_MC.price())
print(american_call.price(), american_put.price(), "\n")

print("delta")
print(eu_call_analytic.delta(), eu_put_analytic.delta())
print(eu_call_binomial.delta(), eu_put_binomial.delta())
print(eu_call_MC.delta(), eu_put_MC.delta(), "\n")

print("gamma")
print(eu_call_analytic.gamma(), eu_put_analytic.gamma())
print(eu_call_binomial.gamma(), eu_put_binomial.gamma())
print(eu_call_MC.gamma(), eu_put_MC.gamma(), "\n")

print("vega")
print(eu_call_analytic.vega(), eu_put_analytic.vega())
print(eu_call_binomial.vega(), eu_put_binomial.vega())
print(eu_call_MC.vega(), eu_put_MC.vega(), "\n")

print("rho")
print(eu_call_analytic.rho(), eu_put_analytic.rho())
print(eu_call_binomial.rho(), eu_put_binomial.rho())
print(eu_call_MC.rho(), eu_put_MC.rho(), "\n")

print("theta")
print(eu_call_analytic.theta(), eu_put_analytic.theta())
print(eu_call_binomial.theta(), eu_put_binomial.theta())
print(eu_call_MC.theta(), eu_put_MC.theta(), "\n")

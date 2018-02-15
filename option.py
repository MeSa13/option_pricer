# from numpy import linspace
# import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import copy


class Asset:
    """asset with a given price and volatility"""

    def __init__(self, price, vol):
        self.price = price
        self.vol = vol

    def __str__(self):
        return "current asset price=%s and volatility=%s" % (self.price, self.vol)

    def simulate(self, time, interest_rate, number_simulation, steps):
        wiener_process = np.random.normal(0.0, 1.0, (number_simulation, steps))
        dt = time / steps
        movement_coefficient = 1 + interest_rate * dt + self.vol * np.sqrt(dt) * wiener_process
        simulated_price = np.ones((number_simulation, 1)) * self.price
        for i in range(steps):
            new_price = simulated_price[:, i] * movement_coefficient[:, i]
            new_price.shape = (number_simulation, 1)
            simulated_price = np.append(simulated_price, new_price, axis=1)
        return simulated_price


class Option:
    """option of a given asset, strike price, maturity time and interest rate"""

    def __init__(self, asset, strike_price, maturity_time, interest_rate):
        self.asset = asset
        self.strike_price = strike_price
        self.maturity_time = maturity_time
        self.interest_rate = interest_rate

    def __str__(self):
        return "Option of an asset with price %s, vol %s, strike price %s, maturity time %s year(s) and risk-free " \
               "interest rate %s per year" % (
                   self.asset.price, self.asset.vol, self.strike_price, self.maturity_time, self.interest_rate)

    def binomial(self, payoff, step=10):
        """option pricing using binomial method. step determines
        the length of the binomial tree."""
        if step == 0:  # value at the final node
            return payoff(self.asset.price)
        dt = self.maturity_time / step
        a = np.exp(self.interest_rate * dt)  # Q: these numbers do not change.
        # Is there a way to carry them and not calculate them every time?
        u = np.exp(self.asset.vol * np.sqrt(dt))
        d = np.exp(-self.asset.vol * np.sqrt(dt))
        p = (a - d) / (u - d)
        u_option = copy.deepcopy(self)  # representing the option in up node
        d_option = copy.deepcopy(self)  # representing the option in down node
        # does not work with copy.copy
        u_option.maturity_time -= dt
        d_option.maturity_time -= dt
        u_option.asset.price = u * self.asset.price
        d_option.asset.price = d * self.asset.price
        return (p * u_option.binomial(payoff, step - 1) + (1 - p) * d_option.binomial(payoff, step - 1)) / a
        # QQ: this is really slow even when step = 20!!

    def monte_carlo(self, payoff, number_simulation=1000, steps=100):
        """option pricing using Monte Carlo method.
        sample: number of simulations
        steps: length of each simulated chain"""
        simulations = self.asset.simulate(self.maturity_time, self.interest_rate, number_simulation, steps)
        asset_price = simulations[:, -1]
        payoff_list = payoff(asset_price)
        price = np.exp(-self.interest_rate * self.maturity_time) * np.mean(payoff_list)
        return price

    def delta_generic(self, pricer, payoff, eps, *args):
        st = np.random.get_state()
        price1 = pricer(self, payoff, *args)
        self.asset.price += eps
        np.random.set_state(st)
        price2 = pricer(self, payoff, *args)
        self.asset.price -= eps
        return (price2 - price1) / eps
    # it works, but I think it's very confusing!

    def gamma_generic(self, pricer, payoff, eps, *args):
        st = np.random.get_state()
        price1 = pricer(self, payoff, *args)
        self.asset.price += eps
        np.random.set_state(st)
        price2 = pricer(self, payoff, *args)
        self.asset.price -= 2*eps
        np.random.set_state(st)
        price3 = pricer(self, payoff, *args)
        self.asset.price += eps
        return (price2 + price3 - 2*price1) / eps ** 2

    def vega_generic(self, pricer, payoff, eps, *args):
        st = np.random.get_state()
        price1 = pricer(self, payoff, *args)
        self.asset.vol += eps
        np.random.set_state(st)
        price2 = pricer(self, payoff, *args)
        self.asset.vol -= eps
        return 0.01 * (price2 - price1) / eps

    def rho_generic(self, pricer, payoff, eps, *args):
        st = np.random.get_state()
        price1 = pricer(self, payoff, *args)
        self.interest_rate += eps
        np.random.set_state(st)
        price2 = pricer(self, payoff, *args)
        self.interest_rate -= eps
        return 0.01 * (price2 - price1) / eps


class PutOption(Option):
    """something here!"""

    def analytic_price(self):
        """price of European options using Black-Scholes-Merton analytic formula."""
        d1 = (np.log(self.asset.price / self.strike_price)
              + (self.interest_rate + self.asset.vol ** 2 / 2) * self.maturity_time) \
             / (self.asset.vol * np.sqrt(self.maturity_time))
        d2 = d1 - self.asset.vol * np.sqrt(self.maturity_time)
        price = self.strike_price * np.exp(-self.interest_rate * self.maturity_time) * scipy.stats.norm.cdf(-d2) \
                - self.asset.price * scipy.stats.norm.cdf(-d1)
        return price

    def payoff(self):
        return lambda s: np.maximum(self.strike_price - s, 0)

    def binomial_price(self, step=10):
        """option pricing using binomial method. len_chain determines
        the length of the binomial tree."""
        return Option.binomial(self, self.payoff(), step)

    def monte_carlo_price(self, number_simulation=1000, steps=100):
        """option pricing using Monte Carlo method.
        number_simulation: number of simulations
        steps: length of each simulated chain"""
        return Option.monte_carlo(self, self.payoff(), number_simulation, steps)

    def delta(self, pricer, *args, eps=10 ** -3):
        return Option.delta_generic(self, pricer, self.payoff(), eps, *args)

    def gamma(self, pricer, *args, eps=10 ** -1):
        return Option.gamma_generic(self, pricer, self.payoff(), eps, *args)

    def vega(self, pricer, *args, eps=10 ** -3):
        return Option.vega_generic(self, pricer, self.payoff(), eps, *args)

    def rho(self, pricer, *args, eps=10 ** -3):
        return Option.rho_generic(self, pricer, self.payoff(), eps, *args)


class CallOption(Option):
    """something here!"""

    def analytic_price(self):
        """price of European options using Black-Scholes-Merton analytic formula."""
        d1 = (np.log(self.asset.price / self.strike_price)
              + (self.interest_rate + self.asset.vol ** 2 / 2) * self.maturity_time) / (
                     self.asset.vol * np.sqrt(self.maturity_time))
        d2 = d1 - self.asset.vol * np.sqrt(self.maturity_time)
        price = - self.strike_price * np.exp(-self.interest_rate * self.maturity_time) * scipy.stats.norm.cdf(d2) \
                + self.asset.price * scipy.stats.norm.cdf(d1)
        return price

    def payoff(self):
        return lambda s: np.maximum(s - self.strike_price, 0)

    def binomial_price(self, step=10):
        """option pricing using binomial method. step determines
        the length of the binomial tree."""
        return Option.binomial(self, self.payoff(), step)

    def monte_carlo_price(self, number_simulation=1000, steps=100):
        """option pricing using Monte Carlo method.
        number_simulation: number of simulations
        steps: length of each simulated chain"""
        return Option.monte_carlo(self, self.payoff(), number_simulation, steps)

    def delta(self, pricer, *args, eps=10 ** -3):
        return Option.delta_generic(self, pricer, self.payoff(), eps, *args)

    def gamma(self, pricer, *args, eps=10 ** -1):
        return Option.gamma_generic(self, pricer, self.payoff(), eps, *args)

    def vega(self, pricer, *args, eps=10 ** -3):
        return Option.vega_generic(self, pricer, self.payoff(), eps, *args)

    def rho(self, pricer, *args, eps=10 ** -3):
        return Option.rho_generic(self, pricer, self.payoff(), eps, *args)


stock = Asset(100.0, 0.3)
call = CallOption(stock, 100.0, 0.1, 0.1)
put = PutOption(stock, 100.0, 0.1, 0.1)

print(call.analytic_price(), put.analytic_price())
print(call.binomial_price(), put.binomial_price())
print(call.monte_carlo_price(), put.monte_carlo_price(), "\n")

print(call.delta(Option.binomial), put.delta(Option.binomial))
print(call.delta(Option.monte_carlo), put.delta(Option.monte_carlo), "\n")

print(call.gamma(Option.binomial), put.gamma(Option.binomial))
print(call.gamma(Option.monte_carlo, 10000), put.gamma(Option.monte_carlo, 10000), "\n")
# QQ: binomial gamma is way off

print(call.vega(Option.binomial), put.vega(Option.binomial))
print(call.vega(Option.monte_carlo), put.vega(Option.monte_carlo), "\n")

print(call.rho(Option.binomial), put.rho(Option.binomial))
print(call.rho(Option.monte_carlo), put.rho(Option.monte_carlo))

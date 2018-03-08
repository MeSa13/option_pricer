# from numpy import linspace
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import time
from abc import ABC, abstractmethod
import copy

# PDE and local volatility
# payoff()
# is monte_carlo error acceptable?


class Asset:
    """Make Asset class

    Attributes:
        price (float): asset price
        vol (function): local volatility of the asset.
    """

    def __init__(self, price, vol):
        """Initialize Asset"""

        self.price = price
        self.vol = vol
        # if a float is passed for vol (e.g. constant volatility) make it a callable object
        if not callable(vol):
            self.vol = lambda x, y: vol

    def simulate(self, duration, growth_rate, number_simulations, steps):
        """Simulate a geometric Brownian motion of the asset price

        Args:
            duration (float): time duration of the simulation
            growth_rate (float): growth rate of the geometric Brownian motion
            number_simulations (int): number of simulations
            steps (int): number of steps in each simulation
        Returns:
            sim_price (numpy.ndarray): an array of size number_simulations * (step+1)
        """
        normal_seeds = np.random.normal(0.0, 1.0, (number_simulations, steps))
        dt = duration / steps
        # Current asset price is the first column in sim_price
        sim_price = np.ones((number_simulations, 1)) * self.price
        vol = self.vol
        t = 0
        for i in range(steps):
            mov_coeff = 1.0 + growth_rate * dt + vol(sim_price[:, i], t) * np.sqrt(dt) * normal_seeds[:, i]
            new_price = sim_price[:, i] * mov_coeff
            # making new_price a column vector
            new_price.shape = (number_simulations, 1)
            sim_price = np.append(sim_price, new_price, axis=1)
            t += dt
        return sim_price

    def sim_plot(self, growth_rate, duration, time_step, num_sim=5):
        """Plot simulation of asset price

        Args:
            growth_rate (float): growth rate of the asset price, usually taken to be the free interest rate
            duration (float): duration of simulation
            time_step (float): time step of each simulation
            num_sim (int): number of simulations, defaults to 5
        """
        t = np.arange(0.0, duration, time_step)
        sim_price = self.simulate(duration, growth_rate, num_sim, len(t) - 1)
        plt.plot(t, np.transpose(sim_price))
        plt.show()


class Option(ABC):
    """Form an Option of a given Asset

    Attributes:
        option_type (str): "Call" or "Put"
        asset (Asset): underlying asset of the option
        strike (float): option strike price
        term (float): option maturity time
        interest_rate (float): risk free interest rate
        pricing method (str): the main pricing method for the option, "Binomial", "MonteCarlo" or "Analytic"
    """

    def __init__(self, option_type, asset, strike, term, interest_rate, pricing_method):
        """Initialize Option

        Args:
            option_type (str): "Call" or "Put"
            asset (Asset): underlying asset of the option
            strike (float): option strike price
            term (float): option maturity time
            interest_rate (float): risk free interest rate
            pricing method (str): the main pricing method for the option, "Binomial", "MonteCarlo" or "Analytic"
        """
        self.option_type = option_type
        assert self.option_type in ["Call", "Put"]
        self.asset = asset
        self.strike = strike
        self.term = term
        self.interest_rate = interest_rate
        self.pricing_method = pricing_method
        assert self.pricing_method in ["Binomial", "MonteCarlo", "Analytic"]

    def payoff(self, s):
        """Return Payoff"""
        if self.option_type == "Put":
            return np.maximum(self.strike - s, 0)
        elif self.option_type == "Call":
            return np.maximum(s - self.strike, 0)

    @abstractmethod
    def price(self, *args):
        pass

    def delta(self, *args, eps=0.01):
        """Calculate delta of the option

        Args:
            *args: QQ
            eps (float): finite difference derivative. Defaults to 0.01
        Returns:
            delta of the option
        """
        # Fix the random generator seed for MonteCarlo pricing
        st = np.random.get_state()
        price1 = self.price(*args)
        self.asset.price += eps
        np.random.set_state(st)
        price2 = self.price(*args)
        self.asset.price -= eps
        return (price2 - price1) / eps

    def gamma(self, *args, eps=2.0):
        """Calculate gamma of the option

        Args:
            *args:
            eps (float): finite difference derivative. Defaults to 2.0
        Returns:
            gamma of the option
        """
        # Fix the random generator seed for MonteCarlo pricing
        st = np.random.get_state()
        price1 = self.price(*args)
        np.random.set_state(st)
        self.asset.price += eps
        price2 = self.price(*args)
        np.random.set_state(st)
        self.asset.price -= 2 * eps
        price3 = self.price(*args)
        self.asset.price += eps
        return (price2 + price3 - 2.0 * price1) / eps ** 2

    def vega(self, *args, eps=0.01):
        """Calculate vega of the option

        Args:
            *args:
            eps (float): finite difference derivative. Defaults to 0.01
        Returns:
            vega of the option
        """
        # Fix the random generator seed for MonteCarlo pricing
        st = np.random.get_state()
        price1 = self.price(*args)
        self.asset.vol += eps
        np.random.set_state(st)
        price2 = self.price(*args)
        self.asset.vol -= eps
        return 0.01 * (price2 - price1) / eps

    def rho(self, *args, eps=0.01):
        """Calculate rho of the option

        Args:
            *args:
            eps (float): finite difference derivative. Defaults to 0.01
        Returns:
            rho of the option per 1% change in interest rate
        """
        # Fix the random generator seed for MonteCarlo pricing
        st = np.random.get_state()
        price1 = self.price(*args)
        self.interest_rate += eps
        np.random.set_state(st)
        price2 = self.price(*args)
        self.interest_rate -= eps
        return 0.01 * (price2 - price1) / eps

    def theta(self, *args, eps=0.001):
        """Calculate theta of the option

        Args:
            *args:
            eps (float): finite difference derivative. Defaults to 0.001
        Returns:
            theta of the option per calendar day
        """
        # Fix the random generator seed for MonteCarlo pricing
        st = np.random.get_state()
        price1 = self.price(*args)
        np.random.set_state(st)
        self.term -= eps
        price2 = self.price(*args)
        self.term += eps
        return (price1 - price2) / (365 * eps)


class EuropeanOption(Option):
    """Form a European Option of a given Asset.

    Attributes:
        option_type (str): "Call" or "Put"
        asset (Asset): underlying asset of the option
        strike (float): option strike price
        term (float): option maturity time
        interest_rate (float): risk free interest rate
        pricing method (str): the main pricing method for the option, "Binomial", "MonteCarlo" or "Analytic"
    """
    def Foo(self, sigma):
        asset = Asset(self.asset.price, sigma)
        option = EuropeanOption(self.option_type, asset, self.strike, self.term, self.interest_rate, "Analytic")
        return option.price()

    def analytic(self):
        """Price a European option with Black-Scholes-Merton analytic formula.

        It assumes a constant volatility.
        """
        vol = self.asset.vol(1.0, 0.0)
        d1 = (np.log(self.asset.price / self.strike)
              + (self.interest_rate + vol ** 2 / 2) * self.term) \
             / (vol * np.sqrt(self.term))
        d2 = d1 - vol * np.sqrt(self.term)
        if self.option_type == "Put":
            return self.strike * np.exp(-self.interest_rate * self.term) * scipy.stats.norm.cdf(-d2) \
                   - self.asset.price * scipy.stats.norm.cdf(-d1)
        elif self.option_type == "Call":
            return - self.strike * np.exp(-self.interest_rate * self.term) * scipy.stats.norm.cdf(d2) \
                   + self.asset.price * scipy.stats.norm.cdf(d1)

    def binomial(self, step=1000):
        """Price a European option with Binomial tree method

        Args:
            step (int): number of layers in the tree. Defaults to 1000.
        Returns:
            price of the European option
        """
        dt = 1.0 * self.term / step
        a = np.exp(self.interest_rate * dt)
        # making the binomial tree as a matrix. [i,j] element represents the price after i up movements and j
        # down movements in price of the asset. Elements with i+j=n correspond to the nth layer of the tree.

        # making the asset price tree.
        asset_tree = np.zeros((step + 1, step + 1))
        asset_tree[0, 0] = self.asset.price
        # evaluating elements resulting from only up movements in the asset price ([i,0]).
        for up_mov in range(1, step + 1):
            # up movement factor in price of the asset
            u = np.exp(self.asset.vol(asset_tree[up_mov - 1, 0], (up_mov - 1) * dt) * np.sqrt(dt))
            asset_tree[up_mov, 0] = asset_tree[up_mov - 1, 0] * u
        # evaluating elements resulting from only down movements in the asset price ([0,i]).
        for down_mov in range(1, step + 1):
            # down movement factor in price of the asset
            d = np.exp(-self.asset.vol(asset_tree[0, down_mov - 1], (down_mov - 1) * dt) * np.sqrt(dt))
            asset_tree[0, down_mov] = asset_tree[0, down_mov - 1] * d
        # evaluating the whole asset_tree, going by each layer of the tree
        for layer in range(1, step + 1):
            for up_mov in range(1, layer):
                down_mov = layer - up_mov
                u = np.exp(self.asset.vol(asset_tree[up_mov - 1, down_mov], (layer - 1) * dt) * np.sqrt(dt))
                d = np.exp(-self.asset.vol(asset_tree[up_mov, down_mov - 1], (layer - 1) * dt) * np.sqrt(dt))
                # taking the average of the price coming from one up movement or one down movement before. For
                # constant volatility these two are the same. For local volatility depending on asset price
                # they are different.
                asset_tree[up_mov, down_mov] = (u * asset_tree[up_mov - 1, down_mov] +
                                                d * asset_tree[up_mov, down_mov - 1]) / 2.0
        # making the option price tree.
        option_price = np.zeros((step + 1, step + 1))
        # pricing the option at maturity. [i,j] elements with i+j=step correspond to maturity.
        for up_mov in range(step + 1):
            option_price[up_mov, step - up_mov] = self.payoff(asset_tree[up_mov, step - up_mov])
        # pricing the option from the maturity layer backwards
        for layer in range(step - 1, -1, -1):
            for up_mov in range(layer + 1):
                down_mov = layer - up_mov
                u = np.exp(self.asset.vol(asset_tree[up_mov, down_mov], layer * dt) * np.sqrt(dt))
                d = np.exp(-self.asset.vol(asset_tree[up_mov, down_mov], layer * dt) * np.sqrt(dt))
                # probability of up movement in the price of the asset
                p = (a - d) / (u - d)
                option_price[up_mov, down_mov] = (p * option_price[up_mov + 1, down_mov]
                                                  + (1 - p) * option_price[up_mov, down_mov + 1]) / a
        return option_price[0, 0]

    def binomial_const_vol(self, step=1000):
        """Price a European option with Binomial tree method

        Args:
            step (int): number of layers in the tree. Defaults to 1000.
        Returns:
            price of the European option
        """
        vol = self.asset.vol(1.0, 0.0)
        dt = 1.0 * self.term / step
        a = np.exp(self.interest_rate * dt)
        # up movement factor in price of the asset
        u = np.exp(vol * np.sqrt(dt))
        # down movement factor in price of the asset
        d = np.exp(-vol * np.sqrt(dt))
        # probability of up movement in price of the asset
        p = (a - d) / (u - d)
        # making the binomial tree as a matrix. [i,j] element represents the option price after i up movements and j
        # down movements in price of the asset. Elements with i+j=n correspond to the nth layer of the tree.
        option_price = np.zeros((step + 1, step + 1))
        # pricing the option at maturity. [i,j] elements with i+j=step correspond to maturity.
        for up_mov in range(step + 1):
            option_price[up_mov, step - up_mov] = self.payoff(u ** up_mov * d ** (step - up_mov) * self.asset.price)
        # pricing the option from maturity layer backwards
        for layer in range(step - 1, -1, -1):
            for up_mov in range(layer + 1):
                down_mov = layer - up_mov
                option_price[up_mov, down_mov] = (p * option_price[up_mov + 1, down_mov]
                                                  + (1 - p) * option_price[up_mov, down_mov + 1]) / a
        return option_price[0, 0]

    def monte_carlo(self, number_simulation=10000, steps=100):
        """Price a European option using Monte Carlo method.

        Args:
            number_simulation (int): number of Monte Carlo simulations. Defaults to 10,000.
            steps (int): length of each Monte Carlo simulations. Defaults to 100.
        Returns:
            price of the European option
        """
        simulations = self.asset.simulate(self.term, self.interest_rate, number_simulation, steps)
        # list of asset prices at maturity QQ: should I include comments like this
        asset_price_list = simulations[:, -1]
        payoff_list = self.payoff(asset_price_list)
        price = np.exp(-self.interest_rate * self.term) * np.mean(payoff_list)
        return price

    def pde_pricing(self, dx, grid_size=4.0, ratio=1.0):
        """Price a European option by solving Black-Scholes-Merton differential equation

        Args:
            dx (float): spatial grid separation
            grid_size (float): size of spatial grid
            ratio (float): controlling time step size
        Returns:
            price of the European option
        """
        dt = ratio * dx ** 2
        t = self.term
        normalized_price = self.asset.price / self.strike
        k = 1.0
        x = np.arange(-grid_size, grid_size + dx, dx)
        if self.option_type == "Call":
            option_price = np.maximum(np.exp(x) - k, 0)
        else:
            option_price = np.maximum(k - np.exp(x), 0)
        while t > 0:
            temp_price = option_price
            t -= dt
            for i in range(1, len(x) - 1):
                sigma = self.asset.vol(np.exp(x[i]), t)
                option_price[i] = temp_price[i] * (1 - sigma ** 2 * dt / dx ** 2 - self.interest_rate * dt) \
                                  + 0.5 * temp_price[i + 1] * \
                                  (sigma ** 2 * dt / dx ** 2 + (self.interest_rate - 0.5 * sigma ** 2) * dt / dx) \
                                  + 0.5 * temp_price[i - 1] * \
                                  (sigma ** 2 * dt / dx ** 2 - (self.interest_rate - 0.5 * sigma ** 2) * dt / dx)
            if self.option_type == "Put":
                option_price[0] *= np.exp(-self.interest_rate * dt)
        x0 = np.log(normalized_price)
        price = self.strike * (option_price[x > x0][0] + option_price[x < x0][-1]) / 2.0
        return price

    def price(self, *args):
        """Return option price using the method specified by pricing_method"""
        if self.pricing_method == "Binomial":
            return self.binomial(*args)
        elif self.pricing_method == "MonteCarlo":
            return self.monte_carlo(*args)
        elif self.pricing_method == "Analytic":
            return self.analytic()


class AmericanOption(Option):
    """Form an AmericanOption of a given Asset.

    Attributes:
        option_type (str): "Call" or "Put"
        asset (Asset): underlying asset of the option
        strike (float): option strike price
        term (float): option maturity time
        interest_rate (float): risk free interest rate
    """

    def __init__(self, option_type, asset, strike, term, interest_rate):
        """Initialize AmericanOption"""
        Option.__init__(self, option_type, asset, strike, term, interest_rate, "Binomial")

    def binomial(self, step=1000):
        """Price an American option with Binomial tree method

        Args:
            step (int): number of layers in the tree. Defaults to 1000.
        Returns:
            price of the American option
        """
        dt = 1.0 * self.term / step
        a = np.exp(self.interest_rate * dt)
        # making the binomial tree as a matrix. [i,j] element represents the price after i up movements and j
        # down movements in price of the asset. Elements with i+j=n correspond to the nth layer of the tree.

        # making the asset price tree.
        asset_tree = np.zeros((step + 1, step + 1))
        asset_tree[0, 0] = self.asset.price
        # evaluating elements resulting from only up movements in the asset price ([i,0]).
        for up_mov in range(1, step + 1):
            # up movement factor in price of the asset
            u = np.exp(self.asset.vol(asset_tree[up_mov - 1, 0], (up_mov - 1) * dt) * np.sqrt(dt))
            asset_tree[up_mov, 0] = asset_tree[up_mov - 1, 0] * u
        # evaluating elements resulting from only down movements in the asset price ([0,i]).
        for down_mov in range(1, step + 1):
            # down movement factor in price of the asset
            d = np.exp(-self.asset.vol(asset_tree[0, down_mov - 1], (down_mov - 1) * dt) * np.sqrt(dt))
            asset_tree[0, down_mov] = asset_tree[0, down_mov - 1] * d
        # evaluating the whole asset_tree, going by each layer of the tree
        for layer in range(1, step + 1):
            for up_mov in range(1, layer):
                down_mov = layer - up_mov
                u = np.exp(self.asset.vol(asset_tree[up_mov - 1, down_mov], (layer - 1) * dt) * np.sqrt(dt))
                d = np.exp(-self.asset.vol(asset_tree[up_mov, down_mov - 1], (layer - 1) * dt) * np.sqrt(dt))
                # taking the average of the price coming from one up movement or one down movement before. For
                # constant volatility these two are the same. For local volatility depending on asset price
                # they are different.
                asset_tree[up_mov, down_mov] = (u * asset_tree[up_mov - 1, down_mov] +
                                                d * asset_tree[up_mov, down_mov - 1]) / 2.0
        # making the option price tree.
        option_price = np.zeros((step + 1, step + 1))
        # pricing the option at maturity. [i,j] elements with i+j=step correspond to maturity.
        for up_mov in range(step + 1):
            option_price[up_mov, step - up_mov] = self.payoff(asset_tree[up_mov, step - up_mov])
        # pricing the option from the maturity layer backwards
        for layer in range(step - 1, -1, -1):
            for up_mov in range(layer + 1):
                down_mov = layer - up_mov
                u = np.exp(self.asset.vol(asset_tree[up_mov, down_mov], layer * dt) * np.sqrt(dt))
                d = np.exp(-self.asset.vol(asset_tree[up_mov, down_mov], layer * dt) * np.sqrt(dt))
                # probability of up movement in the price of the asset
                p = (a - d) / (u - d)
                option_price[up_mov, down_mov] = (p * option_price[up_mov + 1, down_mov]
                                                  + (1 - p) * option_price[up_mov, down_mov + 1]) / a
                current_payoff = self.payoff(asset_tree[up_mov, down_mov])
                option_price[up_mov, down_mov] = max(current_payoff, option_price[up_mov, down_mov])
        return option_price[0, 0]

    def price(self, *args):
        return self.binomial(*args)


class AsianOption(Option):
    """Form an AsianOption of a given Asset.

    Attributes:
        option_type (str): "Call" or "Put"
        asset (Asset): underlying asset of the option
        strike (float): option strike price
        term (float): option maturity time
        interest_rate (float): risk free interest rate
        average_period (float): duration of the average period before maturity
    """

    def __init__(self, option_type, asset, strike, term, interest_rate, average_period):
        assert average_period <= term, "average_period should be less than or equal term"
        self.average_period = average_period
        Option.__init__(self, option_type, asset, strike, term, interest_rate, "MonteCarlo")

    def monte_carlo(self, number_simulation=10000, steps=100):
        """Price an Asian option using Monte Carlo method.

        Args:
            number_simulation (int): number of Monte Carlo simulations. Defaults to 10,000.
            steps (int): length of each Monte Carlo simulations. Defaults to 100.
        Returns:
            price of the Asian option
        """
        simulations = self.asset.simulate(self.term, self.interest_rate, number_simulation, steps)
        # index in simulations corresponding to the beginning of average_period time
        average_period_index = 1 + int((1 - self.average_period / self.term) * steps)
        # asset price list during average_period
        average_period_asset_price_list = simulations[:, average_period_index:]
        # list of average asset price during average_period
        average_period_average_asset_price_list = np.mean(average_period_asset_price_list, axis=1)
        payoff_list = self.payoff(average_period_average_asset_price_list)
        price = np.exp(-self.interest_rate * self.term) * np.mean(payoff_list)
        return price

    def monte_carlo_error(self, number_simulation=10000, steps=100, sample=10):
        """Calculate the error of MonteCarlo pricing method
        Args:
            number_simulation (int): number of Monte Carlo simulations. Defaults to 10,000.
            steps (int): length of each Monte Carlo simulations. Defaults to 100.
            sample (int): number of MonteCarlo runs. Defaults to 10.
        Returns:
            price of the Asian option and its Monte Carlo error.
        """
        # QQ how about adding monte_carlo_error to Option?
        price_list = []
        for i in range(sample):
            price_list.append(self.monte_carlo(number_simulation, steps))
        return np.mean(price_list), np.std(price_list)

    def price(self, *args):
        return self.monte_carlo(*args)


ex_vol = 0.2
ex_asset_price = 100.0
ex_stock = Asset(ex_asset_price, ex_vol)

# ex_stock.sim_plot(0.02, 1.0, 0.01)

ex_strike = 100.0
ex_term = 1.0
ex_interest = 0.02

eu_call_analytic = EuropeanOption("Call", ex_stock, ex_strike, ex_term, ex_interest, "Analytic")
eu_call_binomial = EuropeanOption("Call", ex_stock, ex_strike, ex_term, ex_interest, "Binomial")
eu_call_MC = EuropeanOption("Call", ex_stock, ex_strike, ex_term, ex_interest, "MonteCarlo")
eu_put_analytic = EuropeanOption("Put", ex_stock, ex_strike, ex_term, ex_interest, "Analytic")
eu_put_binomial = EuropeanOption("Put", ex_stock, ex_strike, ex_term, ex_interest, "Binomial")
eu_put_MC = EuropeanOption("Put", ex_stock, ex_strike, ex_term, ex_interest, "MonteCarlo")
american_call = AmericanOption("Call", ex_stock, ex_strike, ex_term, ex_interest)
american_put = AmericanOption("Put", ex_stock, ex_strike, ex_term, ex_interest)

ex_average_period = 0.2
asian_call = AsianOption("Call", ex_stock, ex_strike, ex_term, ex_interest, ex_average_period)
asian_put = AsianOption("Put", ex_stock, ex_strike, ex_term, ex_interest, ex_average_period)

av_period_list = np.arange(0.01, 0.55, 0.05)
rate_list = [0.02, 0.04, 0.08]
sigma_list_list = []
type_list = ["Call", "Put"]
for ty in type_list:
    for rate in rate_list:
        price_list = []
        for av_period in av_period_list:
            asian_call = AsianOption(ty, ex_stock, ex_strike, ex_term, ex_interest, av_period)
            price_list.append(asian_call.monte_carlo_error()[0])
            print("hooray")
        sigma_list = []
        for price in price_list:
            vol1 = ex_vol
            vol2 = 0.001
            stock = Asset(ex_asset_price, vol1)
            eu_call = EuropeanOption(ty, stock, ex_strike, ex_term, ex_interest, "Analytic")
            price1 = eu_call.Foo(vol1)
            price2 = eu_call.Foo(vol2)
            price3 = eu_call.Foo(0.5*(vol1+vol2))
            while abs(price - price3) > 0.05:
                if price3 > price:
                    vol1 = 0.5*(vol1+vol2)
                else:
                    vol2 = 0.5*(vol1+vol2)
                price3 = eu_call.Foo(0.5*(vol1+vol2))
            sigma_list.append(0.5*(vol1+vol2))
        sigma_list_list.append(sigma_list)

plt.plot(av_period_list, sigma_list_list[0], 'b-', av_period_list, sigma_list_list[1], 'g-', av_period_list, sigma_list_list[2], 'r-')
plt.plot(av_period_list, sigma_list_list[3], 'b--', av_period_list, sigma_list_list[4], 'g--', av_period_list, sigma_list_list[5], 'r--')
plt.xlabel("% average period")
plt.ylabel("effective volatility")
plt.show()

# print(eu_call_analytic.pde_pricing(0.01, ratio=1.0))

# print("option price")
# print("European")
# print(eu_call_analytic.price(), eu_put_analytic.price())
# print(eu_call_binomial.price(), eu_put_binomial.price())
# print(eu_call_MC.price(), eu_put_MC.price())
# print("American")
# print(american_call.price(), american_put.price())
# print("Asian")
# print(asian_call.price(), asian_put.price(), "\n")
#
# print("delta")
# print(eu_call_analytic.delta(), eu_put_analytic.delta())
# print(eu_call_binomial.delta(), eu_put_binomial.delta())
# print(eu_call_MC.delta(), eu_put_MC.delta())
# print(american_call.delta(), american_put.delta())
# print(asian_call.delta(), asian_put.delta(), "\n")
#
# print("gamma")
# print(eu_call_analytic.gamma(), eu_put_analytic.gamma())
# print(eu_call_binomial.gamma(), eu_put_binomial.gamma())
# print(eu_call_MC.gamma(), eu_put_MC.gamma())
# print(american_call.gamma(), american_put.gamma())
# print(asian_call.gamma(), asian_put.gamma(), "\n")
#
# print("vega")
# print(eu_call_analytic.vega(), eu_put_analytic.vega())
# print(eu_call_binomial.vega(), eu_put_binomial.vega())
# print(eu_call_MC.vega(), eu_put_MC.vega())
# print(american_call.vega(), american_put.vega())
# print(asian_call.vega(), asian_put.vega(), "\n")
#
# print("rho")
# print(eu_call_analytic.rho(), eu_put_analytic.rho())
# print(eu_call_binomial.rho(), eu_put_binomial.rho())
# print(eu_call_MC.rho(), eu_put_MC.rho())
# print(american_call.rho(), american_put.rho())
# print(asian_call.rho(), asian_put.rho(), "\n")
#
# print("theta")
# print(eu_call_analytic.theta(), eu_put_analytic.theta())
# print(eu_call_binomial.theta(), eu_put_binomial.theta())
# print(eu_call_MC.theta(), eu_put_MC.theta())
# print(american_call.theta(), american_put.theta())
# print(asian_call.theta(), asian_put.theta(), "\n")

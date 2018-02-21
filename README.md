# option_pricer
pricing different kind of options through a variety of methods using python.

At the moment, the option pricer is restricted to European/American call/put options. I use three different methods for pricing:
1- analytic evaluation through Black-Scholes-Merton formula. This is mainly to judge the accuracy of the other methods, and is only viable for Eu options.
2- Monte Carlo method. This does not work for American options.
3- Binomial tree for both American and European options.

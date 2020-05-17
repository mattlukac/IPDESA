import autoencoder.poisson_decoder as pois 

Pois = pois.Decoder(u0=1, u1=2, f=-2)
soln = Pois.solve(100)
print(soln)

# We are allowed to take the data and bin it to construct the SMF
# Optimize for Poisson, so cannot use standard Levenberg-Marquardt, since this assumes Gaussian
# option 1: use algorithm from tutorial 7
# option 2: update Levenberg-Marquardt
# DO ANALYTICAL DERIVATIVES AND USE FOR MINIMIZATION ROUTINE
# DO FUNCTION IN LOG LIKELIHOOD, COULD LEAD TO NEGATIVE VALUES, CHECK FOR THIS

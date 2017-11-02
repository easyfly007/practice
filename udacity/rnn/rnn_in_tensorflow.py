# http://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html
import numpy as np
print("expected cross entropy loss if the model")
print("-learns neither dependency:", -(0.625 *np.log(0.625) + 0.375*np.log(0.375) ))

# learns first dependency only == 0.5191666
print("- learns first dependency: ", -0.5*(0.875*np.log(0.875) + 0.125*np.log(0.125))
	- 0.5*(0.625*np.log(0.625) + 0.375*np.log(0.375)))
print("- learns both dependencies: ", -0.50 * (0.75 * np.log(0.75) + 0.25 * np.log(0.25))
      - 0.25 * (2 * 0.50 * np.log (0.50)) - 0.25 * (0))



import numpy as np
import led
import config

if __name__ == '__main__':
	led.pixels = np.tile(1.0, (3, config.N_PIXELS))
	print(led.pixels)
	led.update()
	led.pixels = np.tile(0.0, (3, config.N_PIXELS))
	print(led.pixels)
	led.update()



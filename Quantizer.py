import math

# levels => number of quantizing levels
def quantize(X, Y, levels):
	observations = []
	quantizing_angle = 2 * math.pi / levels

	for i in range(1, len(X)):
		dx = X[i] - X[i - 1]
		dy = Y[i] - Y[i - 1]

		theta = math.atan2(dy, dx)
		if theta < 0:
			theta += 2 * math.pi

		level = math.floor(theta / quantizing_angle)

		observations.append(int(level))

	return observations

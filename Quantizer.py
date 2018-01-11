import math

# levels => number of quantizing levels
def quantize(X, Y, levels):
	levels *= 2
	observations = []
	quantizing_angle = 2 * math.pi / levels

	mapper = [0] * (levels)
	indx = 1

	#rotating the axes
	for i in range(2, levels, 2):
		mapper[i] = mapper[i-1] = indx
		indx = indx + 1

	for i in range(1, len(X)):
		if X[i] == X[i - 1] and Y[i] == Y[i - 1]:
			continue

		dx = X[i] - X[i - 1]
		dy = Y[i] - Y[i - 1]

		theta = math.atan2(dy, dx)
		if theta < 0:
			theta += 2 * math.pi

		level = math.floor(theta / quantizing_angle)
		cur_level = mapper[int(level)]

		#ignore 2 similar consecutive levels
		if len(observations) > 0 and cur_level == observations[len(observations) - 1]:
			continue
		observations.append(cur_level)

	return observations


import gmsh
import math



def create_mesh(filename, R, R_void, R_refinement, R0, x0, h, h2 ):
	gmsh.initialize()
	xc = R0[0]
	yc = R0[1]

	point1 = gmsh.model.geo.add_point(xc		, yc		, 0, h)         #Center of the domain
	point2 = gmsh.model.geo.add_point(xc + R	, yc		, 0, h)
	point3 = gmsh.model.geo.add_point(xc		, yc + R	, 0, h)
	point4 = gmsh.model.geo.add_point(xc - R	, yc		, 0, h)
	point5 = gmsh.model.geo.add_point(xc		, yc - R	, 0, h)

	circle1 = gmsh.model.geo.add_circle_arc(point2, point1, point3)
	circle2 = gmsh.model.geo.add_circle_arc(point3, point1, point4)
	circle3 = gmsh.model.geo.add_circle_arc(point4, point1, point5)
	circle4 = gmsh.model.geo.add_circle_arc(point5, point1, point2)

	External_boundary = gmsh.model.geo.add_curve_loop([circle1, circle2, circle3, circle4])

	# Creation of the voids

	xc1 = x0[0]
	yc1 = x0[1]

	point6 = gmsh.model.geo.add_point(xc1			, yc1			, 0, h2)         #Center of the domain
	point7 = gmsh.model.geo.add_point(xc1 + R_void	, yc1			, 0, h2)
	point8 = gmsh.model.geo.add_point(xc1			, yc1 + R_void	, 0, h2)
	point9 = gmsh.model.geo.add_point(xc1 - R_void	, yc1			, 0, h2)
	point10 = gmsh.model.geo.add_point(xc1			, yc1 - R_void	, 0, h2)

	circle5 = gmsh.model.geo.add_circle_arc(point7, point6, point8)
	circle6 = gmsh.model.geo.add_circle_arc(point8, point6, point9)
	circle7 = gmsh.model.geo.add_circle_arc(point9, point6, point10)
	circle8 = gmsh.model.geo.add_circle_arc(point10, point6, point7)

	void1 = gmsh.model.geo.add_curve_loop([circle5, circle6, circle7, circle8])


	domain = gmsh.model.geo.add_plane_surface([External_boundary, void1])

	distance = gmsh.model.mesh.field.add("Distance")
	gmsh.model.mesh.field.setNumbers(distance, "CurvesList", [External_boundary])
	gmsh.model.mesh.field.setNumbers(distance, "Sampling", [100])


	# // We then define a `Threshold' field, which uses the return value of the
	# // `Distance' field 1 in order to define a simple change in element size
	# // depending on the computed distances
	# //
	# // SizeMax -                     /------------------
	# //                              /
	# //                             /
	# //                            /
	# // SizeMin -o----------------/
	# //          |                |    |
	# //        Point         DistMin  DistMax
	
	resolution = h/1
	threshold = gmsh.model.mesh.field.add("Threshold")
	gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
	gmsh.model.mesh.field.setNumber(threshold, "LcMin", resolution)
	gmsh.model.mesh.field.setNumber(threshold, "LcMax", 20*resolution)
	gmsh.model.mesh.field.setNumber(threshold, "DistMin", h)
	gmsh.model.mesh.field.setNumber(threshold, "DistMax", h)


	ball = gmsh.model.mesh.field.add("Ball")
	gmsh.model.mesh.field.setNumber(ball, "Radius", R_refinement)
	gmsh.model.mesh.field.setNumber(ball, "VIn", h2)
	gmsh.model.mesh.field.setNumber(ball, "VOut", h)
	gmsh.model.mesh.field.setNumber(ball, "XCenter", 0.0)
	gmsh.model.mesh.field.setNumber(ball, "YCenter", 0.0)
	gmsh.model.mesh.field.setNumber(ball, "Thickness", 0.5)



	minimum = gmsh.model.mesh.field.add("Min")
	gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [threshold, ball])
	gmsh.model.mesh.field.setAsBackgroundMesh(minimum)


	# Create the relevant Gmsh data structures
	# from Gmsh model.
	gmsh.model.geo.synchronize()

	# Generate mesh:
	gmsh.model.mesh.generate()

	# Write mesh data:
	gmsh.write(filename + ".msh")
	gmsh.finalize()

def create_ellipsoid(filename, R, R_refinement, Rx_void, Ry_void, R0, x0, h, h2 ):
	gmsh.initialize()

	xc = R0[0]
	yc = R0[1]

	point1 = gmsh.model.geo.add_point(xc		, yc		, 0, h)         #Center of the domain
	point2 = gmsh.model.geo.add_point(xc + R	, yc		, 0, h)
	point3 = gmsh.model.geo.add_point(xc		, yc + R	, 0, h)
	point4 = gmsh.model.geo.add_point(xc - R	, yc		, 0, h)
	point5 = gmsh.model.geo.add_point(xc		, yc - R	, 0, h)

	circle1 = gmsh.model.geo.add_circle_arc(point2, point1, point3)
	circle2 = gmsh.model.geo.add_circle_arc(point3, point1, point4)
	circle3 = gmsh.model.geo.add_circle_arc(point4, point1, point5)
	circle4 = gmsh.model.geo.add_circle_arc(point5, point1, point2)

	External_boundary = gmsh.model.geo.add_curve_loop([circle1, circle2, circle3, circle4])

	# Creation of the voids


	radii = [Rx_void, Ry_void, 0.0]

	p = [
		gmsh.model.geo.addPoint(*x0, meshSize = h),
		gmsh.model.geo.addPoint(x0[0] + radii[0], x0[1]				, x0[2], meshSize = h),
		gmsh.model.geo.addPoint(x0[0]			, x0[1] + radii[1]	, x0[2], meshSize = h),
		gmsh.model.geo.addPoint(x0[0] - radii[0], x0[1]				, x0[2], meshSize = h),
		gmsh.model.geo.addPoint(x0[0]			, x0[1] - radii[1]	, x0[2], meshSize = h ),
	]
	c = [
		gmsh.model.geo.addEllipseArc(p[1], p[0], p[2], p[2]),  
		gmsh.model.geo.addEllipseArc(p[2], p[0], p[2], p[3]),
		gmsh.model.geo.addEllipseArc(p[3], p[0], p[2], p[4]),
		gmsh.model.geo.addEllipseArc(p[4], p[0], p[2], p[1]),
	]

	ll = [
		gmsh.model.geo.addCurveLoop([c[1], c[2], c[3], c[0]]),
	]


	domain = gmsh.model.geo.add_plane_surface([External_boundary, ll[0]])

	distance = gmsh.model.mesh.field.add("Distance")
	gmsh.model.mesh.field.setNumbers(distance, "CurvesList", [External_boundary])
	gmsh.model.mesh.field.setNumbers(distance, "Sampling", [100])


	resolution = h/1
	threshold = gmsh.model.mesh.field.add("Threshold")
	gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
	gmsh.model.mesh.field.setNumber(threshold, "LcMin", resolution)
	gmsh.model.mesh.field.setNumber(threshold, "LcMax", 20*resolution)
	gmsh.model.mesh.field.setNumber(threshold, "DistMin", h)
	gmsh.model.mesh.field.setNumber(threshold, "DistMax", h)


	ball = gmsh.model.mesh.field.add("Ball")
	gmsh.model.mesh.field.setNumber(ball, "Radius", R_refinement)
	gmsh.model.mesh.field.setNumber(ball, "VIn", h2)
	gmsh.model.mesh.field.setNumber(ball, "VOut", h)
	gmsh.model.mesh.field.setNumber(ball, "XCenter", 0.0)
	gmsh.model.mesh.field.setNumber(ball, "YCenter", 0.0)
	gmsh.model.mesh.field.setNumber(ball, "Thickness", 0.5)



	minimum = gmsh.model.mesh.field.add("Min")
	gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [threshold, ball])
	gmsh.model.mesh.field.setAsBackgroundMesh(minimum)


	# Create the relevant Gmsh data structures
	# from Gmsh model.
	gmsh.model.geo.synchronize()

	# Generate mesh:
	gmsh.model.mesh.generate()

	# Write mesh data:
	gmsh.write(filename + ".msh")
	gmsh.finalize()

def create_perturbed_void(filename, R, R0, R_void, wave_length, perturbation, R_refinement, h, h2 ):
	gmsh.initialize()

	xc = R0[0]
	yc = R0[1]

	point1 = gmsh.model.geo.add_point(xc		, yc		, 0, h)         #Center of the domain
	point2 = gmsh.model.geo.add_point(xc + R	, yc		, 0, h)
	point3 = gmsh.model.geo.add_point(xc		, yc + R	, 0, h)
	point4 = gmsh.model.geo.add_point(xc - R	, yc		, 0, h)
	point5 = gmsh.model.geo.add_point(xc		, yc - R	, 0, h)

	circle1 = gmsh.model.geo.add_circle_arc(point2, point1, point3)
	circle2 = gmsh.model.geo.add_circle_arc(point3, point1, point4)
	circle3 = gmsh.model.geo.add_circle_arc(point4, point1, point5)
	circle4 = gmsh.model.geo.add_circle_arc(point5, point1, point2)

	External_boundary = gmsh.model.geo.add_curve_loop([circle1, circle2, circle3, circle4])

	# Creation of the voids


	nturns = 1.
	npts = 100
	c = perturbation
	p = []
	for i in range(0, npts):
		theta = i * 2 * math.pi * nturns / npts
		r = R_void * (1 + (c/wave_length ) *  math.sin(wave_length * theta))
		p.append(gmsh.model.geo.addPoint(r * math.cos(theta), r * math.sin(theta), 0, h, 1000 + i))
		# p.append(1000 + i)


	q = [i for i in p]
	q.append(1000)

	s1 = gmsh.model.geo.addSpline(q)

	l1 = gmsh.model.geo.addCurveLoop([s1])


	# domain = gmsh.model.geo.add_plane_surface([External_boundary, ll[0]])
	domain = gmsh.model.geo.add_plane_surface([External_boundary, l1])

	distance = gmsh.model.mesh.field.add("Distance")
	gmsh.model.mesh.field.setNumbers(distance, "CurvesList", [External_boundary])
	gmsh.model.mesh.field.setNumbers(distance, "Sampling", [100])


	resolution = h/1
	threshold = gmsh.model.mesh.field.add("Threshold")
	gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
	gmsh.model.mesh.field.setNumber(threshold, "LcMin", resolution)
	gmsh.model.mesh.field.setNumber(threshold, "LcMax", 20*resolution)
	gmsh.model.mesh.field.setNumber(threshold, "DistMin", h)
	gmsh.model.mesh.field.setNumber(threshold, "DistMax", h)


	ball = gmsh.model.mesh.field.add("Ball")
	gmsh.model.mesh.field.setNumber(ball, "Radius", R_refinement)
	gmsh.model.mesh.field.setNumber(ball, "VIn", h2)
	gmsh.model.mesh.field.setNumber(ball, "VOut", h)
	gmsh.model.mesh.field.setNumber(ball, "XCenter", 0.0)
	gmsh.model.mesh.field.setNumber(ball, "YCenter", 0.0)
	gmsh.model.mesh.field.setNumber(ball, "Thickness", 0.5)



	minimum = gmsh.model.mesh.field.add("Min")
	gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [threshold, ball])
	gmsh.model.mesh.field.setAsBackgroundMesh(minimum)


	# Create the relevant Gmsh data structures
	# from Gmsh model.
	gmsh.model.geo.synchronize()

	# Generate mesh:
	gmsh.model.mesh.generate()

	# Write mesh data:
	gmsh.write(filename + ".msh")
	gmsh.finalize()
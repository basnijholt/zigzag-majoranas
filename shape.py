import kwant


class Shape:
    def __init__(self, shape=None):
        if shape is None:
            self.shape = lambda site: True
        else:
            self.shape = shape

    def __call__(self, site):
        return self.shape(site)

    def __add__(self, other_shape):
        return Shape(UNION(self.shape, other_shape.shape))

    def __sub__(self, other_shape):
        return Shape(DIFFERENCE(self.shape, other_shape.shape))

    def __mul__(self, other_shape):
        return Shape(INTERSECTION(self.shape, other_shape.shape))

    def inverse(self):
        return Shape(NOT(self.shape))

    def edge(self):
        return Shape(EDGE(self.shape))

    def outer_edge(self):
        return Shape(OUTER_EDGE(self.shape))

    def interior(self):
        return Shape(INTERIOR(self.shape))

    def __getitem__(self, i):
        if type(i) is tuple:
            dim = len(i)

            if dim == 2:
                curve_x = slice_func(i[0])

                def _shape_x(site):
                    x, y = site.pos
                    return curve_x(x)

                new_shape = self * Shape(_shape_x)

                curve_y = slice_func(i[1])

                def _shape_y(site):
                    x, y = site.pos
                    return curve_y(y)

                new_shape *= Shape(_shape_y)

            elif dim == 3:
                pass
        else:
            curve = slice_func(i)

            def _shape(site):
                x = site.pos
                return curve(x)

            new_shape = self * Shape(_shape)

        return new_shape


def NOT(shape):
    """returns not(shape)"""
    return lambda site: not shape(site)


def UNION(*shapes):
    """returns shape which is an OR function applied to shapes"""
    return lambda site: any(shape(site) for shape in shapes)


def INTERSECTION(*shapes):
    """returns shape which is an AND function applied to shapes"""
    return lambda site: all(shape(site) for shape in shapes)


def DIFFERENCE(shape_A, shape_B):
    """returns shape which is true when the site is in the first shape, but not in the second"""
    return lambda site: shape_A(site) and not shape_B(site)


def EDGE(shape):
    def _shape(site):
        sites = [
            ("wsite", [-1, 0]),
            ("esite", [1, 0]),
            ("nsite", [0, 1]),
            ("ssite", [0, -1]),
            ("nwsite", [-1, 1]),
            ("nesite", [1, 1]),
            ("swsite", [-1, -1]),
            ("sesite", [1, -1]),
        ]
        s = lambda x: kwant.builder.Site(site.family, site.tag + x)
        neighboring_sites = {k: s(x) for k, x in sites}
        return shape(site) and not all(map(shape, neighboring_sites.values()))

    return _shape


def OUTER_EDGE(shape):
    def _shape(site):
        sites = [
            ("wsite", [-1, 0]),
            ("esite", [1, 0]),
            ("nsite", [0, 1]),
            ("ssite", [0, -1]),
            ("nwsite", [-1, 1]),
            ("nesite", [1, 1]),
            ("swsite", [-1, -1]),
            ("sesite", [1, -1]),
        ]
        s = lambda x: kwant.builder.Site(site.family, site.tag + x)
        neighboring_sites = {k: s(x) for k, x in sites}
        return not shape(site) and any(map(shape, neighboring_sites.values()))

    return _shape


def INTERIOR(shape):
    return DIFFERENCE(shape, EDGE(shape))


def TRANSLATE(shape, vector):
    def _shape(site):
        translated_site = kwant.builder.Site(site.family, site.tag - vector)
        return shape(translated_site)

    return _shape


def below_curve(curve):
    def _shape(site):
        x, y = site.pos
        return y < curve(x)

    return Shape(_shape)


def above_curve(curve):
    return Shape(below_curve(curve).inverse())


def left_of_curve(curve):
    def _shape(site):
        x, y = site.pos
        return x < curve(y)

    return _shape


def right_of_curve(curve):
    return NOT(left_of_curve(curve))


def add_to_site_colors(site_colors, marked_sites, color):
    site_colors.update({site: color for site in marked_sites})


def site_color_function(site_color, syst):
    return [site_color[site] for site in syst.sites]


def slice_func(_slice):
    if _slice.start is None and _slice.stop is None:
        return lambda x: True
    elif _slice.stop is None:
        return lambda x: _slice.start <= x
    elif _slice.start is None:
        return lambda x: x < _slice.stop
    else:
        return lambda x: _slice.start <= x < _slice.stop

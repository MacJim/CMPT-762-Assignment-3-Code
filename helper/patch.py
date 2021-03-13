import typing


def get_crop_patch_axes(width: int, height: int, window_width: int, window_height: int, step: int) -> typing.List[typing.Tuple[int, int, int, int]]:
    """
    :param width:
    :param height:
    :param window_height:
    :param window_width:
    :param step:
    :return: (x0, x1, y0, y1)
    """
    if (width > window_width):
        xs = [[0, window_width]]
        while (xs[-1][1] < width):
            last_x = xs[-1]
            last_x0 = last_x[0]
            last_x1 = last_x[1]

            new_x0 = last_x0 + step
            new_x1 = last_x1 + step
            if (new_x1 > width):
                new_x0 -= (new_x1 - width)
                new_x1 = width

            xs.append([new_x0, new_x1])
    else:
        xs = [[0, width]]

    if (height > window_height):
        ys = [[0, window_height]]
        while (ys[-1][1] < height):
            last_y = ys[-1]
            last_y0 = last_y[0]
            last_y1 = last_y[1]

            new_y0 = last_y0 + step
            new_y1 = last_y1 + step
            if (new_y1 > height):
                new_y0 -= (new_y1 - height)
                new_y1 = height

            ys.append([new_y0, new_y1])
    else:
        ys = [[0, height]]

    return_value = []
    for y in ys:
        for x in xs:
            return_value.append(x + y)

    return return_value

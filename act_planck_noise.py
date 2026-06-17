def planck_noise(channel, shape, wcs):

    return noise

def act_noise(channel, shape, wcs, pa):

    return noise

def accurate_noise(telescope, channel, shape, wcs, pa):
    assert telescope in ["act", "planck"]

    if telescope == "planck":
        planck_noise(channel=channel, shape=shape, wcs=wcs)

    else:
        act_noise(channel=channel, shape=shape, wcs=wcs, pa=pa)
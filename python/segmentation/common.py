def otsu(hist):
    total = sum(hist)
    sumB = 0
    wB = 0
    maximum = 0.0
    sum1 = sum([i*j for i,j in enumerate(hist)])
    for i in range(0, len(hist)):
        wB += hist[i]
        wF = total - wB
        if wB == 0 or wF == 0:
            continue
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sum1 - sumB) / wF
        varBetween = wB * wF * (mB - mF) ** 2
        if varBetween > maximum:
            maximum = varBetween
            level = i
    return level
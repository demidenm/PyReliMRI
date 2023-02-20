from numpy import cos, pi, sqrt, logical_and


def tetrachoric_corr(img1, img2):
    """
    Calculates the tetrachoric correlation between two binary vectors, IMG1 and IMG2.

    :param img1: A 1D binary numpy array of length n representing the 1st dichotomous (1/0) variable.
    :param img2: A 1D binary numpy array of length n representing the 2nd dichotomous (1/0) variable..

    Returns: The tetrachoric correlation between the two binary variables.
   """

    assert len(img1) == len(img2), 'Input vectors must have the same length' \
                                   'IMG1 length: {} and IMG2 length: {}'.format(len(img1), len(img2))

    # frequencies of the four possible combinations of IMG1 and IMG2
    A = sum(logical_and(img1 == 0, img2 == 0))  # Both img values are 0
    B = sum(logical_and(img1 == 0, img2 == 1))  # IMG1 is 0, IMG2 is 1
    C = sum(logical_and(img1 == 1, img2 == 0))  # IMG1 is 1, IMG2 is 0
    D = sum(logical_and(img1 == 1, img2 == 1))  # Both variables are 1

    AD = A*D

    return cos(pi/(1+sqrt(AD/B/C))).astype('float64')

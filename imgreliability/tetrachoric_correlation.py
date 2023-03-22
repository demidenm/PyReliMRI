from numpy import cos, pi, sqrt, logical_and, ndarray, nan


def tetrachoric_corr(vec1: ndarray, vec2: ndarray) -> float:
    """
    Calculates the tetrachoric correlation between two binary vectors, vec1 and vec2.

    :param vec1: A 1D binary numpy array of length n representing the 1st dichotomous (1/0) variable.
    :param vec2: A 1D binary numpy array of length n representing the 2nd dichotomous (1/0) variable..

    Returns: The tetrachoric correlation between the two binary variables.
   """
    assert len(vec1) > 0, f"Image 1: ({vec1}) is empty, length should be > 0"
    assert len(vec2) > 0, f"Image 2: ({vec1}) is empty, length should be > 0"
    assert len(vec1) == len(vec2), (
        'Input vectors must have the same length. ',
        f'vec1 length: {len(vec1)} and vec2 length: {len(vec2)}'
    )

    # check for exact replicas
    if (vec1 == vec2).all():
        
        return 1.0
    
    # frequencies of the four possible combinations of vec1 and vec2
    A = sum(logical_and(vec1 == 0, vec2 == 0))
    B = sum(logical_and(vec1 == 0, vec2 == 1))
    C = sum(logical_and(vec1 == 1, vec2 == 0))
    D = sum(logical_and(vec1 == 1, vec2 == 1))

    AD = A*D

    if B == 0 or C == 0:
        return nan
    
    return cos(pi/(1+sqrt(AD/B/C)))

import numpy as np

def editdistance(x, y, insert_cost=1, delete_cost=1, substitute_cost=1):
    """
    Compute weighted edit distance between sequences x and y.
    
    >>> editdistance(['a', 'b', 'c'], ['a', 'b'])
    1
    >>> editdistance(['a'], ['b'])
    1
    >>> editdistance([], [])
    0
    >>> editdistance(['a', 'b'], ['a', 'b'], 1, 1, 1)
    0
    >>> editdistance(['a', 'b', 'c'], ['b', 'd', 'c'])
    2
    >>> editdistance(['a'], ['b'], insert_cost=10, delete_cost=1, substitute_cost=1)
    1
    >>> editdistance(['a', 'b'], ['a', 'b', 'c'], insert_cost=5, delete_cost=1, substitute_cost=1)
    5
    >>> editdistance(['a', 'b'], ['c', 'd'], insert_cost=1, delete_cost=1, substitute_cost=10)
    4
    >>> editdistance(['a', 'b'], ['a', 'd'], insert_cost=1, delete_cost=1, substitute_cost=2.0)
    2.0
    """    
    m, n = len(x), len(y)
    # Infer dtype from weights
    dtype = np.result_type(type(insert_cost), type(delete_cost), type(substitute_cost))

    prev_row = np.arange(n + 1, dtype=dtype) * insert_cost
    curr_row = np.arange(n + 1, dtype=dtype) * insert_cost

    for i in range(1, m + 1):
        # fill row
        curr_row[0] = i * delete_cost  # Reset first column
        for j in range(1, n + 1):
            match_cost = 0 if x[i-1] == y[j-1] else substitute_cost
            curr_row[j] = min(
                prev_row[j]   + delete_cost,      # delete from A
                curr_row[j-1] + insert_cost,      # insert from B
                prev_row[j-1] + match_cost        # substitute or match
            )
        curr_row, prev_row = prev_row, curr_row

    # swap back, because we have just swapped
    curr_row, prev_row = prev_row, curr_row
    return curr_row[-1]

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
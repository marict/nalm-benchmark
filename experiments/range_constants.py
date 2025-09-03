# Range mapping for shorthand notation
RANGE_MAPPING = {
    "sym": ([-2, 2], [[-6, -2], [2, 6]]),          # symmetric around 0
    "neg": ([-2, -1], [-6, -2]),                   # negative moderate  
    "pos": ([1, 2], [2, 6]),                       # positive moderate
    "n10": ([-1.2, -1.1], [-6.1, -1.2]),         # negative narrow (around -1.1)
    "p01": ([0.1, 0.2], [0.2, 2]),                # positive small (0.1-0.2)
    "n01": ([-0.2, -0.1], [-2, -0.2]),           # negative small (-0.2 to -0.1)
    "p11": ([1.1, 1.2], [1.2, 6]),               # positive narrow (around 1.1)
    "n20": ([-20, -10], [-40, -20]),             # negative large (-20 to -10)
    "p20": ([10, 20], [20, 40]),                 # positive large (10-20)
}
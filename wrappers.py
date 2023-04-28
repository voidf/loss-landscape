def lerp(A, B, t): # 0 -> A, 1 -> B
    return A + (B - A) * t

def cat(*args): return '/'.join(args)

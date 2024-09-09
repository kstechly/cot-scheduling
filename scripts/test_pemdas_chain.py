from domain_utils import pemdas

r = pemdas.raw_eq_to_str

x = [[2, '1'], [3, '*'], [3, '+'], [1, '*'], [1, '*'], [9, '/'], [1, '+'], [3, '-'], [4, '*'], [4, '+'], [1, '+'], [1, '*'], [9, '/'], [6, '*'], [1, '*'], [3, '+']]

print(r(x))

y = {"raw_instance":x}
print(pemdas.generate_thoughts_basic(y))
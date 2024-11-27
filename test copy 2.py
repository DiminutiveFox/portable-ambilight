import timeit
import math

number = 222

# Method 1: String Conversion and Length Check
def check_digits_string(number):
    return len(str(number)) == 3

# Method 2: Range Check
def check_digits_range(number):
    return 100 <= number <= 999

# Method 3: Logarithm Method
def check_digits_log(number):
    return math.floor(math.log10(number)) + 1 == 3

# Method 4: Integer Division and Remainder
def check_digits_division(number):
    count = 0
    while number != 0:
        number //= 10
        count += 1
    return count == 3

# Benchmarking each method
time_string = timeit.timeit("check_digits_string(222)", globals=globals(), number=1000000)
time_range = timeit.timeit("check_digits_range(222)", globals=globals(), number=1000000)
time_log = timeit.timeit("check_digits_log(222)", globals=globals(), number=1000000)
time_division = timeit.timeit("check_digits_division(222)", globals=globals(), number=1000000)

print(f"String Conversion: {time_string:.6f} seconds")
print(f"Range Check: {time_range:.6f} seconds")
print(f"Logarithm Method: {time_log:.6f} seconds")
print(f"Integer Division: {time_division:.6f} seconds")

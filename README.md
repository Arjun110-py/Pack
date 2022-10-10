# Pack
## How To Run Your pack file
python interpreter.py
## Examples
##### These are not all implemented yet.
### Hello World File
`
print("Hello World")
`

### Fibonacci Sequence

```
func fibonacci(n) {
    # This expects the "n" parameter to be over 2 or else it will throw an error
    numbers = [0, 1] # 0 and 1 are the starting numbers of the fibonacci sequence
    let prev1 = 0
    let prev2 = 1

    for _ in range(n - 2) {
        new_number = prev1 + prev2
        numbers.append(new_number)
        prev1 = prev2
        prev2 = new_number
    }
    return(numbers)
}

let n = int(input("How many numbers do you want? "))
print(fibonacci(n)) # prints "[0, 1, 1, 2, 3, 5]" if n is 5
```
# Update log
**v0.0**

- math

**v0.1**
###### *This is not available yet.

- strings
- variables
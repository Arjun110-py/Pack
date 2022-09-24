# Pack
## How To Run Your pack file
With file dialog: python interpreter.py

With filename: python interpreter.py (file name)
## Examples
##### These are not all implemented yet.
### Hello World File
`
print("Hello World")
`

### Fibonacci Sequence

```
let prev1 = 0              
let prev2 = 1              
print(prev1)               
print(prev2)               
                           
func add(a, b) {           
    return(a + b)          
}                          
                           
for(i) in range(10) {      
    nth = add(prev1, prev2)
    print(nth)             
    prev2 = nth            
    prev1 = prev2          
}                          ```
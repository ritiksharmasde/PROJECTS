# # part 1 question 1
print("Ritik sharma \n MCA B5")


# import string
str1=input("enter the first name ")
str2=input("enter the second name ")
print((str2[::-1] + " "+ str1[::-1]))

# Take input from the user
user_input = input("Enter a number: ")

# Print the input in different data types
print("Integer:", int(user_input))
print("Float:", float(user_input))
print("Complex:", complex(user_input))




# area of rectangle
length=float(input("Enter the lenght of rectangle : "))
width=float(input("Enter the width of rectangle : "))
area=length*width
# using format 
print(f"The area is {area:.2f}")


# average of 3 numbers
num1=float(input("enter the first number :"))

num2=float(input("enter the second number :"))

num3=float(input("enter the third number :"))

avg=(num1+num2+num3)/3
print("The average of given number is %.2f" %avg)

# conditional and loop statement
while True:
    val = int(input("Enter the value (type -1 to exit): "))
    
    if val == -1:  # Exit condition, you can change -1 to any number you want as an exit signal
        break
    

        continue
    
    elif val > 0:
        print("Positive")
        
    elif val < 0:
        print("Negative")
        
    elif val == 0:
        print("Zero")
    else:
        continue


# # odd even numbers 
val1 = int(input("Enter the first value :"))
                 
val2 = int(input("Enter the  second value :"))
        
if(val1%2==0 and val2%2==0):
    print("both are even")

elif(val1%2!=0 and val2%2!=0):
    print("both are odd")
elif(val1%2!=0 or val2%2!=0):
    print("each one of them is odd")

#  value in different number systems
# Take integer input from the user
num = int(input("Enter an integer: "))

# List of formats to convert the number into
formats = [('Binary', bin), ('Hexadecimal', hex), ('Octal', oct)]

# Loop through each format and print the result
for name, func in formats:
    print(f"{name}: {func(num)[2:].upper()}")


import turtle
wn = turtle.Screen()
wn.bgcolor("red")
tess = turtle.Turtle()
tess.color("white")
tess.shape("turtle")

# print(list(range(5, 60, 2)))
# tess.up()                     
for size in range(100):   
    tess.stamp()              
    tess.forward(size)         
    tess.left(24)       

wn.exitonclick()

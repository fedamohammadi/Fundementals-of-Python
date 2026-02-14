
#Goal 1
# TODO Ask user for their birth year

my_birth_year = int(input("Enter your birth year: "))

zodiac_animals = {
    "Dragon": [1976, 1988, 2000],
    "Snake": [1977, 1989, 2001],
    "Horse": [1978, 1990, 2002],
    "Goat": [1979, 1991, 2003],
    "Tiger": [1974, 1986, 1998],
    "Fox": [1973, 1985, 1997],
    "Dog": [1982, 1994, 2006],
    "Monkey": [1980, 1992, 2004],
    "Rabbit": [1975, 1987, 1999],
    "Pig": [1983, 1995, 2007],
    "Rooster": [1981, 1993, 2005]
}

def get_zodiac_animal(year):
    for animal, years in zodiac_animals.items():
        if year in years:
            return animal
    return None

user_animal = get_zodiac_animal(my_birth_year)



if user_animal:
    print(f"Your birth year is: {my_birth_year}!")
    print(f"Your Chinese Zodiac animal is: {user_animal}!")



if user_animal == "Dragon":
        print("Dragons are clever and resourceful!")
elif user_animal == "Fox":
        print("Foxes are strong!")
elif user_animal == "Tiger":
        print("Tigers are brave and dangerous!")
elif user_animal == "Rabbit":
        print("Rabbits are kind!")
elif user_animal == "Snake":
        print("Snakes are wise!")
elif user_animal == "Horse":
        print("Horses are energetic!")
elif user_animal == "Goat":
        print("Goats climb every mountain!")
elif user_animal == "Monkey":
        print("Monkeys are fun!")
elif user_animal == "Rooster":
        print("Roosters are confident and hardworking!")
elif user_animal == "Dog":
        print("Dogs are loyal and friendly!")
elif user_animal == "Pig":
        print("Pigs are honest and generous!")
else:
    print("Hmm, something went wrong with your birth year. Try again!")



# Goal 2
# TODO Ask the user for your friend's birth year

friend_birth_year = int(input("Enter your friend's birth year: "))
friend_animal = get_zodiac_animal(friend_birth_year)

if friend_animal:
    print(f"Your friend's birth year is: {friend_birth_year}!")
    print(f"Your friend's Chinese Zodiac animal is: {friend_animal}!")



if friend_animal == "Dragon":
        print("Dragons are powerful!")
elif friend_animal == "Fox":
        print("Foxes are cunning!")
elif friend_animal == "Tiger":
        print("Tigers are brave!")
elif friend_animal == "Rabbit":
        print("Rabbits are kind!")
elif friend_animal == "Snake":
        print("Snakes are wise!")
elif friend_animal == "Horse":
        print("Horses are energetic!")
elif friend_animal == "Goat":
        print("Goats are determined!")
elif friend_animal == "Monkey":
        print("Monkeys are playful!")
elif friend_animal == "Rooster":
        print("Roosters are hardworking!")
elif friend_animal == "Dog":
        print("Dogs are loyal!")
elif friend_animal == "Pig":
        print("Pigs are generous!")
else:
        print("Hmm, something went wrong with your friend's birth year. Try again!")


# (Required) Goal 3
# TODO Check for compatibility between your birth year and your friend's birth year


compatibility_chart = {
    "Good match": ["Tiger", "Rabbit"],
    "Bad match": ["Dragon", "Goat"]
}

def get_friend_zodiac_animal(year):
    if year in range(1986, 1999, 12):
        return "Tiger"
    elif year in range(1987, 2000, 12):
        return "Rabbit"
    elif year in range(1988, 2001, 12):
        return "Dragon"
    elif year in range(1991, 2003, 12):
        return "Goat"
    else:
        return None



if friend_animal in compatibility_chart["Good match"]:
    print("Good match! You two are friends, like Dog and Rabbit!")
elif friend_animal in compatibility_chart["Bad match"]:
    print("Bad match... You are not friends like Dog and Dragon.")
else:
    print("You're in between! The start is always hard, but there's hope!")

    
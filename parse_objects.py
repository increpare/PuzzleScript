import json

# Sample dictionary (you can replace this with loading from a JSON file)
objects_dict = {
    "A": [
        {
            "color": "#8a8e91 #7c8184 #82888c",
            "pattern": [
                "11000",
                "02210",
                "22002",
                "10001",
                "02112"
            ]
        }
    ],
    "B": [
        {
            "color": "#ff0000 #00ff00 #0000ff",
            "pattern": [
                "11100",
                "10010",
                "11100",
                "10010",
                "11100"
            ]
        }
    ]
    # Add more objects as needed
}

with open("example_sprites.json", "r") as file:
    objects_dict = json.load(file)

# Iterate through the dictionary keys (OBJECT_NAME)
for obj_name in sorted(objects_dict.keys()):
    obj = objects_dict[obj_name][0]  # Get the first object in the list
    colors = obj["color"].split()  # Split the color string into individual colors
    pattern = obj["pattern"]  # Get the pattern list

    # Print the OBJECT_NAME
    print(obj_name)
    # Print the colors
    print(" ".join(colors))
    # Print the pattern
    for line in pattern:
        print(line)
    print()  # Add a blank line for separation

import re
from numpy import pi

def parse_pi_str(text: str, default: str="2*pi"):
    if text == "" or text is None:
        text = default
        print(f"returning default value of {default}")
    m = re.match(r"^(\d+(\.\d+)?)(\s*\*\s*pi)?$", text)
    if m is None:
        print(f"Invalid argument provided for -v/--v0: '{default}'. Assuming default value '{default}'")
        text = default
        m = re.match(r"^(\d+(\.\d+)?)(\s*\*\s*pi)?$", text)

    if m.lastindex != 3:
        return float(m[1])
    else:
        return float(m[1]) * pi


def get_user_input(prompt, input_type=int, valid_values=None, default=1):
    """
    Helper function to get user input.
    """
    while True:
        try:
            user_input = input(prompt)
            # If the input is empty, use the default value
            if user_input == "":
                print('returning default value of', default)
                return default
            user_input = input_type(user_input)
            if valid_values is None or user_input in valid_values:
                return user_input
            else:
                print("Invalid input. Please enter a valid values: "+str(valid_values))
        except ValueError:
            print("Invalid input type. Please enter a valid value.")

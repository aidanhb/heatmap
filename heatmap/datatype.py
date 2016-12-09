from enum import Enum

# Different methods will need to behave differently depending on the type of data they're working with.
# This Enum class makes it clear what sorts of data this class is designed to display.
# datatype.POINT signifies simple coordinate points without values, over time, such as phone calls
# datatype.VALUE signifies coordinate points with values attached over time, such as temperature samples
class datatype(Enum):
	POINT = 1
	VALUE = 2
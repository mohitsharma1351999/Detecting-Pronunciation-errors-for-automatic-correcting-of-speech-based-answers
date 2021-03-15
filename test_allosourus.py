import allosaurus 
from allosaurus.app import read_recognizer

# A Naive recursive Python program to fin minimum number
# operations to convert str1 to str2


def editDistance(str1, str2, m, n):
	if m == 0:
		return n
	if n == 0:
		return m
	if str1[m-1] == str2[n-1]:
		return editDistance(str1, str2, m-1, n-1)
	return 1 + min(editDistance(str1, str2, m, n-1), # Insert
				editDistance(str1, str2, m-1, n), # Remove
				editDistance(str1, str2, m-1, n-1) # Replace
				)


# load your model
model = read_recognizer()

# run inference -> æ l u s ɔ ɹ s
a = model.recognize('full_intersection/abducted/abducted_pid1.wav')
b = model.recognize('full_intersection/zealous/zealous_pid2.wav')
c = model.recognize('full_intersection/zealous/zealous_pid3.wav')
d = model.recognize('full_intersection/abel/abel_pid1.wav')
e = model.recognize('full_intersection/abel/abel_pid2.wav')


print('1---->', a, '2---->', b, '3---->', c, '4---->',d, '5---->',e)

print("editDistance between same words(zealous and zealous) -------->", editDistance(b, c, len(b), len(c)))
print("editDistance between different classes(abducted and abel) -------->", editDistance(a, d, len(a), len(d)))
print("editDistance between same words(able and able) -------->", editDistance(e, d, len(e), len(d)))

print("editDistance between different words(able and zealous) -------->", editDistance(e, b, len(e), len(b)))
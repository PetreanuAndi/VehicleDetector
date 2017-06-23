import os
import glob

car_folder = 'vehicles/vehicles/'
non_car_folder = 'non-vehicles/non-vehicles/'



def getDatabaseStruct():
	image_types = os.listdir(car_folder)
	cars = []

	for imtype in image_types:
		cars.extend(glob.glob(car_folder+imtype+'/*'))

	print('Number of vehicles found: ',len(cars))
	with open("cars.txt",'w') as f:
		for car in cars:
			f.write(car+'\n')

	image_types = os.listdir(non_car_folder)
	noncars = []

	for imtype in image_types:
		noncars.extend(glob.glob(non_car_folder+imtype+'/*'))

	print('Number of non-vehicles found: ',len(noncars))
	with open("notcars.txt",'w') as f:
		for noncar in noncars:
			f.write(noncar+'\n')

	return cars, noncars

def main():
	getDatabaseStruct()

if __name__ == '__main__':
    main()
import argparse

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', type=str)
	parser.add_argument('--output_dir', type=str)
	args = parser.parse_args()
	return args

if __name__ == '__main__':

	args = get_args()

	print(args.input_dir)
	print(args.output_dir)

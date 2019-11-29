import argparse

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', type=str)
	parser.add_argument('--output_dir', type=str)
	args = parser.parse_args()
	input_dir = args.input_dir
	output_dir = args.output_dir
	print('input_dir : ', input_dir)
	print('output_dir : ', output_dir)

if __name__ == '__main__':
	main()

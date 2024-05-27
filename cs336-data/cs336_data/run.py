from fastwarc import ArchiveIterator
import argparse
from cs336_data.utils import extract_text


def main(
        extract: bool,
        input_path: str,
        output_path: str
):
    with open(output_path, 'w') as f:
        for record in ArchiveIterator(open(input_path, 'rb')):
            output = str(record.http_headers)
            if extract:
                output += extract_text(record.reader.read())
            f.write(output + "\n\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    args = parser.parse_args()
    main(args.extract, args.input_path, args.output_path)
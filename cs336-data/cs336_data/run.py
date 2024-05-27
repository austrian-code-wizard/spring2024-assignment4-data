from fastwarc import ArchiveIterator
import argparse
from tqdm import tqdm
from cs336_data.utils import extract_text, identify_language


def main(
        extract: bool,
        identify: bool,
        input_path: str,
        output_path: str
):
    with open(output_path, 'w') as f:
        for record in tqdm(ArchiveIterator(open(input_path, 'rb')), desc="Processing records"):
            output = str(record.http_headers)
            content = record.reader.read()
            if identify:
                output += str(identify_language(extract_text(content)))
            if extract:
                output += extract_text(content)
            f.write(output + "\n\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--identify', action='store_true')
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    args = parser.parse_args()
    main(args.extract, args.identify, args.input_path, args.output_path)
from fastwarc import ArchiveIterator
import argparse
from tqdm import tqdm
from cs336_data.utils import extract_text, identify_language, mask_emails, mask_phone_numbers, mask_ipv4


def main(
        extract: bool,
        identify: bool,
        mask_pii: bool,
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
                content = extract_text(content)
            if mask_pii:
                content, _ = mask_emails(content)
                content, _ = mask_phone_numbers(content)
                content, _ = mask_ipv4(content)
            output += content
            f.write(output + "\n\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--identify', action='store_true')
    parser.add_argument('--mask-pii', action='store_true')
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    args = parser.parse_args()

    if args.mask_pii and not args.extract:
        raise ValueError("Cannot mask PII without extracting text")

    main(args.extract, args.identify, args.mask_pii, args.input_path, args.output_path)
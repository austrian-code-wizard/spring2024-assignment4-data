from fastwarc import ArchiveIterator
import argparse
from tqdm import tqdm
from cs336_data.utils import extract_text, identify_language, mask_emails, mask_phone_numbers, mask_ipv4, classify_nsfw, classify_toxic_speech, gopher_filters


def main(
        identify: bool,
        mask_pii: bool,
        classify: bool,
        gopher: bool,
        input_path: str,
        output_path: str
):
    with open(output_path, 'w') as f:
        for record in tqdm(ArchiveIterator(open(input_path, 'rb')), desc="Processing records"):
            output = str(record.http_headers)
            content = record.reader.read()
            content = extract_text(content)
            if identify:
                lang, conf = identify_language(content)
                if lang != '__label__en' or conf < 0.65:
                    continue
            if classify:
                nsf, conf = classify_nsfw(content)
                if nsf != '__label__non-nsfw' or conf < 0.9:
                    continue
                toxic, conf = classify_toxic_speech(content)
                if toxic != '__label__non-toxic' or conf < 0.9:
                    continue
            if gopher:
                filter_gopher = gopher_filters(content)
                if not filter_gopher:
                    continue
            if mask_pii:
                content, count1 = mask_emails(content)
                content, count2 = mask_phone_numbers(content)
                content, count3 = mask_ipv4(content)
                if count1 + count2 + count3 == 0:
                    continue
            output += content
            f.write(output + "\n\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--identify', action='store_true')
    parser.add_argument('--mask', action='store_true')
    parser.add_argument('--classify', action='store_true')
    parser.add_argument('--gopher', action='store_true')
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    args = parser.parse_args()

    main(args.identify, args.mask, args.classify, args.gopher, args.input_path, args.output_path)
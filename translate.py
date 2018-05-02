#!/usr/bin/env python
from __future__ import division, unicode_literals
import argparse

from onmt.translate.Translator import make_translator
import onmt.opts


def main(opt):
    translator = make_translator(opt, report_score=True)
    translator.translate(opt.src_dir, opt.src, opt.tgt,
                         opt.batch_size, opt.attn_debug,
                         aux_vec_path=opt.aux_vec_path, retrieved_path=opt.retrieved_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)

    opt = parser.parse_args()
    main(opt)

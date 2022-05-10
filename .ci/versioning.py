#!/usr/bin/env python3

import sys
(version_file, dev_version) = sys.argv[1:]

import versiontag
version = versiontag.get_version(pypi=True).strip()

with open(version_file, "r") as f:
    text = f.read()

text = text.replace(dev_version, version)

with open(version_file, "w") as f:
    f.write(text)

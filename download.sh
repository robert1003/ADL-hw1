#!/usr/bin/env bash

early_zip=140.112.90.197:9763/early.zip

wget "${early_zip}" -O ./temp.zip
unzip temp.zip
mv model.ckpt early/model.ckpt
rm temp.zip

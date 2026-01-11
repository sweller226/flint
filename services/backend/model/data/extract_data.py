import zstandard as zstd

input_file = "GLBX-20260110-WX9PCNR8JL/glbx-mdp3-20160109-20260108.ohlcv-1m.csv.zst"
output_file = "glbx-mdp3-20160109-20260108.ohlcv-1m.csv"

with open(input_file, "rb") as compressed, open(output_file, "wb") as decompressed:
    dctx = zstd.ZstdDecompressor()
    dctx.copy_stream(compressed, decompressed)

print(f"Decompressed to {output_file}")
from astropy.io import fits


def inspect_fits_metadata(file_path):
    """
    检查并打印FITS光谱文件的元信息（头信息和列名）
    """
    with fits.open(file_path) as hdul:
        print(f"文件包含 {len(hdul)} 个HDU (Header/Data Unit)\n")
        for i, hdu in enumerate(hdul):
            print(f"{'=' * 60}")
            print(f"HDU {i}: {type(hdu).__name__}")
            print(f"{'=' * 60}")
            header = hdu.header
            print(f"\n【头信息关键字段】")
            print(f"{'-' * 50}")
            for key in list(header)[:min(100, len(header))]:
                comment = header.comments[key]
                print(f"{key:8} = {header[key]!r:20} # {comment}")
            if hdu.data is not None:
                print(f"\n【数据摘要】")
                print(f"{'-' * 50}")
                print(f"数据形状: {hdu.data.shape}")
                print(f"数据类型: {hdu.data.dtype}")
                if hasattr(hdu, 'columns'):
                    print("\n【表格列信息】")
                    print(f"{'-' > 50}")
                    for col in hdu.columns:
                        unit = f" [{col.unit}]" if col.unit else ""
                        print(f"• {col.name:12}: {col.format}{unit}")

            print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    fits_file = "/home/haokai/stellar_parameters/l/data/fits_ceshi/spec-55859-F5902_sp01-029.fits"
    inspect_fits_metadata(fits_file)